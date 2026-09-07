from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass
class PortalVisualConfig:
    edge_thickness: int = 4
    glow_sigma: float = 14.0
    roi_margin: int = 24
    feather: int = 3
    chromatic_offset: int = 4
    glitch_strength: float = 0.8
    glow_scale: float = 0.5


class PortalRenderer:
    """Composite a rectangular multiverse window over the live camera."""

    def __init__(self, config: PortalVisualConfig | None = None) -> None:
        self.config = config or PortalVisualConfig()
        self._mask_cache: dict[tuple[int, ...], np.ndarray] = {}
        self._edge_cache: dict[tuple[int, int], np.ndarray] = {}
        self._glow_cache: dict[tuple[int, int], np.ndarray] = {}
        self._output_cache: dict[tuple[int, int], np.ndarray] = {}
        self._border_cache: dict[tuple[int, int], np.ndarray] = {}

    @staticmethod
    def _buffer(cache: dict[tuple[int, int], np.ndarray], shape: tuple[int, int, int]) -> np.ndarray:
        key = (shape[1], shape[0])
        buffer = cache.get(key)
        if buffer is None or buffer.shape != shape:
            buffer = np.empty(shape, dtype=np.uint8)
            cache[key] = buffer
        return buffer

    @staticmethod
    def _single(cache: dict[tuple[int, int], np.ndarray], shape: tuple[int, int]) -> np.ndarray:
        key = (shape[1], shape[0])
        buffer = cache.get(key)
        if buffer is None or buffer.shape != shape:
            buffer = np.empty(shape, dtype=np.uint8)
            cache[key] = buffer
        return buffer

    def render(self, frame, dimension, center, width, height, angle, timestamp_ms, intensity=1.0):
        if width <= 0 or height <= 0 or frame.ndim != 3:
            return frame
        intensity = max(0.0, min(1.0, float(intensity)))
        if intensity <= 0.0:
            return frame

        frame_h, frame_w = frame.shape[:2]
        cx, cy = int(round(center[0])), int(round(center[1]))
        half_w = max(1, int(round(width * 0.5)))
        half_h = max(1, int(round(height * 0.5)))
        margin = max(int(self.config.roi_margin), int(self.config.glow_sigma * 2.5))

        x0 = max(0, cx - half_w - margin)
        y0 = max(0, cy - half_h - margin)
        x1 = min(frame_w, cx + half_w + margin + 1)
        y1 = min(frame_h, cy + half_h + margin + 1)
        if x1 <= x0 or y1 <= y0:
            return frame

        local = frame[y0:y1, x0:x1]
        local_h, local_w = local.shape[:2]
        local_center = (cx - x0, cy - y0)

        target_w = max(2, int(round(width)))
        target_h = max(2, int(round(height)))
        if dimension.shape[1] != target_w or dimension.shape[0] != target_h:
            content = cv2.resize(dimension, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        else:
            content = dimension

        mask = self._rectangle_mask(local_w, local_h, local_center, target_w, target_h)
        output = self._buffer(self._output_cache, local.shape)
        output[:] = local

        px0 = max(0, int(round(local_center[0] - target_w * 0.5)))
        py0 = max(0, int(round(local_center[1] - target_h * 0.5)))
        px1 = min(local_w, px0 + target_w)
        py1 = min(local_h, py0 + target_h)
        sx0 = max(0, -int(round(local_center[0] - target_w * 0.5)))
        sy0 = max(0, -int(round(local_center[1] - target_h * 0.5)))
        if px1 > px0 and py1 > py0:
            patch = content[sy0:sy0 + (py1 - py0), sx0:sx0 + (px1 - px0)]
            alpha = mask[py0:py1, px0:px1].astype(np.float32)[:, :, None] / 255.0
            alpha *= intensity
            base = output[py0:py1, px0:px1].astype(np.float32)
            mixed = base * (1.0 - alpha) + patch.astype(np.float32) * alpha
            output[py0:py1, px0:px1] = np.clip(mixed, 0, 255).astype(np.uint8)

        border = self._single(self._border_cache, (local_h, local_w))
        border.fill(0)
        self._draw_border(border, local_center, target_w, target_h, timestamp_ms, intensity)
        glow = self._glow_from_border(border)
        self._add_glow(output, glow, intensity)
        self._composite_border(output, border, intensity)

        frame[y0:y1, x0:x1] = output
        return frame

    def _rectangle_mask(self, width, height, center, target_w, target_h):
        key = (width, height, target_w, target_h, int(center[0]), int(center[1]))
        mask = self._mask_cache.get(key)
        if mask is not None:
            return mask
        mask = np.zeros((height, width), dtype=np.uint8)
        cx, cy = int(round(center[0])), int(round(center[1]))
        x0 = max(0, int(round(cx - target_w * 0.5)))
        y0 = max(0, int(round(cy - target_h * 0.5)))
        x1 = min(width - 1, int(round(cx + target_w * 0.5)) - 1)
        y1 = min(height - 1, int(round(cy + target_h * 0.5)) - 1)
        if x1 >= x0 and y1 >= y0:
            cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)
            feather = max(0, int(self.config.feather))
            if feather:
                kernel = feather * 2 + 1
                mask[:] = cv2.GaussianBlur(mask, (kernel, kernel), feather * 0.55)
        self._mask_cache[key] = mask
        return mask

    def _draw_border(self, image, center, width, height, timestamp_ms, intensity):
        cx, cy = int(round(center[0])), int(round(center[1]))
        x0 = int(round(cx - width * 0.5))
        y0 = int(round(cy - height * 0.5))
        x1 = int(round(cx + width * 0.5))
        y1 = int(round(cy + height * 0.5))
        thickness = max(1, int(round(self.config.edge_thickness * (0.7 + 0.3 * intensity))))
        cv2.rectangle(image, (x0, y0), (x1, y1), 255, thickness, cv2.LINE_AA)

        offset = max(1, int(round(self.config.chromatic_offset * intensity)))
        cv2.rectangle(image, (x0 - offset, y0), (x1 - offset, y1), 135, 1, cv2.LINE_AA)
        cv2.rectangle(image, (x0 + offset, y0), (x1 + offset, y1), 190, 1, cv2.LINE_AA)

        phase = timestamp_ms * 0.01
        for i in range(5):
            edge = i % 4
            fraction = (math.sin(phase * (0.8 + i * 0.13) + i * 2.4) + 1.0) * 0.5
            if edge in (0, 2):
                length = max(12, int(width * (0.04 + 0.08 * fraction)))
                x = x0 + int((width - length) * fraction)
                yy = y0 if edge == 0 else y1
                cv2.line(image, (x, yy), (x + length, yy), 255, 1, cv2.LINE_AA)
            else:
                length = max(12, int(height * (0.04 + 0.08 * fraction)))
                y = y0 + int((height - length) * fraction)
                xx = x1 if edge == 1 else x0
                cv2.line(image, (xx, y), (xx, y + length), 255, 1, cv2.LINE_AA)

    def _glow_from_border(self, border):
        scale = min(max(float(self.config.glow_scale), 0.35), 1.0)
        h, w = border.shape
        if scale >= 0.999:
            blurred = cv2.GaussianBlur(border, (0, 0), max(1.0, self.config.glow_sigma))
            return blurred
        small_w = max(1, int(round(w * scale)))
        small_h = max(1, int(round(h * scale)))
        small = cv2.resize(border, (small_w, small_h), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), max(1.0, self.config.glow_sigma * scale))
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _add_glow(image, glow, intensity):
        image[:, :, 0] = cv2.addWeighted(image[:, :, 0], 1.0, glow, 0.48 * intensity, 0.0)
        image[:, :, 1] = cv2.addWeighted(image[:, :, 1], 1.0, glow, 0.34 * intensity, 0.0)
        image[:, :, 2] = cv2.addWeighted(image[:, :, 2], 1.0, glow, 0.60 * intensity, 0.0)

    @staticmethod
    def _composite_border(image, border, intensity):
        blue = np.roll(border, -3, axis=1)
        red = np.roll(border, 3, axis=1)
        image[:, :, 0] = cv2.addWeighted(image[:, :, 0], 1.0, blue, 0.38 * intensity, 0.0)
        image[:, :, 2] = cv2.addWeighted(image[:, :, 2], 1.0, red, 0.48 * intensity, 0.0)
