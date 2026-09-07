from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass
class PortalVisualConfig:
    edge_thickness: int = 4
    glow_sigma: float = 14.0
    roi_margin: int = 34
    feather: int = 4
    chromatic_offset: int = 4
    glitch_strength: float = 0.8
    glow_scale: float = 0.5


class PortalRenderer:
    """Composite the transformed dimension as a rotatable spatial window."""

    def __init__(self, config: PortalVisualConfig | None = None) -> None:
        self.config = config or PortalVisualConfig()
        self._mask_cache: dict[tuple, np.ndarray] = {}
        self._output_cache: dict[tuple[int, int], np.ndarray] = {}
        self._border_cache: dict[tuple[int, int], np.ndarray] = {}

    @staticmethod
    def _buffer(cache, shape):
        key = (shape[1], shape[0])
        buffer = cache.get(key)
        if buffer is None or buffer.shape != shape:
            buffer = np.empty(shape, dtype=np.uint8)
            cache[key] = buffer
        return buffer

    @staticmethod
    def _single(cache, shape):
        key = (shape[1], shape[0])
        buffer = cache.get(key)
        if buffer is None or buffer.shape != shape:
            buffer = np.empty(shape, dtype=np.uint8)
            cache[key] = buffer
        return buffer

    @staticmethod
    def _rotated_corners(center, width, height, angle):
        cx, cy = float(center[0]), float(center[1])
        rad = math.radians(float(angle))
        c, s = math.cos(rad), math.sin(rad)
        hw, hh = width * 0.5, height * 0.5
        corners = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]], dtype=np.float32)
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        return corners @ rot.T + np.array([cx, cy], dtype=np.float32)

    def render(self, frame, dimension, center, width, height, angle, timestamp_ms, intensity=1.0):
        if width <= 0 or height <= 0 or frame.ndim != 3:
            return frame
        intensity = max(0.0, min(1.0, float(intensity)))
        if intensity <= 0.0:
            return frame

        frame_h, frame_w = frame.shape[:2]
        cx, cy = int(round(center[0])), int(round(center[1]))
        angle = max(-25.0, min(25.0, float(angle)))
        corners_global = self._rotated_corners((cx, cy), width, height, angle)
        min_x = int(math.floor(float(corners_global[:, 0].min())))
        max_x = int(math.ceil(float(corners_global[:, 0].max())))
        min_y = int(math.floor(float(corners_global[:, 1].min())))
        max_y = int(math.ceil(float(corners_global[:, 1].max())))
        margin = max(int(self.config.roi_margin), int(self.config.glow_sigma * 2.5))

        x0 = max(0, min_x - margin)
        y0 = max(0, min_y - margin)
        x1 = min(frame_w, max_x + margin + 1)
        y1 = min(frame_h, max_y + margin + 1)
        if x1 <= x0 or y1 <= y0:
            return frame

        local = frame[y0:y1, x0:x1]
        local_h, local_w = local.shape[:2]
        corners = corners_global - np.array([x0, y0], dtype=np.float32)

        target_w = max(2, int(round(width)))
        target_h = max(2, int(round(height)))
        content = dimension
        if content.shape[1] != target_w or content.shape[0] != target_h:
            content = cv2.resize(content, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        output = self._buffer(self._output_cache, local.shape)
        output[:] = local

        src = np.array([[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1]], dtype=np.float32)
        dst = corners[[0, 1, 2]].astype(np.float32)
        matrix = cv2.getAffineTransform(src, dst)
        warped = cv2.warpAffine(
            content, matrix, (local_w, local_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        mask = self._portal_mask(local_w, local_h, corners, self.config.feather)
        alpha = mask.astype(np.float32)[:, :, None] / 255.0
        alpha *= intensity
        mixed = output.astype(np.float32) * (1.0 - alpha) + warped.astype(np.float32) * alpha
        output[:] = np.clip(mixed, 0, 255).astype(np.uint8)

        border = self._single(self._border_cache, (local_h, local_w))
        border.fill(0)
        self._draw_border(border, corners, timestamp_ms, intensity)
        glow = self._glow_from_border(border)
        self._add_glow(output, glow, intensity)
        self._composite_border(output, border, intensity)

        frame[y0:y1, x0:x1] = output
        return frame

    def _portal_mask(self, width, height, corners, feather):
        rounded = tuple(int(round(v)) for point in corners for v in point)
        key = (width, height, rounded, int(feather))
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached
        mask = np.zeros((height, width), dtype=np.uint8)
        pts = np.round(corners).astype(np.int32)
        cv2.fillConvexPoly(mask, pts, 255, cv2.LINE_AA)
        feather = max(0, int(feather))
        if feather:
            kernel = feather * 2 + 1
            mask[:] = cv2.GaussianBlur(mask, (kernel, kernel), feather * 0.55)
        self._mask_cache[key] = mask
        return mask

    @staticmethod
    def _draw_border(image, corners, timestamp_ms, intensity):
        pts = np.round(corners).astype(np.int32)
        thickness = max(1, int(round(4 * (0.7 + 0.3 * intensity))))
        cv2.polylines(image, [pts], True, 255, thickness, cv2.LINE_AA)
        offset = max(1, int(round(4 * intensity)))
        shift = np.array([offset, 0], dtype=np.int32)
        cv2.polylines(image, [pts - shift], True, 135, 1, cv2.LINE_AA)
        cv2.polylines(image, [pts + shift], True, 190, 1, cv2.LINE_AA)

        phase = timestamp_ms * 0.01
        for i in range(6):
            a = i % 4
            fraction = (math.sin(phase * (0.8 + i * 0.13) + i * 2.4) + 1.0) * 0.5
            p0 = pts[a]
            p1 = pts[(a + 1) % 4]
            start = p0 + (p1 - p0) * fraction
            end = start + (p1 - p0) * (0.04 + 0.08 * fraction)
            cv2.line(image, tuple(np.round(start).astype(int)), tuple(np.round(end).astype(int)), 255, 1, cv2.LINE_AA)

    def _glow_from_border(self, border):
        scale = min(max(float(self.config.glow_scale), 0.35), 1.0)
        h, w = border.shape
        if scale >= 0.999:
            return cv2.GaussianBlur(border, (0, 0), max(1.0, self.config.glow_sigma))
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
