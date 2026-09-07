from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass
class PortalVisualConfig:
    edge_thickness: int = 5
    glow_sigma: float = 15.0
    roi_margin: int = 46
    feather: int = 5
    chromatic_offset: int = 5
    glitch_strength: float = 1.0
    glow_scale: float = 0.48
    depth_rings: int = 6
    energy_particles: int = 24


class PortalRenderer:
    """Render a cinematic dimensional aperture with depth, energy and signal artifacts."""

    def __init__(self, config: PortalVisualConfig | None = None) -> None:
        self.config = config or PortalVisualConfig()
        self._mask_cache: dict[tuple, np.ndarray] = {}
        self._output_cache: dict[tuple[int, int], np.ndarray] = {}
        self._border_cache: dict[tuple[int, int], np.ndarray] = {}
        self._seed = np.random.default_rng(7319).random((max(24, self.config.energy_particles), 4), dtype=np.float32)

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
        x0, y0 = max(0, min_x - margin), max(0, min_y - margin)
        x1, y1 = min(frame_w, max_x + margin + 1), min(frame_h, max_y + margin + 1)
        if x1 <= x0 or y1 <= y0:
            return frame

        local = frame[y0:y1, x0:x1]
        local_h, local_w = local.shape[:2]
        corners = corners_global - np.array([x0, y0], dtype=np.float32)
        local_center = (cx - x0, cy - y0)
        target_w, target_h = max(2, int(round(width))), max(2, int(round(height)))
        content = dimension if dimension.shape[:2] == (target_h, target_w) else cv2.resize(dimension, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        output = self._buffer(self._output_cache, local.shape)
        output[:] = local
        src = np.array([[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1]], dtype=np.float32)
        matrix = cv2.getAffineTransform(src, corners[[0, 1, 2]].astype(np.float32))
        warped = cv2.warpAffine(content, matrix, (local_w, local_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        mask = self._portal_mask(local_w, local_h, corners, self.config.feather)
        alpha = (mask.astype(np.float32) / 255.0 * intensity)[:, :, None]
        output[:] = np.clip(output.astype(np.float32) * (1.0 - alpha) + warped.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

        self._dimensional_tunnel(output, local_center, width, height, angle, timestamp_ms, intensity)
        self._energy_field(output, local_center, width, height, angle, timestamp_ms, intensity)
        self._particle_field(output, local_center, width, height, angle, timestamp_ms, intensity)
        self._signal_artifacts(output, local_center, width, height, timestamp_ms, intensity)

        border = self._single(self._border_cache, (local_h, local_w))
        border.fill(0)
        self._draw_border(border, corners, timestamp_ms, intensity)
        glow = self._glow_from_border(border)
        self._add_glow(output, glow, intensity)
        self._composite_border(output, border, intensity)
        frame[y0:y1, x0:x1] = output
        return frame

    def _portal_mask(self, width, height, corners, feather):
        key = (width, height, tuple(int(round(v)) for p in corners for v in p), int(feather))
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.round(corners).astype(np.int32), 255, cv2.LINE_AA)
        feather = max(0, int(feather))
        if feather:
            k = feather * 2 + 1
            mask[:] = cv2.GaussianBlur(mask, (k, k), feather * 0.55)
        self._mask_cache[key] = mask
        return mask

    @staticmethod
    def _quad(center, width, height, angle, scale):
        return PortalRenderer._rotated_corners(center, width * scale, height * scale, angle)

    def _dimensional_tunnel(self, image, center, width, height, angle, timestamp_ms, intensity):
        """Nested planes + perspective rays give the portal an actual tunnel impression."""
        phase = timestamp_ms * 0.001
        layers = max(2, int(self.config.depth_rings))
        for i in range(layers):
            z = 0.91 - i * 0.13
            if z < 0.30:
                break
            wobble = math.sin(phase * (1.0 + i * 0.17) + i * 1.8) * (1.5 + i * 0.35) * intensity
            c = (center[0] + wobble, center[1] + math.cos(phase * 0.8 + i) * 1.5 * intensity)
            outer = self._quad(c, width * 0.96, height * 0.96, angle, z)
            inner = self._quad(c, width * 0.96, height * 0.96, angle, max(0.30, z - 0.12))
            value = int(48 + 58 * intensity * (1.0 - i / max(layers, 1)))
            cv2.polylines(image, [np.round(outer).astype(np.int32)], True, value, 1 if i else 2, cv2.LINE_AA)
            if i == 0:
                for j in range(4):
                    cv2.line(image, tuple(np.round(outer[j]).astype(int)), tuple(np.round(inner[j]).astype(int)), int(45 + 45 * intensity), 1, cv2.LINE_AA)

    def _energy_field(self, image, center, width, height, angle, timestamp_ms, intensity):
        """Animated energy fragments orbit and pulse along the aperture."""
        phase = timestamp_ms * 0.003
        cx, cy = center
        hw, hh = width * 0.5, height * 0.5
        for i in range(max(6, self.config.energy_particles // 2)):
            edge = i % 4
            f = (math.sin(phase * (0.55 + i * 0.057) + i * 1.77) + 1.0) * 0.5
            drift = math.sin(phase * 1.9 + i) * 6.0
            length = 0.035 + 0.09 * ((math.sin(phase * 0.8 + i * 0.7) + 1.0) * 0.5)
            if edge == 0:
                p0, p1 = (cx - hw * 0.75 + drift, cy - hh), (cx - hw * 0.75 + width * length + drift, cy - hh)
            elif edge == 1:
                p0, p1 = (cx + hw, cy - hh * 0.75 + drift), (cx + hw, cy - hh * 0.75 + height * length + drift)
            elif edge == 2:
                p0, p1 = (cx + hw * 0.75 - drift, cy + hh), (cx + hw * 0.75 - width * length - drift, cy + hh)
            else:
                p0, p1 = (cx - hw, cy + hh * 0.75 - drift), (cx - hw, cy + hh * 0.75 - height * length - drift)
            cv2.line(image, tuple(map(int, p0)), tuple(map(int, p1)), int(145 + 90 * intensity), 1, cv2.LINE_AA)

    def _particle_field(self, image, center, width, height, angle, timestamp_ms, intensity):
        """Stable particles flow through the aperture instead of being regenerated each frame."""
        cx, cy = center
        phase = timestamp_ms * 0.001
        count = min(len(self._seed), max(0, int(self.config.energy_particles)))
        for i in range(count):
            p = self._seed[i]
            u = (float(p[0]) + phase * (0.018 + (i % 5) * 0.002)) % 1.0
            v = (float(p[1]) + phase * (0.014 + (i % 7) * 0.0015)) % 1.0
            depth = 0.55 + 0.45 * ((math.sin(phase * (0.7 + p[2]) + i) + 1.0) * 0.5)
            x = int(cx + (u - 0.5) * width * (0.68 + depth * 0.50))
            y = int(cy + (v - 0.5) * height * (0.68 + depth * 0.50))
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                r = 1 if p[3] < 0.84 else 2
                cv2.circle(image, (x, y), r, int(100 + 135 * intensity), -1, cv2.LINE_AA)

    def _signal_artifacts(self, image, center, width, height, timestamp_ms, intensity):
        """Glitch bars, scanlines and tiny signal tears make the dimension feel unstable."""
        cx, cy = center
        phase = timestamp_ms * 0.05
        h, w = image.shape[:2]
        for i in range(5):
            y = int(cy + ((phase * (0.5 + i * 0.12) + i * height * 0.31) % max(height, 1)) - height * 0.5)
            if 0 <= y < h:
                x0 = max(0, int(cx - width * 0.47))
                x1 = min(w - 1, int(cx + width * 0.47))
                cv2.line(image, (x0, y), (x1, y), int(22 + 65 * intensity), 1 + int(intensity), cv2.LINE_AA)
        for i in range(3):
            y = int(cy + math.sin(phase * 0.6 + i * 2.4) * height * 0.32)
            if 0 <= y < h:
                span = int(width * (0.07 + i * 0.035))
                x0 = max(0, int(cx - span))
                x1 = min(w - 1, int(cx + span))
                if x1 > x0:
                    row = image[y:y + 1, x0:x1 + 1]
                    shift = (-1 if i % 2 else 1) * max(1, int(self.config.chromatic_offset * intensity))
                    row[:] = np.roll(row, shift, axis=1)

    @staticmethod
    def _draw_border(image, corners, timestamp_ms, intensity):
        pts = np.round(corners).astype(np.int32)
        thickness = max(1, int(round(5 * (0.72 + 0.28 * intensity))))
        cv2.polylines(image, [pts], True, 255, thickness, cv2.LINE_AA)
        offset = max(1, int(round(5 * intensity)))
        shift = np.array([offset, 0], dtype=np.int32)
        cv2.polylines(image, [pts - shift], True, 125, 1, cv2.LINE_AA)
        cv2.polylines(image, [pts + shift], True, 205, 1, cv2.LINE_AA)
        phase = timestamp_ms * 0.012
        for i in range(12):
            edge = i % 4
            fraction = (math.sin(phase * (0.72 + i * 0.105) + i * 2.13) + 1.0) * 0.5
            start = corners[edge] + (corners[(edge + 1) % 4] - corners[edge]) * fraction
            end = start + (corners[(edge + 1) % 4] - corners[edge]) * (0.025 + 0.07 * fraction)
            cv2.line(image, tuple(np.round(start).astype(int)), tuple(np.round(end).astype(int)), 255, 1, cv2.LINE_AA)
        for i in range(4):
            p = pts[i]
            a = pts[(i - 1) % 4]
            b = pts[(i + 1) % 4]
            cv2.line(image, tuple((p + (a - p) * 0.13).astype(int)), tuple(p), 220, 2, cv2.LINE_AA)
            cv2.line(image, tuple(p), tuple((p + (b - p) * 0.13).astype(int)), 255, 2, cv2.LINE_AA)

    def _glow_from_border(self, border):
        scale = min(max(float(self.config.glow_scale), 0.35), 1.0)
        h, w = border.shape
        if scale >= 0.999:
            return cv2.GaussianBlur(border, (0, 0), max(1.0, self.config.glow_sigma))
        sw, sh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        small = cv2.resize(border, (sw, sh), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), max(1.0, self.config.glow_sigma * scale))
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _add_glow(image, glow, intensity):
        image[:, :, 0] = cv2.addWeighted(image[:, :, 0], 1.0, glow, 0.52 * intensity, 0.0)
        image[:, :, 1] = cv2.addWeighted(image[:, :, 1], 1.0, glow, 0.37 * intensity, 0.0)
        image[:, :, 2] = cv2.addWeighted(image[:, :, 2], 1.0, glow, 0.70 * intensity, 0.0)

    @staticmethod
    def _composite_border(image, border, intensity):
        offset = max(2, int(round(5 * intensity)))
        blue = np.roll(border, -offset, axis=1)
        red = np.roll(border, offset, axis=1)
        image[:, :, 0] = cv2.addWeighted(image[:, :, 0], 1.0, blue, 0.42 * intensity, 0.0)
        image[:, :, 2] = cv2.addWeighted(image[:, :, 2], 1.0, red, 0.54 * intensity, 0.0)
