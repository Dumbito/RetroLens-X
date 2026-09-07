from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass
class PortalVisualConfig:
    edge_thickness: int = 5
    glow_sigma: float = 15.0
    roi_margin: int = 52
    feather: int = 6
    chromatic_offset: int = 6
    glitch_strength: float = 1.0
    glow_scale: float = 0.46
    depth_rings: int = 7
    energy_particles: int = 28
    shape_points: int = 64


class PortalRenderer:
    """Render a cinematic organic dimensional aperture instead of a rigid rectangle."""

    def __init__(self, config: PortalVisualConfig | None = None) -> None:
        self.config = config or PortalVisualConfig()
        self._mask_cache: dict[tuple, np.ndarray] = {}
        self._output_cache: dict[tuple[int, int], np.ndarray] = {}
        self._border_cache: dict[tuple[int, int], np.ndarray] = {}
        count = max(28, int(self.config.energy_particles))
        self._seed = np.random.default_rng(7319).random((count, 4), dtype=np.float32)

    @staticmethod
    def _buffer(cache, shape):
        key = (shape[1], shape[0])
        buf = cache.get(key)
        if buf is None or buf.shape != shape:
            buf = np.empty(shape, dtype=np.uint8)
            cache[key] = buf
        return buf

    @staticmethod
    def _rotated_corners(center, width, height, angle):
        cx, cy = float(center[0]), float(center[1])
        r = math.radians(float(angle))
        c, s = math.cos(r), math.sin(r)
        hw, hh = width * 0.5, height * 0.5
        p = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]], np.float32)
        return p @ np.array([[c, -s], [s, c]], np.float32).T + np.array([cx, cy], np.float32)

    @staticmethod
    def _organic_points(center, width, height, angle, t, count=64, scale=1.0):
        cx, cy = float(center[0]), float(center[1])
        rx, ry = width * 0.5 * scale, height * 0.5 * scale
        n = max(32, int(count))
        theta = np.linspace(0.0, math.tau, n, endpoint=False, dtype=np.float32)
        wobble = (
            1.0
            + 0.045 * np.sin(theta * 3.0 + t * 1.7)
            + 0.028 * np.sin(theta * 5.0 - t * 1.13)
            + 0.018 * np.sin(theta * 7.0 + t * 0.77)
        )
        x = np.cos(theta) * rx * wobble
        y = np.sin(theta) * ry * wobble
        r = math.radians(float(angle))
        c, s = math.cos(r), math.sin(r)
        return np.column_stack((x * c - y * s + cx, x * s + y * c + cy)).astype(np.float32)

    def render(self, frame, dimension, center, width, height, angle, timestamp_ms, intensity=1.0):
        if width <= 0 or height <= 0 or frame.ndim != 3:
            return frame
        intensity = max(0.0, min(1.0, float(intensity)))
        if intensity <= 0.0:
            return frame

        h, w = frame.shape[:2]
        cx, cy = int(round(center[0])), int(round(center[1]))
        angle = max(-25.0, min(25.0, float(angle)))
        t = timestamp_ms * 0.001
        outer_global = self._organic_points((cx, cy), width, height, angle, t, self.config.shape_points)
        margin = max(int(self.config.roi_margin), int(self.config.glow_sigma * 2.5))
        x0 = max(0, int(math.floor(outer_global[:, 0].min())) - margin)
        y0 = max(0, int(math.floor(outer_global[:, 1].min())) - margin)
        x1 = min(w, int(math.ceil(outer_global[:, 0].max())) + margin + 1)
        y1 = min(h, int(math.ceil(outer_global[:, 1].max())) + margin + 1)
        if x1 <= x0 or y1 <= y0:
            return frame

        local = frame[y0:y1, x0:x1]
        local_center = (cx - x0, cy - y0)
        points = outer_global - np.array([x0, y0], np.float32)
        target_w, target_h = max(2, int(round(width))), max(2, int(round(height)))
        content = dimension if dimension.shape[:2] == (target_h, target_w) else cv2.resize(dimension, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        output = self._buffer(self._output_cache, local.shape)
        output[:] = local
        rotated = cv2.warpAffine(
            content,
            cv2.getRotationMatrix2D((target_w * 0.5, target_h * 0.5), -angle, 1.0),
            (target_w, target_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        quad = self._rotated_corners(local_center, width, height, angle)
        src = np.array([[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1]], np.float32)
        affine = cv2.getAffineTransform(src, quad[[0, 1, 2]].astype(np.float32))
        warped = cv2.warpAffine(rotated, affine, (local.shape[1], local.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        mask = self._organic_mask(local.shape[1], local.shape[0], points, self.config.feather)
        alpha = (mask.astype(np.float32) / 255.0 * intensity)[:, :, None]
        output[:] = np.clip(output.astype(np.float32) * (1.0 - alpha) + warped.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

        self._dimensional_tunnel(output, local_center, width, height, angle, t, intensity)
        self._energy_field(output, local_center, width, height, angle, t, intensity)
        self._particle_field(output, local_center, width, height, angle, t, intensity)
        self._signal_artifacts(output, local_center, width, height, t, intensity)

        border = self._buffer(self._border_cache, (local.shape[0], local.shape[1]))
        border.fill(0)
        self._draw_border(border, points, t, intensity)
        glow = self._glow_from_border(border)
        self._add_glow(output, glow, intensity)
        self._composite_border(output, border, intensity)
        frame[y0:y1, x0:x1] = output
        return frame

    def _organic_mask(self, width, height, points, feather):
        key = (width, height, tuple(np.round(points[::4]).astype(np.int16).ravel()), int(feather))
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached
        mask = np.zeros((height, width), np.uint8)
        cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 255, cv2.LINE_AA)
        f = max(0, int(feather))
        if f:
            mask[:] = cv2.GaussianBlur(mask, (f * 2 + 1, f * 2 + 1), f * 0.55)
        self._mask_cache[key] = mask
        return mask

    def _dimensional_tunnel(self, image, center, width, height, angle, t, intensity):
        for i in range(max(2, int(self.config.depth_rings))):
            scale = 0.92 - i * 0.105
            if scale < 0.30:
                break
            c = (center[0] + math.sin(t * (1.0 + i * 0.17) + i * 1.9) * 1.8 * intensity, center[1] + math.cos(t * 0.8 + i) * 1.8 * intensity)
            pts = self._organic_points(c, width, height, angle, t + i * 0.12, self.config.shape_points, scale)
            value = int(45 + 62 * intensity * (1.0 - i / max(self.config.depth_rings, 1)))
            cv2.polylines(image, [np.round(pts).astype(np.int32)], True, value, 1 if i else 2, cv2.LINE_AA)
            if i == 0:
                inner = self._organic_points(c, width, height, angle, t + 0.2, self.config.shape_points, 0.56)
                for j in range(0, len(pts), max(1, len(pts) // 8)):
                    cv2.line(image, tuple(np.round(pts[j]).astype(int)), tuple(np.round(inner[j]).astype(int)), int(38 + 48 * intensity), 1, cv2.LINE_AA)

    def _energy_field(self, image, center, width, height, angle, t, intensity):
        outer = self._organic_points(center, width * 1.03, height * 1.03, angle, t, self.config.shape_points)
        n = len(outer)
        for i in range(max(8, self.config.energy_particles // 2)):
            start = int((t * (8.0 + i * 0.35) + i * 23.0) % n)
            length = int(n * (0.018 + 0.022 * ((math.sin(t * 1.4 + i) + 1.0) * 0.5)))
            idx = [(start + j) % n for j in range(max(2, length))]
            cv2.polylines(image, [np.round(outer[idx]).astype(np.int32)], False, int(145 + 90 * intensity), 1, cv2.LINE_AA)

    def _particle_field(self, image, center, width, height, angle, t, intensity):
        cx, cy = center
        r = math.radians(angle)
        c, s = math.cos(r), math.sin(r)
        for i, p in enumerate(self._seed[: self.config.energy_particles]):
            u = (float(p[0]) + t * (0.018 + (i % 5) * 0.002)) % 1.0
            v = (float(p[1]) + t * (0.014 + (i % 7) * 0.0015)) % 1.0
            depth = 0.55 + 0.45 * ((math.sin(t * (0.7 + p[2]) + i) + 1.0) * 0.5)
            x, y = (u - 0.5) * width * (0.70 + depth * 0.50), (v - 0.5) * height * (0.70 + depth * 0.50)
            px, py = int(cx + x * c - y * s), int(cy + x * s + y * c)
            if 0 <= px < image.shape[1] and 0 <= py < image.shape[0]:
                cv2.circle(image, (px, py), 1 if p[3] < 0.84 else 2, int(95 + 145 * intensity), -1, cv2.LINE_AA)

    def _signal_artifacts(self, image, center, width, height, t, intensity):
        cx, cy = center
        h, w = image.shape[:2]
        for i in range(5):
            y = int(cy + ((t * 42 * (0.5 + i * 0.12) + i * height * 0.31) % max(height, 1)) - height * 0.5)
            if 0 <= y < h:
                x0, x1 = max(0, int(cx - width * 0.45)), min(w - 1, int(cx + width * 0.45))
                cv2.line(image, (x0, y), (x1, y), int(20 + 65 * intensity), 1 + int(intensity), cv2.LINE_AA)
        for i in range(3):
            y = int(cy + math.sin(t * 2.0 + i * 2.4) * height * 0.32)
            if 0 <= y < h:
                span = int(width * (0.07 + i * 0.035))
                x0, x1 = max(0, int(cx - span)), min(w - 1, int(cx + span))
                if x1 > x0:
                    row = image[y:y + 1, x0:x1 + 1]
                    row[:] = np.roll(row, (-1 if i % 2 else 1) * max(1, int(self.config.chromatic_offset * intensity)), axis=1)

    @staticmethod
    def _draw_border(image, points, t, intensity):
        pts = np.round(points).astype(np.int32)
        cv2.polylines(image, [pts], True, 255, max(1, int(round(5 * (0.7 + 0.3 * intensity)))), cv2.LINE_AA)
        for shift, value in [(-5, 120), (5, 205)]:
            cv2.polylines(image, [pts + np.array([shift, 0], np.int32)], True, value, 1, cv2.LINE_AA)
        n = len(points)
        for i in range(14):
            a = int((t * (7.0 + i * 0.2) + i * 17.0) % n)
            length = max(2, int(n * (0.012 + 0.025 * ((math.sin(t + i) + 1.0) * 0.5))))
            idx = [(a + j) % n for j in range(length)]
            cv2.polylines(image, [pts[idx]], False, 255, 1, cv2.LINE_AA)

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
        image[:, :, 0] = cv2.addWeighted(image[:, :, 0], 1.0, np.roll(border, -offset, axis=1), 0.42 * intensity, 0.0)
        image[:, :, 2] = cv2.addWeighted(image[:, :, 2], 1.0, np.roll(border, offset, axis=1), 0.54 * intensity, 0.0)
