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
    feather: int = 7
    chromatic_offset: int = 6
    glitch_strength: float = 1.0
    glow_scale: float = 0.46
    depth_rings: int = 8
    energy_particles: int = 34
    shape_points: int = 72
    feedback_alpha: float = 0.18
    lens_strength: float = 0.10
    scanline_strength: float = 0.055
    aberration_strength: float = 0.55
    motion_echo: float = 0.35
    fracture_count: int = 16
    fracture_length: float = 54.0
    fracture_strength: float = 0.82


class PortalRenderer:
    """Cinematic organic aperture with clipped temporal/VFX layers."""

    def __init__(self, config: PortalVisualConfig | None = None) -> None:
        self.config = config or PortalVisualConfig()
        self._mask_cache: dict[tuple, np.ndarray] = {}
        self._output_cache: dict[tuple[int, int], np.ndarray] = {}
        self._border_cache: dict[tuple[int, int], np.ndarray] = {}
        self._feedback_cache: dict[tuple[int, int], np.ndarray] = {}
        self._fracture_cache: dict[tuple[int, int], np.ndarray] = {}
        self._seed = np.random.default_rng(7319).random((max(34, self.config.energy_particles), 4), dtype=np.float32)

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
    def _organic_points(center, width, height, angle, t, count=72, scale=1.0, deformation=1.0):
        cx, cy = float(center[0]), float(center[1])
        rx, ry = width * 0.5 * scale, height * 0.5 * scale
        n = max(40, int(count))
        theta = np.linspace(0.0, math.tau, n, endpoint=False, dtype=np.float32)
        wobble = 1.0 + deformation * (0.055 * np.sin(theta * 3.0 + t * 1.7) + 0.032 * np.sin(theta * 5.0 - t * 1.13) + 0.022 * np.sin(theta * 7.0 + t * 0.77) + 0.012 * np.sin(theta * 11.0 - t * 1.91))
        x, y = np.cos(theta) * rx * wobble, np.sin(theta) * ry * wobble
        r = math.radians(float(angle))
        c, s = math.cos(r), math.sin(r)
        return np.column_stack((x * c - y * s + cx, x * s + y * c + cy)).astype(np.float32)

    def render(self, frame, dimension, center, width, height, angle, timestamp_ms, intensity=1.0, motion=0.0):
        if width <= 0 or height <= 0 or frame.ndim != 3:
            return frame
        intensity = max(0.0, min(1.0, float(intensity)))
        motion = max(0.0, min(1.0, float(motion)))
        if intensity <= 0.0:
            return frame
        h, w = frame.shape[:2]
        cx, cy = int(round(center[0])), int(round(center[1]))
        angle = max(-25.0, min(25.0, float(angle)))
        t = timestamp_ms * 0.001
        pulse = 1.0 + 0.018 * math.sin(t * 5.0)
        outer_global = self._organic_points((cx, cy), width * pulse, height * pulse, angle, t, self.config.shape_points, 1.0, 1.0)
        margin = max(int(self.config.roi_margin), int(self.config.glow_sigma * 2.5), int(self.config.fracture_length) + 18)
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
        rotated = cv2.warpAffine(content, cv2.getRotationMatrix2D((target_w * 0.5, target_h * 0.5), -angle, 1.0), (target_w, target_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        quad = self._rotated_corners(local_center, width, height, angle)
        src = np.array([[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1]], np.float32)
        affine = cv2.getAffineTransform(src, quad[[0, 1, 2]].astype(np.float32))
        warped = cv2.warpAffine(rotated, affine, (local.shape[1], local.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mask = self._organic_mask(local.shape[1], local.shape[0], points, self.config.feather)
        alpha = (mask.astype(np.float32) / 255.0 * intensity)[:, :, None]
        output[:] = np.clip(output.astype(np.float32) * (1.0 - alpha) + warped.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
        interior_base = output.copy()
        self._dimensional_tunnel(output, local_center, width, height, angle, t, intensity)
        self._energy_field(output, local_center, width, height, angle, t, intensity)
        self._particle_field(output, local_center, width, height, angle, t, intensity)
        self._signal_artifacts(output, local_center, width, height, angle, t, intensity, motion)
        self._temporal_feedback(output, intensity, motion)
        self._lens_warp(output, local_center, width, height, intensity)
        self._scanlines(output, intensity)
        self._chromatic_split(output, intensity, motion)
        self._core_lensing(output, local_center, width, height, angle, t, intensity)
        self._clip_interior(output, interior_base, mask)
        border = self._buffer(self._border_cache, (local.shape[0], local.shape[1]))
        border.fill(0)
        self._draw_border(border, points, t, intensity, motion)
        glow = self._glow_from_border(border)
        self._add_glow(output, glow, intensity)
        self._composite_border(output, border, intensity, mask)
        self._draw_multiverse_fractures(output, local_center, width, height, angle, t, intensity, motion, mask)
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

    @staticmethod
    def _clip_interior(image, base, mask):
        alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
        image[:] = np.clip(base.astype(np.float32) * (1.0 - alpha) + image.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

    def _dimensional_tunnel(self, image, center, width, height, angle, t, intensity):
        rings = max(3, int(self.config.depth_rings))
        for i in range(rings):
            scale = 0.91 - i * 0.085
            if scale < 0.30:
                break
            drift = 1.8 * intensity * math.sin(t * (1.0 + i * 0.16) + i * 1.7)
            c = (center[0] + drift, center[1] + 1.6 * intensity * math.cos(t * 0.8 + i))
            pts = self._organic_points(c, width, height, angle, t + i * 0.13, self.config.shape_points, scale, 1.0 + i * 0.025)
            value = int(42 + 65 * intensity * (1.0 - i / rings))
            cv2.polylines(image, [np.round(pts).astype(np.int32)], True, value, 1 if i else 2, cv2.LINE_AA)
            if i in (0, 2, 4):
                inner = self._organic_points(c, width, height, angle, t + 0.2, self.config.shape_points, max(0.34, scale - 0.15))
                step = max(1, len(pts) // 8)
                for j in range(0, len(pts), step):
                    end = pts[j] * 0.58 + inner[j] * 0.42
                    cv2.line(image, tuple(np.round(pts[j]).astype(int)), tuple(np.round(end).astype(int)), int(38 + 42 * intensity), 1, cv2.LINE_AA)

    def _energy_field(self, image, center, width, height, angle, t, intensity):
        outer = self._organic_points(center, width * 1.015, height * 1.015, angle, t, self.config.shape_points, 1.0, 1.0)
        n = len(outer)
        for i in range(max(10, self.config.energy_particles // 2)):
            start = int((t * (7.0 + i * 0.31) + i * 23.0) % n)
            length = max(2, int(n * (0.012 + 0.018 * ((math.sin(t * 1.4 + i) + 1.0) * 0.5))))
            idx = [(start + j) % n for j in range(length)]
            cv2.polylines(image, [np.round(outer[idx]).astype(np.int32)], False, int(145 + 90 * intensity), 1, cv2.LINE_AA)

    def _particle_field(self, image, center, width, height, angle, t, intensity):
        cx, cy = center
        r = math.radians(angle)
        c, s = math.cos(r), math.sin(r)
        for i, p in enumerate(self._seed[: self.config.energy_particles]):
            u = (float(p[0]) + t * (0.018 + (i % 5) * 0.002)) % 1.0
            v = (float(p[1]) + t * (0.014 + (i % 7) * 0.0015)) % 1.0
            depth = 0.55 + 0.45 * ((math.sin(t * (0.7 + p[2]) + i) + 1.0) * 0.5)
            x = (u - 0.5) * width * (0.68 + depth * 0.48)
            y = (v - 0.5) * height * (0.68 + depth * 0.48)
            px, py = int(cx + x * c - y * s), int(cy + x * s + y * c)
            if 0 <= px < image.shape[1] and 0 <= py < image.shape[0]:
                cv2.circle(image, (px, py), 1 if p[3] < 0.84 else 2, int(95 + 145 * intensity), -1, cv2.LINE_AA)

    def _signal_artifacts(self, image, center, width, height, angle, t, intensity, motion):
        cx, cy = center
        h, w = image.shape[:2]
        r = math.radians(angle)
        c, s = math.cos(r), math.sin(r)
        count = 7 + int(5 * motion)
        for i in range(count):
            local_y = ((t * 38.0 * (0.55 + i * 0.09) + i * height * 0.27) % max(height, 1)) - height * 0.5
            local_x = math.sin(t * 1.7 + i * 2.1) * width * (0.18 + 0.08 * motion)
            span = width * (0.12 + 0.025 * (i % 4) + 0.04 * motion)
            p0 = (cx + (local_x - span) * c - local_y * s, cy + (local_x - span) * s + local_y * c)
            p1 = (cx + (local_x + span) * c - local_y * s, cy + (local_x + span) * s + local_y * c)
            if (0 <= p0[0] < w or 0 <= p1[0] < w) and (0 <= p0[1] < h or 0 <= p1[1] < h):
                cv2.line(image, tuple(np.round(p0).astype(int)), tuple(np.round(p1).astype(int)), int(20 + 80 * intensity), 1 + int(intensity + motion), cv2.LINE_AA)

    def _temporal_feedback(self, image, intensity, motion):
        previous = self._feedback_cache.get((image.shape[1], image.shape[0]))
        if previous is None or previous.shape != image.shape:
            previous = np.zeros_like(image)
            self._feedback_cache[(image.shape[1], image.shape[0])] = previous
        alpha = min(0.42, self.config.feedback_alpha + motion * self.config.motion_echo)
        if np.any(previous):
            ghost = cv2.GaussianBlur(previous, (0, 0), 0.6 + 1.8 * motion)
            image[:] = cv2.addWeighted(image, 1.0 - alpha, ghost, alpha, 0.0)
        previous[:] = image

    def _lens_warp(self, image, center, width, height, intensity):
        strength = self.config.lens_strength * intensity
        if strength <= 0.0:
            return
        h, w = image.shape[:2]
        cx, cy = center
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        dx, dy = xx - cx, yy - cy
        nx = dx / max(width * 0.5, 1.0)
        ny = dy / max(height * 0.5, 1.0)
        radius = np.minimum(1.0, np.sqrt(nx * nx + ny * ny))
        factor = 1.0 - strength * radius * radius
        image[:] = cv2.remap(image, cx + dx * factor, cy + dy * factor, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def _scanlines(self, image, intensity):
        strength = self.config.scanline_strength * intensity
        if strength <= 0.0:
            return
        rows = np.arange(image.shape[0], dtype=np.float32)
        factor = 1.0 - strength * (0.5 + 0.5 * (np.cos(rows * math.pi) + 1.0) * 0.5)
        image[:] = np.clip(image.astype(np.float32) * factor[:, None, None], 0, 255).astype(np.uint8)

    def _chromatic_split(self, image, intensity, motion):
        amount = int(round(self.config.chromatic_offset * (0.35 + self.config.aberration_strength * intensity + 0.35 * motion)))
        if amount <= 0:
            return
        shifted_r = np.roll(image[:, :, 2], amount, axis=1)
        shifted_b = np.roll(image[:, :, 0], -amount, axis=1)
        edge = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 45, 110)
        mask = edge.astype(bool)
        image[:, :, 2][mask] = shifted_r[mask]
        image[:, :, 0][mask] = shifted_b[mask]

    def _core_lensing(self, image, center, width, height, angle, t, intensity):
        pulse = 0.5 + 0.5 * math.sin(t * 4.2)
        inner = self._organic_points(center, width * (0.30 + pulse * 0.035), height * (0.30 + pulse * 0.035), angle, -t * 0.55, self.config.shape_points, 0.9, 0.7)
        cv2.polylines(image, [np.round(inner).astype(np.int32)], True, int(35 + 55 * intensity), 1, cv2.LINE_AA)

    @staticmethod
    def _draw_border(image, points, t, intensity, motion):
        pts = np.round(points).astype(np.int32)
        cv2.polylines(image, [pts], True, int(180 + 70 * intensity), 5, cv2.LINE_AA)
        n = len(pts)
        fragments = 12 + int(8 * motion)
        for i in range(fragments):
            start = int((t * (5.0 + i * 0.27) + i * 17.0) % n)
            length = max(2, int(n * (0.008 + 0.012 * ((math.sin(t + i) + 1.0) * 0.5))))
            idx = [(start + j) % n for j in range(length)]
            cv2.polylines(image, [pts[idx]], False, int(210 + 45 * intensity), 1 + int(motion * 2), cv2.LINE_AA)

    def _glow_from_border(self, border):
        return cv2.GaussianBlur(border, (0, 0), self.config.glow_sigma)

    def _add_glow(self, image, glow, intensity):
        if glow.ndim == 2:
            glow = glow[:, :, None]
        image[:] = np.clip(image.astype(np.float32) + glow.astype(np.float32) * self.config.glow_scale * intensity, 0, 255).astype(np.uint8)

    def _composite_border(self, image, border, intensity, mask):
        border_rgb = border[:, :, None] if border.ndim == 2 else border
        border_alpha = np.minimum(1.0, border.astype(np.float32) / 255.0 * (0.45 + 0.55 * intensity))
        edge_only = np.maximum(0.0, border_alpha - (mask.astype(np.float32) / 255.0) * 0.35)[:, :, None]
        image[:] = np.clip(image.astype(np.float32) * (1.0 - edge_only) + border_rgb.astype(np.float32) * edge_only, 0, 255).astype(np.uint8)

    def _draw_multiverse_fractures(self, image, center, width, height, angle, t, intensity, motion, portal_mask):
        """Draw localized dimensional cracks outside the aperture, never a full ROI overlay."""
        h, w = image.shape[:2]
        if h < 20 or w < 20:
            return
        count = int(self.config.fracture_count + motion * 8.0)
        layer = self._fracture_cache.get((w, h))
        if layer is None or layer.shape != image.shape:
            layer = np.zeros_like(image)
            self._fracture_cache[(w, h)] = layer
        layer.fill(0)

        cx, cy = float(center[0]), float(center[1])
        r = math.radians(angle)
        c, s = math.cos(r), math.sin(r)
        points = self._organic_points(center, width * 1.01, height * 1.01, angle, t, self.config.shape_points, 1.0, 1.0)
        n = len(points)

        for i in range(count):
            idx = int((i * 37.0 + t * (4.0 + i * 0.17)) % n)
            px, py = float(points[idx, 0]), float(points[idx, 1])
            nx, ny = px - cx, py - cy
            norm = math.hypot(nx, ny)
            if norm < 1.0:
                continue
            nx, ny = nx / norm, ny / norm
            tangent_x, tangent_y = -ny, nx
            base = 10.0 + 13.0 * ((math.sin(t * 1.7 + i * 2.3) + 1.0) * 0.5)
            length = self.config.fracture_length * (0.55 + 0.65 * ((math.sin(i * 1.91 + t * 0.7) + 1.0) * 0.5))
            length *= 0.82 + 0.38 * motion
            segments = 4 + (i % 3)
            crack = [(px + nx * base, py + ny * base)]
            for j in range(segments):
                u = (j + 1) / segments
                jitter = math.sin(i * 4.7 + j * 2.6 + t * 2.1) * (5.0 + 5.0 * motion)
                bend = (math.sin(i * 2.2 + j * 1.7 + t) * 0.035 + 0.018) * length
                dist = base + u * length
                crack.append((px + nx * dist + tangent_x * (jitter + bend), py + ny * dist + tangent_y * (jitter + bend)))
            pts = np.round(np.asarray(crack, np.float32)).astype(np.int32)
            if np.max(pts[:, 0]) < 0 or np.min(pts[:, 0]) >= w or np.max(pts[:, 1]) < 0 or np.min(pts[:, 1]) >= h:
                continue
            phase = (i * 0.73 + t * 1.4) % math.tau
            red = int(150 + 90 * ((math.sin(phase) + 1.0) * 0.5) * intensity)
            blue = int(175 + 80 * ((math.cos(phase * 1.3) + 1.0) * 0.5) * intensity)
            green = int(115 + 75 * motion)
            cv2.polylines(layer, [pts], False, (blue, green, red), 1 + int(motion), cv2.LINE_AA)
            if i % 3 == 0:
                cv2.line(layer, tuple(pts[1]), tuple(pts[min(2, len(pts) - 1)]), (255, 255, 255), 1, cv2.LINE_AA)

        # Keep cracks outside the aperture, with a small overlap at the edge for a torn-reality look.
        outer = cv2.dilate(portal_mask, np.ones((9, 9), np.uint8), iterations=1)
        ring = np.clip(outer.astype(np.int16) - cv2.erode(portal_mask, np.ones((5, 5), np.uint8), iterations=1).astype(np.int16), 0, 255).astype(np.uint8)
        alpha = (ring.astype(np.float32) / 255.0 * self.config.fracture_strength * intensity)[:, :, None]
        glow = cv2.GaussianBlur(layer, (0, 0), 4.0 + 2.0 * motion)
        glow_alpha = alpha * 0.42
        image[:] = np.clip(image.astype(np.float32) + glow.astype(np.float32) * glow_alpha, 0, 255).astype(np.uint8)
        image[:] = np.clip(image.astype(np.float32) * (1.0 - alpha) + layer.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
