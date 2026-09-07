from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass
class PortalVisualConfig:
    edge_thickness: int = 4
    glow_sigma: float = 18.0
    ring_count: int = 3
    particle_count: int = 140
    particle_seed: int = 42
    roi_margin: int = 8
    ring_blur_sigma: float = 2.5
    mask_scale: float = 0.5
    glow_scale: float = 0.5
    energy_arc_count: int = 3


class PortalRenderer:
    """Render portal content and bounded cinematic VFX."""

    def __init__(self, config: PortalVisualConfig | None = None) -> None:
        self.config = config or PortalVisualConfig()
        rng = np.random.default_rng(self.config.particle_seed)
        count = max(0, int(self.config.particle_count))
        self.particle_phase = rng.uniform(0.0, 2.0 * math.pi, count)
        self.particle_radius = rng.uniform(0.82, 1.18, count)
        self.particle_angle = rng.uniform(0.0, 2.0 * math.pi, count)
        self.particle_speed = rng.uniform(0.35, 1.25, count)
        self.particle_size = rng.uniform(1.0, 3.0, count)
        self._theta_cache: dict[int, np.ndarray] = {}
        self._grid_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self._canvas_cache: dict[tuple[int, int], np.ndarray] = {}
        self._content_cache: dict[tuple[int, int], np.ndarray] = {}
        self._ring_cache: dict[tuple[int, int], np.ndarray] = {}
        self._rim_cache: dict[tuple[int, int], np.ndarray] = {}
        self._glow_cache: dict[tuple[int, int], np.ndarray] = {}
        self._alpha_cache: dict[tuple[int, int], np.ndarray] = {}
        self._alpha_float_cache: dict[tuple[int, int], np.ndarray] = {}
        self._blend_cache: dict[tuple[int, int], np.ndarray] = {}

    def _buffer(self, cache: dict[tuple[int, int], np.ndarray], shape: tuple[int, int, int]) -> np.ndarray:
        key = (shape[1], shape[0])
        buffer = cache.get(key)
        if buffer is None or buffer.shape != shape:
            buffer = np.empty(shape, dtype=np.uint8)
            cache[key] = buffer
        return buffer

    def _single_buffer(self, cache: dict[tuple[int, int], np.ndarray], shape: tuple[int, int]) -> np.ndarray:
        key = (shape[1], shape[0])
        buffer = cache.get(key)
        if buffer is None or buffer.shape != shape:
            buffer = np.empty(shape, dtype=np.uint8)
            cache[key] = buffer
        return buffer

    def _float_buffer(self, cache: dict[tuple[int, int], np.ndarray], shape: tuple[int, int]) -> np.ndarray:
        key = (shape[1], shape[0])
        buffer = cache.get(key)
        if buffer is None or buffer.shape != shape:
            buffer = np.empty(shape, dtype=np.float32)
            cache[key] = buffer
        return buffer

    def render(self, frame, dimension, center, width, height, angle, timestamp_ms, intensity=1.0):
        if width <= 0 or height <= 0 or frame.ndim != 3:
            return frame

        intensity = max(0.0, min(1.0, float(intensity)))
        if intensity <= 0.0:
            return frame

        frame_h, frame_w = frame.shape[:2]
        cx, cy = int(round(float(center[0]))), int(round(float(center[1])))
        angle_rad = math.radians(float(angle))
        rx = max(float(width) * 0.5, 1.0)
        ry = max(float(height) * 0.5, 1.0)
        ca = abs(math.cos(angle_rad))
        sa = abs(math.sin(angle_rad))
        margin = max(
            int(self.config.roi_margin),
            int(math.ceil(self.config.glow_sigma * 3.0 + self.config.edge_thickness + 4)),
        )

        half_w = int(math.ceil(rx * ca + ry * sa)) + margin
        half_h = int(math.ceil(rx * sa + ry * ca)) + margin
        x0 = max(0, cx - half_w)
        y0 = max(0, cy - half_h)
        x1 = min(frame_w, cx + half_w + 1)
        y1 = min(frame_h, cy + half_h + 1)
        if x1 <= x0 or y1 <= y0:
            return frame

        local_center = (cx - x0, cy - y0)
        local_h = y1 - y0
        local_w = x1 - x0
        t = float(timestamp_ms) * 0.001
        local_frame = frame[y0:y1, x0:x1]

        mask = self._organic_mask((local_h, local_w), local_center, width, height, angle_rad, t)
        content = self._prepare_content(
            dimension, width, height, angle_rad, local_center, local_w, local_h
        )

        alpha_u8 = self._single_buffer(self._alpha_cache, mask.shape)
        cv2.normalize(mask, alpha_u8, 1.0 / 255.0, 0.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        alpha = self._float_buffer(self._alpha_float_cache, mask.shape)
        alpha[:] = alpha_u8.astype(np.float32)
        if intensity < 0.999:
            alpha *= intensity

        local_output = self._buffer(self._canvas_cache, local_frame.shape)
        np.multiply(local_frame, 1.0 - alpha[..., None], out=local_output, casting="unsafe")
        np.add(local_output, content * alpha[..., None], out=local_output, casting="unsafe")
        np.clip(local_output, 0, 255, out=local_output)

        edges = cv2.Canny(mask, 70, 180)
        glow = self._glow_from_edges(edges)
        self._add_glow(local_output, glow, intensity)

        energy = self._buffer(self._ring_cache, local_output.shape)
        energy.fill(0)
        self._draw_rings(energy, local_center, width, height, angle_rad, t, intensity)
        self._draw_particles(energy, local_center, width, height, angle_rad, t, intensity)

        blend = self._buffer(self._blend_cache, local_output.shape)
        cv2.addWeighted(local_output, 1.0, energy, 0.82 * intensity, 0.0, dst=blend)
        local_output, blend = blend, local_output

        rim = self._buffer(self._rim_cache, local_output.shape)
        rim.fill(0)
        self._draw_rim(rim, local_center, width, height, angle_rad, t, intensity)
        cv2.addWeighted(local_output, 1.0, rim, 0.95 * intensity, 0.0, dst=blend)
        local_output = blend

        frame[y0:y1, x0:x1] = local_output
        return frame

    def _glow_from_edges(self, edges):
        sigma = max(float(self.config.glow_sigma), 0.5)
        radius = max(1, int(round(sigma * 1.5)))
        kernel = radius * 2 + 1
        glow = self._single_buffer(self._glow_cache, edges.shape)
        cv2.blur(edges, (kernel, kernel), dst=glow)
        return glow

    @staticmethod
    def _add_glow(image, glow, intensity=1.0):
        weights = (0.55, 0.41, 0.19)
        for channel, weight in enumerate(weights):
            image[:, :, channel] = cv2.addWeighted(
                image[:, :, channel], 1.0, glow, weight * intensity, 0.0
            )

    def _prepare_content(self, dimension, width, height, angle, center, local_w, local_h):
        target_w = max(1, int(round(float(width))))
        target_h = max(1, int(round(float(height))))
        if dimension.shape[1] != target_w or dimension.shape[0] != target_h:
            resized = cv2.resize(dimension, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized = dimension

        key = (local_w, local_h)
        canvas = self._content_cache.get(key)
        if canvas is None or canvas.shape != (local_h, local_w, 3):
            canvas = np.empty((local_h, local_w, 3), dtype=np.uint8)
            self._content_cache[key] = canvas
        canvas.fill(0)

        cx, cy = center
        x0 = int(round(cx - target_w * 0.5))
        y0 = int(round(cy - target_h * 0.5))
        x1 = x0 + target_w
        y1 = y0 + target_h
        dst_x0 = max(0, x0)
        dst_y0 = max(0, y0)
        dst_x1 = min(local_w, x1)
        dst_y1 = min(local_h, y1)
        src_x0 = dst_x0 - x0
        src_y0 = dst_y0 - y0
        src_x1 = src_x0 + max(0, dst_x1 - dst_x0)
        src_y1 = src_y0 + max(0, dst_y1 - dst_y0)
        if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
            return canvas

        if abs(angle) < 1e-6:
            transformed = resized
        elif abs(abs(angle) - math.pi * 0.5) < 1e-6:
            k = 1 if angle > 0 else 3
            transformed = np.rot90(resized, k=k).copy()
            transformed = cv2.resize(transformed, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        else:
            matrix = cv2.getRotationMatrix2D(
                ((target_w - 1) * 0.5, (target_h - 1) * 0.5),
                -math.degrees(angle),
                1.0,
            )
            transformed = cv2.warpAffine(
                resized,
                matrix,
                (target_w, target_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )

        canvas[dst_y0:dst_y1, dst_x0:dst_x1] = transformed[src_y0:src_y1, src_x0:src_x1]
        return canvas

    def _organic_mask(self, shape, center, width, height, angle, t):
        h, w = shape
        scale = min(max(float(self.config.mask_scale), 0.25), 1.0)
        if scale >= 0.999:
            return self._organic_mask_full(shape, center, width, height, angle, t)

        small_w = max(1, int(round(w * scale)))
        small_h = max(1, int(round(h * scale)))
        small_center = (center[0] * scale, center[1] * scale)
        small_mask = self._organic_mask_full(
            (small_h, small_w),
            small_center,
            float(width) * scale,
            float(height) * scale,
            angle,
            t,
        )
        return cv2.resize(small_mask, (w, h), interpolation=cv2.INTER_LINEAR)

    def _organic_mask_full(self, shape, center, width, height, angle, t):
        h, w = shape
        cx, cy = float(center[0]), float(center[1])
        key = (w, h)
        grid = self._grid_cache.get(key)
        if grid is None:
            yy, xx = np.mgrid[0:h, 0:w]
            grid = (xx.astype(np.float32), yy.astype(np.float32))
            self._grid_cache[key] = grid
        xx, yy = grid

        dx = xx - cx
        dy = yy - cy
        ca = math.cos(angle)
        sa = math.sin(angle)
        xr = dx * ca + dy * sa
        yr = -dx * sa + dy * ca
        rx = max(float(width) * 0.5, 1.0)
        ry = max(float(height) * 0.5, 1.0)
        theta = np.arctan2(yr, xr)
        wave = (
            1.0
            + 0.075 * np.sin(theta * 5.0 + t * 3.2)
            + 0.045 * np.sin(theta * 9.0 - t * 2.1)
            + 0.025 * np.sin(theta * 14.0 + t * 4.7)
        )
        radius = np.sqrt((xr / rx) ** 2 + (yr / ry) ** 2)
        boundary = radius / wave
        alpha = np.clip((1.0 - boundary) / 0.035 + 0.5, 0.0, 1.0)
        return (alpha * 255.0).astype(np.uint8)

    def _ellipse_points(self, center, width, height, angle, t, phase=0.0, scale=1.0, count=180):
        theta = self._theta_cache.get(count)
        if theta is None:
            theta = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False, dtype=np.float32)
            self._theta_cache[count] = theta
        cx, cy = center
        wave = 1.0 + 0.06 * np.sin(theta * 5.0 + t * 3.0 + phase) + 0.035 * np.sin(theta * 11.0 - t * 2.2 + phase * 1.7)
        x = width * 0.5 * scale * wave * np.cos(theta)
        y = height * 0.5 * scale * wave * np.sin(theta)
        ca = math.cos(angle)
        sa = math.sin(angle)
        return np.column_stack((x * ca - y * sa + cx, x * sa + y * ca + cy)).astype(np.int32)

    def _draw_rings(self, image, center, width, height, angle, t, intensity=1.0):
        base = image
        for i in range(self.config.ring_count):
            pulse = 1.0 + 0.035 * math.sin(t * (2.0 + i * 0.55) + i)
            scale = (1.0 + (i - 1) * 0.045) * pulse
            pts = self._ellipse_points(center, width, height, angle, t, i * 1.9, scale, 160)
            thickness = 2 if i == 1 else 1
            if intensity < 0.999:
                thickness = max(1, int(round(thickness * (0.65 + 0.35 * intensity))))
            cv2.polylines(base, [pts], True, (90, 170, 245), thickness, cv2.LINE_AA)

        arc_count = max(0, int(self.config.energy_arc_count))
        if arc_count:
            center_point = (int(round(center[0])), int(round(center[1])))
            rotation = math.degrees(angle)
            scale_step = 0.035
            for i in range(arc_count):
                phase = t * (55.0 + i * 11.0) + i * 83.0
                start = phase % 360.0
                span = 48.0 + 10.0 * math.sin(t * 2.0 + i)
                axes = (
                    max(1, int(width * (0.42 + i * scale_step))),
                    max(1, int(height * (0.42 + i * scale_step))),
                )
                cv2.ellipse(
                    base,
                    center_point,
                    axes,
                    rotation,
                    start,
                    start + span,
                    (120, 205, 255),
                    1,
                    cv2.LINE_AA,
                )

        if self.config.ring_blur_sigma > 0:
            blur = cv2.GaussianBlur(base, (0, 0), self.config.ring_blur_sigma)
            image[:] = cv2.addWeighted(image, 1.0, blur, 0.45 * intensity, 0.0)

    def _draw_rim(self, image, center, width, height, angle, t, intensity=1.0):
        pts = self._ellipse_points(center, width, height, angle, t, 0.0, 1.0, 220)
        thickness = max(1, int(round(self.config.edge_thickness * (0.55 + 0.45 * intensity))))
        cv2.polylines(image, [pts], True, (180, 225, 255), thickness, cv2.LINE_AA)

    def _draw_particles(self, image, center, width, height, angle, t, intensity=1.0):
        cx, cy = center
        ca = math.cos(angle)
        sa = math.sin(angle)
        for i in range(len(self.particle_phase)):
            a = self.particle_angle[i] + t * self.particle_speed[i] * 0.22
            pulse = 1.0 + 0.14 * math.sin(t * 3.0 + self.particle_phase[i])
            rx = width * 0.5 * self.particle_radius[i] * pulse
            ry = height * 0.5 * self.particle_radius[i] * pulse
            local_x = math.cos(a) * rx
            local_y = math.sin(a) * ry
            x = int(cx + local_x * ca - local_y * sa)
            y = int(cy + local_x * sa + local_y * ca)
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                size = max(1, int(self.particle_size[i] * (0.55 + 0.45 * intensity)))
                brightness = int(130 + 90 * (0.5 + 0.5 * math.sin(t * 5.0 + self.particle_phase[i])))
                brightness = int(brightness * (0.35 + 0.65 * intensity))
                cv2.circle(image, (x, y), size, (brightness // 2, brightness, 255), -1)
