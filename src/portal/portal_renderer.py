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


class PortalRenderer:
    """Render portal content and VFX inside a bounded region of interest."""

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

    def render(self, frame, dimension, center, width, height, angle, timestamp_ms):
        if width <= 0 or height <= 0 or frame.ndim != 3:
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

        mask = self._organic_mask(
            (local_h, local_w),
            local_center,
            width,
            height,
            angle_rad,
            t,
        )

        content = self._prepare_content(
            dimension,
            width,
            height,
            angle_rad,
            local_center,
            local_w,
            local_h,
        )

        alpha = mask.astype(np.float32) / 255.0
        local_output = (
            local_frame.astype(np.float32) * (1.0 - alpha[..., None])
            + content.astype(np.float32) * alpha[..., None]
        ).astype(np.uint8)

        edges = cv2.Canny(mask, 70, 180)
        glow = cv2.GaussianBlur(edges, (0, 0), self.config.glow_sigma)
        self._add_glow(local_output, glow)

        energy = np.zeros_like(local_output)
        self._draw_rings(energy, local_center, width, height, angle_rad, t)
        self._draw_particles(energy, local_center, width, height, angle_rad, t)
        local_output = cv2.addWeighted(local_output, 1.0, energy, 0.82, 0.0)

        rim = np.zeros_like(local_output)
        self._draw_rim(rim, local_center, width, height, angle_rad, t)
        local_output = cv2.addWeighted(local_output, 1.0, rim, 0.95, 0.0)

        frame[y0:y1, x0:x1] = local_output
        return frame

    @staticmethod
    def _add_glow(image, glow):
        for channel, weight in enumerate((0.55, 0.41, 0.19)):
            image[:, :, channel] = cv2.addWeighted(
                image[:, :, channel], 1.0, glow, weight, 0.0
            )

    @staticmethod
    def _prepare_content(dimension, width, height, angle, center, local_w, local_h):
        target_w = max(1, int(round(float(width))))
        target_h = max(1, int(round(float(height))))
        if dimension.shape[1] != target_w or dimension.shape[0] != target_h:
            resized = cv2.resize(dimension, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized = dimension

        # Keep the content inside the same axis-aligned bounding box as the
        # portal mask. At 90° the portal dimensions intentionally stay WxH:
        # the visual rotates, but the mask geometry remains stable.
        if abs(math.sin(angle)) < 1e-6:
            transformed = resized
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

        canvas = np.zeros((local_h, local_w, 3), dtype=np.uint8)
        cx, cy = center
        x = int(round(cx - target_w * 0.5))
        y = int(round(cy - target_h * 0.5))
        src_x0 = max(0, -x)
        src_y0 = max(0, -y)
        dst_x0 = max(0, x)
        dst_y0 = max(0, y)
        copy_w = min(target_w - src_x0, local_w - dst_x0)
        copy_h = min(target_h - src_y0, local_h - dst_y0)
        if copy_w > 0 and copy_h > 0:
            canvas[dst_y0:dst_y0 + copy_h, dst_x0:dst_x0 + copy_w] = transformed[
                src_y0:src_y0 + copy_h,
                src_x0:src_x0 + copy_w,
            ]
        return canvas

    def _organic_mask(self, shape, center, width, height, angle, t):
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
        wave = 1.0 + 0.06 * np.sin(theta * 5.0 + t * 3.0 + phase) + 0.035 * np.sin(
            theta * 11.0 - t * 2.2 + phase * 1.7
        )
        x = width * 0.5 * scale * wave * np.cos(theta)
        y = height * 0.5 * scale * wave * np.sin(theta)
        ca = math.cos(angle)
        sa = math.sin(angle)
        return np.column_stack((x * ca - y * sa + cx, x * sa + y * ca + cy)).astype(np.int32)

    def _draw_rings(self, image, center, width, height, angle, t):
        for i in range(self.config.ring_count):
            pulse = 1.0 + 0.035 * math.sin(t * (2.0 + i * 0.55) + i)
            scale = (1.0 + (i - 1) * 0.045) * pulse
            pts = self._ellipse_points(center, width, height, angle, t, i * 1.9, scale, 160)
            layer = np.zeros_like(image)
            cv2.polylines(layer, [pts], True, (90, 170, 245), 2 if i == 1 else 1, cv2.LINE_AA)
            if self.config.ring_blur_sigma > 0:
                blur = cv2.GaussianBlur(layer, (0, 0), self.config.ring_blur_sigma)
                image = cv2.addWeighted(image, 1.0, blur, 0.45, 0.0)
            image[:] = cv2.addWeighted(image, 1.0, layer, 0.75, 0.0)

    def _draw_rim(self, image, center, width, height, angle, t):
        pts = self._ellipse_points(center, width, height, angle, t, 0.0, 1.0, 220)
        cv2.polylines(image, [pts], True, (180, 225, 255), self.config.edge_thickness, cv2.LINE_AA)

    def _draw_particles(self, image, center, width, height, angle, t):
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
                size = max(1, int(self.particle_size[i]))
                brightness = int(
                    130 + 90 * (0.5 + 0.5 * math.sin(t * 5.0 + self.particle_phase[i]))
                )
                cv2.circle(image, (x, y), size, (brightness // 2, brightness, 255), -1)
