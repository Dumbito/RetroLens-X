from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass
class PortalSceneConfig:
    stars: int = 90
    dust: int = 34
    planets: int = 3
    opacity: float = 0.62
    parallax: float = 0.10


class Portal3DScene:
    """Lightweight procedural pseudo-3D scene for the portal interior."""

    def __init__(self, config: PortalSceneConfig | None = None) -> None:
        self.config = config or PortalSceneConfig()
        rng = np.random.default_rng(4817)
        self._stars = rng.random((self.config.stars, 4), dtype=np.float32)
        self._dust = rng.random((self.config.dust, 4), dtype=np.float32)
        self._planet_seed = rng.random((self.config.planets, 5), dtype=np.float32)

    def render(self, width: int, height: int, timestamp_ms: int, view_x: float = 0.0, view_y: float = 0.0, intensity: float = 1.0) -> np.ndarray:
        width, height = max(32, int(width)), max(32, int(height))
        t = timestamp_ms * 0.001
        image = np.zeros((height, width, 3), np.uint8)
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        nx = (xx - width * 0.5) / max(width * 0.5, 1.0)
        ny = (yy - height * 0.5) / max(height * 0.5, 1.0)
        radial = np.sqrt(nx * nx + ny * ny)
        vignette = np.clip(1.0 - radial * 0.72, 0.0, 1.0)
        image[:, :, 0] = np.clip(7 + 18 * vignette, 0, 255)
        image[:, :, 1] = np.clip(7 + 17 * vignette, 0, 255)
        image[:, :, 2] = np.clip(10 + 24 * vignette, 0, 255)

        for i, p in enumerate(self._stars):
            depth = 0.25 + 0.75 * float(p[2])
            sx = ((float(p[0]) - 0.5) * width + view_x * width * self.config.parallax / depth) % width
            sy = ((float(p[1]) - 0.5) * height + view_y * height * self.config.parallax / depth) % height
            value = int(80 + 145 * (0.55 + 0.45 * math.sin(t * (0.7 + p[2]) + i)))
            cv2.circle(image, (int(sx), int(sy)), 1 if p[3] < 0.94 else 2, (value, value, min(255, value + 20)), -1, cv2.LINE_AA)

        for i, p in enumerate(self._dust):
            phase = t * (0.10 + p[2] * 0.08) + p[0] * math.tau
            x = (float(p[0]) * width + math.sin(phase) * width * 0.08 + view_x * width * 0.06) % width
            y = (float(p[1]) * height + math.cos(phase * 0.8) * height * 0.05 + view_y * height * 0.04) % height
            cv2.circle(image, (int(x), int(y)), 1, (120, 112, 102), -1, cv2.LINE_AA)

        cx = width * 0.5 + view_x * width * 0.045
        cy = height * 0.5 + view_y * height * 0.035
        max_r = min(width, height) * 0.42
        for r in range(int(max_r), 2, -7):
            a = max(0.0, 1.0 - r / max_r)
            value = int(18 + 50 * a * intensity)
            cv2.ellipse(image, (int(cx), int(cy)), (r, max(2, int(r * 0.38))), 0, 0, 360, (value, value + 2, value + 5), 1, cv2.LINE_AA)

        for i, p in enumerate(self._planet_seed):
            orbit = min(width, height) * (0.16 + 0.08 * i)
            phase = t * (0.12 + 0.06 * i) + float(p[0]) * math.tau
            px = cx + math.cos(phase) * orbit + view_x * width * (0.025 + 0.015 * i)
            py = cy + math.sin(phase) * orbit * 0.42 + view_y * height * (0.02 + 0.012 * i)
            radius = max(7, int(min(width, height) * (0.035 + 0.012 * p[1])))
            self._planet(image, (px, py), radius, i, intensity)

        glow = np.zeros_like(image)
        cv2.circle(glow, (int(cx), int(cy)), max(4, int(min(width, height) * 0.06)), (100, 100, 100), -1, cv2.LINE_AA)
        glow = cv2.GaussianBlur(glow, (0, 0), max(3.0, min(width, height) * 0.035))
        return cv2.addWeighted(image, 1.0, glow, 0.38 * intensity, 0.0)

    @staticmethod
    def _planet(image: np.ndarray, center: tuple[float, float], radius: int, index: int, intensity: float) -> None:
        cx, cy = int(center[0]), int(center[1])
        y0, y1 = max(0, cy - radius), min(image.shape[0], cy + radius + 1)
        x0, x1 = max(0, cx - radius), min(image.shape[1], cx + radius + 1)
        if x1 <= x0 or y1 <= y0:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        nx = (xx - cx) / radius
        ny = (yy - cy) / radius
        rr = nx * nx + ny * ny
        inside = rr <= 1.0
        nz = np.sqrt(np.clip(1.0 - rr, 0.0, 1.0))
        diffuse = np.clip(nx * -0.45 + ny * -0.35 + nz * 0.82, 0.0, 1.0)
        rim = np.clip(1.0 - nz, 0.0, 1.0) ** 2
        palette = ((150, 135, 122), (126, 140, 146), (156, 147, 126))[index % 3]
        local = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
        for channel, base in enumerate(palette):
            local[:, :, channel] = np.clip(base * (0.20 + 0.82 * diffuse) + 28 * rim * intensity, 0, 255)
        local[~inside] = 0
        current = image[y0:y1, x0:x1]
        image[y0:y1, x0:x1] = np.maximum(current, local)
