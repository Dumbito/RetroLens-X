from __future__ import annotations

import math

import cv2
import numpy as np


class DimensionalUniverse:
    """Procedural cosmic scene: stars, nebulae, galaxy core, planets and depth particles."""

    def __init__(self, work_scale: float = 0.42, max_width: int = 420, max_height: int = 300, seed: int = 7319):
        self.work_scale = float(np.clip(work_scale, 0.25, 0.75))
        self.max_width = int(max_width)
        self.max_height = int(max_height)
        rng = np.random.default_rng(seed)
        self._stars = rng.random((180, 5), dtype=np.float32)
        self._dust = rng.random((70, 5), dtype=np.float32)
        self._planets = np.array([
            [0.73, 0.28, 0.045, 0.82, 0.15],
            [0.24, 0.66, 0.026, 0.48, 0.72],
            [0.80, 0.70, 0.018, 0.63, 0.34],
        ], dtype=np.float32)
        self._size = None
        self._canvas = None
        self._nebula = None
        self._glow = None

    def _ensure(self, w: int, h: int):
        if self._size == (w, h):
            return
        self._size = (w, h)
        self._canvas = np.zeros((h, w, 3), np.uint8)
        self._nebula = np.empty((h, w), np.float32)
        self._glow = np.empty((h, w), np.float32)

    def render(self, width: int, height: int, timestamp_ms: int, view_x=0.0, view_y=0.0, intensity=1.0, turbulence=0.0):
        width, height = max(2, int(width)), max(2, int(height))
        ww = min(self.max_width, max(32, int(width * self.work_scale)))
        wh = min(self.max_height, max(24, int(height * self.work_scale)))
        self._ensure(ww, wh)
        canvas = self._canvas
        canvas.fill(0)
        t = timestamp_ms * 0.001
        vx = float(np.clip(view_x, -1, 1))
        vy = float(np.clip(view_y, -1, 1))
        yy, xx = np.mgrid[0:wh, 0:ww].astype(np.float32)
        nx, ny = xx / max(ww - 1, 1), yy / max(wh - 1, 1)

        # Layered nebula: several soft moving fields create a deep colored cloud.
        n1 = np.sin(nx * 8.0 + t * 0.35) * np.cos(ny * 6.0 - t * 0.28)
        n2 = np.sin((nx + ny) * 15.0 - t * 0.55)
        n3 = np.cos((nx * 2.2 - ny * 3.1) * 11.0 + t * 0.22)
        cloud = np.clip((n1 * 0.34 + n2 * 0.22 + n3 * 0.18 + 0.58), 0.0, 1.0)
        diagonal = np.exp(-((ny - (0.80 - 0.46 * nx + 0.025 * np.sin(t))) ** 2) * 28.0)
        radial = np.sqrt((nx - 0.52) ** 2 + (ny - 0.52) ** 2)
        nebula = np.clip(cloud * diagonal * 1.35 + np.exp(-radial * 7.0) * 0.20, 0, 1)
        self._nebula[:] = nebula

        # BGR cosmic palette with magenta/cyan highlights.
        canvas[:, :, 0] = np.clip(32 + nebula * 150, 0, 255).astype(np.uint8)
        canvas[:, :, 1] = np.clip(12 + nebula * 82, 0, 255).astype(np.uint8)
        canvas[:, :, 2] = np.clip(45 + nebula * 190, 0, 255).astype(np.uint8)

        # Galaxy spiral: rotating arms around the center.
        gcx, gcy = ww * (0.52 + vx * 0.035), wh * (0.53 + vy * 0.035)
        gx, gy = xx - gcx, yy - gcy
        gr = np.sqrt(gx * gx + gy * gy) / max(ww, wh)
        ga = np.arctan2(gy, gx)
        arm1 = np.exp(-((np.sin(ga * 2.0 + gr * 30.0 - t * 0.7)) ** 2) / 0.10) * np.exp(-gr * 5.0)
        arm2 = np.exp(-((np.sin(ga * 2.0 + gr * 30.0 - t * 0.7 + math.pi)) ** 2) / 0.10) * np.exp(-gr * 5.0)
        galaxy = np.clip((arm1 + arm2) * 0.34 + np.exp(-gr * 13.0) * 0.72, 0, 1)
        canvas[:, :, 0] = np.clip(canvas[:, :, 0].astype(np.float32) + galaxy * 70, 0, 255).astype(np.uint8)
        canvas[:, :, 2] = np.clip(canvas[:, :, 2].astype(np.float32) + galaxy * 120, 0, 255).astype(np.uint8)

        # Fixed-depth stars with slow parallax, plus larger foreground stars.
        for i, p in enumerate(self._stars):
            depth = 0.35 + float(p[2]) * 0.9
            sx = ((float(p[0]) + vx * 0.06 * depth + t * (0.002 + (i % 3) * 0.0007)) % 1.0) * ww
            sy = ((float(p[1]) + vy * 0.05 * depth + t * (0.001 + (i % 4) * 0.0005)) % 1.0) * wh
            if ((sx - gcx) / max(ww * 0.5, 1)) ** 2 + ((sy - gcy) / max(wh * 0.5, 1)) ** 2 < 0.08:
                continue
            radius = 1 if p[3] < 0.88 else 2
            value = int(105 + 145 * float(p[4]) * float(intensity))
            cv2.circle(canvas, (int(sx), int(sy)), radius, (value, value, min(255, value + 25)), -1, cv2.LINE_AA)

        # Cosmic dust streams flowing toward the vanishing point.
        for p in self._dust:
            depth = 0.3 + float(p[2]) * 1.4
            angle = float(p[0]) * math.tau + t * (0.10 + depth * 0.12)
            radius = (0.12 + float(p[1]) * 0.55) * min(ww, wh)
            px = gcx + math.cos(angle) * radius
            py = gcy + math.sin(angle) * radius * 0.62
            length = 2 + int(depth * 8)
            ex = px + math.cos(angle) * length
            ey = py + math.sin(angle) * length * 0.62
            if -10 < px < ww + 10 and -10 < py < wh + 10:
                cv2.line(canvas, (int(px), int(py)), (int(ex), int(ey)), (95, 70, 165), 1, cv2.LINE_AA)

        # Three planets/moons at different depths; no external assets required.
        for pxn, pyn, radius_n, depth, phase in self._planets:
            px = int(((pxn + vx * 0.055 * depth) % 1.0) * ww)
            py = int(((pyn + vy * 0.045 * depth) % 1.0) * wh)
            r = max(3, int(min(ww, wh) * radius_n * (0.65 + depth * 0.55)))
            cv2.circle(canvas, (px, py), r + 2, (55, 35, 105), -1, cv2.LINE_AA)
            cv2.circle(canvas, (px - r // 3, py - r // 4), r, (75, 45, 125), -1, cv2.LINE_AA)
            cv2.ellipse(canvas, (px - r // 4, py - r // 5), (max(1, r // 3), max(1, r // 5)), -18, 0, 360, (125, 75, 155), 1, cv2.LINE_AA)

        # Bright dimensional core.
        core = np.exp(-((xx - gcx) ** 2 + (yy - gcy) ** 2) / max((min(ww, wh) * 0.085) ** 2, 1.0))
        glow = np.clip(core * (0.55 + 0.45 * math.sin(t * 4.0) * 0.15 + turbulence * 0.25), 0, 1)
        canvas[:, :, 0] = np.clip(canvas[:, :, 0].astype(np.float32) + glow * 120, 0, 255).astype(np.uint8)
        canvas[:, :, 2] = np.clip(canvas[:, :, 2].astype(np.float32) + glow * 150, 0, 255).astype(np.uint8)

        # Directional motion vignette makes the scene feel like a tunnel.
        vignette = np.clip(1.25 - radial * 1.25, 0.25, 1.0)
        canvas[:] = np.clip(canvas.astype(np.float32) * vignette[..., None], 0, 255).astype(np.uint8)

        if (ww, wh) != (width, height):
            return cv2.resize(canvas, (width, height), interpolation=cv2.INTER_LINEAR)
        return canvas.copy()
