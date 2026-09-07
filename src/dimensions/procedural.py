from __future__ import annotations

import cv2
import numpy as np


class ProceduralDimension:
    """Procedural cosmic scene: deep space, stars, nebulae and planets."""

    def __init__(self, work_scale: float = 0.5, max_work_width: int = 420, max_work_height: int = 300):
        self.work_scale = float(np.clip(work_scale, 0.25, 1.0))
        self.max_work_width = max(64, int(max_work_width))
        self.max_work_height = max(48, int(max_work_height))
        self._grid_shape = None
        self._nx = self._ny = self._radial = None
        self._result = self._energy = None
        self._star_x = self._star_y = self._star_r = self._star_b = None
        self._star_phase = self._star_twinkle = None
        self._nebula = self._dust = self._planet = self._scratch = None

    def _ensure_grid(self, w: int, h: int) -> None:
        shape = (h, w)
        if self._grid_shape == shape:
            return
        y, x = np.mgrid[0:h, 0:w]
        self._nx = x.astype(np.float32) / max(w - 1, 1)
        self._ny = y.astype(np.float32) / max(h - 1, 1)
        dx = self._nx - 0.5
        dy = self._ny - 0.5
        self._radial = np.sqrt(dx * dx + dy * dy, dtype=np.float32)
        self._result = np.empty((h, w, 3), dtype=np.uint8)
        self._energy = np.empty((h, w), dtype=np.float32)
        rng = np.random.default_rng(1729)
        count = max(180, min(700, int(w * h / 130)))
        self._star_x = rng.uniform(0.02, 0.98, count).astype(np.float32)
        self._star_y = rng.uniform(0.02, 0.98, count).astype(np.float32)
        self._star_r = rng.choice(np.array([0.45, 0.65, 0.9, 1.25, 1.8], np.float32), count).astype(np.float32)
        self._star_b = rng.uniform(0.55, 1.0, count).astype(np.float32)
        self._star_phase = rng.uniform(0.0, 6.28318, count).astype(np.float32)
        self._star_twinkle = rng.uniform(0.4, 2.0, count).astype(np.float32)
        self._nebula = np.empty((h, w), np.float32)
        self._dust = np.empty((h, w), np.float32)
        self._planet = np.empty((h, w), np.float32)
        self._scratch = np.empty((h, w), np.float32)
        self._grid_shape = shape

    @staticmethod
    def _gaussian(x, y, cx, cy, sx, sy):
        return np.exp(-(((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2) * 0.5)

    def _draw_stars(self, image, t, vx, vy):
        h, w = image.shape[:2]
        for i in range(len(self._star_x)):
            x = int((self._star_x[i] + vx * 0.035) % 1.0 * (w - 1))
            y = int((self._star_y[i] + vy * 0.025) % 1.0 * (h - 1))
            twinkle = 0.72 + 0.28 * np.sin(t * self._star_twinkle[i] + self._star_phase[i])
            radius = float(self._star_r[i] * (0.75 + 0.25 * twinkle))
            brightness = int(np.clip(125 + 120 * self._star_b[i] * twinkle, 80, 255))
            if radius > 1.35:
                cv2.line(image, (max(0, x - int(radius * 2)), y), (min(w - 1, x + int(radius * 2)), y), (brightness // 2, brightness // 2, brightness), 1)
                cv2.line(image, (x, max(0, y - int(radius * 2))), (x, min(h - 1, y + int(radius * 2))), (brightness, brightness // 2, brightness // 2), 1)
            cv2.circle(image, (x, y), max(1, int(round(radius))), (brightness, brightness, brightness), -1)

    def render(self, width: int, height: int, timestamp_ms: int, view_x: float = 0.0, view_y: float = 0.0, view_angle: float = 0.0) -> np.ndarray:
        width = max(1, int(width))
        height = max(1, int(height))
        requested_w = max(1, int(round(width * self.work_scale)))
        requested_h = max(1, int(round(height * self.work_scale)))
        w = max(1, min(requested_w, self.max_work_width))
        h = max(1, min(requested_h, self.max_work_height))
        self._ensure_grid(w, h)

        nx, ny = self._nx, self._ny
        t = timestamp_ms * 0.001
        vx = float(np.clip(view_x, -1, 1))
        vy = float(np.clip(view_y, -1, 1))

        # Deep-space base: nearly black, with restrained natural warm/cool gradients.
        neb = self._nebula
        neb[:] = 0.0
        for cx, cy, sx, sy, strength in ((0.28, 0.30, 0.26, 0.17, 0.70), (0.73, 0.62, 0.30, 0.20, 0.52), (0.48, 0.82, 0.22, 0.13, 0.38)):
            neb += self._gaussian(nx, ny, cx + np.sin(t * 0.05) * 0.02, cy, sx, sy) * strength
        neb *= 0.5 + 0.5 * np.sin(nx * 9.0 + ny * 6.0 + t * 0.10)
        neb = np.clip(neb, 0, 1)

        dust = self._dust
        dust[:] = np.sin(nx * 43.0 + ny * 7.0 + t * 0.35) * np.sin(ny * 31.0 - nx * 5.0)
        dust *= 0.5
        dust += 0.5
        dust *= neb

        result = self._result
        result[:, :, 0] = np.clip(4 + neb * 34 + dust * 18, 0, 255).astype(np.uint8)
        result[:, :, 1] = np.clip(6 + neb * 44 + dust * 25, 0, 255).astype(np.uint8)
        result[:, :, 2] = np.clip(8 + neb * 48 + dust * 28, 0, 255).astype(np.uint8)

        # Broad warm stellar-cloud lane: keeps the palette from becoming only blue/purple.
        warm = np.exp(-((ny - (0.48 + 0.11 * np.sin(nx * 4.5 + t * 0.08))) / 0.115) ** 2)
        result[:, :, 0] = np.clip(result[:, :, 0].astype(np.float32) + warm * 32, 0, 255).astype(np.uint8)
        result[:, :, 1] = np.clip(result[:, :, 1].astype(np.float32) + warm * 24, 0, 255).astype(np.uint8)
        result[:, :, 2] = np.clip(result[:, :, 2].astype(np.float32) + warm * 10, 0, 255).astype(np.uint8)

        self._draw_stars(result, t, vx, vy)

        # Three subtle planets: one large limb, two distant bodies. They are composited
        # procedurally and remain secondary to the live camera portal.
        for cx, cy, radius, light, rgb in ((0.74, 0.30, 0.105, (-0.35, -0.35), (112, 142, 168)), (0.22, 0.72, 0.055, (-0.5, -0.4), (156, 126, 82)), (0.86, 0.78, 0.032, (-0.6, -0.5), (105, 125, 132))):
            px = nx - (cx + vx * radius * 0.12)
            py = ny - (cy + vy * radius * 0.10)
            rr = np.sqrt(px * px + py * py)
            mask = rr < radius
            if not np.any(mask):
                continue
            z = np.clip(np.sqrt(np.maximum(0.0, 1.0 - (rr / radius) ** 2)), 0, 1)
            lightx, lighty = light
            lambert = np.clip(((-px * lightx) + (-py * lighty)) / radius, 0, 1) * z
            atmosphere = np.clip((1.0 - rr / radius) * 0.65, 0, 1)
            for c, base in enumerate(rgb):
                plane = result[:, :, c].astype(np.float32)
                shade = base * (0.10 + 0.90 * lambert)
                plane[mask] = np.clip(plane[mask] * 0.22 + shade[mask], 0, 255)
                plane[mask] += atmosphere[mask] * (24 if c == 2 else 10)
                result[:, :, c] = np.clip(plane, 0, 255).astype(np.uint8)
            # faint limb rim
            rim = np.clip((rr - radius * 0.91) / (radius * 0.09), 0, 1)
            rim *= mask
            result[:, :, 0] = np.clip(result[:, :, 0].astype(np.float32) + rim * 18, 0, 255).astype(np.uint8)
            result[:, :, 1] = np.clip(result[:, :, 1].astype(np.float32) + rim * 22, 0, 255).astype(np.uint8)
            result[:, :, 2] = np.clip(result[:, :, 2].astype(np.float32) + rim * 28, 0, 255).astype(np.uint8)

        # Central dimensional core and subtle concentric depth rings.
        dx = nx - 0.5 + vx * 0.05
        dy = ny - 0.5 + vy * 0.05
        r = np.sqrt(dx * dx + dy * dy)
        core = np.exp(-r * 10.0)
        rings = 0.5 + 0.5 * np.sin(r * 34.0 - t * 1.8 + np.sin(np.arctan2(dy, dx) * 5.0) * 1.5)
        rings *= np.exp(-r * 3.2) * 0.16
        result[:, :, 0] = np.clip(result[:, :, 0].astype(np.float32) + core * 28 + rings * 15, 0, 255).astype(np.uint8)
        result[:, :, 1] = np.clip(result[:, :, 1].astype(np.float32) + core * 42 + rings * 20, 0, 255).astype(np.uint8)
        result[:, :, 2] = np.clip(result[:, :, 2].astype(np.float32) + core * 50 + rings * 24, 0, 255).astype(np.uint8)

        if result.shape[:2] != (height, width):
            result = cv2.resize(result, (width, height), interpolation=cv2.INTER_LINEAR)
        return result
