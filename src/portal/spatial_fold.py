from __future__ import annotations

import math

import cv2
import numpy as np


class SpatialFoldEngine:
    """Apply a localized, animated spatial fold around the portal."""

    def __init__(self, strength=0.34, falloff=1.35, margin=40, work_scale=0.70):
        if strength < 0.0:
            raise ValueError("strength debe ser >= 0")
        if falloff <= 0.0:
            raise ValueError("falloff debe ser > 0")
        if margin < 0:
            raise ValueError("margin debe ser >= 0")
        if not 0.35 <= work_scale <= 1.0:
            raise ValueError("work_scale debe estar entre 0.35 y 1.0")
        self.strength = float(strength)
        self.falloff = float(falloff)
        self.margin = int(margin)
        self.work_scale = float(work_scale)

    def apply(self, frame, center, width, height, angle=0.0, intensity=1.0, motion=0.0, timestamp_ms=0):
        if frame.ndim != 3 or width <= 0 or height <= 0:
            return frame

        intensity = max(0.0, min(1.0, float(intensity)))
        if intensity <= 0.001:
            return frame

        strength = self.strength * intensity
        strength *= 1.0 + min(0.55, max(0.0, float(motion)) * 0.40)
        if strength <= 0.001:
            return frame

        h, w = frame.shape[:2]
        cx, cy = float(center[0]), float(center[1])
        angle = max(-25.0, min(25.0, float(angle)))
        half_w = width * 0.5
        half_h = height * 0.5

        extra_x = int(abs(math.sin(math.radians(angle))) * half_h + self.margin)
        extra_y = int(abs(math.sin(math.radians(angle))) * half_w + self.margin)
        radius_x = int(half_w + extra_x)
        radius_y = int(half_h + extra_y)

        x0 = max(0, int(round(cx - radius_x)))
        y0 = max(0, int(round(cy - radius_y)))
        x1 = min(w, int(round(cx + radius_x + 1)))
        y1 = min(h, int(round(cy + radius_y + 1)))
        if x1 - x0 < 16 or y1 - y0 < 16:
            return frame

        local = frame[y0:y1, x0:x1]
        lh, lw = local.shape[:2]
        scale = self.work_scale
        sw = max(16, int(round(lw * scale)))
        sh = max(16, int(round(lh * scale)))
        small = cv2.resize(local, (sw, sh), interpolation=cv2.INTER_AREA) if scale < 0.999 else local

        local_cx = (cx - x0) * (sw / lw)
        local_cy = (cy - y0) * (sh / lh)
        sx = np.arange(sw, dtype=np.float32)[None, :]
        sy = np.arange(sh, dtype=np.float32)[:, None]
        dx = (sx - local_cx) / max(half_w * (sw / lw), 1.0)
        dy = (sy - local_cy) / max(half_h * (sh / lh), 1.0)
        r = np.sqrt(dx * dx + dy * dy)

        ring = np.exp(-((r - 1.02) ** 2) * (self.falloff * 5.5))
        core = np.exp(-(r * r) * 1.7)
        wave = np.sin(r * 12.0 - timestamp_ms * 0.010) * 0.5 + 0.5
        fold = ring * (0.62 + 0.24 * core + 0.14 * wave)

        radial = strength * (16.0 + 5.0 * wave) * fold * np.clip(r, 0.0, 1.8)
        ux = np.divide(dx, r + 1e-5)
        uy = np.divide(dy, r + 1e-5)
        tangent = strength * (7.0 + 3.0 * wave) * fold

        # A subtle angular shear gives the neighborhood a folded-sheet feel.
        shear = strength * 3.5 * fold * np.sin(math.radians(angle))
        map_x = sx - radial * ux - tangent * uy + shear * dy
        map_y = sy - radial * uy + tangent * ux - shear * dx

        warped = cv2.remap(small, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        if scale < 0.999:
            warped = cv2.resize(warped, (lw, lh), interpolation=cv2.INTER_LINEAR)

        yy, xx = np.ogrid[:lh, :lw]
        edge_x = np.minimum(xx + 1, lw - xx) / max(lw * 0.18, 1.0)
        edge_y = np.minimum(yy + 1, lh - yy) / max(lh * 0.18, 1.0)
        edge_alpha = np.clip(np.minimum(edge_x, edge_y), 0.0, 1.0)
        edge_alpha = edge_alpha * edge_alpha * (3.0 - 2.0 * edge_alpha)
        alpha = (0.78 * strength * edge_alpha)[..., None]
        local[:] = np.clip(local.astype(np.float32) * (1.0 - alpha) + warped.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
        frame[y0:y1, x0:x1] = local
        return frame
