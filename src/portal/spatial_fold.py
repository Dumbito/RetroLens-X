from __future__ import annotations

import math

import cv2
import numpy as np


class SpatialFoldEngine:
    """Apply a localized spatial-fold distortion around the portal.

    The effect is intentionally ROI-based: only the neighborhood around the
    portal is remapped, keeping the rest of the camera frame untouched.
    """

    def __init__(
        self,
        strength: float = 0.34,
        falloff: float = 1.35,
        margin: int = 34,
        work_scale: float = 0.70,
    ) -> None:
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
        self._map_cache: dict[tuple, tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]] = {}

    def apply(self, frame, center, width, height, angle=0.0, intensity=1.0, motion=0.0):
        if frame.ndim != 3 or width <= 0 or height <= 0:
            return frame

        intensity = max(0.0, min(1.0, float(intensity)))
        if intensity <= 0.001:
            return frame

        strength = self.strength * intensity
        strength *= 1.0 + min(0.45, max(0.0, float(motion)) * 0.35)
        if strength <= 0.001:
            return frame

        h, w = frame.shape[:2]
        cx, cy = float(center[0]), float(center[1])
        angle = max(-25.0, min(25.0, float(angle)))
        half_w = width * 0.5
        half_h = height * 0.5

        # The distortion extends beyond the portal edge, but decays quickly.
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

        # Ring-shaped displacement: strongest close to the portal boundary,
        # fading both toward the center and toward the outer neighborhood.
        ring = np.exp(-((r - 1.02) ** 2) * (self.falloff * 5.5))
        core = np.exp(-(r * r) * 1.7)
        fold = ring * (0.72 + 0.28 * core)

        radial = strength * 16.0 * fold * np.clip(r, 0.0, 1.8)
        ux = np.divide(dx, r + 1e-5)
        uy = np.divide(dy, r + 1e-5)

        # Tangential component makes the fold feel like a sheet bending in 3D.
        tangent = strength * 7.0 * fold
        map_x = (sx - radial * ux - tangent * uy).astype(np.float32)
        map_y = (sy - radial * uy + tangent * ux).astype(np.float32)

        warped = cv2.remap(small, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        if scale < 0.999:
            warped = cv2.resize(warped, (lw, lh), interpolation=cv2.INTER_LINEAR)

        # Keep the fold localized with a soft elliptical falloff at the ROI edge.
        yy, xx = np.ogrid[:lh, :lw]
        edge_x = np.minimum(xx + 1, lw - xx) / max(lw * 0.18, 1.0)
        edge_y = np.minimum(yy + 1, lh - yy) / max(lh * 0.18, 1.0)
        edge_alpha = np.clip(np.minimum(edge_x, edge_y), 0.0, 1.0)
        edge_alpha = edge_alpha * edge_alpha * (3.0 - 2.0 * edge_alpha)
        alpha = (0.82 * strength * edge_alpha)[..., None]
        local[:] = np.clip(local.astype(np.float32) * (1.0 - alpha) + warped.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
        frame[y0:y1, x0:x1] = local
        return frame
