from __future__ import annotations

import math
import cv2
import numpy as np


class BackgroundFX:
    """Localized reality-warp effects around the portal, kept outside its interior."""

    def __init__(self, work_scale=0.55, margin=120):
        self.work_scale = float(work_scale)
        self.margin = int(margin)
        self._cache = {}

    def apply(self, frame, center, width, height, angle, intensity, motion=0.0, waves=(), hands=(), flash=0.0, timestamp_ms=0):
        if intensity <= 0.001 or width <= 0 or height <= 0:
            return frame
        h, w = frame.shape[:2]
        cx, cy = float(center[0]), float(center[1])
        pad = int(max(self.margin, width * 0.16, height * 0.55))
        x0, y0 = max(0, int(cx - width * 0.5 - pad)), max(0, int(cy - height * 0.5 - pad))
        x1, y1 = min(w, int(cx + width * 0.5 + pad)), min(h, int(cy + height * 0.5 + pad))
        if x1 - x0 < 24 or y1 - y0 < 24:
            return frame
        roi = frame[y0:y1, x0:x1]
        rh, rw = roi.shape[:2]
        scale = self.work_scale
        sw, sh = max(24, int(rw * scale)), max(24, int(rh * scale))
        small = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_AREA) if scale < .999 else roi.copy()
        lc = ((cx - x0) * sw / rw, (cy - y0) * sh / rh)
        yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
        dx, dy = xx - lc[0], yy - lc[1]
        rx, ry = max(20.0, width * .58 * sw / rw), max(20.0, height * 1.15 * sh / rh)
        r = np.sqrt((dx / rx) ** 2 + (dy / ry) ** 2)
        ring = np.exp(-((r - .72) ** 2) * 9.0)
        core = np.exp(-(r ** 2) * 2.0)
        t = timestamp_ms * .001
        ripple = .5 + .5 * np.sin(r * 22.0 - t * 5.0)
        strength = (.22 + .52 * float(intensity)) * (1.0 + .5 * min(1.0, float(motion)))
        radial = ring * (3.0 + 10.0 * strength) + core * 2.0 * strength * ripple
        ux = dx / (np.sqrt(dx * dx + dy * dy) + 1e-4)
        uy = dy / (np.sqrt(dx * dx + dy * dy) + 1e-4)
        # Fold the real camera image toward/away from the anomaly.
        map_x = xx - ux * radial
        map_y = yy - uy * radial
        warped = cv2.remap(small, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        alpha = np.clip(ring * (.35 + .45 * intensity) + core * .10 * intensity, 0.0, .72)[..., None]
        small = np.clip(small.astype(np.float32) * (1 - alpha) + warped.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

        # Local chromatic reality shear.
        chroma = np.clip(ring * (1.5 + 5.0 * intensity), 0, 5).astype(np.float32)
        b, g, rr = cv2.split(small)
        map_left = np.clip(xx - chroma, 0, sw - 1).astype(np.float32)
        map_right = np.clip(xx + chroma, 0, sw - 1).astype(np.float32)
        b = cv2.remap(b, map_left, yy, cv2.INTER_LINEAR)
        rr = cv2.remap(rr, map_right, yy, cv2.INTER_LINEAR)
        small = cv2.merge((b, g, rr))

        # Ripple rings emitted by fast fingertip movement.
        for wx, wy, wr, energy in waves:
            lx, ly = (wx - x0) * sw / rw, (wy - y0) * sh / rh
            rrng = np.sqrt((xx - lx) ** 2 + (yy - ly) ** 2)
            band = np.exp(-((rrng - wr * scale) ** 2) / 18.0) * max(0.0, energy)
            band = (band * (80.0 * intensity)).astype(np.uint8)
            small = cv2.add(small, np.repeat(band[:, :, None], 3, axis=2))

        # Sparse dimensional dust outside the portal.
        seed = int((timestamp_ms // 80) % 9973)
        rng = np.random.default_rng(seed)
        count = int(18 + 24 * intensity + 18 * motion)
        for _ in range(count):
            px = int(rng.uniform(0, sw))
            py = int(rng.uniform(0, sh))
            if ((px - lc[0]) / rx) ** 2 + ((py - lc[1]) / ry) ** 2 < .55:
                continue
            cv2.circle(small, (px, py), 1, int(90 + 100 * intensity), -1, cv2.LINE_AA)

        # Opening impulse: short radial flash in the surrounding space.
        if flash > .01:
            flash_alpha = np.clip(ring * flash * .28, 0, .28)[..., None]
            small = np.clip(small.astype(np.float32) * (1 - flash_alpha) + 255 * flash_alpha, 0, 255).astype(np.uint8)

        result = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_LINEAR) if scale < .999 else small
        # Keep the effect strongest near the anomaly and fade at the ROI edges.
        edge_x = np.minimum(np.arange(rw)[None, :] + 1, rw - np.arange(rw)[None, :]) / max(rw * .16, 1)
        edge_y = np.minimum(np.arange(rh)[:, None] + 1, rh - np.arange(rh)[:, None]) / max(rh * .16, 1)
        edge = np.clip(np.minimum(edge_x, edge_y), 0, 1)
        edge = edge * edge * (3 - 2 * edge)
        # Only a localized neighborhood is modified; no full-frame overlay.
        blend = (.72 * edge)[..., None]
        roi[:] = np.clip(roi.astype(np.float32) * (1 - blend) + result.astype(np.float32) * blend, 0, 255).astype(np.uint8)
        frame[y0:y1, x0:x1] = roi
        return frame
