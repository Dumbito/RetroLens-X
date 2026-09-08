from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass
class PortalVisualConfig:
    edge_thickness: int = 4
    glow_sigma: float = 13.0
    roi_margin: int = 42
    feather: int = 3
    chromatic_offset: int = 5
    glow_scale: float = 0.38
    scanline_strength: float = 0.065
    aberration_strength: float = 0.55
    filter_opacity: float = 0.16
    glass_noise: float = 0.025
    corner_glow: float = 1.0
    max_effect_pixels: int = 260_000
    min_effect_scale: float = 0.50


class PortalRenderer:
    """Rectangular holographic video filter with adaptive-resolution VFX."""

    def __init__(self, config: PortalVisualConfig | None = None) -> None:
        self.config = config or PortalVisualConfig()
        self._output_cache: dict[tuple[int, int, int], np.ndarray] = {}
        self._border_cache: dict[tuple[int, int, int], np.ndarray] = {}
        self._effect_cache: dict[tuple[int, int, int], np.ndarray] = {}
        self._rng = np.random.default_rng(7319)

    @staticmethod
    def _buffer(cache, shape):
        key = (shape[1], shape[0], shape[2] if len(shape) == 3 else 1)
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
        rotation = np.array([[c, -s], [s, c]], np.float32)
        return p @ rotation.T + np.array([cx, cy], np.float32)

    def _effect_scale(self, width, height):
        pixels = max(1, int(width) * int(height))
        if pixels <= self.config.max_effect_pixels:
            return 1.0
        return max(self.config.min_effect_scale, math.sqrt(self.config.max_effect_pixels / pixels))

    def render(self, frame, dimension, center, width, height, angle, timestamp_ms, intensity=1.0, motion=0.0):
        if width <= 0 or height <= 0 or frame.ndim != 3:
            return frame
        intensity = max(0.0, min(1.0, float(intensity)))
        motion = max(0.0, min(1.0, float(motion)))
        if intensity <= 0.0:
            return frame

        frame_h, frame_w = frame.shape[:2]
        cx, cy = int(round(center[0])), int(round(center[1]))
        angle = max(-25.0, min(25.0, float(angle)))
        target_w = max(2, int(round(width)))
        target_h = max(2, int(round(height)))
        t = timestamp_ms * 0.001

        quad_global = self._rotated_corners((cx, cy), target_w, target_h, angle)
        margin = max(int(self.config.roi_margin), int(self.config.glow_sigma * 2.5))
        x0 = max(0, int(math.floor(quad_global[:, 0].min())) - margin)
        y0 = max(0, int(math.floor(quad_global[:, 1].min())) - margin)
        x1 = min(frame_w, int(math.ceil(quad_global[:, 0].max())) + margin + 1)
        y1 = min(frame_h, int(math.ceil(quad_global[:, 1].max())) + margin + 1)
        if x1 <= x0 or y1 <= y0:
            return frame

        local = frame[y0:y1, x0:x1]
        quad = quad_global - np.array([x0, y0], np.float32)

        content = dimension
        if content.shape[:2] != (target_h, target_w):
            content = cv2.resize(content, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        src = np.array(
            [[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]],
            np.float32,
        )
        transform = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
        warped = cv2.warpPerspective(
            content,
            transform,
            (local.shape[1], local.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

        mask = self._rectangle_mask(local.shape[1], local.shape[0], quad, self.config.feather)
        alpha = (mask.astype(np.float32) / 255.0 * intensity)[:, :, None]
        output = self._buffer(self._output_cache, local.shape)
        output[:] = np.clip(
            local.astype(np.float32) * (1.0 - alpha) + warped.astype(np.float32) * alpha,
            0,
            255,
        ).astype(np.uint8)

        # Expensive image-space effects run at a capped resolution for large portals.
        # The final composite remains full-resolution, so the portal can still fill the screen.
        scale = self._effect_scale(target_w, target_h)
        effect_w = max(2, int(round(local.shape[1] * scale)))
        effect_h = max(2, int(round(local.shape[0] * scale)))
        if scale < 0.999:
            effect = self._buffer(self._effect_cache, (effect_h, effect_w, 3))
            cv2.resize(output, (effect_w, effect_h), dst=effect, interpolation=cv2.INTER_AREA)
        else:
            effect = output

        tint = np.empty_like(effect)
        tint[:] = (150, 48, 205)
        tint_alpha = self.config.filter_opacity * (0.72 + 0.28 * intensity)
        filtered = cv2.addWeighted(effect, 1.0 - tint_alpha, tint, tint_alpha, 0.0)
        self._scanlines(filtered, intensity)
        self._glass_noise(filtered, intensity, t)
        self._chromatic_split(filtered, intensity, motion)

        if scale < 0.999:
            filtered_full = self._buffer(self._output_cache, local.shape)
            cv2.resize(filtered, (local.shape[1], local.shape[0]), dst=filtered_full, interpolation=cv2.INTER_LINEAR)
            filtered = filtered_full
        self._clip_to_mask(output, local, filtered, mask, intensity)

        border = self._buffer(self._border_cache, (local.shape[0], local.shape[1], 3))
        border.fill(0)
        self._draw_hologram_border(border, quad, t, intensity, motion)

        # Glow is also rendered at reduced resolution when the portal becomes large.
        glow_scale = self._effect_scale(target_w, target_h)
        if glow_scale < 0.999:
            glow_w = max(2, int(round(border.shape[1] * glow_scale)))
            glow_h = max(2, int(round(border.shape[0] * glow_scale)))
            glow_small = self._buffer(self._effect_cache, (glow_h, glow_w, 3))
            cv2.resize(border, (glow_w, glow_h), dst=glow_small, interpolation=cv2.INTER_AREA)
            glow_small = cv2.GaussianBlur(glow_small, (0, 0), max(2.0, self.config.glow_sigma * glow_scale))
            glow = cv2.resize(glow_small, (border.shape[1], border.shape[0]), interpolation=cv2.INTER_LINEAR)
        else:
            glow = cv2.GaussianBlur(border, (0, 0), self.config.glow_sigma)

        output[:] = np.clip(
            output.astype(np.float32) + glow.astype(np.float32) * self.config.glow_scale * intensity,
            0,
            255,
        ).astype(np.uint8)
        self._composite_border(output, border, intensity)
        self._corner_sparks(output, quad, t, intensity, motion)

        frame[y0:y1, x0:x1] = output
        return frame

    def _rectangle_mask(self, width, height, quad, feather):
        mask = np.zeros((height, width), np.uint8)
        cv2.fillConvexPoly(mask, np.round(quad).astype(np.int32), 255, cv2.LINE_AA)
        f = max(0, int(feather))
        if f:
            mask[:] = cv2.GaussianBlur(mask, (f * 2 + 1, f * 2 + 1), f * 0.55)
        return mask

    @staticmethod
    def _clip_to_mask(output, base, filtered, mask, intensity):
        alpha = (mask.astype(np.float32) / 255.0 * intensity)[:, :, None]
        output[:] = np.clip(
            base.astype(np.float32) * (1.0 - alpha) + filtered.astype(np.float32) * alpha,
            0,
            255,
        ).astype(np.uint8)

    def _scanlines(self, image, intensity):
        strength = self.config.scanline_strength * intensity
        rows = np.arange(image.shape[0], dtype=np.float32)
        modulation = 0.5 + 0.5 * np.cos(rows * math.pi)
        factor = 1.0 - strength * modulation
        image[:] = np.clip(image.astype(np.float32) * factor[:, None, None], 0, 255).astype(np.uint8)

    def _glass_noise(self, image, intensity, t):
        if self.config.glass_noise <= 0:
            return
        h, w = image.shape[:2]
        noise = self._rng.normal(0.0, 255.0 * self.config.glass_noise * intensity, (h, w, 1)).astype(np.float32)
        image[:] = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    def _chromatic_split(self, image, intensity, motion):
        amount = int(round(self.config.chromatic_offset * (0.45 + self.config.aberration_strength * intensity + 0.30 * motion)))
        if amount <= 0:
            return
        shifted_r = np.roll(image[:, :, 2], amount, axis=1)
        shifted_b = np.roll(image[:, :, 0], -amount, axis=1)
        edge = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 45, 110)
        edge_mask = edge > 0
        image[:, :, 2][edge_mask] = shifted_r[edge_mask]
        image[:, :, 0][edge_mask] = shifted_b[edge_mask]

    @staticmethod
    def _draw_hologram_border(image, quad, t, intensity, motion):
        pts = np.round(quad).astype(np.int32)
        cv2.polylines(image, [pts], True, (235, 70, 255), 2 + int(intensity * 2), cv2.LINE_AA)
        cv2.polylines(image, [pts], True, (255, 190, 255), 1, cv2.LINE_AA)
        edges = list(zip(pts, np.roll(pts, -1, axis=0)))
        for i, (a, b) in enumerate(edges):
            phase = (t * (0.55 + i * 0.13) + i * 0.9) % 1.0
            length = 0.08 + 0.06 * motion
            start = phase
            end = min(1.0, start + length)
            p0 = a.astype(np.float32) * (1.0 - start) + b.astype(np.float32) * start
            p1 = a.astype(np.float32) * (1.0 - end) + b.astype(np.float32) * end
            cv2.line(image, tuple(np.round(p0).astype(int)), tuple(np.round(p1).astype(int)), (255, 255, 255), 1 + int(motion), cv2.LINE_AA)

    def _composite_border(self, image, border, intensity):
        alpha = (np.max(border.astype(np.float32), axis=2) / 255.0 * (0.50 + 0.50 * intensity))[:, :, None]
        image[:] = np.clip(
            image.astype(np.float32) * (1.0 - alpha) + border.astype(np.float32) * alpha,
            0,
            255,
        ).astype(np.uint8)

    def _corner_sparks(self, image, quad, t, intensity, motion):
        pts = np.round(quad).astype(np.int32)
        radius = int(4 + 4 * intensity + 3 * motion)
        for i, (x, y) in enumerate(pts):
            pulse = 0.5 + 0.5 * math.sin(t * 5.0 + i * 1.7)
            r = max(1, int(radius * (0.55 + 0.45 * pulse)))
            cv2.circle(image, (int(x), int(y)), r, (255, 95, 245), 1, cv2.LINE_AA)
            if pulse > 0.78:
                cv2.line(image, (x - r * 2, y), (x + r * 2, y), (255, 220, 255), 1, cv2.LINE_AA)
                cv2.line(image, (x, y - r * 2), (x, y + r * 2), (255, 220, 255), 1, cv2.LINE_AA)
