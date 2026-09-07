from __future__ import annotations

import cv2
import numpy as np


class ProceduralDimension:
    """Animated procedural dimension with layered depth/parallax."""

    def __init__(self, work_scale: float = 0.5):
        self.work_scale = float(np.clip(work_scale, 0.25, 1.0))
        self._grid_shape: tuple[int, int] | None = None
        self._nx: np.ndarray | None = None
        self._ny: np.ndarray | None = None
        self._radial: np.ndarray | None = None
        self._angle: np.ndarray | None = None
        self._result: np.ndarray | None = None
        self._layer: np.ndarray | None = None

    def _ensure_grid(self, work_width: int, work_height: int) -> None:
        shape = (work_height, work_width)
        if self._grid_shape == shape:
            return

        y, x = np.mgrid[0:work_height, 0:work_width]
        nx = x / max(work_width - 1, 1)
        ny = y / max(work_height - 1, 1)
        self._nx = nx.astype(np.float32)
        self._ny = ny.astype(np.float32)
        self._radial = np.sqrt((self._nx - 0.5) ** 2 + (self._ny - 0.5) ** 2).astype(np.float32)
        self._angle = np.arctan2(self._ny - 0.5, self._nx - 0.5).astype(np.float32)
        self._result = np.empty((work_height, work_width, 3), dtype=np.uint8)
        self._layer = np.empty((work_height, work_width), dtype=np.float32)
        self._grid_shape = shape

    def render(
        self,
        width: int,
        height: int,
        timestamp_ms: int,
        view_x: float = 0.0,
        view_y: float = 0.0,
        view_angle: float = 0.0,
    ) -> np.ndarray:
        width = max(1, int(width))
        height = max(1, int(height))
        work_width = max(1, int(round(width * self.work_scale)))
        work_height = max(1, int(round(height * self.work_scale)))

        self._ensure_grid(work_width, work_height)
        nx = self._nx
        ny = self._ny
        radial = self._radial
        angle = self._angle
        result = self._result
        layer = self._layer
        assert nx is not None and ny is not None and radial is not None
        assert angle is not None and result is not None and layer is not None

        t = timestamp_ms * 0.001
        vx = float(np.clip(view_x, -1.0, 1.0))
        vy = float(np.clip(view_y, -1.0, 1.0))
        va = float(view_angle)

        # Camera-relative parallax. Near layers move more than distant layers.
        px = nx + vx * (0.055 + radial * 0.12)
        py = ny + vy * (0.045 + radial * 0.10)
        local_r = np.sqrt((px - 0.5) ** 2 + (py - 0.5) ** 2)
        local_a = np.arctan2(py - 0.5, px - 0.5) + va * 0.22

        # Deep animated field: several frequencies create motion without a flat TV/static look.
        wave_a = np.sin(px * 15.0 + t * 1.8)
        wave_b = np.sin(py * 11.0 - t * 1.35)
        wave_c = np.sin((px + py) * 21.0 + t * 2.4)
        wave_d = np.sin(local_a * 7.0 - local_r * 30.0 - t * 3.2)
        energy = (wave_a * 0.22 + wave_b * 0.18 + wave_c * 0.20 + wave_d * 0.40)
        energy = (energy + 1.0) * 0.5

        # Vortex-like depth layers. The changing radius makes the interior feel recessed.
        layer.fill(0.0)
        depth_phase = (local_r * 8.5 - t * 0.75 + local_a * 0.18) % 1.0
        rings = np.exp(-((depth_phase - 0.5) ** 2) / 0.045)
        core = np.exp(-local_r * 9.0)
        edge_falloff = np.clip(1.15 - local_r * 2.0, 0.0, 1.0)
        energy = np.clip(energy * 0.62 + rings * 0.28 + core * 0.35, 0.0, 1.0)
        energy *= edge_falloff

        # Fine luminous filaments moving toward the center add depth cues.
        filaments = 0.5 + 0.5 * np.sin(local_a * 13.0 + local_r * 42.0 - t * 4.0)
        energy = np.clip(energy + filaments * core * 0.22, 0.0, 1.0)

        # BGR palette: dark interior with cyan/blue energy and a warm central highlight.
        np.multiply(energy, 205.0, out=result[:, :, 0], casting="unsafe")
        np.multiply(np.sqrt(energy), 225.0, out=result[:, :, 1], casting="unsafe")
        blue = np.clip(energy * 255.0 + core * 35.0, 0.0, 255.0)
        np.copyto(result[:, :, 2], blue.astype(np.uint8))

        # Subtle central hotspot gives the portal a sense of looking into a volume.
        result[:, :, 1] = np.clip(result[:, :, 1].astype(np.float32) + core * 22.0, 0, 255).astype(np.uint8)

        if result.shape[:2] != (height, width):
            return cv2.resize(result, (width, height), interpolation=cv2.INTER_LINEAR)
        return result
