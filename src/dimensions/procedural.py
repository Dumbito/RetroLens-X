from __future__ import annotations

import cv2
import numpy as np


class ProceduralDimension:
    """Animated procedural content rendered at a reduced internal resolution."""

    def __init__(self, work_scale: float = 0.5):
        self.work_scale = float(np.clip(work_scale, 0.25, 1.0))
        self.phase = 0.0
        self._grid_shape: tuple[int, int] | None = None
        self._nx: np.ndarray | None = None
        self._ny: np.ndarray | None = None
        self._radial: np.ndarray | None = None
        self._wave_x: np.ndarray | None = None
        self._wave_y: np.ndarray | None = None
        self._wave_xy: np.ndarray | None = None
        self._radial_weight: np.ndarray | None = None
        self._result: np.ndarray | None = None

    def _ensure_grid(self, work_width: int, work_height: int) -> None:
        shape = (work_height, work_width)
        if self._grid_shape == shape:
            return
        y, x = np.mgrid[0:work_height, 0:work_width]
        nx = x / max(work_width - 1, 1)
        ny = y / max(work_height - 1, 1)
        self._nx = nx
        self._ny = ny
        self._radial = np.sqrt((nx - 0.5) ** 2 + (ny - 0.5) ** 2)
        self._wave_x = nx * 18.0
        self._wave_y = ny * 14.0
        self._wave_xy = (nx + ny) * 24.0
        self._radial_weight = np.clip(1.2 - self._radial * 1.8, 0.0, 1.0)
        self._result = np.empty((work_height, work_width, 3), dtype=np.uint8)
        self._grid_shape = shape

    def render(self, width: int, height: int, timestamp_ms: int) -> np.ndarray:
        width = max(1, int(width))
        height = max(1, int(height))
        work_width = max(1, int(round(width * self.work_scale)))
        work_height = max(1, int(round(height * self.work_scale)))

        self._ensure_grid(work_width, work_height)
        wave_x = self._wave_x
        wave_y = self._wave_y
        wave_xy = self._wave_xy
        radial_weight = self._radial_weight
        result = self._result
        assert wave_x is not None and wave_y is not None and wave_xy is not None
        assert radial_weight is not None and result is not None

        t = timestamp_ms * 0.001
        wave1 = np.sin(wave_x + t * 2.0)
        wave2 = np.sin(wave_y - t * 1.5)
        wave3 = np.sin(wave_xy + t * 3.0)

        energy = (wave1 + wave2 + wave3) * (1.0 / 3.0)
        energy = (energy + 1.0) * 0.5
        energy *= radial_weight
        energy = np.clip(energy, 0.0, 1.0)

        np.multiply(energy, 255.0, out=result[:, :, 2], casting="unsafe")
        np.multiply(energy, energy, out=wave1)
        np.sqrt(energy, out=wave1)
        np.power(energy, 0.7, out=wave1)
        np.multiply(wave1, 180.0, out=result[:, :, 1], casting="unsafe")
        np.subtract(1.0, energy, out=wave1)
        np.multiply(wave1, 220.0, out=result[:, :, 0], casting="unsafe")

        if result.shape[:2] != (height, width):
            return cv2.resize(result, (width, height), interpolation=cv2.INTER_LINEAR)
        return result
