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
        self._grid_shape = shape

    def render(self, width: int, height: int, timestamp_ms: int) -> np.ndarray:
        width = max(1, int(width))
        height = max(1, int(height))
        work_width = max(1, int(round(width * self.work_scale)))
        work_height = max(1, int(round(height * self.work_scale)))

        self._ensure_grid(work_width, work_height)
        nx = self._nx
        ny = self._ny
        radial = self._radial
        assert nx is not None and ny is not None and radial is not None

        t = timestamp_ms * 0.001
        wave1 = np.sin(nx * 18.0 + t * 2.0)
        wave2 = np.sin(ny * 14.0 - t * 1.5)
        wave3 = np.sin((nx + ny) * 24.0 + t * 3.0)

        energy = (wave1 + wave2 + wave3) * (1.0 / 3.0)
        energy = (energy + 1.0) * 0.5
        energy *= np.clip(1.2 - radial * 1.8, 0.0, 1.0)
        energy = np.clip(energy, 0.0, 1.0)

        r = (energy * 255.0).astype(np.uint8)
        g = (np.power(energy, 0.7) * 180.0).astype(np.uint8)
        b = ((1.0 - energy) * 220.0).astype(np.uint8)
        result = cv2.merge((b, g, r))

        if result.shape[:2] != (height, width):
            result = cv2.resize(result, (width, height), interpolation=cv2.INTER_LINEAR)
        return result
