import cv2
import numpy as np


class ProceduralDimension:
    """Animated procedural content rendered at a reduced internal resolution."""

    def __init__(self, work_scale: float = 0.5):
        self.work_scale = float(np.clip(work_scale, 0.25, 1.0))
        self.phase = 0.0

    def render(self, width: int, height: int, timestamp_ms: int) -> np.ndarray:
        width = max(1, int(width))
        height = max(1, int(height))

        work_width = max(1, int(round(width * self.work_scale)))
        work_height = max(1, int(round(height * self.work_scale)))

        y, x = np.mgrid[0:work_height, 0:work_width]
        nx = x / max(work_width - 1, 1)
        ny = y / max(work_height - 1, 1)
        t = timestamp_ms * 0.001

        wave1 = np.sin(nx * 18.0 + t * 2.0)
        wave2 = np.sin(ny * 14.0 - t * 1.5)
        wave3 = np.sin((nx + ny) * 24.0 + t * 3.0)

        energy = (wave1 + wave2 + wave3) / 3.0
        energy = (energy + 1.0) * 0.5

        radial = np.sqrt((nx - 0.5) ** 2 + (ny - 0.5) ** 2)
        energy *= np.clip(1.2 - radial * 1.8, 0.0, 1.0)
        energy = np.clip(energy, 0.0, 1.0)

        r = (energy * 255).astype(np.uint8)
        g = (np.power(energy, 0.7) * 180).astype(np.uint8)
        b = ((1.0 - energy) * 220).astype(np.uint8)

        result = cv2.merge((b, g, r))
        if result.shape[1] != width or result.shape[0] != height:
            result = cv2.resize(result, (width, height), interpolation=cv2.INTER_LINEAR)
        return result
