import cv2
import numpy as np


class ProceduralDimension:
    def __init__(self):
        self.phase = 0.0

    def render(self, width: int, height: int, timestamp_ms: int) -> np.ndarray:
        y, x = np.mgrid[0:height, 0:width]

        t = timestamp_ms * 0.001

        nx = x / max(width, 1)
        ny = y / max(height, 1)

        wave1 = np.sin(nx * 18.0 + t * 2.0)
        wave2 = np.sin(ny * 14.0 - t * 1.5)
        wave3 = np.sin((nx + ny) * 24.0 + t * 3.0)

        energy = (wave1 + wave2 + wave3) / 3.0
        energy = (energy + 1.0) * 0.5

        radial = np.sqrt((nx - 0.5) ** 2 + (ny - 0.5) ** 2)
        energy *= np.clip(1.2 - radial * 1.8, 0.0, 1.0)

        r = (energy * 255).astype(np.uint8)
        g = ((energy ** 0.7) * 180).astype(np.uint8)
        b = ((1.0 - energy) * 220).astype(np.uint8)

        return cv2.merge((b, g, r))
