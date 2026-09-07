from dataclasses import dataclass
from math import atan2, degrees, hypot

import cv2
import numpy as np


@dataclass
class PortalState:
    active: bool
    center: tuple[int, int]
    width: int
    height: int
    angle: float


class PortalEngine:
    def __init__(self, min_width: int = 140, max_width: int = 900):
        self.min_width = min_width
        self.max_width = max_width
        self.state = PortalState(False, (0, 0), 0, 0, 0.0)

    def update(self, hands, timestamp_ms: int) -> PortalState:
        if len(hands) < 2:
            self.state = PortalState(False, (0, 0), 0, 0, 0.0)
            return self.state

        centers = []
        for hand in hands[:2]:
            points = hand.pixel_landmarks
            palm = points[9]
            centers.append(palm)

        left = centers[0]
        right = centers[1]

        center_x = int((left[0] + right[0]) / 2)
        center_y = int((left[1] + right[1]) / 2)

        distance = hypot(right[0] - left[0], right[1] - left[1])
        width = int(np.clip(distance * 1.65, self.min_width, self.max_width))
        height = int(width * 0.72)

        angle = degrees(atan2(right[1] - left[1], right[0] - left[0]))

        self.state = PortalState(
            active=True,
            center=(center_x, center_y),
            width=width,
            height=height,
            angle=angle,
        )
        return self.state

    def render(self, frame: np.ndarray, dimension: np.ndarray) -> np.ndarray:
        if not self.state.active:
            return frame

        result = frame.copy()
        h, w = frame.shape[:2]

        dimension = cv2.resize(dimension, (w, h), interpolation=cv2.INTER_LINEAR)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(
            mask,
            self.state.center,
            (self.state.width // 2, self.state.height // 2),
            self.state.angle,
            0,
            360,
            255,
            -1,
        )

        inner = cv2.GaussianBlur(mask, (0, 0), 8)
        alpha = inner.astype(np.float32) / 255.0
        alpha = alpha[..., None]

        result = (
            frame.astype(np.float32) * (1.0 - alpha)
            + dimension.astype(np.float32) * alpha
        ).astype(np.uint8)

        edge = cv2.Canny(mask, 80, 160)
        glow = cv2.GaussianBlur(edge, (0, 0), 14)
        glow = cv2.cvtColor(glow, cv2.COLOR_GRAY2BGR)

        result = cv2.addWeighted(result, 1.0, glow, 0.65, 0)

        return result
