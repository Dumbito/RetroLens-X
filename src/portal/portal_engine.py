from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees, hypot


@dataclass
class PortalState:
    active: bool
    center: tuple[int, int]
    width: int
    height: int
    angle: float


class PortalEngine:
    def __init__(self, min_width: int = 140, max_width: int = 900):
        if min_width <= 0:
            raise ValueError("min_width debe ser > 0")
        if max_width < min_width:
            raise ValueError("max_width debe ser >= min_width")
        self.min_width = int(min_width)
        self.max_width = int(max_width)
        self.state = PortalState(False, (0, 0), 0, 0, 0.0)

    def update(self, hands, timestamp_ms: int = 0) -> PortalState:
        if len(hands) < 2:
            self.state = PortalState(False, (0, 0), 0, 0, 0.0)
            return self.state

        first = hands[0].pixel_landmarks
        second = hands[1].pixel_landmarks
        if len(first) <= 9 or len(second) <= 9:
            self.state = PortalState(False, (0, 0), 0, 0, 0.0)
            return self.state

        left = first[9]
        right = second[9]
        center_x = int(round((left[0] + right[0]) * 0.5))
        center_y = int(round((left[1] + right[1]) * 0.5))
        distance = hypot(right[0] - left[0], right[1] - left[1])
        width = int(max(self.min_width, min(self.max_width, distance * 1.65)))
        height = max(1, int(round(width * 0.72)))
        angle = degrees(atan2(right[1] - left[1], right[0] - left[0]))

        self.state = PortalState(
            active=True,
            center=(center_x, center_y),
            width=width,
            height=height,
            angle=angle,
        )
        return self.state
