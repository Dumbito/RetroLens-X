from __future__ import annotations

from dataclasses import dataclass
from math import hypot


@dataclass
class PortalState:
    active: bool
    center: tuple[int, int]
    width: int
    height: int
    angle: float


class PortalEngine:
    """Track a rectangular interdimensional window between the index fingertips."""

    def __init__(
        self,
        min_width: int = 120,
        max_width: int = 900,
        aspect_ratio: float = 0.58,
        smoothing: float = 0.24,
    ):
        if min_width <= 0:
            raise ValueError("min_width debe ser > 0")
        if max_width < min_width:
            raise ValueError("max_width debe ser >= min_width")
        if not 0.0 < smoothing <= 1.0:
            raise ValueError("smoothing debe estar en (0, 1]")
        if not 0.2 <= aspect_ratio <= 1.0:
            raise ValueError("aspect_ratio debe estar entre 0.2 y 1.0")
        self.min_width = int(min_width)
        self.max_width = int(max_width)
        self.aspect_ratio = float(aspect_ratio)
        self.smoothing = float(smoothing)
        self.state = PortalState(False, (0, 0), 0, 0, 0.0)
        self._smoothed_center: tuple[float, float] | None = None
        self._smoothed_width: float | None = None
        self._smoothed_height: float | None = None

    def reset(self) -> None:
        self.state = PortalState(False, (0, 0), 0, 0, 0.0)
        self._smoothed_center = None
        self._smoothed_width = None
        self._smoothed_height = None

    def update(self, hands, timestamp_ms: int = 0) -> PortalState:
        if len(hands) < 2:
            self.reset()
            return self.state

        first = hands[0].pixel_landmarks
        second = hands[1].pixel_landmarks
        if len(first) <= 8 or len(second) <= 8:
            self.reset()
            return self.state

        # Index fingertips are the two corners/anchors of the gesture.
        p1 = first[8]
        p2 = second[8]
        distance = hypot(p2[0] - p1[0], p2[1] - p1[1])
        width = float(max(self.min_width, min(self.max_width, distance * 1.55)))
        height = max(60.0, width * self.aspect_ratio)
        center_x = (p1[0] + p2[0]) * 0.5
        center_y = (p1[1] + p2[1]) * 0.5

        factor = self.smoothing
        if self._smoothed_center is None:
            self._smoothed_center = (center_x, center_y)
            self._smoothed_width = width
            self._smoothed_height = height
        else:
            old_x, old_y = self._smoothed_center
            self._smoothed_center = (
                old_x + (center_x - old_x) * factor,
                old_y + (center_y - old_y) * factor,
            )
            assert self._smoothed_width is not None
            assert self._smoothed_height is not None
            self._smoothed_width += (width - self._smoothed_width) * factor
            self._smoothed_height += (height - self._smoothed_height) * factor

        smooth_x, smooth_y = self._smoothed_center
        smooth_width = int(round(self._smoothed_width))
        smooth_height = max(1, int(round(self._smoothed_height)))
        self.state = PortalState(
            active=True,
            center=(int(round(smooth_x)), int(round(smooth_y))),
            width=smooth_width,
            height=smooth_height,
            angle=0.0,
        )
        return self.state
