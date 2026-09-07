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
    def __init__(
        self,
        min_width: int = 140,
        max_width: int = 900,
        smoothing: float = 0.22,
    ):
        if min_width <= 0:
            raise ValueError("min_width debe ser > 0")
        if max_width < min_width:
            raise ValueError("max_width debe ser >= min_width")
        if not 0.0 < smoothing <= 1.0:
            raise ValueError("smoothing debe estar en (0, 1]")
        self.min_width = int(min_width)
        self.max_width = int(max_width)
        self.smoothing = float(smoothing)
        self.state = PortalState(False, (0, 0), 0, 0, 0.0)
        self._smoothed_center: tuple[float, float] | None = None
        self._smoothed_width: float | None = None
        self._smoothed_height: float | None = None
        self._smoothed_angle: float | None = None

    @staticmethod
    def _smooth_angle(previous: float, current: float, factor: float) -> float:
        delta = (current - previous + 180.0) % 360.0 - 180.0
        return previous + delta * factor

    def update(self, hands, timestamp_ms: int = 0) -> PortalState:
        if len(hands) < 2:
            self.state = PortalState(False, (0, 0), 0, 0, 0.0)
            self._smoothed_center = None
            self._smoothed_width = None
            self._smoothed_height = None
            self._smoothed_angle = None
            return self.state

        first = hands[0].pixel_landmarks
        second = hands[1].pixel_landmarks
        if len(first) <= 9 or len(second) <= 9:
            self.state = PortalState(False, (0, 0), 0, 0, 0.0)
            return self.state

        left = first[9]
        right = second[9]
        center_x = (left[0] + right[0]) * 0.5
        center_y = (left[1] + right[1]) * 0.5
        distance = hypot(right[0] - left[0], right[1] - left[1])
        width = float(max(self.min_width, min(self.max_width, distance * 1.65)))
        height = max(1.0, width * 0.72)
        angle = degrees(atan2(right[1] - left[1], right[0] - left[0]))

        factor = self.smoothing
        if self._smoothed_center is None:
            self._smoothed_center = (center_x, center_y)
            self._smoothed_width = width
            self._smoothed_height = height
            self._smoothed_angle = angle
        else:
            old_x, old_y = self._smoothed_center
            self._smoothed_center = (
                old_x + (center_x - old_x) * factor,
                old_y + (center_y - old_y) * factor,
            )
            assert self._smoothed_width is not None
            assert self._smoothed_height is not None
            assert self._smoothed_angle is not None
            self._smoothed_width += (width - self._smoothed_width) * factor
            self._smoothed_height += (height - self._smoothed_height) * factor
            self._smoothed_angle = self._smooth_angle(
                self._smoothed_angle,
                angle,
                factor,
            )

        smooth_x, smooth_y = self._smoothed_center
        smooth_width = int(round(self._smoothed_width))
        smooth_height = max(1, int(round(self._smoothed_height)))
        smooth_angle = self._smoothed_angle

        self.state = PortalState(
            active=True,
            center=(int(round(smooth_x)), int(round(smooth_y))),
            width=smooth_width,
            height=smooth_height,
            angle=smooth_angle,
        )
        return self.state
