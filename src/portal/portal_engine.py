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
    """Track and stabilize the rectangular multiverse window."""

    def __init__(
        self,
        min_width: int = 120,
        max_width: int = 900,
        aspect_ratio: float = 0.58,
        smoothing: float = 0.32,
        max_angle: float = 18.0,
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
        self.max_angle = float(max_angle)
        self.state = PortalState(False, (0, 0), 0, 0, 0.0)
        self._smoothed_center: tuple[float, float] | None = None
        self._smoothed_width: float | None = None
        self._smoothed_height: float | None = None
        self._smoothed_angle: float | None = None
        self._velocity: tuple[float, float] = (0.0, 0.0)
        self._last_timestamp_ms: int | None = None

    def reset(self) -> None:
        self.state = PortalState(False, (0, 0), 0, 0, 0.0)
        self._smoothed_center = None
        self._smoothed_width = None
        self._smoothed_height = None
        self._smoothed_angle = None
        self._velocity = (0.0, 0.0)
        self._last_timestamp_ms = None

    def update(self, hands, timestamp_ms: int = 0) -> PortalState:
        if len(hands) < 2:
            self.reset()
            return self.state

        first = hands[0].pixel_landmarks
        second = hands[1].pixel_landmarks
        if len(first) <= 8 or len(second) <= 8:
            self.reset()
            return self.state

        p1 = first[8]
        p2 = second[8]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        distance = hypot(dx, dy)
        width = float(max(self.min_width, min(self.max_width, distance * 1.55)))
        height = max(60.0, width * self.aspect_ratio)
        center_x = (p1[0] + p2[0]) * 0.5
        center_y = (p1[1] + p2[1]) * 0.5

        raw_angle = degrees(atan2(dy, dx))
        if raw_angle > 90.0:
            raw_angle -= 180.0
        elif raw_angle < -90.0:
            raw_angle += 180.0
        raw_angle = max(-self.max_angle, min(self.max_angle, raw_angle))

        if self._smoothed_center is None:
            self._smoothed_center = (center_x, center_y)
            self._smoothed_width = width
            self._smoothed_height = height
            self._smoothed_angle = raw_angle
            self._velocity = (0.0, 0.0)
        else:
            old_x, old_y = self._smoothed_center
            if self._last_timestamp_ms is not None:
                dt = max(0.001, min((timestamp_ms - self._last_timestamp_ms) / 1000.0, 0.10))
                measured_vx = (center_x - old_x) / dt
                measured_vy = (center_y - old_y) / dt
                # Smooth velocity separately; this gives the window a little
                # inertia without making it lag behind quick hand movement.
                velocity_factor = 0.38
                vx = self._velocity[0] + (measured_vx - self._velocity[0]) * velocity_factor
                vy = self._velocity[1] + (measured_vy - self._velocity[1]) * velocity_factor
                self._velocity = (vx, vy)
                prediction = min(dt * 0.035, 0.018)
                target_x = center_x + vx * prediction
                target_y = center_y + vy * prediction
            else:
                target_x, target_y = center_x, center_y

            # Adaptive smoothing: quick movements get a more responsive filter,
            # while slow movements remain very stable.
            speed = hypot(self._velocity[0], self._velocity[1])
            speed_factor = min(1.0, speed / 900.0)
            factor = min(0.62, self.smoothing + speed_factor * 0.22)
            self._smoothed_center = (
                old_x + (target_x - old_x) * factor,
                old_y + (target_y - old_y) * factor,
            )
            assert self._smoothed_width is not None
            assert self._smoothed_height is not None
            assert self._smoothed_angle is not None
            size_factor = min(0.58, self.smoothing + speed_factor * 0.12)
            self._smoothed_width += (width - self._smoothed_width) * size_factor
            self._smoothed_height += (height - self._smoothed_height) * size_factor
            self._smoothed_angle += (raw_angle - self._smoothed_angle) * min(0.55, factor)

        self._last_timestamp_ms = timestamp_ms
        smooth_x, smooth_y = self._smoothed_center
        self.state = PortalState(
            active=True,
            center=(int(round(smooth_x)), int(round(smooth_y))),
            width=int(round(self._smoothed_width)),
            height=max(1, int(round(self._smoothed_height))),
            angle=float(self._smoothed_angle),
        )
        return self.state
