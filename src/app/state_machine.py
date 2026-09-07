from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PortalStateMachine:
    """Gesture-driven lifecycle for the aperture."""

    arm_ratio: float = 0.72
    open_ratio: float = 1.05
    close_ratio: float = 0.78
    lost_grace_seconds: float = 0.60
    arm_hold_seconds: float = 0.08
    phase: str = "READY"
    armed_since: float | None = None
    lost_since: float | None = None

    def update(self, hands, distance, scale, now, portal, physics, timestamp_ms):
        if distance is not None and scale is not None:
            arm_distance = scale * self.arm_ratio
            open_distance = scale * self.open_ratio
            close_distance = scale * self.close_ratio

            if self.phase == "READY":
                if distance <= arm_distance:
                    if self.armed_since is None:
                        self.armed_since = now
                    elif now - self.armed_since >= self.arm_hold_seconds:
                        self.phase = "ARMED"
                        self.lost_since = None
                else:
                    self.armed_since = None

            elif self.phase == "ARMED":
                if distance >= open_distance:
                    self.phase = "OPEN"
                    self.lost_since = None
                    portal.reset()
                    portal.update(hands, timestamp_ms)
                    physics.reset()
                    physics.trigger_open(portal.state.center)
                elif distance > arm_distance * 1.65:
                    self.phase = "READY"
                    self.armed_since = None

            elif self.phase == "OPEN":
                if distance <= close_distance:
                    physics.trigger_close()
                    self._reset_runtime(portal)

        elif self.phase == "OPEN":
            if self.lost_since is None:
                self.lost_since = now
            elif now - self.lost_since >= self.lost_grace_seconds:
                self._reset_runtime(portal)
        else:
            self.phase = "READY"
            self.armed_since = None

        return self.phase

    def _reset_runtime(self, portal):
        self.phase = "READY"
        self.armed_since = None
        self.lost_since = None
        portal.reset()
