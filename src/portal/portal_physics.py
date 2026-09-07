from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class PortalPhysicsState:
    speed: float = 0.0
    pressure: float = 0.0
    stability: float = 1.0
    pulse: float = 0.0
    turbulence: float = 0.0
    opening_ratio: float = 0.0
    wave_energy: float = 0.0
    hand_velocity: tuple[float, float] = (0.0, 0.0)


class PortalPhysics:
    """Small temporal physics layer for hand-reactive dimensional effects."""

    def __init__(self):
        self.state = PortalPhysicsState()
        self._prev_center = None
        self._prev_hands = {}
        self._waves = []
        self._event_flash = 0.0
        self._last_time = None

    def reset(self):
        self.state = PortalPhysicsState()
        self._prev_center = None
        self._prev_hands.clear()
        self._waves.clear()
        self._event_flash = 0.0
        self._last_time = None

    def update(self, hands, center, width, height, timestamp_ms, opening_ratio=1.0):
        t = timestamp_ms * 0.001
        if self._last_time is None:
            dt = 1.0 / 30.0
        else:
            dt = max(0.001, min(0.08, t - self._last_time))
        self._last_time = t

        speed = 0.0
        vx = vy = 0.0
        if self._prev_center is not None:
            vx = (center[0] - self._prev_center[0]) / dt
            vy = (center[1] - self._prev_center[1]) / dt
            speed = math.hypot(vx, vy)
        self._prev_center = center

        speed_n = min(1.0, speed / 1100.0)
        pressure = 0.0
        if width > 0:
            pressure = min(1.0, max(0.0, 1.0 - opening_ratio))
        turbulence = min(1.0, speed_n * 0.9 + pressure * 0.55)
        stability_target = max(0.0, 1.0 - turbulence * 0.72)
        stability = self.state.stability + (stability_target - self.state.stability) * min(1.0, dt * 8.0)
        pulse = 0.5 + 0.5 * math.sin(t * (3.6 + turbulence * 4.0))
        self._event_flash = max(0.0, self._event_flash - dt * 2.8)

        if hands:
            for idx, hand in enumerate(hands[:2]):
                pts = hand.pixel_landmarks
                if len(pts) > 8:
                    p = pts[8]
                    previous = self._prev_hands.get(idx)
                    if previous is not None:
                        hvx = (p[0] - previous[0]) / dt
                        hvy = (p[1] - previous[1]) / dt
                        hspeed = math.hypot(hvx, hvy)
                        if hspeed > 420.0:
                            self._waves.append([float(p[0]), float(p[1]), 0.0, min(1.0, hspeed / 1300.0)])
                    self._prev_hands[idx] = p

        self._waves = [[x, y, radius + dt * (150.0 + energy * 180.0), energy - dt * 1.25] for x, y, radius, energy in self._waves if energy - dt * 1.25 > 0.02]
        wave_energy = min(1.0, sum(w[3] for w in self._waves))
        self.state = PortalPhysicsState(speed_n, pressure, stability, pulse, turbulence, float(opening_ratio), wave_energy, (vx, vy))
        return self.state

    def trigger_open(self):
        self._event_flash = 1.0
        self._waves.extend([[0.0, 0.0, 0.0, 0.0]])

    def consume_flash(self):
        return self._event_flash

    @property
    def waves(self):
        return tuple(self._waves)
