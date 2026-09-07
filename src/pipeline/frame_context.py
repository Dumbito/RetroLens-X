from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class FrameContext:
    """Shared per-frame state passed between runtime stages.

    Keeping transient state in one object prevents the application loop from
    accumulating long argument lists as new effects are added.
    """

    frame: Any
    timestamp_ms: int
    delta_time: float
    hands: tuple[Any, ...] = ()
    hand_points: tuple[tuple[int, int], ...] = ()
    phase: str = "READY"
    distance: float | None = None
    hand_scale: float | None = None
    opening_ratio: float = 0.0
    intensity: float = 0.0
    motion: float = 0.0
    view_x: float = 0.0
    view_y: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def portal_state(self) -> Any:
        return self.metadata.get("portal_state")

    @portal_state.setter
    def portal_state(self, value: Any) -> None:
        self.metadata["portal_state"] = value

    @property
    def physics_state(self) -> Any:
        return self.metadata.get("physics_state")

    @physics_state.setter
    def physics_state(self, value: Any) -> None:
        self.metadata["physics_state"] = value

    @property
    def roi(self) -> tuple[int, int, int, int] | None:
        return self.metadata.get("roi")

    @roi.setter
    def roi(self, value: tuple[int, int, int, int] | None) -> None:
        self.metadata["roi"] = value
