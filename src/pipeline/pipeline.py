from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from .frame_context import FrameContext


class FrameEffect(Protocol):
    name: str

    def __call__(self, context: FrameContext) -> FrameContext: ...


@dataclass(slots=True)
class CallableEffect:
    name: str
    function: Callable[[FrameContext], FrameContext]

    def __call__(self, context: FrameContext) -> FrameContext:
        return self.function(context)


class EffectPipeline:
    """Ordered, inspectable effect chain.

    Effects are ordinary Python callables, so the OpenCV renderer remains the
    default backend while a future GPU backend can be inserted without
    rewriting the runtime.
    """

    def __init__(self, effects: list[FrameEffect] | None = None):
        self.effects: list[FrameEffect] = list(effects or [])

    def add(self, effect: FrameEffect) -> "EffectPipeline":
        self.effects.append(effect)
        return self

    def clear(self) -> None:
        self.effects.clear()

    def run(self, context: FrameContext) -> FrameContext:
        for effect in self.effects:
            context = effect(context)
        return context

    def names(self) -> tuple[str, ...]:
        return tuple(effect.name for effect in self.effects)
