"""Reusable real-time frame pipeline primitives."""

from .frame_context import FrameContext
from .pipeline import EffectPipeline, FrameEffect

__all__ = ["EffectPipeline", "FrameContext", "FrameEffect"]
