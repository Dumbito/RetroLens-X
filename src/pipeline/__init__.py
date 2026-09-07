"""Reusable real-time frame pipeline primitives."""

from .frame_context import FrameContext
from .pipeline import CallableEffect, EffectPipeline, FrameEffect

__all__ = ["CallableEffect", "EffectPipeline", "FrameContext", "FrameEffect"]
