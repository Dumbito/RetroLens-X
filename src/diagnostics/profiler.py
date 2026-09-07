from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Iterator


@dataclass(slots=True)
class RollingStats:
    samples: deque[float] = field(default_factory=lambda: deque(maxlen=180))

    def add(self, milliseconds: float) -> None:
        self.samples.append(float(milliseconds))

    @property
    def average(self) -> float:
        return sum(self.samples) / len(self.samples) if self.samples else 0.0

    @property
    def p95(self) -> float:
        if not self.samples:
            return 0.0
        values = sorted(self.samples)
        index = min(len(values) - 1, int(len(values) * 0.95))
        return values[index]


class PipelineProfiler:
    """Low-overhead rolling stage profiler for real-time debugging."""

    def __init__(self, window: int = 180):
        self.stats: dict[str, RollingStats] = {}
        self.frame_times = RollingStats(deque(maxlen=window))
        self._frame_started = 0.0

    def begin_frame(self) -> None:
        self._frame_started = perf_counter()

    def end_frame(self) -> None:
        if self._frame_started:
            self.frame_times.add((perf_counter() - self._frame_started) * 1000.0)

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        started = perf_counter()
        try:
            yield
        finally:
            self.stats.setdefault(name, RollingStats()).add((perf_counter() - started) * 1000.0)

    @property
    def fps(self) -> float:
        average_ms = self.frame_times.average
        return 1000.0 / average_ms if average_ms > 0.0 else 0.0

    def summary(self) -> dict[str, float]:
        result = {"fps": self.fps, "frame_avg_ms": self.frame_times.average, "frame_p95_ms": self.frame_times.p95}
        for name, stats in self.stats.items():
            result[f"{name}_avg_ms"] = stats.average
            result[f"{name}_p95_ms"] = stats.p95
        return result
