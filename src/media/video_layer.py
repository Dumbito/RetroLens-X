from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class VideoLayer:
    """Looping local video layer for the interior of the portal."""

    def __init__(self, path: str = "assets/media/portal.mp4", opacity: float = 0.68) -> None:
        self.path = Path(path)
        self.opacity = max(0.0, min(1.0, float(opacity)))
        self._capture: cv2.VideoCapture | None = None
        self._last_frame: np.ndarray | None = None
        self._opened_path: Path | None = None

    def _ensure_open(self) -> bool:
        if self._capture is not None and self._opened_path == self.path and self._capture.isOpened():
            return True
        self.close()
        if not self.path.exists():
            return False
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            capture.release()
            return False
        self._capture = capture
        self._opened_path = self.path
        return True

    def render(
        self,
        width: int,
        height: int,
        timestamp_ms: int,
        view_x: float = 0.0,
        view_y: float = 0.0,
    ) -> np.ndarray | None:
        width, height = max(2, int(width)), max(2, int(height))
        if not self._ensure_open():
            return None

        assert self._capture is not None
        ok, frame = self._capture.read()
        if not ok:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._capture.read()
        if not ok or frame is None:
            return self._last_frame.copy() if self._last_frame is not None else None

        # Preserve the scene's wide composition while filling the portal.
        source_h, source_w = frame.shape[:2]
        target_ratio = width / max(height, 1)
        source_ratio = source_w / max(source_h, 1)
        if source_ratio > target_ratio:
            crop_w = max(2, int(round(source_h * target_ratio)))
            max_x = max(0, source_w - crop_w)
            offset_x = int(round((max_x * 0.5) + view_x * max_x * 0.16))
            offset_x = max(0, min(max_x, offset_x))
            frame = frame[:, offset_x : offset_x + crop_w]
        else:
            crop_h = max(2, int(round(source_w / target_ratio)))
            max_y = max(0, source_h - crop_h)
            offset_y = int(round((max_y * 0.5) + view_y * max_y * 0.10))
            offset_y = max(0, min(max_y, offset_y))
            frame = frame[offset_y : offset_y + crop_h]

        layer = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        self._last_frame = layer.copy()
        return layer

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
        self._capture = None
        self._opened_path = None

    def __del__(self) -> None:
        self.close()
