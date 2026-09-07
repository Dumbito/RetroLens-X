from __future__ import annotations

import cv2
import numpy as np


class MultiverseDimension:
    """Turn the camera view inside the portal into an alternate-reality window."""

    def __init__(self, work_scale: float = 0.60) -> None:
        self.work_scale = min(max(float(work_scale), 0.35), 1.0)
        self._work: np.ndarray | None = None
        self._channel: np.ndarray | None = None
        self._edges: np.ndarray | None = None
        self._output: np.ndarray | None = None
        self._scanlines: np.ndarray | None = None
        self._size: tuple[int, int] | None = None

    def _ensure(self, width: int, height: int) -> None:
        key = (width, height)
        if self._size == key:
            return
        shape = (height, width, 3)
        self._work = np.empty(shape, dtype=np.uint8)
        self._channel = np.empty((height, width), dtype=np.uint8)
        self._edges = np.empty((height, width), dtype=np.uint8)
        self._output = np.empty(shape, dtype=np.uint8)
        self._scanlines = np.empty((height, width), dtype=np.uint8)
        self._size = key

    def render(self, frame, width, height, center, timestamp_ms, view_x=0.0, view_y=0.0, view_angle=0.0):
        target_w = max(2, int(round(width)))
        target_h = max(2, int(round(height)))
        work_w = max(2, int(round(target_w * self.work_scale)))
        work_h = max(2, int(round(target_h * self.work_scale)))
        self._ensure(work_w, work_h)
        assert self._work is not None
        assert self._channel is not None
        assert self._edges is not None
        assert self._output is not None
        assert self._scanlines is not None

        cx, cy = int(center[0]), int(center[1])
        x0 = max(0, cx - target_w // 2)
        y0 = max(0, cy - target_h // 2)
        x1 = min(frame.shape[1], x0 + target_w)
        y1 = min(frame.shape[0], y0 + target_h)
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            return np.zeros((target_h, target_w, 3), dtype=np.uint8)

        resized = cv2.resize(crop, (work_w, work_h), interpolation=cv2.INTER_LINEAR)
        self._work[:] = resized
        self._work[:] = (self._work // 24) * 24
        cv2.convertScaleAbs(self._work, alpha=1.18, beta=-14, dst=self._work)

        shift = int(round(3.0 + 2.0 * np.sin(timestamp_ms * 0.004)))
        shift_x = int(round(view_x * 3.0))
        self._output[:] = self._work
        blue = self._work[:, :, 0]
        green = self._work[:, :, 1]
        red = self._work[:, :, 2]

        total_blue = shift + shift_x
        if total_blue >= 0:
            sx = min(work_w - 1, total_blue)
            self._output[:, sx:, 0] = blue[:, :work_w - sx]
        else:
            sx = min(work_w - 1, -total_blue)
            self._output[:, :work_w - sx, 0] = blue[:, sx:]

        self._output[:, :, 1] = green
        total_red = shift - shift_x
        if total_red >= 0:
            sx = min(work_w - 1, total_red)
            self._output[:, sx:, 2] = red[:, :work_w - sx]
        else:
            sx = min(work_w - 1, -total_red)
            self._output[:, :work_w - sx, 2] = red[:, sx:]

        gray = cv2.cvtColor(self._output, cv2.COLOR_BGR2GRAY)
        self._edges[:] = cv2.Canny(gray, 70, 150)
        self._channel[:] = cv2.GaussianBlur(self._edges, (0, 0), 0.8)
        self._output[:, :, 1] = cv2.addWeighted(self._output[:, :, 1], 1.0, self._channel, 0.30, 0.0)
        self._output[:, :, 2] = cv2.addWeighted(self._output[:, :, 2], 1.0, self._edges, 0.18, 0.0)

        yy = np.arange(work_h, dtype=np.int32)[:, None]
        scan = ((yy + int(timestamp_ms * 0.04)) % 6 < 1).astype(np.uint8) * 18
        self._scanlines[:] = np.broadcast_to(scan, (work_h, work_w))
        for channel_index in range(3):
            self._output[:, :, channel_index] = cv2.subtract(
                self._output[:, :, channel_index], self._scanlines
            )

        return cv2.resize(self._output, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
