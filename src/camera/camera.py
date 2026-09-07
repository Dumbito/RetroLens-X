from dataclasses import dataclass
import subprocess

import cv2


@dataclass
class CameraConfig:
    index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    fourcc: str = "MJPG"
    disable_dynamic_exposure_fps: bool = True


class Camera:
    def __init__(self, config: CameraConfig | None = None):
        self.config = config or CameraConfig()
        self.capture = None
        self._configure_v4l2()
        self._open()

    def _configure_v4l2(self):
        if not self.config.disable_dynamic_exposure_fps:
            return

        device = f"/dev/video{self.config.index}"
        try:
            subprocess.run(
                ["v4l2-ctl", "-d", device, "--set-ctrl=exposure_dynamic_framerate=0"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            pass

    def _open(self):
        self.capture = cv2.VideoCapture(self.config.index, cv2.CAP_V4L2)

        if not self.capture.isOpened():
            self.capture.release()
            self.capture = cv2.VideoCapture(self.config.index)

        if not self.capture.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara {self.config.index}.")

        fourcc = cv2.VideoWriter_fourcc(*self.config.fourcc)
        self.capture.set(cv2.CAP_PROP_FOURCC, fourcc)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self.capture.set(cv2.CAP_PROP_FPS, self.config.fps)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self):
        success, frame = self.capture.read()
        if not success:
            raise RuntimeError("No se pudo leer un frame de la cámara.")
        return frame

    def release(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
