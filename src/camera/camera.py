from dataclasses import dataclass

import cv2


@dataclass
class CameraConfig:
    index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30


class Camera:
    def __init__(self, config: CameraConfig | None = None):
        self.config = config or CameraConfig()
        self.capture = cv2.VideoCapture(self.config.index)

        if not self.capture.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara {self.config.index}.")

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self.capture.set(cv2.CAP_PROP_FPS, self.config.fps)

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
