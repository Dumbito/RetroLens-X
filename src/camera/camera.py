from dataclasses import dataclass
import subprocess
import threading

import cv2


@dataclass
class CameraConfig:
    index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    fourcc: str = "MJPG"
    disable_dynamic_exposure_fps: bool = True
    threaded: bool = False


class Camera:
    def __init__(self, config: CameraConfig | None = None):
        self.config = config or CameraConfig()
        self.capture = None
        self._running = False
        self._thread = None
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._frame_ready = threading.Event()
        self._configure_v4l2()
        self._open()
        if self.config.threaded:
            self._start_reader()

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

    def _start_reader(self):
        self._running = True
        self._thread = threading.Thread(
            target=self._reader_loop,
            name="retrolens-camera",
            daemon=True,
        )
        self._thread.start()

    def _reader_loop(self):
        while self._running and self.capture is not None:
            success, frame = self.capture.read()
            if not success:
                continue
            with self._frame_lock:
                self._latest_frame = frame
                self._frame_ready.set()

    def read(self):
        if not self.config.threaded:
            success, frame = self.capture.read()
            if not success:
                raise RuntimeError("No se pudo leer un frame de la cámara.")
            return frame

        if not self._frame_ready.wait(timeout=1.0):
            raise RuntimeError("Tiempo de espera agotado esperando un frame de la cámara.")
        with self._frame_lock:
            if self._latest_frame is None:
                raise RuntimeError("La cámara no entregó un frame válido.")
            return self._latest_frame.copy()

    def release(self):
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self._frame_ready.clear()
        with self._frame_lock:
            self._latest_frame = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
