from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions


@dataclass
class Hand:
    """Representa una mano detectada."""

    landmarks: list
    handedness: str
    score: float


class HandTracker:
    """Detector de manos basado en MediaPipe HandLandmarker."""

    def __init__(
        self,
        model_path: str | Path,
        num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo: {self.model_path}"
            )

        base_options = BaseOptions(
            model_asset_path=str(self.model_path)
        )

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.landmarker = vision.HandLandmarker.create_from_options(
            options
        )

    def detect(self, frame, timestamp_ms: int) -> list[Hand]:
        """
        Detecta manos en un frame de OpenCV.

        Args:
            frame: Frame BGR procedente de OpenCV.
            timestamp_ms: Timestamp monotónico en milisegundos.

        Returns:
            Lista de objetos Hand.
        """

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        result = self.landmarker.detect_for_video(
            mp_image,
            timestamp_ms,
        )

        hands = []

        if not result.hand_landmarks:
            return hands

        for index, landmarks in enumerate(result.hand_landmarks):
            handedness = "Unknown"
            score = 0.0

            if (
                result.handedness
                and index < len(result.handedness)
                and result.handedness[index]
            ):
                category = result.handedness[index][0]

                handedness = category.category_name
                score = category.score

            hands.append(
                Hand(
                    landmarks=landmarks,
                    handedness=handedness,
                    score=score,
                )
            )

        return hands

    def close(self):
        """Libera los recursos de MediaPipe."""
        self.landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()