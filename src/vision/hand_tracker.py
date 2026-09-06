from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

@dataclass
class Hand:
    landmarks: list
    pixel_landmarks: list[tuple[int, int]]
    handedness: str
    score: float
    bbox: tuple[int, int, int, int]

class HandTracker:
    def __init__(self, model_path: str | Path, num_hands: int = 2, min_detection_confidence: float = 0.5, min_hand_presence_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"No se encontró el modelo: {self.model_path}")
        base_options = BaseOptions(model_asset_path=str(self.model_path))
        options = vision.HandLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.VIDEO, num_hands=num_hands, min_hand_detection_confidence=min_detection_confidence, min_hand_presence_confidence=min_hand_presence_confidence, min_tracking_confidence=min_tracking_confidence)
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame, timestamp_ms: int) -> list[Hand]:
        height, width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        hands = []
        if not result.hand_landmarks:
            return hands
        for index, landmarks in enumerate(result.hand_landmarks):
            pixel_landmarks = [(max(0, min(width - 1, int(landmark.x * width))), max(0, min(height - 1, int(landmark.y * height)))) for landmark in landmarks]
            xs = [point[0] for point in pixel_landmarks]
            ys = [point[1] for point in pixel_landmarks]
            bbox = (min(xs), min(ys), max(xs), max(ys))
            handedness = "Unknown"
            score = 0.0
            if result.handedness and index < len(result.handedness) and result.handedness[index]:
                category = result.handedness[index][0]
                handedness = category.category_name
                score = category.score
            hands.append(Hand(landmarks=landmarks, pixel_landmarks=pixel_landmarks, handedness=handedness, score=score, bbox=bbox))
        return hands

    def close(self):
        self.landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
