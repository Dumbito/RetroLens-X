from dataclasses import dataclass
from enum import Enum
from math import hypot

from src.vision.hand_tracker import Hand


class GestureType(str, Enum):
    NONE = "none"
    OPEN_HAND = "open_hand"
    FIST = "fist"
    PINCH = "pinch"
    POINT = "point"
    PEACE = "peace"


class GestureState(str, Enum):
    NONE = "none"
    DOWN = "down"
    HOLD = "hold"
    RELEASE = "release"


@dataclass
class Gesture:
    type: GestureType
    hand_index: int
    confidence: float
    state: GestureState
    duration_ms: int


class GestureEngine:
    def __init__(self, release_timeout_ms: int = 120):
        self.release_timeout_ms = release_timeout_ms
        self.active: dict[int, tuple[GestureType, int, int]] = {}

    def detect(self, hands: list[Hand], timestamp_ms: int) -> list[Gesture]:
        gestures = []
        seen = set()

        for index, hand in enumerate(hands):
            seen.add(index)
            gesture_type, confidence = self._classify_hand(hand)

            previous = self.active.get(index)

            if gesture_type == GestureType.NONE:
                if previous is not None:
                    previous_type, start_ms, last_ms = previous
                    if timestamp_ms - last_ms >= self.release_timeout_ms:
                        duration = max(0, timestamp_ms - start_ms)
                        gestures.append(Gesture(previous_type, index, 0.0, GestureState.RELEASE, duration))
                        del self.active[index]
                continue

            if previous is None or previous[0] != gesture_type:
                self.active[index] = (gesture_type, timestamp_ms, timestamp_ms)
                gestures.append(Gesture(gesture_type, index, confidence, GestureState.DOWN, 0))
                continue

            _, start_ms, _ = previous
            self.active[index] = (gesture_type, start_ms, timestamp_ms)
            duration = max(0, timestamp_ms - start_ms)
            gestures.append(Gesture(gesture_type, index, confidence, GestureState.HOLD, duration))

        for index in list(self.active):
            if index not in seen:
                gesture_type, start_ms, last_ms = self.active[index]
                if timestamp_ms - last_ms >= self.release_timeout_ms:
                    duration = max(0, timestamp_ms - start_ms)
                    gestures.append(Gesture(gesture_type, index, 0.0, GestureState.RELEASE, duration))
                    del self.active[index]

        return gestures

    def _classify_hand(self, hand: Hand) -> tuple[GestureType, float]:
        points = hand.pixel_landmarks

        if len(points) < 21:
            return GestureType.NONE, 0.0

        wrist = points[0]
        thumb_tip = points[4]
        index_tip = points[8]

        pinch_distance = hypot(
            thumb_tip[0] - index_tip[0],
            thumb_tip[1] - index_tip[1],
        )

        palm_size = hypot(
            wrist[0] - points[9][0],
            wrist[1] - points[9][1],
        )

        if palm_size <= 0:
            return GestureType.NONE, 0.0

        if pinch_distance / palm_size < 0.45:
            return GestureType.PINCH, 0.9

        extended = [
            self._finger_extended(points, 8, 6),
            self._finger_extended(points, 12, 10),
            self._finger_extended(points, 16, 14),
            self._finger_extended(points, 20, 18),
        ]

        count = sum(extended)

        if count == 4:
            return GestureType.OPEN_HAND, 0.95

        if count == 0:
            return GestureType.FIST, 0.95

        if extended[0] and not any(extended[1:]):
            return GestureType.POINT, 0.9

        if extended[0] and extended[1] and not any(extended[2:]):
            return GestureType.PEACE, 0.9

        return GestureType.NONE, 0.5

    @staticmethod
    def _finger_extended(points, tip_index: int, pip_index: int) -> bool:
        return points[tip_index][1] < points[pip_index][1]
