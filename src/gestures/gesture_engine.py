from dataclasses import dataclass
from enum import Enum
from math import acos, degrees, hypot

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

@dataclass
class _Track:
    key: str
    landmarks: list = None
    gesture_type: GestureType = GestureType.NONE
    candidate_type: GestureType = GestureType.NONE
    candidate_since_ms: int = 0
    start_ms: int = 0
    last_seen_ms: int = 0
    confidence: float = 0.0

    def __post_init__(self):
        if self.landmarks is None:
            self.landmarks = []
class GestureEngine:
    def __init__(self, release_timeout_ms=220, switch_stability_ms=180, open_gate_ms=250, open_release_ms=350, smoothing_alpha=0.45):
        self.release_timeout_ms = release_timeout_ms
        self.switch_stability_ms = switch_stability_ms
        self.open_gate_ms = open_gate_ms
        self.open_release_ms = open_release_ms
        self.smoothing_alpha = max(0.10, min(0.90, smoothing_alpha))
        self._tracks = {}
        self._two_hand_open = False
        self._open_since_ms = None
        self._open_lost_since_ms = None

    @property
    def two_hand_open(self):
        return self._two_hand_open

    @property
    def tracked_hands(self):
        return self._tracks

    def detect(self, hands, timestamp_ms):
        observations = []
        used_keys = set()

        for index, hand in enumerate(hands):
            key = self._identity_key(hand, index, used_keys)
            used_keys.add(key)
            track = self._tracks.get(key)
            if track is None:
                track = _Track(key=key)
                self._tracks[key] = track

            points = self._points(hand)
            points = self._smooth(track.landmarks, points)
            track.landmarks = points
            track.last_seen_ms = timestamp_ms

            gesture_type, confidence = self._classify(points)
            gesture = self._update_track(track, gesture_type, confidence, timestamp_ms, index)
            observations.append(gesture)

        self._update_gate(observations, timestamp_ms)
        self._expire(used_keys, timestamp_ms)
        return observations

    def _identity_key(self, hand, index, used_keys):
        handedness = getattr(hand, "handedness", "Unknown")
        if handedness in ("Left", "Right") and handedness not in used_keys:
            return handedness
        return f"hand_{index}"
    @staticmethod
    def _points(hand):
        return [(float(lm.x), float(lm.y), float(getattr(lm, "z", 0.0))) for lm in hand.landmarks]

    def _smooth(self, previous, current):
        if not previous or len(previous) != len(current):
            return current
        a = self.smoothing_alpha
        b = 1.0 - a
        return [(previous[i][0] * b + current[i][0] * a, previous[i][1] * b + current[i][1] * a, previous[i][2] * b + current[i][2] * a) for i in range(len(current))]
    def _classify(self, p):
        if len(p) < 21:
            return GestureType.NONE, 0.0

        palm = max(self._dist(p[0], p[9]), 1e-6)

        fingers = [
            self._finger_extended_score(p, tip, pip, dip, mcp, palm)
            for tip, pip, dip, mcp in (
                (8, 6, 7, 5),
                (12, 10, 11, 9),
                (16, 14, 15, 13),
                (20, 18, 19, 17),
            )
        ]

        extended = [score >= 0.50 for score in fingers]
        count = sum(extended)
        thumb = self._thumb_extended_score(p, palm)
        pinch_distance = self._dist(p[4], p[8]) / palm

        if count == 4 and thumb >= 0.35:
            confidence = min(1.0, 0.70 + 0.30 * (sum(fingers) / 4.0))
            return GestureType.OPEN_HAND, confidence

        if pinch_distance < 0.28 and fingers[0] >= 0.45:
            confidence = 1.0 - min(1.0, pinch_distance / 0.28)
            return GestureType.PINCH, 0.70 + 0.30 * confidence

        if count == 0 and thumb < 0.45:
            confidence = 1.0 - sum(fingers) / 4.0
            return GestureType.FIST, 0.60 + 0.40 * confidence

        if extended[0] and extended[1] and not extended[2] and not extended[3]:
            confidence = (fingers[0] + fingers[1]) / 2.0
            return GestureType.PEACE, max(0.0, min(1.0, confidence))

        if extended[0] and not extended[1] and not extended[2] and not extended[3]:
            return GestureType.POINT, max(0.0, min(1.0, fingers[0]))

        return GestureType.NONE, 0.20
    @staticmethod
    def _finger_extended_score(p, tip, pip, dip, mcp, palm):
        a_pip = GestureEngine._angle(p[mcp], p[pip], p[dip])
        a_dip = GestureEngine._angle(p[pip], p[dip], p[tip])
        tip_distance = GestureEngine._dist(p[tip], p[0]) / palm
        mcp_distance = GestureEngine._dist(p[mcp], p[0]) / palm
        angle_score = min(1.0, max(0.0, (a_pip - 95.0) / 80.0))
        dip_score = min(1.0, max(0.0, (a_dip - 95.0) / 80.0))
        distance_score = min(1.0, max(0.0, (tip_distance - mcp_distance) / 0.75))
        return 0.40 * angle_score + 0.35 * dip_score + 0.25 * distance_score

    @staticmethod
    def _thumb_extended_score(p, palm):
        angle = GestureEngine._angle(p[1], p[2], p[3])
        tip_distance = GestureEngine._dist(p[4], p[9]) / palm
        angle_score = min(1.0, max(0.0, (angle - 100.0) / 70.0))
        distance_score = min(1.0, max(0.0, (tip_distance - 0.30) / 0.55))
        return 0.55 * angle_score + 0.45 * distance_score

    @staticmethod
    def _angle(a, b, c):
        bax = a[0] - b[0]
        bay = a[1] - b[1]
        bcx = c[0] - b[0]
        bcy = c[1] - b[1]
        denom = max(1e-9, hypot(bax, bay) * hypot(bcx, bcy))
        cosine = (bax * bcx + bay * bcy) / denom
        cosine = max(-1.0, min(1.0, cosine))
        return degrees(acos(cosine))

    @staticmethod
    def _dist(a, b):
        return hypot(a[0] - b[0], a[1] - b[1])
    def _update_track(self, track, candidate, confidence, now, hand_index):
        current = track.gesture_type

        if current == candidate:
            track.last_seen_ms = now
            track.confidence = confidence
            state = GestureState.HOLD if current != GestureType.NONE else GestureState.NONE
            return Gesture(current, hand_index, confidence, state, max(0, now - track.start_ms))

        if candidate != track.candidate_type:
            track.candidate_type = candidate
            track.candidate_since_ms = now

        stable_for = now - track.candidate_since_ms

        if stable_for >= self.switch_stability_ms:
            previous = current
            track.gesture_type = candidate
            track.start_ms = now
            track.last_seen_ms = now
            track.confidence = confidence
            state = GestureState.DOWN if candidate != GestureType.NONE else GestureState.RELEASE
            if candidate == GestureType.NONE and previous == GestureType.NONE:
                state = GestureState.NONE
            return Gesture(candidate, hand_index, confidence, state, 0)

        if current != GestureType.NONE:
            track.last_seen_ms = now
            track.confidence = max(track.confidence * 0.92, confidence * 0.65)
            return Gesture(current, hand_index, track.confidence, GestureState.HOLD, max(0, now - track.start_ms))

        return Gesture(GestureType.NONE, hand_index, confidence, GestureState.NONE, 0)

    def _update_gate(self, observations, now):
        open_hands = [g for g in observations if g.type == GestureType.OPEN_HAND and g.confidence >= 0.65]
        both_open = len(open_hands) >= 2

        if both_open:
            self._open_lost_since_ms = None
            if self._open_since_ms is None:
                self._open_since_ms = now
            if not self._two_hand_open and now - self._open_since_ms >= self.open_gate_ms:
                self._two_hand_open = True
        else:
            self._open_since_ms = None
            if self._two_hand_open:
                if self._open_lost_since_ms is None:
                    self._open_lost_since_ms = now
                elif now - self._open_lost_since_ms >= self.open_release_ms:
                    self._two_hand_open = False

    def _expire(self, used_keys, now):
        expired = []
        for key, track in self._tracks.items():
            if key not in used_keys and now - track.last_seen_ms > self.release_timeout_ms:
                expired.append(key)
        for key in expired:
            del self._tracks[key]
