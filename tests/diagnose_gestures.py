import time
import cv2
from src.camera.camera import Camera, CameraConfig
from src.vision.hand_tracker import HandTracker
from src.gestures.gesture_engine import GestureEngine, GestureType

MODEL = "assets/models/hand_landmarker.task"
camera = Camera(CameraConfig(width=1280, height=720, fps=30))
tracker = HandTracker(MODEL, num_hands=2)
engine = GestureEngine()
start = time.monotonic()
last_report = 0.0
frames = 0

try:
    while time.monotonic() - start < 15.0:
        frame = camera.read()
        timestamp_ms = int((time.monotonic() - start) * 1000)
        hands = tracker.detect(frame, timestamp_ms)
        gestures = engine.detect(hands, timestamp_ms)
        frames += 1

        for i, hand in enumerate(hands):
            p = engine._points(hand)
            track = engine.tracked_hands.get(hand.handedness)
            if track is None:
                continue
            fingers = [
                engine._finger_extended_score(p, 8, 6, 7, 5, max(engine._dist(p[0], p[9]), 1e-6)),
                engine._finger_extended_score(p, 12, 10, 11, 9, max(engine._dist(p[0], p[9]), 1e-6)),
                engine._finger_extended_score(p, 16, 14, 15, 13, max(engine._dist(p[0], p[9]), 1e-6)),
                engine._finger_extended_score(p, 20, 18, 19, 17, max(engine._dist(p[0], p[9]), 1e-6)),
            ]
            palm = max(engine._dist(p[0], p[9]), 1e-6)
            thumb = engine._thumb_extended_score(p, palm)
            pinch = engine._dist(p[4], p[8]) / palm
            gesture = gestures[i].type.value if i < len(gestures) else "?"
            now = time.monotonic()
            if now - last_report >= 0.35:
                print(f"{hand.handedness:>7} | G={gesture:<9} | F={fingers[0]:.2f},{fingers[1]:.2f},{fingers[2]:.2f},{fingers[3]:.2f} | T={thumb:.2f} | P={pinch:.2f}")
                last_report = now

        cv2.putText(frame, f"HANDS: {len(hands)}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("RetroLens-X v0.5.3 Diagnostic", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    tracker.close()
    camera.release()
    cv2.destroyAllWindows()

