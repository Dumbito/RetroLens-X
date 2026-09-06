import time
import cv2
from src.camera.camera import Camera
from src.vision.hand_tracker import HandTracker
from src.gestures.gesture_engine import GestureEngine

CONNECTIONS = [
(0, 1), (1, 2), (2, 3), (3, 4),
(0, 5), (5, 6), (6, 7), (7, 8),
(5, 9), (9, 10), (10, 11), (11, 12),
(9, 13), (13, 14), (14, 15), (15, 16),
(13, 17), (17, 18), (18, 19), (19, 20),
(0, 17)]


def draw_hand_rig(frame, hand):
    points = hand.pixel_landmarks
    for start, end in CONNECTIONS:
        cv2.line(frame, points[start], points[end], (255, 180, 0), 2)
    for point in points:
        cv2.circle(frame, point, 5, (0, 255, 255), -1)
    x1, y1, x2, y2 = hand.bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


def main():
    camera = Camera()
    tracker = HandTracker("assets/models/hand_landmarker.task")
    gestures = GestureEngine()

    try:
        while True:
            frame = camera.read()
            timestamp_ms = time.monotonic_ns() // 1_000_000

            hands = tracker.detect(frame, timestamp_ms)
            detected = gestures.detect(hands, timestamp_ms)

            for hand, gesture in zip(hands, detected):
                draw_hand_rig(frame, hand)
                x1, y1, _, _ = hand.bbox
                label = f"{gesture.type.value} | {gesture.state.value} | {gesture.duration_ms}ms"
                cv2.putText(frame, label, (x1, max(30, y1 - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

            cv2.putText(frame, "RetroLens-X | GestureEngine v0.2", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, "Q / ESC = salir", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("RetroLens-X | GestureEngine v0.2", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    finally:
        tracker.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
