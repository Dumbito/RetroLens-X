import time

import cv2

from src.camera.camera import Camera, CameraConfig
from src.vision.hand_tracker import HandTracker
from src.gestures.gesture_engine import GestureEngine
from src.portal.portal_engine import PortalEngine
from src.portal.portal_renderer import PortalRenderer
from src.dimensions import ProceduralDimension


CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def draw_hand_rig(frame, hand):
    points = hand.pixel_landmarks
    for start, end in CONNECTIONS:
        cv2.line(frame, points[start], points[end], (255, 180, 0), 2)
    for point in points:
        cv2.circle(frame, point, 5, (0, 255, 255), -1)
    x1, y1, x2, y2 = hand.bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


def main():
    camera = Camera(CameraConfig(threaded=True))
    tracker = HandTracker("assets/models/hand_landmarker.task")
    gestures = GestureEngine()
    portal = PortalEngine()
    renderer = PortalRenderer()
    dimension = ProceduralDimension()
    show_hand_rig = False

    try:
        while True:
            frame = camera.read()
            timestamp_ms = time.monotonic_ns() // 1_000_000

            hands = tracker.detect(frame, timestamp_ms)
            gestures.detect(hands, timestamp_ms)

            if show_hand_rig:
                for hand in hands:
                    draw_hand_rig(frame, hand)

            if gestures.two_hand_open:
                portal.update(hands, timestamp_ms)
            else:
                portal.state.active = False

            if portal.state.active:
                state = portal.state
                portal_dimension = dimension.render(
                    state.width,
                    state.height,
                    timestamp_ms,
                )
                frame = renderer.render(
                    frame,
                    portal_dimension,
                    state.center,
                    state.width,
                    state.height,
                    state.angle,
                    timestamp_ms,
                )

            status = "PORTAL ACTIVE" if portal.state.active else "PORTAL STANDBY"
            cv2.putText(
                frame,
                status,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                "2 OPEN HANDS = OPEN PORTAL",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )
            cv2.putText(
                frame,
                "H = HAND RIG | Q / ESC = EXIT",
                (20, 92),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )

            cv2.imshow("RetroLens-X", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("h"):
                show_hand_rig = not show_hand_rig
            elif key in (ord("q"), 27):
                break

    finally:
        tracker.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
