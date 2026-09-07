import math
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

    portal_intensity = 0.0
    has_portal_geometry = False
    open_confirm_frames = 0
    lost_frames = 0
    OPEN_CONFIRM_FRAMES = 3
    LOST_GRACE_FRAMES = 7
    last_time = time.monotonic()

    try:
        while True:
            frame = camera.read()
            now = time.monotonic()
            delta_time = min(max(now - last_time, 0.0), 0.1)
            last_time = now
            timestamp_ms = time.monotonic_ns() // 1_000_000

            hands = tracker.detect(frame, timestamp_ms)
            gestures.detect(hands, timestamp_ms)

            if show_hand_rig:
                for hand in hands:
                    draw_hand_rig(frame, hand)

            target_active = gestures.two_hand_open

            if target_active:
                open_confirm_frames = min(open_confirm_frames + 1, OPEN_CONFIRM_FRAMES)
                lost_frames = 0
                if open_confirm_frames >= OPEN_CONFIRM_FRAMES:
                    state = portal.update(hands, timestamp_ms)
                    has_portal_geometry = state.active
            else:
                open_confirm_frames = 0
                if has_portal_geometry:
                    lost_frames += 1
                    if lost_frames > LOST_GRACE_FRAMES:
                        has_portal_geometry = False
                        portal.state.active = False
                else:
                    lost_frames = 0

            target_intensity = 1.0 if has_portal_geometry else 0.0
            smoothing = 1.0 - math.exp(-delta_time * 10.0)
            portal_intensity += (target_intensity - portal_intensity) * smoothing

            if has_portal_geometry and portal_intensity > 0.005:
                state = portal.state
                frame_h, frame_w = frame.shape[:2]

                view_x = ((state.center[0] / max(frame_w - 1, 1)) - 0.5) * 2.0
                view_y = ((state.center[1] / max(frame_h - 1, 1)) - 0.5) * 2.0
                view_x = float(max(-1.0, min(1.0, view_x)))
                view_y = float(max(-1.0, min(1.0, view_y)))
                view_angle = math.radians(float(state.angle))

                portal_dimension = dimension.render(
                    state.width,
                    state.height,
                    timestamp_ms,
                    view_x=view_x,
                    view_y=view_y,
                    view_angle=view_angle,
                )
                frame = renderer.render(
                    frame,
                    portal_dimension,
                    state.center,
                    state.width,
                    state.height,
                    state.angle,
                    timestamp_ms,
                    portal_intensity,
                )

            status = "PORTAL ACTIVE" if has_portal_geometry else "PORTAL STANDBY"
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
