import math
import time

import cv2

from src.camera.camera import Camera, CameraConfig
from src.vision.hand_tracker import HandTracker
from src.gestures.gesture_engine import GestureEngine
from src.portal.portal_engine import PortalEngine
from src.portal.portal_renderer import PortalRenderer
from src.dimensions import MultiverseDimension


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


def fingertip_distance(hands):
    if len(hands) < 2:
        return None
    first = hands[0].pixel_landmarks
    second = hands[1].pixel_landmarks
    if len(first) <= 8 or len(second) <= 8:
        return None
    return math.hypot(second[8][0] - first[8][0], second[8][1] - first[8][1])


def hand_scale(hands):
    values = []
    for hand in hands[:2]:
        points = hand.pixel_landmarks
        if len(points) > 9:
            value = math.hypot(points[9][0] - points[0][0], points[9][1] - points[0][1])
            if value > 1.0:
                values.append(value)
    return sum(values) / len(values) if values else None


def main():
    camera = Camera(CameraConfig(threaded=True))
    tracker = HandTracker("assets/models/hand_landmarker.task")
    gestures = GestureEngine()
    portal = PortalEngine(min_width=120, max_width=900, aspect_ratio=0.58, smoothing=0.24)
    renderer = PortalRenderer()
    dimension = MultiverseDimension(work_scale=0.60)
    show_hand_rig = False

    portal_intensity = 0.0
    phase = "READY"
    lost_since = None
    last_time = time.monotonic()

    # Much more forgiving fingertip gesture:
    # 1) bring index fingertips close together to arm,
    # 2) separate them to open,
    # 3) bring them close again to close.
    ARM_RATIO = 0.72
    OPEN_RATIO = 1.05
    CLOSE_RATIO = 0.78
    LOST_GRACE_SECONDS = 0.60
    ARM_HOLD_SECONDS = 0.08

    armed_since = None
    armed_distance = None

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

            distance = fingertip_distance(hands)
            scale = hand_scale(hands)

            if distance is not None and scale is not None:
                arm_distance = scale * ARM_RATIO
                open_distance = scale * OPEN_RATIO
                close_distance = scale * CLOSE_RATIO

                if phase == "READY":
                    if distance <= arm_distance:
                        if armed_since is None:
                            armed_since = now
                            armed_distance = distance
                        elif now - armed_since >= ARM_HOLD_SECONDS:
                            phase = "ARMED"
                            lost_since = None
                    else:
                        armed_since = None
                        armed_distance = None

                elif phase == "ARMED":
                    armed_distance = distance
                    if distance >= open_distance:
                        phase = "OPEN"
                        lost_since = None
                        portal.reset()
                        portal.update(hands, timestamp_ms)
                    elif distance > arm_distance * 1.65:
                        phase = "READY"
                        armed_since = None
                        armed_distance = None

                elif phase == "OPEN":
                    if distance <= close_distance:
                        phase = "READY"
                        lost_since = None
                        portal.reset()
                        armed_since = None
                        armed_distance = None
                    else:
                        portal.update(hands, timestamp_ms)
                        lost_since = None

            elif phase == "OPEN":
                if lost_since is None:
                    lost_since = now
                elif now - lost_since >= LOST_GRACE_SECONDS:
                    phase = "READY"
                    lost_since = None
                    armed_since = None
                    armed_distance = None
                    portal.reset()
            else:
                phase = "READY"
                armed_since = None
                armed_distance = None

            target_intensity = 1.0 if phase == "OPEN" else 0.0
            smoothing = 1.0 - math.exp(-delta_time * 12.0)
            portal_intensity += (target_intensity - portal_intensity) * smoothing

            if phase == "OPEN" and portal_intensity > 0.005:
                state = portal.state
                frame_h, frame_w = frame.shape[:2]
                view_x = ((state.center[0] / max(frame_w - 1, 1)) - 0.5) * 2.0
                view_y = ((state.center[1] / max(frame_h - 1, 1)) - 0.5) * 2.0
                view_x = float(max(-1.0, min(1.0, view_x)))
                view_y = float(max(-1.0, min(1.0, view_y)))

                portal_dimension = dimension.render(
                    frame,
                    state.width,
                    state.height,
                    state.center,
                    timestamp_ms,
                    view_x=view_x,
                    view_y=view_y,
                )
                frame = renderer.render(
                    frame,
                    portal_dimension,
                    state.center,
                    state.width,
                    state.height,
                    0.0,
                    timestamp_ms,
                    portal_intensity,
                )

            if phase == "OPEN":
                status = "MULTIVERSE WINDOW"
                hint = "JUNTA LOS INDICES PARA CERRAR"
            elif phase == "ARMED":
                status = "FRAME ARMED"
                hint = "SEPARA LOS INDICES PARA ABRIR"
            else:
                status = "FRAME READY"
                hint = "JUNTA LAS PUNTAS DE LOS INDICES"

            cv2.putText(frame, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, hint, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            if distance is not None and scale is not None:
                cv2.putText(frame, f"INDEX DIST: {distance:.0f} / HAND SCALE: {scale:.0f}", (20, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "H = HAND RIG | Q / ESC = EXIT", (20, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

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
