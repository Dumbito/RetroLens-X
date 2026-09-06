import time
from pathlib import Path

import cv2

from src.vision.hand_tracker import HandTracker


MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "assets"
    / "models"
    / "hand_landmarker.task"
)


def draw_hand(frame, hand):
    height, width = frame.shape[:2]

    points = []

    for landmark in hand.landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)

        points.append((x, y))

        cv2.circle(
            frame,
            (x, y),
            4,
            (0, 255, 0),
            -1,
        )

    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17),
    ]

    for start, end in connections:
        cv2.line(
            frame,
            points[start],
            points[end],
            (255, 255, 255),
            2,
        )


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: No se pudo abrir la cámara.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("RetroLens-X | HandTracker test")
    print("Q o ESC para salir.")

    start_time = time.perf_counter()
    frame_count = 0

    with HandTracker(MODEL_PATH) as tracker:

        while True:
            success, frame = cap.read()

            if not success:
                print("ERROR: No se pudo leer el frame.")
                break

            frame = cv2.flip(frame, 1)

            timestamp_ms = time.monotonic_ns() // 1_000_000

            hands = tracker.detect(
                frame,
                timestamp_ms,
            )

            for hand in hands:
                draw_hand(frame, hand)

            frame_count += 1

            elapsed = time.perf_counter() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0

            cv2.putText(
                frame,
                f"Hands: {len(hands)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                "RetroLens-X | HandTracker",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
