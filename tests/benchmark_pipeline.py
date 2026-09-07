import argparse
import time

from src.camera.camera import Camera, CameraConfig
from src.vision.hand_tracker import HandTracker
from src.gestures.gesture_engine import GestureEngine, GestureType
from src.portal.portal_engine import PortalEngine
from src.dimensions.procedural import ProceduralDimension
from src.portal.portal_renderer import PortalRenderer

MODEL = "assets/models/hand_landmarker.task"
DURATION = 15.0


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark RetroLens-X pipeline")
    parser.add_argument(
        "--force-portal",
        action="store_true",
        help="Render the portal every frame using a fixed geometry, isolating active-pipeline cost.",
    )
    return parser.parse_args()


def avg(values):
    return sum(values) / len(values) * 1000 if values else 0.0


def percentile(values, percentile):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((percentile / 100) * (len(ordered) - 1)))))
    return ordered[index] * 1000


def main():
    args = parse_args()

    camera = Camera(CameraConfig(threaded=True))
    tracker = HandTracker(MODEL)
    gestures = GestureEngine()
    portal = PortalEngine()
    dimension = ProceduralDimension()
    renderer = PortalRenderer()

    camera_times = []
    vision_times = []
    gesture_times = []
    portal_times = []
    dimension_times = []
    renderer_times = []
    active_total_times = []
    total_times = []
    active_frames = 0
    frames = 0
    start = time.perf_counter()
    timestamp_ms = 0

    print("=== BENCHMARK PIPELINE ===")
    print("Modo cámara: latest-frame threaded capture")
    print("Modo portal:", "FORCED ACTIVE" if args.force_portal else "LIVE GESTURE")
    if not args.force_portal:
        print("Activa el portal manteniendo DOS manos abiertas frente a la cámara.")
    print("Duración:", DURATION, "s")

    try:
        while time.perf_counter() - start < DURATION:
            total_start = time.perf_counter()

            t = time.perf_counter()
            frame = camera.read()
            camera_times.append(time.perf_counter() - t)

            timestamp_ms += 33
            t = time.perf_counter()
            hands = tracker.detect(frame, timestamp_ms)
            vision_times.append(time.perf_counter() - t)

            t = time.perf_counter()
            detected = gestures.detect(hands, timestamp_ms)
            gesture_times.append(time.perf_counter() - t)

            open_indices = [g.hand_index for g in detected if g.type == GestureType.OPEN_HAND]
            open_hands = [hands[i] for i in open_indices if 0 <= i < len(hands)]

            t = time.perf_counter()
            if len(open_hands) >= 2:
                portal.update(open_hands[:2], timestamp_ms)
            else:
                portal.state.active = False
            portal_times.append(time.perf_counter() - t)

            if args.force_portal:
                portal.state.active = True
                portal.state.center = (640, 360)
                portal.state.width = 650
                portal.state.height = 468
                portal.state.angle = 0.0

            if portal.state.active:
                active_frames += 1
                state = portal.state
                active_start = time.perf_counter()

                t = time.perf_counter()
                portal_dimension = dimension.render(state.width, state.height, timestamp_ms)
                dimension_times.append(time.perf_counter() - t)

                t = time.perf_counter()
                renderer.render(
                    frame,
                    portal_dimension,
                    state.center,
                    state.width,
                    state.height,
                    state.angle,
                    timestamp_ms,
                )
                renderer_times.append(time.perf_counter() - t)
                active_total_times.append(time.perf_counter() - active_start)
            else:
                dimension_times.append(0.0)
                renderer_times.append(0.0)

            total_times.append(time.perf_counter() - total_start)
            frames += 1
    finally:
        tracker.close()
        camera.release()

    elapsed = time.perf_counter() - start

    print()
    print("=== RESULTADOS ===")
    print("Frames:", frames)
    print("Frames portal activo:", active_frames)
    print("Cobertura portal:", f"{active_frames / max(frames, 1) * 100:.1f}%")
    print("FPS pipeline:", f"{frames / elapsed:.2f}")
    print("Camera read:", f"{avg(camera_times):.3f} ms")
    print("MediaPipe:", f"{avg(vision_times):.2f} ms")
    print("Gestures:", f"{avg(gesture_times):.3f} ms")
    print("Portal:", f"{avg(portal_times):.3f} ms")
    print("Dimension activo:", f"{avg(dimension_times):.2f} ms")
    print("Renderer activo:", f"{avg(renderer_times):.2f} ms")
    print("Active pipeline:", f"{avg(active_total_times):.2f} ms")
    print("Active pipeline P95:", f"{percentile(active_total_times, 95):.2f} ms")
    print("TOTAL loop:", f"{avg(total_times):.2f} ms")
    print("TOTAL loop P95:", f"{percentile(total_times, 95):.2f} ms")


if __name__ == "__main__":
    main()
