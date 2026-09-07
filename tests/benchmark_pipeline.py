import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.camera.camera import Camera, CameraConfig
from src.vision.hand_tracker import HandTracker
from src.gestures.gesture_engine import GestureEngine, GestureType
from src.portal.portal_engine import PortalEngine
from src.dimensions.procedural import ProceduralDimension
from src.portal.portal_renderer import PortalRenderer

MODEL = "assets/models/hand_landmarker.task"
DURATION = 8.0
WARMUP = 2.0
RUNS = 3


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark RetroLens-X pipeline")
    parser.add_argument(
        "--force-portal",
        action="store_true",
        help="Render the portal every frame using a fixed geometry, isolating active-pipeline cost.",
    )
    parser.add_argument("--runs", type=int, default=RUNS, help="Number of measured runs.")
    parser.add_argument("--duration", type=float, default=DURATION, help="Seconds per measured run.")
    parser.add_argument("--warmup", type=float, default=WARMUP, help="Warmup seconds before each measured run.")
    return parser.parse_args()


def avg(values):
    return sum(values) / len(values) * 1000 if values else 0.0


def percentile(values, percentile_value):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, int(round((percentile_value / 100) * (len(ordered) - 1)))),
    )
    return ordered[index] * 1000


def run_once(force_portal, duration, warmup):
    camera = Camera(CameraConfig(threaded=True))
    tracker = HandTracker(MODEL)
    gestures = GestureEngine()
    portal = PortalEngine()
    dimension = ProceduralDimension()
    renderer = PortalRenderer()
    timestamp_ms = 0

    def process_frame(measure=False):
        nonlocal timestamp_ms
        total_start = time.perf_counter()

        t = time.perf_counter()
        frame = camera.read()
        camera_time = time.perf_counter() - t

        timestamp_ms += 33
        t = time.perf_counter()
        hands = tracker.detect(frame, timestamp_ms)
        vision_time = time.perf_counter() - t

        t = time.perf_counter()
        detected = gestures.detect(hands, timestamp_ms)
        gesture_time = time.perf_counter() - t

        open_indices = [g.hand_index for g in detected if g.type == GestureType.OPEN_HAND]
        open_hands = [hands[i] for i in open_indices if 0 <= i < len(hands)]

        t = time.perf_counter()
        if len(open_hands) >= 2:
            portal.update(open_hands[:2], timestamp_ms)
        else:
            portal.state.active = False
        portal_time = time.perf_counter() - t

        if force_portal:
            portal.state.active = True
            portal.state.center = (640, 360)
            portal.state.width = 650
            portal.state.height = 468
            portal.state.angle = 0.0

        dimension_time = 0.0
        renderer_time = 0.0
        active_total_time = 0.0
        active = portal.state.active
        if active:
            state = portal.state
            active_start = time.perf_counter()

            t = time.perf_counter()
            portal_dimension = dimension.render(state.width, state.height, timestamp_ms)
            dimension_time = time.perf_counter() - t

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
            renderer_time = time.perf_counter() - t
            active_total_time = time.perf_counter() - active_start

        total_time = time.perf_counter() - total_start
        if not measure:
            return None
        return (
            camera_time,
            vision_time,
            gesture_time,
            portal_time,
            dimension_time,
            renderer_time,
            active_total_time,
            total_time,
            active,
        )

    try:
        warmup_start = time.perf_counter()
        while time.perf_counter() - warmup_start < warmup:
            process_frame(measure=False)

        samples = []
        active_frames = 0
        start = time.perf_counter()
        while time.perf_counter() - start < duration:
            sample = process_frame(measure=True)
            samples.append(sample)
            active_frames += int(sample[-1])
    finally:
        tracker.close()
        camera.release()

    elapsed = time.perf_counter() - start
    columns = list(zip(*samples)) if samples else [[] for _ in range(9)]
    dimension_values = [v for v, active in zip(columns[4], columns[8]) if active]
    renderer_values = [v for v, active in zip(columns[5], columns[8]) if active]
    active_values = [v for v in columns[6] if v > 0]
    return {
        "frames": len(samples),
        "active_frames": active_frames,
        "fps": len(samples) / elapsed if elapsed > 0 else 0.0,
        "camera": avg(columns[0]),
        "vision": avg(columns[1]),
        "gesture": avg(columns[2]),
        "portal": avg(columns[3]),
        "dimension": avg(dimension_values),
        "renderer": avg(renderer_values),
        "active_total": avg(active_values),
        "active_p95": percentile(active_values, 95),
        "total": avg(columns[7]),
        "total_p95": percentile(columns[7], 95),
    }


def median(results, key):
    return statistics.median(result[key] for result in results)


def main():
    args = parse_args()
    runs = max(1, args.runs)
    duration = max(0.5, args.duration)
    warmup = max(0.0, args.warmup)

    print("=== BENCHMARK PIPELINE ===")
    print("Modo cámara: latest-frame threaded capture")
    print("Modo portal:", "FORCED ACTIVE" if args.force_portal else "LIVE GESTURE")
    if not args.force_portal:
        print("Activa el portal manteniendo DOS manos abiertas frente a la cámara.")
    print(f"Runs: {runs} | Warmup: {warmup:.1f}s | Duración: {duration:.1f}s/run")

    results = []
    for index in range(runs):
        result = run_once(args.force_portal, duration, warmup)
        results.append(result)
        print(
            f"Run {index + 1}: {result['fps']:.2f} FPS | "
            f"active {result['active_total']:.2f} ms | "
            f"renderer {result['renderer']:.2f} ms | "
            f"MediaPipe {result['vision']:.2f} ms"
        )

    print()
    print("=== MEDIANA DE RUNS ===")
    print("FPS pipeline:", f"{median(results, 'fps'):.2f}")
    print("Camera read:", f"{median(results, 'camera'):.3f} ms")
    print("MediaPipe:", f"{median(results, 'vision'):.2f} ms")
    print("Gestures:", f"{median(results, 'gesture'):.3f} ms")
    print("Portal:", f"{median(results, 'portal'):.3f} ms")
    print("Dimension activo:", f"{median(results, 'dimension'):.2f} ms")
    print("Renderer activo:", f"{median(results, 'renderer'):.2f} ms")
    print("Active pipeline:", f"{median(results, 'active_total'):.2f} ms")
    print("Active pipeline P95:", f"{median(results, 'active_p95'):.2f} ms")
    print("TOTAL loop:", f"{median(results, 'total'):.2f} ms")
    print("TOTAL loop P95:", f"{median(results, 'total_p95'):.2f} ms")
    print(
        "Cobertura portal:",
        f"{median(results, 'active_frames') / max(median(results, 'frames'), 1) * 100:.1f}%",
    )


if __name__ == "__main__":
    main()
