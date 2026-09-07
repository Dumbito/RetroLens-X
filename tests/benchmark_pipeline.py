import time
import cv2
from src.camera.camera import Camera
from src.vision.hand_tracker import HandTracker
from src.gestures.gesture_engine import GestureEngine, GestureType
from src.portal.portal_engine import PortalEngine
from src.dimensions.procedural import ProceduralDimension
from src.portal.portal_renderer import PortalRenderer

MODEL = "assets/models/hand_landmarker.task"
DURATION = 15.0

camera = Camera()
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
total_times = []
active_frames = 0
frames = 0
start = time.perf_counter()
timestamp_ms = 0

print("=== BENCHMARK PIPELINE ===")
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

        if portal.state.active:
            active_frames += 1
            state = portal.state

            t = time.perf_counter()
            portal_dimension = dimension.render(state.width, state.height, timestamp_ms)
            dimension_times.append(time.perf_counter() - t)

            t = time.perf_counter()
            output = renderer.render(frame, portal_dimension, state.center, state.width, state.height, state.angle, timestamp_ms)
            renderer_times.append(time.perf_counter() - t)
        else:
            dimension_times.append(0.0)
            renderer_times.append(0.0)

        total_times.append(time.perf_counter() - total_start)
        frames += 1
finally:
    tracker.close()
    camera.release()

elapsed = time.perf_counter() - start

def avg(values):
    return sum(values) / len(values) * 1000 if values else 0.0

print()
print("=== RESULTADOS ===")
print("Frames:", frames)
print("Frames portal activo:", active_frames)
print("Cobertura portal:", f"{active_frames / max(frames, 1) * 100:.1f}%")
print("FPS pipeline:", f"{frames / elapsed:.2f}")
print("Camera:", f"{avg(camera_times):.2f} ms")
print("MediaPipe:", f"{avg(vision_times):.2f} ms")
print("Gestures:", f"{avg(gesture_times):.3f} ms")
print("Portal:", f"{avg(portal_times):.3f} ms")
print("Dimension:", f"{avg(dimension_times):.2f} ms")
print("Renderer:", f"{avg(renderer_times):.2f} ms")
print("TOTAL:", f"{avg(total_times):.2f} ms")
