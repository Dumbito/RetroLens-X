import time
import cv2
from src.camera.camera import Camera
from src.vision.hand_tracker import HandTracker
from src.gestures.gesture_engine import GestureEngine, GestureType
from src.portal.portal_engine import PortalEngine
from src.dimensions import ProceduralDimension
from src.portal.portal_renderer import PortalRenderer

def main():
    camera = Camera()
    tracker = HandTracker("assets/models/hand_landmarker.task")
    gestures = GestureEngine()
    portal = PortalEngine()
    dimension = ProceduralDimension()
    renderer = PortalRenderer()
    samples = []
    try:
        print("=== RETROLENS-X PROFILER ===")
        print("Activa el portal con las dos manos abiertas.")
        print("Midiendo durante 10 segundos...")
        end_time = time.perf_counter() + 10.0
        while time.perf_counter() < end_time:
            t0 = time.perf_counter()
            frame = camera.read()
            t1 = time.perf_counter()
            timestamp_ms = time.monotonic_ns() // 1_000_000
            hands = tracker.detect(frame, timestamp_ms)
            t2 = time.perf_counter()
            detected = gestures.detect(hands, timestamp_ms)
            t3 = time.perf_counter()
            open_hand_indices = [g.hand_index for g in detected if g.type == GestureType.OPEN_HAND]
            open_hands = [hands[i] for i in open_hand_indices if 0 <= i < len(hands)]
            if len(open_hands) >= 2:
                portal.update(open_hands[:2], timestamp_ms)
            else:
                portal.state.active = False
            t4 = time.perf_counter()
            if portal.state.active:
                h, w = frame.shape[:2]
                portal_dimension = dimension.render(w, h, timestamp_ms)
                t5 = time.perf_counter()
                state = portal.state
                renderer.render(frame, portal_dimension, state.center, state.width, state.height, state.angle, timestamp_ms)
                t6 = time.perf_counter()
            else:
                t5 = t4
                t6 = t4
            samples.append(((t1-t0)*1000, (t2-t1)*1000, (t3-t2)*1000, (t4-t3)*1000, (t5-t4)*1000, (t6-t5)*1000, t6-t0, portal.state.active))
    finally:
        tracker.close()
        camera.release()
        cv2.destroyAllWindows()
    if not samples:
        print("ERROR: no se obtuvieron frames.")
        return
    avg = [sum(x[i] for x in samples) / len(samples) for i in range(6)]
    fps = len(samples) / sum(x[6] for x in samples)
    active = sum(1 for x in samples if x[7])
    print()
    print("=== RESULTADOS ===")
    print(f"Frames:       {len(samples)}")
    print(f"FPS pipeline: {fps:.2f}")
    print(f"Camera:       {avg[0]:.2f} ms")
    print(f"MediaPipe:    {avg[1]:.2f} ms")
    print(f"Gestures:     {avg[2]:.2f} ms")
    print(f"Portal:       {avg[3]:.2f} ms")
    print(f"Dimension:    {avg[4]:.2f} ms")
    print(f"Renderer:     {avg[5]:.2f} ms")
    print(f"Portal activo en {active}/{len(samples)} frames")

if __name__ == "__main__":
    main()
