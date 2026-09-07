import time
from src.camera.camera import Camera

camera = Camera()
print("=== CAMERA BENCHMARK ===")
print("Backend:", camera.capture.get(42))
print("FOURCC:", int(camera.capture.get(6)))
print("Width:", camera.capture.get(3))
print("Height:", camera.capture.get(4))
print("Requested FPS:", camera.capture.get(5))
times = []
for _ in range(100):
    t0 = time.perf_counter()
    frame = camera.read()
    times.append(time.perf_counter() - t0)
camera.release()
avg = sum(times) / len(times)
print()
print("=== RESULTADOS ===")
print(f"Frames: {len(times)}")
print(f"Average read: {avg * 1000:.2f} ms")
print(f"Capture FPS: {1.0 / avg:.2f}")
print(f"Min read: {min(times) * 1000:.2f} ms")
print(f"Max read: {max(times) * 1000:.2f} ms")
