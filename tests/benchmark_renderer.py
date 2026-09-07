import time

import numpy as np

from src.portal.portal_renderer import PortalRenderer


WIDTH = 1280
HEIGHT = 720
PORTAL_WIDTH = 650
PORTAL_HEIGHT = 468
ITERATIONS = 120
WARMUP = 20


def main():
    renderer = PortalRenderer()
    dimension = np.full((PORTAL_HEIGHT, PORTAL_WIDTH, 3), 128, dtype=np.uint8)
    center = (WIDTH // 2, HEIGHT // 2)
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    for _ in range(WARMUP):
        renderer.render(frame.copy(), dimension, center, PORTAL_WIDTH, PORTAL_HEIGHT, 0.0, 1000)

    times = []
    for i in range(ITERATIONS):
        test_frame = frame.copy()
        start = time.perf_counter()
        renderer.render(
            test_frame,
            dimension,
            center,
            PORTAL_WIDTH,
            PORTAL_HEIGHT,
            0.0,
            2000 + i * 33,
        )
        times.append(time.perf_counter() - start)

    ordered = sorted(times)
    mean_ms = sum(times) / len(times) * 1000
    p50_ms = ordered[len(ordered) // 2] * 1000
    p95_ms = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))] * 1000
    fps = 1000.0 / mean_ms if mean_ms else 0.0

    print("=== RENDERER MICROBENCHMARK ===")
    print("Frame:", f"{WIDTH}x{HEIGHT}")
    print("Portal:", f"{PORTAL_WIDTH}x{PORTAL_HEIGHT}")
    print("Iterations:", ITERATIONS)
    print("Mean:", f"{mean_ms:.2f} ms")
    print("P50:", f"{p50_ms:.2f} ms")
    print("P95:", f"{p95_ms:.2f} ms")
    print("Renderer-only FPS:", f"{fps:.2f}")


if __name__ == "__main__":
    main()
