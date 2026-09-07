import time

import numpy as np

from src.portal.portal_renderer import PortalRenderer


WIDTH = 1280
HEIGHT = 720
PORTAL_WIDTH = 650
PORTAL_HEIGHT = 468
ITERATIONS = 80
WARMUP = 10


def measure(label, fn):
    for _ in range(WARMUP):
        fn()
    values = []
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        fn()
        values.append(time.perf_counter() - start)
    mean = sum(values) / len(values) * 1000
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))] * 1000
    print(f"{label}: mean={mean:.2f} ms p95={p95:.2f} ms")
    return mean


def main():
    renderer = PortalRenderer()
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    center = (WIDTH // 2, HEIGHT // 2)
    dimension = np.full((PORTAL_HEIGHT, PORTAL_WIDTH, 3), 128, dtype=np.uint8)
    t = 12345.0
    angle = 0.0

    local_w = int(PORTAL_WIDTH + 2 * max(8, int(np.ceil(renderer.config.glow_sigma * 3 + renderer.config.edge_thickness + 4))))
    local_h = int(PORTAL_HEIGHT + 2 * max(8, int(np.ceil(renderer.config.glow_sigma * 3 + renderer.config.edge_thickness + 4))))
    local_center = (local_w // 2, local_h // 2)

    mask = renderer._organic_mask((local_h, local_w), local_center, PORTAL_WIDTH, PORTAL_HEIGHT, angle, t * 0.001)
    content = renderer._prepare_content(dimension, PORTAL_WIDTH, PORTAL_HEIGHT, angle, local_center, local_w, local_h)

    print("=== RENDERER STAGE PROFILE ===")
    print("Local ROI:", f"{local_w}x{local_h}")
    print("Rings:", renderer.config.ring_count)
    print("Particles:", renderer.config.particle_count)
    print()

    measure(
        "Organic mask",
        lambda: renderer._organic_mask((local_h, local_w), local_center, PORTAL_WIDTH, PORTAL_HEIGHT, angle, t * 0.001),
    )
    measure(
        "Prepare content",
        lambda: renderer._prepare_content(dimension, PORTAL_WIDTH, PORTAL_HEIGHT, angle, local_center, local_w, local_h),
    )
    measure("Canny", lambda: __import__("cv2").Canny(mask, 70, 180))

    edges = __import__("cv2").Canny(mask, 70, 180)
    measure(
        "Gaussian glow",
        lambda: __import__("cv2").GaussianBlur(edges, (0, 0), renderer.config.glow_sigma),
    )

    glow = __import__("cv2").GaussianBlur(edges, (0, 0), renderer.config.glow_sigma)
    image = np.zeros_like(content, dtype=np.float32)
    measure("Add glow", lambda: renderer._add_glow(image.copy(), glow))

    energy = np.zeros_like(content)
    measure(
        "Rings",
        lambda: renderer._draw_rings(energy.copy(), local_center, PORTAL_WIDTH, PORTAL_HEIGHT, angle, t * 0.001),
    )
    measure(
        "Particles",
        lambda: renderer._draw_particles(energy.copy(), local_center, PORTAL_WIDTH, PORTAL_HEIGHT, angle, t * 0.001),
    )
    rim = np.zeros_like(content)
    measure(
        "Rim",
        lambda: renderer._draw_rim(rim.copy(), local_center, PORTAL_WIDTH, PORTAL_HEIGHT, angle, t * 0.001),
    )

    print()
    measure(
        "Full renderer",
        lambda: renderer.render(
            frame.copy(),
            dimension,
            center,
            PORTAL_WIDTH,
            PORTAL_HEIGHT,
            angle,
            12345,
        ),
    )


if __name__ == "__main__":
    main()
