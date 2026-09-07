# RetroLens-X

Code-first real-time webcam AR portal. The project combines MediaPipe hand tracking, stabilized portal geometry, procedural dimensions, localized VFX and a composable render graph. No TouchDesigner dependency.

## Architecture

```text
Camera
  ↓
HandTracker → GestureEngine
  ↓
PortalStateMachine → PortalEngine → PortalPhysics
  ↓
FrameContext
  ↓
EffectPipeline
  ├─ BackgroundFX
  ├─ SpatialFold
  ├─ Dimensions
  └─ PortalRenderer
  ↓
OpenCV display
```

The runtime is intentionally split into independent stages. `FrameContext` carries per-frame state, while `EffectPipeline` makes render stages composable. This keeps the current OpenCV implementation intact and creates a clean seam for a future OpenGL/GLSL compositor.

## Engineering patterns adopted

The architecture was informed by several open-source real-time projects rather than copying their code:

- **Latest-frame / drop-queue processing:** real-time systems should prefer current frames over processing an ever-growing backlog. This principle is used by the camera's threaded reader and is documented by ConceptCodes' Portal and NeHeGL's Camera Effects.
- **Temporal stabilization:** normalized hand geometry, adaptive smoothing and velocity prediction reduce jitter without making fast movements feel sluggish.
- **Effect graphs:** VJ-9000 demonstrates the value of a composable GPU effect chain; RetroLens-X now has the same conceptual separation while remaining backend-neutral.
- **Clipped interaction regions:** Finger Frame applies effects only inside the hand-defined frame; RetroLens-X extends that idea with an organic aperture and ROI-local processing.
- **Reduced-resolution rendering:** expensive procedural dimensions and VFX run at bounded internal resolution before being composited into the camera frame.
- **Profiling as a first-class feature:** the runtime includes rolling FPS, average and P95 frame timing plus stage timings, toggled with `P`.

These references are design inputs, not bundled dependencies or copied assets.

## Current visual pipeline

The portal is a rectangular interdimensional aperture controlled by the two index fingertips:

1. Bring the index fingertips together to arm the frame.
2. Separate them to open it.
3. Moving the hands moves and rotates the stabilized portal.
4. Bringing the fingertips together closes it.
5. The interior combines the live camera transformation, procedural comic styling and a neutral/warm cosmic universe.
6. Background distortion, spatial folding, waves, particles, chromatic edges and localized signal artifacts react to portal motion and pressure.

Expensive operations stay ROI-bound; heavy full-frame processing is deliberately avoided.

## Development environment

- CachyOS / Linux
- Python 3.11
- OpenCV / OpenCV contrib
- MediaPipe Tasks API
- NumPy
- pytest

Dependencies are pinned in `requirements.txt` for reproducibility. The hand-landmarker model is a local binary asset and remains excluded from Git.

## Controls

- Bring index fingertips together: arm/close
- Separate index fingertips: open
- `H`: toggle hand landmark rig
- `P`: toggle live profiler
- `Q` / `Esc`: exit

## Validation

```bash
python -m compileall -q src tests
pytest -q tests/test_core_components.py tests/test_runtime_architecture.py
```

For hardware benchmarks:

```bash
QT_QPA_PLATFORM=xcb python -m tests.benchmark_v4l2
QT_QPA_PLATFORM=xcb python -m tests.benchmark_pipeline
```

A fresh local benchmark is required before publishing a new FPS figure.

## Repository research notes

High-value references reviewed during the architecture pass:

- `sophiamyang/finger-frame-effect` — hand-defined quad, exponential smoothing, presence fade and effect clipping.
- `NeHeGL/Camera-Effects` — native Python real-time effects and explicit protection against frame backlog.
- `gantasmo/VJ-9000` — source/effect separation, composable GPU chain and render-scale tiers.
- `ConceptCodes/portal` — drop queues, temporal smoothing, preallocated output buffers and independent output sinks.

The implementation remains native to RetroLens-X; these repositories supplied patterns and ideas only.

## License

This project is an independent implementation. See the repository history for project provenance and development notes.
