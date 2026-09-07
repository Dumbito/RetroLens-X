# RetroLens-X

Code-first real-time webcam AR portal project inspired by RetroLens, rebuilt as a modular system for gesture-driven portals and interactive dimensions.

## Architecture

Camera → HandTracker → GestureEngine → PortalEngine → PortalRenderer → Dimension

## Current status

The core camera, MediaPipe hand tracking, gesture detection, portal geometry, procedural dimension, renderer, and integration pipeline are implemented.

The current codebase also includes regression tests and continuous integration for compilation and core components.

## Performance work

The original baseline was approximately 11 FPS for the full pipeline at 1280×720. The main bottleneck was the portal renderer, which previously allocated and processed full-frame masks, effects, glows, rings, and particle layers.

The renderer now limits expensive VFX work to a portal-local region of interest (ROI). The procedural dimension is also rendered at reduced internal resolution and scaled to the portal size. The main loop no longer draws the hand rig unless explicitly enabled with `H`.

A fresh local benchmark is required before publishing a new FPS figure.

## Development environment

- CachyOS / Linux
- Python 3.11
- OpenCV / OpenCV contrib
- MediaPipe Tasks API
- NumPy
- pytest

Dependencies are pinned in `requirements.txt` for reproducibility.

The hand-landmarker model is downloaded locally and intentionally excluded from Git because it is a binary asset.

## Controls

- Two open hands: activate portal
- `H`: toggle hand landmark rig
- `Q` / `Esc`: exit

## Validation

```bash
python -m compileall -q src tests
pytest -q tests/test_core_components.py
```

For the complete camera and pipeline benchmarks:

```bash
QT_QPA_PLATFORM=xcb python -m tests.benchmark_v4l2
QT_QPA_PLATFORM=xcb python -m tests.benchmark_pipeline
```

## License

This project is an independent implementation. See the repository history for project provenance and development notes.
