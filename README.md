# RetroLens-X

Code-first real-time webcam AR portal project inspired by RetroLens, rebuilt as a modular system for gesture-driven portals and interactive dimensions.

## Architecture

Camera → HandTracker → GestureEngine → PortalEngine → PortalRenderer → Dimension

## Current status

Early development. The camera, MediaPipe hand tracking, gesture detection, portal geometry, procedural dimension, renderer, and integration pipeline are implemented and locally benchmarked.

## Development environment

- CachyOS / Linux
- Python 3.11
- OpenCV
- MediaPipe Tasks API
- NumPy

The hand-landmarker model is downloaded locally and is intentionally excluded from Git because it is a binary asset.

## Performance baseline

The current local full-pipeline benchmark measured approximately 11 FPS, with the portal renderer as the main CPU-side bottleneck. Optimization work is ongoing.

## License

This project is an independent implementation. See the repository history for project provenance and development notes.
