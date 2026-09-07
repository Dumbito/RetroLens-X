import math

import numpy as np

from src.dimensions.procedural import ProceduralDimension
from src.gestures.gesture_engine import GestureEngine, GestureType
from src.portal.portal_engine import PortalEngine
from src.portal.portal_renderer import PortalRenderer


class _Landmark:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class _Hand:
    def __init__(self, points, handedness="Unknown"):
        self.landmarks = [_Landmark(*point) for point in points]
        self.pixel_landmarks = [(int(x * 1000), int(y * 1000)) for x, y, *_ in points]
        self.handedness = handedness


def _open_hand(offset_x=0.0):
    points = [
        (0.35 + offset_x, 0.75), (0.28 + offset_x, 0.68), (0.22 + offset_x, 0.58),
        (0.18 + offset_x, 0.48), (0.15 + offset_x, 0.40), (0.42 + offset_x, 0.55),
        (0.42 + offset_x, 0.43), (0.42 + offset_x, 0.31), (0.42 + offset_x, 0.20),
        (0.50 + offset_x, 0.55), (0.50 + offset_x, 0.41), (0.50 + offset_x, 0.29),
        (0.50 + offset_x, 0.18), (0.58 + offset_x, 0.57), (0.59 + offset_x, 0.44),
        (0.60 + offset_x, 0.32), (0.61 + offset_x, 0.22), (0.65 + offset_x, 0.61),
        (0.68 + offset_x, 0.50), (0.70 + offset_x, 0.39), (0.72 + offset_x, 0.30),
    ]
    return _Hand(points)


def test_gesture_engine_accepts_empty_frame():
    engine = GestureEngine()
    assert engine.detect([], 1000) == []


def test_portal_engine_geometry_is_bounded():
    hands = [_open_hand(-0.12), _open_hand(0.12)]
    engine = PortalEngine(min_width=140, max_width=900)
    state = engine.update(hands, 1000)
    assert state.active
    assert 140 <= state.width <= 900
    assert state.height == int(state.width * 0.72)
    assert math.isfinite(state.angle)


def test_procedural_dimension_shape_and_range():
    image = ProceduralDimension(work_scale=0.5).render(320, 180, 1000)
    assert image.shape == (180, 320, 3)
    assert image.dtype == np.uint8


def test_portal_renderer_preserves_frame_shape():
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    dimension = ProceduralDimension(work_scale=0.5).render(140, 100, 1000)
    renderer = PortalRenderer()
    result = renderer.render(frame, dimension, (160, 90), 140, 100, 25.0, 1000)
    assert result.shape == frame.shape
    assert result.dtype == np.uint8


def test_portal_renderer_accepts_degree_angles():
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    dimension = np.full((100, 140, 3), 128, dtype=np.uint8)
    renderer = PortalRenderer()
    result = renderer.render(frame, dimension, (160, 90), 140, 100, 90.0, 1000)
    assert np.any(result != frame)
