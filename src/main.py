import math
import time

import cv2
import numpy as np

from src.app.state_machine import PortalStateMachine
from src.camera.camera import Camera, CameraConfig
from src.diagnostics.profiler import PipelineProfiler
from src.dimensions import MultiverseDimension, ProceduralDimension
from src.dimensions.universe import DimensionalUniverse
from src.gestures.gesture_engine import GestureEngine
from src.pipeline import CallableEffect, EffectPipeline, FrameContext
from src.portal.portal_engine import PortalEngine
from src.portal.portal_physics import PortalPhysics
from src.portal.portal_renderer import PortalRenderer
from src.portal.spatial_fold import SpatialFoldEngine
from src.vfx.background_fx import BackgroundFX


CONNECTIONS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]


def draw_hand_rig(frame, hand):
    points = hand.pixel_landmarks
    for start, end in CONNECTIONS:
        cv2.line(frame, points[start], points[end], (255, 180, 0), 2)
    for point in points:
        cv2.circle(frame, point, 5, (0, 255, 255), -1)
    x1, y1, x2, y2 = hand.bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


def fingertip_distance(hands):
    if len(hands) < 2:
        return None
    first, second = hands[0].pixel_landmarks, hands[1].pixel_landmarks
    if len(first) <= 8 or len(second) <= 8:
        return None
    return math.hypot(second[8][0] - first[8][0], second[8][1] - first[8][1])


def hand_scale(hands):
    values = []
    for hand in hands[:2]:
        points = hand.pixel_landmarks
        if len(points) > 9:
            value = math.hypot(points[9][0] - points[0][0], points[9][1] - points[0][1])
            if value > 1.0:
                values.append(value)
    return sum(values) / len(values) if values else None


def hand_points(hands):
    return tuple(hand.pixel_landmarks[8] for hand in hands[:2] if len(hand.pixel_landmarks) > 8)


def blend_dimensions(base, overlay, weight):
    weight = float(max(0.0, min(1.0, weight)))
    if base.shape != overlay.shape:
        overlay = cv2.resize(overlay, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.clip(base.astype(np.float32) * (1.0 - weight) + overlay.astype(np.float32) * weight, 0, 255).astype(np.uint8)


def build_portal_pipeline(background, fold, renderer, camera_dimension, comic_dimension, universe):
    def environment(ctx):
        if ctx.phase != "OPEN" or ctx.intensity <= 0.005:
            return ctx
        state, physics = ctx.portal_state, ctx.physics_state
        ctx.frame = background.apply(ctx.frame, state.center, state.width, state.height, state.angle, ctx.intensity, motion=ctx.motion, waves=physics and ctx.metadata.get("waves", ()), hands=ctx.hand_points, flash=ctx.metadata.get("flash", 0.0), timestamp_ms=ctx.timestamp_ms)
        return ctx

    def fold_stage(ctx):
        if ctx.phase != "OPEN" or ctx.intensity <= 0.005:
            return ctx
        state = ctx.portal_state
        ctx.frame = fold.apply(ctx.frame, state.center, state.width, state.height, angle=state.angle, intensity=ctx.intensity, motion=ctx.motion, timestamp_ms=ctx.timestamp_ms)
        return ctx

    def dimensions(ctx):
        if ctx.phase != "OPEN" or ctx.intensity <= 0.005:
            return ctx
        state, physics = ctx.portal_state, ctx.physics_state
        camera_layer = camera_dimension.render(ctx.frame, state.width, state.height, state.center, ctx.timestamp_ms, view_x=ctx.view_x, view_y=ctx.view_y, view_angle=state.angle)
        comic = comic_dimension.render(state.width, state.height, ctx.timestamp_ms, view_x=ctx.view_x, view_y=ctx.view_y, view_angle=state.angle)
        cosmos = universe.render(state.width, state.height, ctx.timestamp_ms, view_x=ctx.view_x, view_y=ctx.view_y, intensity=ctx.intensity, turbulence=physics.turbulence)
        cosmic_weight = min(0.88, 0.72 + 0.10 * physics.stability + 0.05 * physics.pulse)
        comic_weight = 0.16 + 0.06 * physics.turbulence
        layered = blend_dimensions(camera_layer, comic, comic_weight)
        layered = blend_dimensions(layered, cosmos, cosmic_weight)
        gain = 0.82 + 0.18 * (1.0 - physics.pressure) + 0.08 * physics.pulse
        ctx.metadata["layered"] = np.clip(layered.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        return ctx

    def final_composite(ctx):
        if ctx.phase != "OPEN" or ctx.intensity <= 0.005:
            return ctx
        state, physics = ctx.portal_state, ctx.physics_state
        layered = ctx.metadata["layered"]
        if physics and physics.collapse > 0.01:
            layered = cv2.GaussianBlur(layered, (0, 0), 1.0 + 4.0 * physics.collapse)
        ctx.frame = renderer.render(ctx.frame, layered, state.center, state.width, state.height, state.angle, ctx.timestamp_ms, ctx.intensity, motion=ctx.motion)
        return ctx

    return EffectPipeline([CallableEffect("background", environment), CallableEffect("spatial_fold", fold_stage), CallableEffect("dimensions", dimensions), CallableEffect("portal_composite", final_composite)])


def draw_hud(frame, phase, distance, scale, profiler, show_profiler):
    status, hint = (("MULTIVERSE WINDOW", "JUNTA LOS INDICES PARA CERRAR") if phase == "OPEN" else ("FRAME ARMED", "SEPARA LOS INDICES PARA ABRIR") if phase == "ARMED" else ("FRAME READY", "JUNTA LAS PUNTAS DE LOS INDICES"))
    cv2.putText(frame, status, (20,35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.putText(frame, hint, (20,65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    if distance is not None and scale is not None:
        cv2.putText(frame, f"INDEX DIST: {distance:.0f} / HAND SCALE: {scale:.0f}", (20,92), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(frame, "H = HAND RIG | P = PROFILER | Q / ESC = EXIT", (20,118), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
    if show_profiler:
        stats = profiler.summary()
        cv2.putText(frame, f"FPS {stats['fps']:.1f} | FRAME {stats['frame_avg_ms']:.1f}ms | P95 {stats['frame_p95_ms']:.1f}ms", (20,145), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,255), 1)


def main():
    camera = Camera(CameraConfig(threaded=True))
    tracker = HandTracker("assets/models/hand_landmarker.task")
    gestures = GestureEngine()
    portal = PortalEngine(min_width=120, max_width=900, aspect_ratio=0.58, smoothing=0.24)
    physics = PortalPhysics()
    background = BackgroundFX(work_scale=0.55, margin=125)
    fold = SpatialFoldEngine(strength=0.34, falloff=1.35, margin=40, work_scale=0.70)
    renderer = PortalRenderer()
    camera_dimension = MultiverseDimension(work_scale=0.60)
    comic_dimension = ProceduralDimension(work_scale=0.46, max_work_width=360, max_work_height=260)
    universe = DimensionalUniverse(work_scale=0.42, max_width=420, max_height=300)
    portal_pipeline = build_portal_pipeline(background, fold, renderer, camera_dimension, comic_dimension, universe)
    lifecycle = PortalStateMachine()
    profiler = PipelineProfiler()
    show_hand_rig = False
    show_profiler = False
    portal_intensity = 0.0
    last_time = time.monotonic()
    previous_center = None

    try:
        while True:
            profiler.begin_frame()
            frame = camera.read()
            now = time.monotonic()
            delta_time = min(max(now - last_time, 0.0), 0.1)
            last_time = now
            timestamp_ms = time.monotonic_ns() // 1_000_000
            with profiler.measure("tracking"):
                hands = tracker.detect(frame, timestamp_ms)
                gestures.detect(hands, timestamp_ms)
            if show_hand_rig:
                for hand in hands:
                    draw_hand_rig(frame, hand)
            distance, scale = fingertip_distance(hands), hand_scale(hands)
            old_phase = lifecycle.phase
            lifecycle.update(hands, distance, scale, now, portal, physics, timestamp_ms)
            if lifecycle.phase == "OPEN" and old_phase != "OPEN":
                previous_center = portal.state.center
            if lifecycle.phase == "OPEN" and distance is not None and scale is not None:
                state = portal.update(hands, timestamp_ms)
                open_reference = max(scale * lifecycle.open_ratio, 1.0)
                opening_ratio = max(0.0, min(1.35, distance / open_reference))
                physics_state = physics.update(hands, state.center, state.width, state.height, timestamp_ms, opening_ratio=opening_ratio)
                motion = physics_state.speed
                if previous_center is not None and delta_time > 0.0:
                    px_s = math.hypot(state.center[0] - previous_center[0], state.center[1] - previous_center[1]) / delta_time
                    motion = max(motion, min(1.0, px_s / 900.0))
                previous_center = state.center
            else:
                opening_ratio, physics_state, motion = 0.0, physics.state, physics.state.speed
                if lifecycle.phase != "OPEN":
                    previous_center = None
            target_intensity = 1.0 if lifecycle.phase == "OPEN" else 0.0
            portal_intensity += (target_intensity - portal_intensity) * (1.0 - math.exp(-delta_time * 12.0))
            ctx = FrameContext(frame=frame, timestamp_ms=timestamp_ms, delta_time=delta_time, hands=tuple(hands), hand_points=hand_points(hands), phase=lifecycle.phase, distance=distance, hand_scale=scale, opening_ratio=opening_ratio, intensity=portal_intensity, motion=motion)
            if lifecycle.phase == "OPEN":
                state = portal.state
                frame_h, frame_w = frame.shape[:2]
                ctx.view_x = max(-1.0, min(1.0, ((state.center[0] / max(frame_w - 1, 1)) - 0.5) * 2.0))
                ctx.view_y = max(-1.0, min(1.0, ((state.center[1] / max(frame_h - 1, 1)) - 0.5) * 2.0))
                ctx.portal_state, ctx.physics_state = state, physics_state
                ctx.metadata["waves"], ctx.metadata["flash"] = physics.waves, physics.consume_flash()
            with profiler.measure("pipeline"):
                ctx = portal_pipeline.run(ctx)
            frame = ctx.frame
            draw_hud(frame, lifecycle.phase, distance, scale, profiler, show_profiler)
            cv2.imshow("RetroLens-X", frame)
            profiler.end_frame()
            key = cv2.waitKey(1) & 0xFF
            if key == ord("h"):
                show_hand_rig = not show_hand_rig
            elif key == ord("p"):
                show_profiler = not show_profiler
            elif key in (ord("q"), 27):
                break
    finally:
        tracker.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
