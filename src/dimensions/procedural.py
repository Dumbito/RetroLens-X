from __future__ import annotations

import cv2
import numpy as np


class ProceduralDimension:
    """Animated procedural dimension with layered depth/parallax."""

    def __init__(self, work_scale: float = 0.5):
        self.work_scale = float(np.clip(work_scale, 0.25, 1.0))
        self._grid_shape: tuple[int, int] | None = None
        self._nx: np.ndarray | None = None
        self._ny: np.ndarray | None = None
        self._radial: np.ndarray | None = None
        self._angle: np.ndarray | None = None
        self._result: np.ndarray | None = None
        self._energy: np.ndarray | None = None
        self._local_r: np.ndarray | None = None
        self._local_a: np.ndarray | None = None
        self._core: np.ndarray | None = None
        self._edge_falloff: np.ndarray | None = None
        self._px: np.ndarray | None = None
        self._py: np.ndarray | None = None
        self._wave_a: np.ndarray | None = None
        self._wave_b: np.ndarray | None = None
        self._wave_c: np.ndarray | None = None
        self._wave_d: np.ndarray | None = None
        self._rings: np.ndarray | None = None
        self._filaments: np.ndarray | None = None

    def _ensure_grid(self, work_width: int, work_height: int) -> None:
        shape = (work_height, work_width)
        if self._grid_shape == shape:
            return

        y, x = np.mgrid[0:work_height, 0:work_width]
        nx = x.astype(np.float32) / max(work_width - 1, 1)
        ny = y.astype(np.float32) / max(work_height - 1, 1)
        dx = nx - 0.5
        dy = ny - 0.5

        self._nx = nx
        self._ny = ny
        self._radial = np.sqrt(dx * dx + dy * dy, dtype=np.float32)
        self._angle = np.arctan2(dy, dx, dtype=np.float32)
        self._result = np.empty((work_height, work_width, 3), dtype=np.uint8)
        self._energy = np.empty((work_height, work_width), dtype=np.float32)
        self._local_r = np.empty_like(nx)
        self._local_a = np.empty_like(nx)
        self._core = np.empty_like(nx)
        self._edge_falloff = np.empty_like(nx)
        self._px = np.empty_like(nx)
        self._py = np.empty_like(nx)
        self._wave_a = np.empty_like(nx)
        self._wave_b = np.empty_like(nx)
        self._wave_c = np.empty_like(nx)
        self._wave_d = np.empty_like(nx)
        self._rings = np.empty_like(nx)
        self._filaments = np.empty_like(nx)
        self._grid_shape = shape

    def render(
        self,
        width: int,
        height: int,
        timestamp_ms: int,
        view_x: float = 0.0,
        view_y: float = 0.0,
        view_angle: float = 0.0,
    ) -> np.ndarray:
        width = max(1, int(width))
        height = max(1, int(height))
        work_width = max(1, int(round(width * self.work_scale)))
        work_height = max(1, int(round(height * self.work_scale)))

        self._ensure_grid(work_width, work_height)
        nx = self._nx
        ny = self._ny
        radial = self._radial
        base_angle = self._angle
        result = self._result
        energy = self._energy
        local_r = self._local_r
        local_a = self._local_a
        core = self._core
        edge_falloff = self._edge_falloff
        px = self._px
        py = self._py
        wave_a = self._wave_a
        wave_b = self._wave_b
        wave_c = self._wave_c
        wave_d = self._wave_d
        rings = self._rings
        filaments = self._filaments
        assert all(x is not None for x in (
            nx, ny, radial, base_angle, result, energy, local_r, local_a,
            core, edge_falloff, px, py, wave_a, wave_b, wave_c, wave_d,
            rings, filaments,
        ))

        t = timestamp_ms * 0.001
        vx = float(np.clip(view_x, -1.0, 1.0))
        vy = float(np.clip(view_y, -1.0, 1.0))
        va = float(view_angle)

        # Reuse preallocated buffers to avoid per-frame temporary arrays.
        np.multiply(radial, 0.12, out=px)
        np.add(px, 0.055, out=px)
        np.multiply(px, vx, out=px)
        np.add(nx, px, out=px)

        np.multiply(radial, 0.10, out=py)
        np.add(py, 0.045, out=py)
        np.multiply(py, vy, out=py)
        np.add(ny, py, out=py)

        dx = local_r
        dy = local_a
        np.subtract(px, 0.5, out=dx)
        np.subtract(py, 0.5, out=dy)
        np.multiply(dx, dx, out=local_r)
        np.multiply(dy, dy, out=local_a)
        np.add(local_r, local_a, out=local_r)
        np.sqrt(local_r, out=local_r)
        np.arctan2(dy, dx, out=local_a)
        local_a += va * 0.22

        np.sin(px * 15.0 + t * 1.8, out=wave_a)
        np.sin(py * 11.0 - t * 1.35, out=wave_b)
        np.sin((px + py) * 21.0 + t * 2.4, out=wave_c)
        np.sin(local_a * 7.0 - local_r * 30.0 - t * 3.2, out=wave_d)

        np.multiply(wave_a, 0.22, out=energy)
        energy += wave_b * 0.18
        energy += wave_c * 0.20
        energy += wave_d * 0.40
        energy += 1.0
        energy *= 0.5

        # Animated depth rings.
        np.multiply(local_r, 8.5, out=rings)
        rings -= t * 0.75
        rings += local_a * 0.18
        rings %= 1.0
        rings -= 0.5
        np.multiply(rings, rings, out=rings)
        rings *= -1.0 / 0.045
        np.exp(rings, out=rings)

        np.multiply(local_r, -9.0, out=core)
        np.exp(core, out=core)

        np.multiply(local_r, -2.0, out=edge_falloff)
        edge_falloff += 1.15
        np.clip(edge_falloff, 0.0, 1.0, out=edge_falloff)

        energy *= 0.62
        energy += rings * 0.28
        energy += core * 0.35
        np.clip(energy, 0.0, 1.0, out=energy)
        energy *= edge_falloff

        # Fine moving filaments.
        np.multiply(local_a, 13.0, out=filaments)
        filaments += local_r * 42.0 - t * 4.0
        np.sin(filaments, out=filaments)
        filaments *= 0.5
        filaments += 0.5
        filaments *= core
        filaments *= 0.22
        energy += filaments
        np.clip(energy, 0.0, 1.0, out=energy)

        # BGR palette.
        np.multiply(energy, 205.0, out=wave_a)
        result[:, :, 0] = wave_a.astype(np.uint8)
        np.sqrt(energy, out=wave_b)
        wave_b *= 225.0
        result[:, :, 1] = wave_b.astype(np.uint8)
        np.multiply(energy, 255.0, out=wave_c)
        wave_c += core * 35.0
        np.clip(wave_c, 0.0, 255.0, out=wave_c)
        result[:, :, 2] = wave_c.astype(np.uint8)

        # Central hotspot.
        wave_d[:] = result[:, :, 1]
        wave_d += core * 22.0
        np.clip(wave_d, 0.0, 255.0, out=wave_d)
        result[:, :, 1] = wave_d.astype(np.uint8)

        if result.shape[:2] != (height, width):
            return cv2.resize(result, (width, height), interpolation=cv2.INTER_LINEAR)
        return result
