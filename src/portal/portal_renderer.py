from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass
class PortalVisualConfig:
    edge_thickness: int = 4
    glow_sigma: float = 18.0
    ring_count: int = 3
    particle_count: int = 140
    particle_seed: int = 42


class PortalRenderer:
    def __init__(self, config: PortalVisualConfig | None = None) -> None:
        self.config = config or PortalVisualConfig()

        rng = np.random.default_rng(self.config.particle_seed)
        count = self.config.particle_count

        self.particle_phase = rng.uniform(0.0, 2.0 * math.pi, count)
        self.particle_radius = rng.uniform(0.82, 1.18, count)
        self.particle_angle = rng.uniform(0.0, 2.0 * math.pi, count)
        self.particle_speed = rng.uniform(0.35, 1.25, count)
        self.particle_size = rng.uniform(1.0, 3.0, count)

    def render(
        self,
        frame: np.ndarray,
        dimension: np.ndarray,
        center: tuple[int, int],
        width: int,
        height: int,
        angle: float,
        timestamp_ms: int,
    ) -> np.ndarray:
        if width <= 0 or height <= 0:
            return frame

        h, w = frame.shape[:2]
        t = timestamp_ms / 1000.0

        mask = self._organic_mask(
            (h, w),
            center,
            width,
            height,
            angle,
            t,
        )

        content = cv2.resize(
            dimension,
            (w, h),
            interpolation=cv2.INTER_LINEAR,
        )

        alpha = mask.astype(np.float32) / 255.0
        alpha = alpha[..., None]

        output = (
            frame.astype(np.float32) * (1.0 - alpha)
            + content.astype(np.float32) * alpha
        ).astype(np.uint8)

        energy = np.zeros_like(frame)
        self._draw_rings(
            energy,
            center,
            width,
            height,
            angle,
            t,
        )

        self._draw_particles(
            energy,
            center,
            width,
            height,
            angle,
            t,
        )

        edges = cv2.Canny(mask, 70, 180)
        glow = cv2.GaussianBlur(
            edges,
            (0, 0),
            self.config.glow_sigma,
        )

        glow_layer = np.zeros_like(frame)
        glow_layer[:, :, 0] = glow
        glow_layer[:, :, 1] = np.clip(glow * 0.75, 0, 255).astype(np.uint8)
        glow_layer[:, :, 2] = np.clip(glow * 0.35, 0, 255).astype(np.uint8)

        output = cv2.addWeighted(
            output,
            1.0,
            glow_layer,
            0.55,
            0.0,
        )

        output = cv2.addWeighted(
            output,
            1.0,
            energy,
            0.82,
            0.0,
        )

        rim = np.zeros_like(frame)
        self._draw_rim(
            rim,
            center,
            width,
            height,
            angle,
            t,
        )

        output = cv2.addWeighted(
            output,
            1.0,
            rim,
            0.95,
            0.0,
        )

        return output

    def _organic_mask(
        self,
        shape: tuple[int, int],
        center: tuple[int, int],
        width: int,
        height: int,
        angle: float,
        t: float,
    ) -> np.ndarray:
        h, w = shape
        cx, cy = center

        yy, xx = np.mgrid[0:h, 0:w]

        dx = xx - cx
        dy = yy - cy

        ca = math.cos(angle)
        sa = math.sin(angle)

        xr = dx * ca + dy * sa
        yr = -dx * sa + dy * ca

        rx = max(width * 0.5, 1.0)
        ry = max(height * 0.5, 1.0)

        theta = np.arctan2(yr, xr)

        wave = (
            1.0
            + 0.075 * np.sin(theta * 5.0 + t * 3.2)
            + 0.045 * np.sin(theta * 9.0 - t * 2.1)
            + 0.025 * np.sin(theta * 14.0 + t * 4.7)
        )

        radius = np.sqrt(
            (xr / rx) ** 2
            + (yr / ry) ** 2
        )

        boundary = radius / wave

        softness = 0.035

        alpha = np.clip(
            (1.0 - boundary) / softness + 0.5,
            0.0,
            1.0,
        )

        return (alpha * 255.0).astype(np.uint8)

    def _ellipse_points(
        self,
        center: tuple[int, int],
        width: int,
        height: int,
        angle: float,
        t: float,
        phase: float = 0.0,
        scale: float = 1.0,
        count: int = 180,
    ) -> np.ndarray:
        cx, cy = center

        theta = np.linspace(
            0.0,
            2.0 * math.pi,
            count,
            endpoint=False,
        )

        wave = (
            1.0
            + 0.06 * np.sin(theta * 5.0 + t * 3.0 + phase)
            + 0.035 * np.sin(theta * 11.0 - t * 2.2 + phase * 1.7)
        )

        x = (
            width
            * 0.5
            * scale
            * wave
            * np.cos(theta)
        )

        y = (
            height
            * 0.5
            * scale
            * wave
            * np.sin(theta)
        )

        ca = math.cos(angle)
        sa = math.sin(angle)

        xr = x * ca - y * sa
        yr = x * sa + y * ca

        return np.column_stack(
            (xr + cx, yr + cy)
        ).astype(np.int32)

    def _draw_rings(
        self,
        image: np.ndarray,
        center: tuple[int, int],
        width: int,
        height: int,
        angle: float,
        t: float,
    ) -> None:
        for i in range(self.config.ring_count):
            pulse = (
                1.0
                + 0.035
                * math.sin(
                    t * (2.0 + i * 0.55) + i
                )
            )

            scale = (
                1.0
                + (i - 1) * 0.045
            ) * pulse

            pts = self._ellipse_points(
                center,
                width,
                height,
                angle,
                t,
                i * 1.9,
                scale,
                160,
            )

            layer = np.zeros_like(image)

            cv2.polylines(
                layer,
                [pts],
                True,
                (90, 170, 245),
                2 if i == 1 else 1,
                cv2.LINE_AA,
            )

            blur = cv2.GaussianBlur(
                layer,
                (0, 0),
                4.0 + i * 2.5,
            )

            image[:] = cv2.addWeighted(
                image,
                1.0,
                blur,
                0.45,
                0.0,
            )

            image[:] = cv2.addWeighted(
                image,
                1.0,
                layer,
                0.75,
                0.0,
            )

    def _draw_rim(
        self,
        image: np.ndarray,
        center: tuple[int, int],
        width: int,
        height: int,
        angle: float,
        t: float,
    ) -> None:
        pts = self._ellipse_points(
            center,
            width,
            height,
            angle,
            t,
            0.0,
            1.0,
            220,
        )

        cv2.polylines(
            image,
            [pts],
            True,
            (180, 225, 255),
            self.config.edge_thickness,
            cv2.LINE_AA,
        )

    def _draw_particles(
        self,
        image: np.ndarray,
        center: tuple[int, int],
        width: int,
        height: int,
        angle: float,
        t: float,
    ) -> None:
        cx, cy = center

        ca = math.cos(angle)
        sa = math.sin(angle)

        for i in range(self.config.particle_count):
            a = (
                self.particle_angle[i]
                + t
                * self.particle_speed[i]
                * 0.22
            )

            pulse = (
                1.0
                + 0.14
                * math.sin(
                    t * 3.0
                    + self.particle_phase[i]
                )
            )

            rx = (
                width
                * 0.5
                * self.particle_radius[i]
                * pulse
            )

            ry = (
                height
                * 0.5
                * self.particle_radius[i]
                * pulse
            )

            local_x = math.cos(a) * rx
            local_y = math.sin(a) * ry

            x = int(
                cx
                + local_x * ca
                - local_y * sa
            )

            y = int(
                cy
                + local_x * sa
                + local_y * ca
            )

            if (
                0 <= x < image.shape[1]
                and 0 <= y < image.shape[0]
            ):
                size = max(
                    1,
                    int(self.particle_size[i]),
                )

                brightness = int(
                    130
                    + 90
                    * (
                        0.5
                        + 0.5
                        * math.sin(
                            t * 5.0
                            + self.particle_phase[i]
                        )
                    )
                )

                cv2.circle(
                    image,
                    (x, y),
                    size,
                    (
                        brightness // 2,
                        brightness,
                        255,
                    ),
                    -1,
                )
