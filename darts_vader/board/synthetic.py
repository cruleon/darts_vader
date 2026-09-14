"""Synthetic scenes with ground truth: a dartboard seen by a virtual pinhole camera.

Used to measure detector accuracy from arbitrary viewing angles and to try the tools without a
real board.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from . import geometry as g

# BGR colours
BLACK = (28, 28, 28)
CREAM = (185, 222, 236)
RED = (45, 35, 200)
GREEN = (50, 140, 25)
WIRE = (175, 175, 175)
SURROUND = (18, 18, 18)


def render_board_texture(ppm: float = 3.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fronto-parallel board texture. Returns (bgr, alpha, texture -> model homography)."""
    size = int(round(2 * g.R_BOARD * ppm))
    c = size / 2
    yy, xx = np.mgrid[0:size, 0:size]
    x, y = (xx + 0.5 - c) / ppm, (yy + 0.5 - c) / ppm
    r = np.hypot(x, y)
    theta = np.degrees(np.arctan2(x, -y)) % 360
    even = (((theta + 9) // 18).astype(int) % 20) % 2 == 0

    tex = np.empty((size, size, 3), np.uint8)
    tex[:] = SURROUND
    inner = r <= g.R_DOUBLE_OUT
    tex[inner & even] = BLACK
    tex[inner & ~even] = CREAM
    rings = ((r >= g.R_TREBLE_IN) & (r <= g.R_TREBLE_OUT)) | ((r >= g.R_DOUBLE_IN) & inner)
    tex[rings & even] = RED
    tex[rings & ~even] = GREEN
    tex[r <= g.R_BULL] = GREEN
    tex[r <= g.R_BULLSEYE] = RED

    shift = 4
    fp = lambda v: int(round(v * (1 << shift)))  # noqa: E731  (fixed-point coordinates)
    for R in g.RING_RADII:
        cv2.circle(tex, (fp(c), fp(c)), fp(R * ppm), WIRE, 2, cv2.LINE_AA, shift)
    for k in range(20):
        (ax, ay), (bx, by) = g.model_points([g.R_BULL, g.R_DOUBLE_OUT], 9 + 18 * k) * ppm + c
        cv2.line(tex, (fp(ax), fp(ay)), (fp(bx), fp(by)), WIRE, 2, cv2.LINE_AA, shift)
    for k, number in enumerate(g.SECTOR_ORDER):
        px, py = g.model_points(g.R_NUMBERS, 18 * k)[0] * ppm + c
        text, scale, th = str(number), 0.12 * ppm * 4, max(1, int(ppm))
        (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
        cv2.putText(tex, text, (int(px - tw / 2), int(py + tht / 2)), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (230, 230, 230), th, cv2.LINE_AA)

    alpha = (r <= g.R_BOARD).astype(np.float32)
    tex_to_model = np.array([[1 / ppm, 0, -c / ppm], [0, 1 / ppm, -c / ppm], [0, 0, 1.0]])
    return tex, alpha, tex_to_model


def _rot(axis: str, deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class SyntheticScene:
    def __init__(self, width=1280, height=720, focal=1000.0, seed=0):
        self.size = (width, height)
        self.K = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.0]])
        self.rng = np.random.default_rng(seed)
        self.tex, self.alpha, self.tex_to_model = render_board_texture()
        self.background = self._make_background()

    def _make_background(self) -> np.ndarray:
        w, h = self.size
        low = self.rng.uniform(70, 150, (6, 10, 3)).astype(np.float32)
        low[..., 0] *= 0.8  # slightly warm wall
        bg = cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC)
        bg += self.rng.normal(0, 6, (h, w, 3)).astype(np.float32)
        # distractors: a red and a green poster
        cv2.rectangle(bg, (int(0.03 * w), int(0.05 * h)), (int(0.14 * w), int(0.35 * h)), (40, 40, 180), -1)
        cv2.rectangle(bg, (int(0.86 * w), int(0.60 * h)), (int(0.97 * w), int(0.92 * h)), (60, 150, 40), -1)
        return np.clip(bg, 0, 255)

    def model_to_image(self, yaw, pitch, roll, distance, offset=(0.0, 0.0)) -> np.ndarray:
        R = _rot("z", roll) @ _rot("x", pitch) @ _rot("y", yaw)
        t = np.array([offset[0], offset[1], distance])
        return self.K @ np.column_stack([R[:, 0], R[:, 1], t])

    def render(self, yaw=0.0, pitch=0.0, roll=0.0, distance=750.0, offset=(0.0, 0.0),
               noise=3.0, gain=1.0) -> tuple[np.ndarray, np.ndarray]:
        """Returns (BGR frame, ground-truth image -> model homography)."""
        M = self.model_to_image(yaw, pitch, roll, distance, offset)
        W = M @ self.tex_to_model
        board = cv2.warpPerspective(self.tex, W, self.size, flags=cv2.INTER_LINEAR).astype(np.float32)
        a = cv2.warpPerspective(self.alpha, W, self.size, flags=cv2.INTER_LINEAR)[..., None]
        frame = self.background * (1 - a) + board * a
        # uneven lighting
        w, h = self.size
        xs = np.linspace(-1, 1, w)[None, :, None]
        frame *= gain * (1.0 - 0.18 * xs)
        frame += self.rng.normal(0, noise, frame.shape)
        frame = cv2.GaussianBlur(np.clip(frame, 0, 255).astype(np.uint8), (3, 3), 0)
        return frame, np.linalg.inv(M)


class SyntheticVideo:
    """Video source with a camera moving around the board (``cv2.VideoCapture``-like API)."""

    def __init__(self, fps=30.0, **scene_kwargs):
        self.scene = SyntheticScene(**scene_kwargs)
        self.fps = fps
        self.t = 0.0
        self.last_truth: np.ndarray | None = None

    def isOpened(self) -> bool:  # noqa: N802  (mirrors cv2.VideoCapture)
        return True

    def pose(self, t: float) -> dict:
        return dict(
            yaw=50 * math.sin(0.35 * t),
            pitch=30 * math.sin(0.23 * t + 1.0),
            roll=8 * math.sin(0.17 * t),
            distance=780 + 180 * math.sin(0.29 * t),
            offset=(60 * math.sin(0.41 * t), 40 * math.cos(0.31 * t)),
        )

    def read(self):
        frame, self.last_truth = self.scene.render(**self.pose(self.t))
        self.t += 1.0 / self.fps
        return True, frame

    def release(self) -> None:
        pass
