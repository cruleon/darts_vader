"""Board-centred image views used by the dart tip model.

* ``rect``: fronto-parallel rectified board (``RECT_PPM`` px/mm, ``RECT_SIZE`` square).
* ``camera``: square crop of the original frame around the board; darts keep the appearance
  they have from the camera.
* ``camera_canon``: the ``camera`` crop rotated so that the side of the board closest to the
  camera is at the bottom. The viewing direction becomes the same for every camera position and
  the model only has to learn how steep the view is.

Every view comes with a ``px_to_mm`` homography from view pixels to board millimetres.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..board import geometry as g
from ..board.detector import BoardState

VIEWS = ("rect", "camera", "camera_canon")

RECT_PPM = 2.0
RECT_SIZE = int(round(2 * g.R_BOARD * RECT_PPM))
MM_TO_RECT_PX = np.array([[RECT_PPM, 0, RECT_SIZE / 2], [0, RECT_PPM, RECT_SIZE / 2], [0, 0, 1.0]])
RECT_PX_TO_MM = np.linalg.inv(MM_TO_RECT_PX)

CROP_SIZE = 800


def rectify(frame: np.ndarray, state: BoardState) -> np.ndarray:
    """Fronto-parallel view of the board."""
    return cv2.warpPerspective(frame, MM_TO_RECT_PX @ state.H, (RECT_SIZE, RECT_SIZE), flags=cv2.INTER_LINEAR)


def board_crop(frame: np.ndarray, state: BoardState, size: int = CROP_SIZE, margin: float = 1.06):
    """Square crop of the original image around the board (surround included), resized to
    ``size × size``. Areas outside the image are black.
    Returns ``(crop, A)`` where A is the 3x3 homography image pixels -> crop pixels."""
    pts = state.to_image(g.model_points(g.R_BOARD, np.arange(0, 360, 5)))
    lo, hi = pts.min(0), pts.max(0)
    center = (lo + hi) / 2
    half = (hi - lo).max() / 2 * margin
    x0, y0 = np.floor(center - half).astype(int)
    side = max(1, int(np.ceil(2 * half)))
    h, w = frame.shape[:2]
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(w, x0 + side), min(h, y0 + side)
    canvas = np.zeros((side, side, 3), np.uint8)
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = frame[sy0:sy1, sx0:sx1]
    crop = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
    s = size / side
    # pixel-centre convention of cv2.resize: x_crop = s * (x - x0 + 0.5) - 0.5
    A = np.array([[s, 0, s * (0.5 - x0) - 0.5], [0, s, s * (0.5 - y0) - 0.5], [0, 0, 1.0]])
    return crop, A


def view_geometry(px_to_mm: np.ndarray) -> tuple[np.ndarray, float]:
    """From a pixel -> mm homography: the unit direction (in pixels) from the board centre towards
    the side closest to the camera (the one that looks largest), and the view tilt in degrees."""
    to_px = np.linalg.inv(px_to_mm)
    angles = np.arange(0.0, 360.0, 10.0)
    p = g.apply_homography(to_px, g.model_points(g.R_DOUBLE_OUT, angles))
    q = g.apply_homography(to_px, g.model_points(g.R_DOUBLE_OUT + 5.0, angles))
    c = g.apply_homography(to_px, [(0.0, 0.0)])[0]
    scale = np.linalg.norm(q - p, axis=1)
    u = (p - c) / np.maximum(np.linalg.norm(p - c, axis=1, keepdims=True), 1e-9)
    d = ((scale - scale.mean())[:, None] * u).sum(axis=0)
    (_, _), (a1, a2), _ = cv2.fitEllipse(p.astype(np.float32))
    tilt = float(np.degrees(np.arccos(np.clip(min(a1, a2) / max(a1, a2), 0.0, 1.0))))
    n = float(np.linalg.norm(d))
    return (d / n if n > 1e-9 else np.array([0.0, 1.0])), tilt


def canonical_rotation(px_to_mm: np.ndarray, size: int) -> np.ndarray:
    """3x3 rotation about the centre of a ``size × size`` crop that brings the side of the board
    closest to the camera to the bottom."""
    d, _ = view_geometry(px_to_mm)
    phi = float(np.degrees(np.arctan2(d[1], d[0])))
    return np.vstack([cv2.getRotationMatrix2D((size / 2, size / 2), phi - 90.0, 1.0), [0.0, 0.0, 1.0]])


def canonical_crop(frame: np.ndarray, state: BoardState, size: int = CROP_SIZE):
    """``board_crop`` rotated by ``canonical_rotation``. Returns ``(crop, px_to_mm)``."""
    crop, A = board_crop(frame, state, size)
    px_to_mm = state.H @ np.linalg.inv(A)
    R = canonical_rotation(px_to_mm, size)
    return cv2.warpAffine(crop, R[:2], (size, size), flags=cv2.INTER_LINEAR), px_to_mm @ np.linalg.inv(R)


def render_view(frame: np.ndarray, state: BoardState, view: str = "camera", size: int = CROP_SIZE):
    """The board in the requested view. Returns ``(image, px_to_mm)``."""
    if view == "rect":
        return rectify(frame, state), RECT_PX_TO_MM
    if view == "camera":
        crop, A = board_crop(frame, state, size)
        return crop, state.H @ np.linalg.inv(A)
    if view == "camera_canon":
        return canonical_crop(frame, state, size)
    raise ValueError(f"unknown view {view!r}, expected one of {VIEWS}")
