"""Automatic board orientation from the ring of printed sector numbers.

Board detection recovers the full geometry but the red/green pattern repeats every 36°, so it
cannot tell which red sector is the 20. Once the player has confirmed the 20, the number ring (the
band between the double ring and the outer edge of the board) is sampled in board coordinates and
kept as a template. Board coordinates remove perspective, so the template matches the same board
seen from any camera position: a later detection is rotated to whichever of the 20 sector shifts
makes its number ring correlate best with the template.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import geometry as g
from .detector import BoardState

RING_RADII = np.arange(g.R_DOUBLE_OUT + 5.0, g.R_BOARD - 3.0, 1.0)  # mm
ANGLES = np.arange(0.0, 360.0, 0.5)  # degrees clockwise from the 20
MIN_SCORE = 0.30  # correlation of the best rotation
MIN_MARGIN = 0.10  # lead of the best rotation over the second best


@dataclass
class OrientationMatch:
    sectors: int  # rotate the detected board by this many sectors (see BoardState.rotated)
    score: float
    margin: float

    @property
    def confident(self) -> bool:
        return self.score >= MIN_SCORE and self.margin >= MIN_MARGIN


def _gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame


def number_ring(frame: np.ndarray, board: BoardState) -> np.ndarray:
    """Number ring as a (radii × angles) strip, normalised against lighting changes."""
    rr, aa = np.meshgrid(RING_RADII, ANGLES, indexing="ij")
    pts = board.to_image(g.model_points(rr, aa)).reshape(len(RING_RADII), len(ANGLES), 2).astype(np.float32)
    strip = cv2.remap(_gray(frame), pts[..., 0], pts[..., 1], cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float32)
    strip -= strip.mean(axis=0, keepdims=True)  # lighting varies around the ring, not across it
    return strip / max(float(strip.std()), 1e-6)


def match_orientation(frame: np.ndarray, board: BoardState, template: np.ndarray) -> OrientationMatch:
    """Best sector rotation of `board` according to the number-ring template."""
    gray = _gray(frame)
    scores = np.array([float(np.mean(number_ring(gray, board.rotated(k)) * template)) for k in range(20)])
    order = np.argsort(scores)[::-1]
    return OrientationMatch(int(order[0]), float(scores[order[0]]), float(scores[order[0]] - scores[order[1]]))


def load_template(path: Path) -> np.ndarray | None:
    try:
        template = np.load(path)
    except (OSError, ValueError):
        return None
    return template if template.shape == (len(RING_RADII), len(ANGLES)) else None
