"""Standard dartboard geometry (regulation measurements), in millimetres.

Model coordinates: origin at the centre of the bull, x to the right and y downwards (as in
images). Angles are in degrees, measured clockwise from the top: sector 20 is centred at 0°.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Sector numbers clockwise from the top.
SECTOR_ORDER = (20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5)
SECTOR_ANGLE = 18.0

R_BULLSEYE = 6.35
R_BULL = 15.9
R_TREBLE_IN = 99.0
R_TREBLE_OUT = 107.0
R_DOUBLE_IN = 162.0
R_DOUBLE_OUT = 170.0
R_NUMBERS = 195.0  # radius of the printed sector numbers
R_BOARD = 225.0  # outer edge of the black surround (approximate)

RING_RADII = (R_BULLSEYE, R_BULL, R_TREBLE_IN, R_TREBLE_OUT, R_DOUBLE_IN, R_DOUBLE_OUT)


@dataclass(frozen=True)
class Hit:
    number: int  # 1..20, 25 for the bull, 0 for a miss
    multiplier: int  # 0 miss, 1 single, 2 double (or bullseye), 3 treble

    @property
    def score(self) -> int:
        return self.number * self.multiplier

    @property
    def label(self) -> str:
        if self.multiplier == 0:
            return "MISS"
        if self.number == 25:
            return "BULL" if self.multiplier == 2 else "25"
        return "SDT"[self.multiplier - 1] + str(self.number)


def polar(x: float, y: float) -> tuple[float, float]:
    """Model coordinates -> (radius in mm, angle in degrees clockwise from the top)."""
    return math.hypot(x, y), math.degrees(math.atan2(x, -y)) % 360.0


def sector_index(theta: float) -> int:
    """Index in SECTOR_ORDER of the sector containing the angle `theta`."""
    return int(((theta + SECTOR_ANGLE / 2) % 360.0) // SECTOR_ANGLE) % 20


def score_model_point(x: float, y: float, rings=RING_RADII) -> Hit:
    """Score of a model point. `rings` are the radii of the six wires (bullseye, bull, treble
    inner/outer, double inner/outer); nominal measurements by default."""
    r_bullseye, r_bull, t_in, t_out, d_in, d_out = rings
    r, theta = polar(x, y)
    if r <= r_bullseye:
        return Hit(25, 2)
    if r <= r_bull:
        return Hit(25, 1)
    if r > d_out:
        return Hit(0, 0)
    number = SECTOR_ORDER[sector_index(theta)]
    if t_in <= r <= t_out:
        return Hit(number, 3)
    if r >= d_in:
        return Hit(number, 2)
    return Hit(number, 1)


def hit_region(x: float, y: float, rings=RING_RADII) -> tuple[float, float, float, float] | None:
    """Segment containing a model point: (inner radius, outer radius, start angle, end angle)."""
    r_bullseye, r_bull, t_in, t_out, d_in, d_out = rings
    r, theta = polar(x, y)
    if r > d_out:
        return None
    if r <= r_bullseye:
        return 0.0, r_bullseye, 0.0, 360.0
    if r <= r_bull:
        return r_bullseye, r_bull, 0.0, 360.0
    bounds = (r_bull, t_in, t_out, d_in, d_out)
    for r_in, r_out in zip(bounds, bounds[1:]):
        if r_in <= r <= r_out:
            break
    start = sector_index(theta) * SECTOR_ANGLE - SECTOR_ANGLE / 2
    return r_in, r_out, start, start + SECTOR_ANGLE


def model_points(r, theta_deg) -> np.ndarray:
    """(N, 2) model points from radii and angles (scalars or arrays, broadcast together)."""
    r, t = np.broadcast_arrays(np.asarray(r, float), np.deg2rad(np.asarray(theta_deg, float)))
    return np.stack([r * np.sin(t), -r * np.cos(t)], axis=-1).reshape(-1, 2)


def rotation(deg: float) -> np.ndarray:
    """Homography rotating the model clockwise by `deg` degrees."""
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def apply_homography(H: np.ndarray, pts) -> np.ndarray:
    """Apply a 3x3 homography to (N, 2) points."""
    pts = np.asarray(pts, float).reshape(-1, 2)
    q = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return q[:, :2] / q[:, 2:3]
