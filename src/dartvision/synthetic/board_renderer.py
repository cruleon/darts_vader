"""Rendering del bersaglio + marker ArUco sul "piano verita'" (vista
dall'alto, coerente con la geometria di ``config/board_config.yaml``).

Condiviso tra ``scripts/generate_synthetic_video.py`` (che poi proietta
questo piano verso una vista camera plausibile, per validare
calibrazione e frame-diff) e ``dartvision.synthetic.pose_dataset`` (che
lavora direttamente su questo piano, dominio in cui opera anche
``MLTipDetector`` a runtime).
"""

from __future__ import annotations

import cv2
import numpy as np

from dartvision.calibration.aruco_detector import resolve_dictionary
from dartvision.config import AppConfig

COLOR_BLACK = (35, 35, 35)
COLOR_CREAM = (215, 215, 230)
COLOR_RED = (40, 40, 195)
COLOR_GREEN = (60, 130, 60)


def render_plane_image(config: AppConfig) -> np.ndarray:
    """Disegna bersaglio + marker sul piano "verita'" (vista dall'alto)."""
    plane = config.rectified_plane
    board = config.board
    size = plane.size_px
    img = np.full((size, size, 3), 60, dtype=np.uint8)

    center = tuple(round(v) for v in plane.mm_to_px(0.0, 0.0))
    mm_per_px = plane.mm_per_px

    def r_px(r_mm: float) -> int:
        return max(1, round(r_mm / mm_per_px))

    sector_width = 360.0 / board.sector_count
    ring_bands = [
        (board.double_outer_radius_mm, True),
        (board.double_inner_radius_mm, False),
        (board.triple_outer_radius_mm, True),
        (board.triple_inner_radius_mm, False),
    ]

    for i in range(board.sector_count):
        center_angle = board.sector0_offset_deg + i * sector_width
        start_cv = (center_angle - sector_width / 2) - 90
        end_cv = (center_angle + sector_width / 2) - 90
        parity = i % 2
        for radius_mm, is_multiplier_ring in ring_bands:
            if is_multiplier_ring:
                color = COLOR_RED if parity == 0 else COLOR_GREEN
            else:
                color = COLOR_BLACK if parity == 0 else COLOR_CREAM
            cv2.ellipse(
                img, center, (r_px(radius_mm), r_px(radius_mm)), 0,
                start_cv, end_cv, color, -1,
            )

    cv2.circle(img, center, r_px(board.outer_bull_radius_mm), COLOR_GREEN, -1)
    cv2.circle(img, center, r_px(board.inner_bull_radius_mm), COLOR_RED, -1)
    return img


def _paste_clipped(dst: np.ndarray, patch: np.ndarray, top_left_x: int, top_left_y: int) -> None:
    """Incolla ``patch`` su ``dst`` ritagliando ai bordi se necessario."""
    h, w = patch.shape[:2]
    dst_h, dst_w = dst.shape[:2]

    x0, y0 = max(0, top_left_x), max(0, top_left_y)
    x1, y1 = min(dst_w, top_left_x + w), min(dst_h, top_left_y + h)
    if x0 >= x1 or y0 >= y1:
        return

    src_x0, src_y0 = x0 - top_left_x, y0 - top_left_y
    dst[y0:y1, x0:x1] = patch[src_y0 : src_y0 + (y1 - y0), src_x0 : src_x0 + (x1 - x0)]


def paste_markers(plane_img: np.ndarray, config: AppConfig) -> None:
    dictionary = resolve_dictionary(config.aruco.dictionary)
    mm_per_px = config.rectified_plane.mm_per_px
    side_px = max(8, round(config.aruco.marker_length_mm / mm_per_px))
    quiet = max(2, side_px // 6)

    for marker in config.aruco.markers:
        marker_img = cv2.aruco.generateImageMarker(dictionary, marker.id, side_px)
        padded = cv2.copyMakeBorder(
            marker_img, quiet, quiet, quiet, quiet, cv2.BORDER_CONSTANT, value=255
        )
        padded_bgr = cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)

        cx, cy = config.rectified_plane.mm_to_px(marker.x_mm, marker.y_mm)
        full_side = padded.shape[0]
        top_left_x = round(cx - full_side / 2)
        top_left_y = round(cy - full_side / 2)
        _paste_clipped(plane_img, padded_bgr, top_left_x, top_left_y)
