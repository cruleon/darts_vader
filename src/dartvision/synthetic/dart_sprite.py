"""Sprite di una freccetta stilizzata (punta + fusto + alette) sul piano
raddrizzato, per generare dataset di training realistici.

A differenza del semplice cerchio pieno usato da
``scripts/generate_synthetic_video.py`` (che serve solo a validare
calibrazione/frame-diff, dove la forma esatta non conta), qui la forma
conta: un modello di keypoint deve imparare a distinguere la punta da
fusto/alette, non solo "qualcosa e' cambiato".
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from dartvision.config import AppConfig

# Dimensioni fisiche approssimative della porzione di freccetta visibile
# sopra il bersaglio dopo l'impatto (mm).
_BARREL_LENGTH_MM = 22.0
_BARREL_WIDTH_MM = 3.0
_FLIGHT_LENGTH_MM = 22.0
_FLIGHT_WIDTH_MM = 18.0
_TIP_RADIUS_MM = 1.2
_TIP_COLOR = (15, 15, 15)

# Palette di colori (BGR) tra cui scegliere casualmente fusto/alette:
# diversita' di colore aiuta il modello a non ancorarsi a un colore
# specifico durante il training.
_BARREL_COLOR_PALETTE = [(90, 90, 90), (60, 60, 130), (110, 70, 40), (40, 40, 40)]
_FLIGHT_COLOR_PALETTE = [
    (30, 30, 200),
    (200, 60, 30),
    (30, 160, 30),
    (20, 20, 20),
    (200, 200, 30),
]


def _direction_vector(angle_deg: float) -> np.ndarray:
    """Vettore unitario per ``angle_deg``, stessa convenzione del resto
    del progetto: senso orario a partire da "in alto" (-y)."""
    rad = math.radians(angle_deg)
    return np.array([math.sin(rad), -math.cos(rad)], dtype=np.float64)


def place_dart(
    plane_img: np.ndarray,
    config: AppConfig,
    x_mm: float,
    y_mm: float,
    angle_deg: float,
    rng: np.random.Generator,
) -> tuple[tuple[float, float], tuple[float, float, float, float]]:
    """Disegna una freccetta stilizzata con punta in ``(x_mm, y_mm)`` e
    fusto/alette orientati lungo ``angle_deg``, direttamente su
    ``plane_img`` (in-place). Ritorna ``(tip_px, bbox_px)``: l'etichetta
    e' esatta per costruzione, non ricavata a posteriori con un'euristica.
    """
    mm_per_px = config.rectified_plane.mm_per_px
    tip_px = np.array(config.rectified_plane.mm_to_px(x_mm, y_mm), dtype=np.float64)
    direction = _direction_vector(angle_deg)
    perpendicular = np.array([-direction[1], direction[0]], dtype=np.float64)

    barrel_length_px = _BARREL_LENGTH_MM / mm_per_px
    barrel_width_px = _BARREL_WIDTH_MM / mm_per_px
    flight_length_px = _FLIGHT_LENGTH_MM / mm_per_px
    flight_width_px = _FLIGHT_WIDTH_MM / mm_per_px
    tip_radius_px = max(1.0, _TIP_RADIUS_MM / mm_per_px)

    barrel_end_px = tip_px + direction * barrel_length_px
    flight_end_px = barrel_end_px + direction * flight_length_px

    barrel_color = _BARREL_COLOR_PALETTE[rng.integers(len(_BARREL_COLOR_PALETTE))]
    flight_color = _FLIGHT_COLOR_PALETTE[rng.integers(len(_FLIGHT_COLOR_PALETTE))]

    # Fusto: rettangolo sottile da punta a inizio aletta.
    barrel_poly = np.array(
        [
            tip_px + perpendicular * (barrel_width_px / 2),
            barrel_end_px + perpendicular * (barrel_width_px / 2),
            barrel_end_px - perpendicular * (barrel_width_px / 2),
            tip_px - perpendicular * (barrel_width_px / 2),
        ]
    )
    cv2.fillPoly(plane_img, [barrel_poly.astype(np.int32)], barrel_color)

    # Aletta: forma a "kite" aperta a ventaglio dal fondo del fusto.
    flight_poly = np.array(
        [
            barrel_end_px,
            barrel_end_px + perpendicular * (flight_width_px / 2),
            flight_end_px,
            barrel_end_px - perpendicular * (flight_width_px / 2),
        ]
    )
    cv2.fillPoly(plane_img, [flight_poly.astype(np.int32)], flight_color)

    # Punta: sopra fusto/aletta, e' l'estremo piu' vicino al centro
    # bersaglio (coerente con l'euristica classica select_tip_point).
    cv2.circle(plane_img, tuple(np.round(tip_px).astype(int)), round(tip_radius_px), _TIP_COLOR, -1)

    all_points = np.vstack([barrel_poly, flight_poly, tip_px[None, :]])
    x_min, y_min = all_points.min(axis=0)
    x_max, y_max = all_points.max(axis=0)

    return (float(tip_px[0]), float(tip_px[1])), (float(x_min), float(y_min), float(x_max), float(y_max))
