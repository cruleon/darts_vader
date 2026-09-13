"""Distanza di chiusura: quanto un tiro ha mancato il doppio richiesto.

Applicabile solo quando il residuo permette la chiusura con UN singolo
doppio specifico (residuo pari, 2-40): e' il caso descritto nelle
specifiche del progetto. Il bullseye come chiusura alternativa (residuo
50) non ha un "doppio specifico" a cui riferire una distanza laterale
(il bersaglio e' un punto, non un segmento d'anello) e resta fuori da
questa metrica.
"""

from __future__ import annotations

import math

from dartvision.config import BoardConfig
from dartvision.scoring.board_geometry import polar_to_cartesian, sector_center_angle

MIN_CHECKOUT_DOUBLE = 2
MAX_CHECKOUT_DOUBLE = 40


def required_double_sector(remaining: int) -> int | None:
    """Numero del settore il cui doppio chiuderebbe esattamente ``remaining``
    con una sola freccetta, o ``None`` se nessun doppio singolo lo consente."""
    if remaining % 2 != 0 or not MIN_CHECKOUT_DOUBLE <= remaining <= MAX_CHECKOUT_DOUBLE:
        return None
    return remaining // 2


def double_target_point_mm(sector: int, board: BoardConfig) -> tuple[float, float]:
    """Centro (mm) del segmento di doppio del settore dato."""
    angle = sector_center_angle(sector, board)
    radius = (board.double_inner_radius_mm + board.double_outer_radius_mm) / 2.0
    return polar_to_cartesian(angle, radius)


def closing_distance_mm(
    x_mm: float, y_mm: float, remaining_before_throw: int, board: BoardConfig
) -> float | None:
    """Distanza euclidea (mm) tra l'impatto e il centro del doppio che
    avrebbe chiuso ``remaining_before_throw``; ``None`` se non applicabile."""
    sector = required_double_sector(remaining_before_throw)
    if sector is None:
        return None
    target_x, target_y = double_target_point_mm(sector, board)
    return math.hypot(x_mm - target_x, y_mm - target_y)
