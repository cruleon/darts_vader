"""Conversioni geometriche pure tra coordinate cartesiane (mm, origine
al centro del bersaglio) e coordinate polari del bersaglio.

Convenzione angolare: gradi, senso ORARIO, 0 = "in alto" (verso -y),
coerente con la numerazione ufficiale del bersaglio (il 20 in alto,
poi in senso orario). Nessuna funzione qui sa nulla di OpenCV,
immagini o pixel: e' logica pura, testabile senza visione.
"""

from __future__ import annotations

import math

from dartvision.config import BoardConfig


def cartesian_to_polar(x_mm: float, y_mm: float) -> tuple[float, float]:
    """(x, y) mm -> (angolo in gradi orari da 'in alto' [0, 360), raggio mm)."""
    radius = math.hypot(x_mm, y_mm)
    angle = math.degrees(math.atan2(x_mm, -y_mm)) % 360.0
    return angle, radius


def polar_to_cartesian(angle_deg: float, radius_mm: float) -> tuple[float, float]:
    """Inversa di :func:`cartesian_to_polar`."""
    theta = math.radians(angle_deg)
    return radius_mm * math.sin(theta), -radius_mm * math.cos(theta)


def sector_number_at_angle(angle_deg: float, board: BoardConfig) -> int:
    """Numero del settore (es. 20, 1, 18, ...) alla data direzione angolare."""
    sector_width = 360.0 / board.sector_count
    shifted = (angle_deg - board.sector0_offset_deg + sector_width / 2.0) % 360.0
    index = int(shifted // sector_width)
    return board.sector_order[index]


def sector_center_angle(sector_number: int, board: BoardConfig) -> float:
    """Angolo (gradi, orario da 'in alto') del centro del settore dato."""
    index = board.sector_order.index(sector_number)
    sector_width = 360.0 / board.sector_count
    return (board.sector0_offset_deg + index * sector_width) % 360.0
