"""Conversione di un punto d'impatto (mm) nel punteggio del bersaglio."""

from __future__ import annotations

from dataclasses import dataclass

from dartvision.config import BoardConfig
from dartvision.scoring.board_geometry import cartesian_to_polar, sector_number_at_angle

RING_BULLSEYE = "bullseye"  # bull interno, 50 punti
RING_BULL = "bull"  # bull esterno, 25 punti
RING_SINGLE_INNER = "single_inner"
RING_TRIPLE = "triple"
RING_SINGLE_OUTER = "single_outer"
RING_DOUBLE = "double"
RING_MISS = "miss"


@dataclass(frozen=True)
class ScoredThrow:
    """Un impatto gia' convertito in punteggio.

    Porta con se' anche le coordinate (cartesiane e polari) di origine:
    servono al livello di persistenza (che le salva cosi' come sono) e
    al calcolo della distanza di chiusura, senza dover ricalcolare la
    geometria altrove.
    """

    x_mm: float
    y_mm: float
    angle_deg: float
    radius_mm: float
    sector: int | None  # None per bull/bullseye/miss
    ring: str
    multiplier: int  # 0 (miss), 1, 2, 3; per i bull e' convenzionalmente 1
    points: int


def score_point(x_mm: float, y_mm: float, board: BoardConfig) -> ScoredThrow:
    angle, radius = cartesian_to_polar(x_mm, y_mm)

    if radius <= board.inner_bull_radius_mm:
        return ScoredThrow(x_mm, y_mm, angle, radius, None, RING_BULLSEYE, 1, 50)
    if radius <= board.outer_bull_radius_mm:
        return ScoredThrow(x_mm, y_mm, angle, radius, None, RING_BULL, 1, 25)
    if radius > board.double_outer_radius_mm:
        return ScoredThrow(x_mm, y_mm, angle, radius, None, RING_MISS, 0, 0)

    sector = sector_number_at_angle(angle, board)
    if radius <= board.triple_inner_radius_mm:
        ring, multiplier = RING_SINGLE_INNER, 1
    elif radius <= board.triple_outer_radius_mm:
        ring, multiplier = RING_TRIPLE, 3
    elif radius <= board.double_inner_radius_mm:
        ring, multiplier = RING_SINGLE_OUTER, 1
    else:
        ring, multiplier = RING_DOUBLE, 2

    return ScoredThrow(x_mm, y_mm, angle, radius, sector, ring, multiplier, sector * multiplier)
