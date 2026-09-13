"""Statistiche aggregate su uno storico di tiri gia' persistiti.

Logica pura: opera solo su ``ThrowRecord`` (nessun accesso al DB, nessuna
dipendenza da Streamlit), quindi testabile senza fixture pesanti.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby

from dartvision.persistence.models import ThrowRecord


@dataclass(frozen=True)
class GameStats:
    num_darts: int
    num_turns: int
    total_points_scored: int  # esclude i punti dei turni andati bust
    average_per_dart: float
    average_per_turn: float
    bust_count: int  # numero di TURNI andati bust, non di freccette
    best_turn_number: int | None
    best_turn_points: int
    checkout_attempts: int  # tiri con un target di chiusura applicabile
    average_closing_distance_mm: float | None


def compute_stats(throws: list[ThrowRecord]) -> GameStats:
    num_darts = len(throws)
    if num_darts == 0:
        return GameStats(0, 0, 0, 0.0, 0.0, 0, None, 0, 0, None)

    turn_numbers = sorted({t.turn_number for t in throws})
    num_turns = len(turn_numbers)

    total_points_scored = sum(t.points for t in throws if not t.is_bust)
    average_per_dart = total_points_scored / num_darts
    average_per_turn = total_points_scored / num_turns if num_turns else 0.0

    bust_count = 0
    best_turn_number: int | None = None
    best_turn_points = -1
    for turn_number, group in groupby(
        sorted(throws, key=lambda t: t.turn_number), key=lambda t: t.turn_number
    ):
        group_throws = list(group)
        if group_throws[0].is_bust:
            bust_count += 1
            continue
        turn_points = sum(t.points for t in group_throws)
        if turn_points > best_turn_points:
            best_turn_points = turn_points
            best_turn_number = turn_number

    closing_distances = [t.closing_distance_mm for t in throws if t.closing_distance_mm is not None]
    average_closing_distance = (
        sum(closing_distances) / len(closing_distances) if closing_distances else None
    )

    return GameStats(
        num_darts=num_darts,
        num_turns=num_turns,
        total_points_scored=total_points_scored,
        average_per_dart=average_per_dart,
        average_per_turn=average_per_turn,
        bust_count=bust_count,
        best_turn_number=best_turn_number,
        best_turn_points=max(best_turn_points, 0),
        checkout_attempts=len(closing_distances),
        average_closing_distance_mm=average_closing_distance,
    )
