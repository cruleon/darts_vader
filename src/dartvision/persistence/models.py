"""Record persistenti: mappano 1:1 le righe delle tabelle SQLite.

``throw_record_from_outcome`` e' l'unico punto di raccordo tra il
Livello 3 (scoring, ``DartOutcome``) e questo livello: e' una funzione
pura (nessun accesso al DB), cosi' resta facile da testare.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from dartvision.scoring.game_501 import DartOutcome


@dataclass(frozen=True)
class GameRecord:
    id: int | None
    started_at: datetime
    starting_score: int
    player1_name: str
    player2_name: str
    winner_player_number: int | None  # 1, 2, o None se non ancora conclusa
    finished_at: datetime | None

    def player_name(self, player_number: int) -> str:
        return self.player1_name if player_number == 1 else self.player2_name


@dataclass(frozen=True)
class ThrowRecord:
    id: int | None
    game_id: int
    player_number: int  # 1 o 2
    turn_number: int  # turno progressivo DI QUEL giocatore (1,2,3,... indipendente dall'altro)
    throw_number: int  # 1-3, posizione della freccetta nel turno
    timestamp: datetime
    x_mm: float
    y_mm: float
    angle_deg: float
    radius_mm: float
    sector: int | None
    ring: str
    multiplier: int
    points: int  # punti "grezzi" del tiro (indipendenti dal bust)
    is_bust: bool  # riflette l'esito dell'intero turno a cui appartiene
    is_checkout: bool
    closing_distance_mm: float | None
    remaining_before: int
    remaining_after: int


def throw_record_from_outcome(
    game_id: int,
    player_number: int,
    turn_number: int,
    throw_number: int,
    outcome: DartOutcome,
    timestamp: datetime,
    throw_id: int | None = None,
) -> ThrowRecord:
    scored = outcome.scored
    return ThrowRecord(
        id=throw_id,
        game_id=game_id,
        player_number=player_number,
        turn_number=turn_number,
        throw_number=throw_number,
        timestamp=timestamp,
        x_mm=scored.x_mm,
        y_mm=scored.y_mm,
        angle_deg=scored.angle_deg,
        radius_mm=scored.radius_mm,
        sector=scored.sector,
        ring=scored.ring,
        multiplier=scored.multiplier,
        points=scored.points,
        is_bust=outcome.is_bust,
        is_checkout=outcome.is_checkout,
        closing_distance_mm=outcome.closing_distance_mm,
        remaining_before=outcome.remaining_before,
        remaining_after=outcome.remaining_after,
    )
