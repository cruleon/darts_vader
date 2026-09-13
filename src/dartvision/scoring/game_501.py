"""Regole della modalita' 501.

Il bust e' un concetto di TURNO, non di singola freccetta: se una
freccetta del turno lo farebbe scattare, l'intero turno (comprese le
freccette precedenti dello stesso turno, che a quel punto avevano
gia' sottratto punti) viene annullato e il residuo torna al valore
di inizio turno. Le freccette successive alla chiusura o al bust,
se fisicamente lanciate e rilevate, vengono comunque registrate (per
le statistiche) ma non influiscono piu' sul punteggio.
"""

from __future__ import annotations

from dataclasses import dataclass

from dartvision.config import BoardConfig
from dartvision.scoring.closing import closing_distance_mm
from dartvision.scoring.score import RING_BULLSEYE, RING_DOUBLE, ScoredThrow

STARTING_SCORE_DEFAULT = 501

_VALID_FINISH_RINGS = (RING_DOUBLE, RING_BULLSEYE)


@dataclass(frozen=True)
class DartOutcome:
    scored: ScoredThrow
    remaining_before: int
    remaining_after: int
    points_scored: int  # 0 se il turno e' andato bust
    is_bust: bool  # riflette l'esito dell'INTERO turno, non solo di questa freccetta
    is_checkout: bool
    closing_distance_mm: float | None


@dataclass(frozen=True)
class TurnOutcome:
    darts: tuple[DartOutcome, ...]
    remaining_before: int
    remaining_after: int
    turn_points: int
    is_bust: bool
    is_checkout: bool


class Game501:
    """Stato e regole di una singola 'leg' di 501."""

    def __init__(self, board: BoardConfig, starting_score: int = STARTING_SCORE_DEFAULT) -> None:
        self._board = board
        self.starting_score = starting_score
        self.remaining = starting_score
        self.finished = False
        self.turns: list[TurnOutcome] = []

    def play_turn(self, throws: list[ScoredThrow]) -> TurnOutcome:
        if self.finished:
            raise ValueError("La partita e' gia' conclusa: nessun altro turno puo' essere giocato")
        if not 1 <= len(throws) <= 3:
            raise ValueError("Un turno e' composto da 1 a 3 freccette")

        remaining_before_turn = self.remaining
        running = remaining_before_turn
        turn_is_bust = False
        turn_is_checkout = False
        dart_outcomes: list[DartOutcome] = []

        for scored in throws:
            distance = closing_distance_mm(scored.x_mm, scored.y_mm, running, self._board)

            if turn_is_bust:
                dart_outcomes.append(
                    DartOutcome(scored, running, running, 0, True, False, distance)
                )
                continue

            candidate = running - scored.points
            valid_finish = scored.ring in _VALID_FINISH_RINGS
            dart_busts = candidate < 0 or candidate == 1 or (candidate == 0 and not valid_finish)

            if dart_busts:
                turn_is_bust = True
                dart_outcomes.append(
                    DartOutcome(scored, running, running, 0, True, False, distance)
                )
                continue

            remaining_before_dart = running
            running = candidate
            is_checkout = running == 0
            turn_is_checkout = turn_is_checkout or is_checkout

            dart_outcomes.append(
                DartOutcome(
                    scored, remaining_before_dart, running, scored.points, False, is_checkout, distance
                )
            )

            if is_checkout:
                break

        if turn_is_bust:
            final_remaining = remaining_before_turn
            turn_points = 0
            dart_outcomes = [
                DartOutcome(
                    d.scored, remaining_before_turn, remaining_before_turn, 0, True, False,
                    d.closing_distance_mm,
                )
                for d in dart_outcomes
            ]
        else:
            final_remaining = running
            turn_points = remaining_before_turn - final_remaining

        self.remaining = final_remaining
        if turn_is_checkout:
            self.finished = True

        outcome = TurnOutcome(
            darts=tuple(dart_outcomes),
            remaining_before=remaining_before_turn,
            remaining_after=final_remaining,
            turn_points=turn_points,
            is_bust=turn_is_bust,
            is_checkout=turn_is_checkout,
        )
        self.turns.append(outcome)
        return outcome
