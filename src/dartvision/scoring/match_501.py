"""Partita 501 a due giocatori: alterna il turno tra due leg indipendenti.

Ogni giocatore ha il proprio ``Game501`` (proprio residuo, propri bust):
``Match501`` si limita a decidere di chi e' il turno e a dichiarare un
vincitore al primo checkout valido, senza reimplementare le regole di
punteggio gia' testate in ``game_501``.
"""

from __future__ import annotations

from dataclasses import dataclass

from dartvision.config import BoardConfig
from dartvision.scoring.game_501 import STARTING_SCORE_DEFAULT, Game501, TurnOutcome
from dartvision.scoring.score import ScoredThrow


@dataclass(frozen=True)
class MatchTurnResult:
    player_number: int  # 1 o 2: chi ha giocato QUESTO turno
    outcome: TurnOutcome


class Match501:
    def __init__(
        self,
        board: BoardConfig,
        player1_name: str,
        player2_name: str,
        starting_score: int = STARTING_SCORE_DEFAULT,
    ) -> None:
        if not player1_name.strip() or not player2_name.strip():
            raise ValueError("I nomi dei due giocatori non possono essere vuoti")

        self.player_names: dict[int, str] = {1: player1_name, 2: player2_name}
        self.games: dict[int, Game501] = {
            1: Game501(board, starting_score),
            2: Game501(board, starting_score),
        }
        self.current_player = 1
        self.winner: int | None = None

    @property
    def finished(self) -> bool:
        return self.winner is not None

    def player_name(self, player_number: int) -> str:
        return self.player_names[player_number]

    def play_turn(self, throws: list[ScoredThrow]) -> MatchTurnResult:
        if self.finished:
            raise ValueError("La partita e' gia' conclusa: nessun altro turno puo' essere giocato")

        player = self.current_player
        outcome = self.games[player].play_turn(throws)

        if outcome.is_checkout:
            self.winner = player

        self.current_player = 2 if player == 1 else 1
        return MatchTurnResult(player_number=player, outcome=outcome)
