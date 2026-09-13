"""Orchestratore: l'unico modulo che conosce tutti i livelli.

Calibrazione, detection, scoring e persistenza restano ciascuno
ignari degli altri; questo modulo li collega in un ciclo che
processa una sorgente di frame dall'inizio alla fine, alternando il
turno tra due giocatori fino a quando uno dei due chiude la partita.

Limite noto (scelta di scope, non svista): un turno viene chiuso
automaticamente al terzo impatto rilevato, oppure alla fine dello
stream se ne restano meno di 3 in sospeso (es. il video finisce dopo
2 freccette). Non c'e' modo di rilevare la chiusura di un turno da
1-2 freccette a META' di un video/stream piu' lungo: per un file
registrato i turni sono quasi sempre completi, quindi non e' un
problema pratico oggi. E' pero' il punto naturale in cui la futura
estensione a webcam live dovra' aggiungere un'euristica di confine
turno (es. timeout di inattivita', o la mano che entra a ritirare le
freccette), quando servira' davvero.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from dartvision.config import AppConfig
from dartvision.detection.interface import ImpactDetector
from dartvision.input.frame_source import FrameSource
from dartvision.persistence.repository import DartRepository
from dartvision.scoring.game_501 import STARTING_SCORE_DEFAULT
from dartvision.scoring.match_501 import Match501, MatchTurnResult
from dartvision.scoring.score import score_point


class _CalibratorLike:
    """Protocollo implicito: qualunque oggetto con un metodo ``calibrate``
    compatibile (tipicamente un ``Calibrator``, o un doppio di test)."""

    def calibrate(self, frame): ...  # pragma: no cover - solo per documentazione


@dataclass
class PipelineResult:
    game_id: int
    turns: list[MatchTurnResult] = field(default_factory=list)


def run_pipeline(
    source: FrameSource,
    config: AppConfig,
    calibrator: _CalibratorLike,
    detector: ImpactDetector,
    repository: DartRepository,
    player1_name: str,
    player2_name: str,
    starting_score: int = STARTING_SCORE_DEFAULT,
    on_turn: Callable[[MatchTurnResult], None] | None = None,
) -> PipelineResult:
    """Processa ``source`` dall'inizio alla fine, giocando una partita 501
    a due giocatori (turni alternati, vince chi chiude per primo)."""
    match = Match501(config.board, player1_name, player2_name, starting_score=starting_score)
    game_id = repository.create_game(
        starting_score=starting_score, player1_name=player1_name, player2_name=player2_name
    )

    turns: list[MatchTurnResult] = []
    pending_throws = []
    turn_numbers = {1: 0, 2: 0}  # turno progressivo indipendente per giocatore

    def flush_turn() -> None:
        if not pending_throws or match.finished:
            return
        player_about_to_play = match.current_player
        result = match.play_turn(list(pending_throws))
        turn_numbers[player_about_to_play] += 1
        repository.add_turn_outcome(
            game_id, player_about_to_play, turn_numbers[player_about_to_play], result.outcome
        )
        turns.append(result)
        pending_throws.clear()
        if on_turn is not None:
            on_turn(result)

    prev_frame = None
    for frame in source.frames():
        if match.finished:
            break

        result = calibrator.calibrate(frame)
        if not result.success:
            prev_frame = frame
            continue

        if prev_frame is not None:
            impact = detector.detect(prev_frame, frame, result.homography)
            if impact is not None:
                scored = score_point(impact.x_mm, impact.y_mm, config.board)
                pending_throws.append(scored)
                if len(pending_throws) == 3:
                    flush_turn()

        prev_frame = frame

    flush_turn()  # eventuale turno incompleto rimasto a fine sorgente

    return PipelineResult(game_id=game_id, turns=turns)
