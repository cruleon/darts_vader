"""Esegue l'intera pipeline (calibrazione -> detection -> scoring ->
persistenza) live da webcam, invece che da un video registrato.

Una webcam non finisce mai da sola: Ctrl+C interrompe lo stream in modo
pulito (l'eventuale turno parziale in corso viene comunque chiuso e
salvato, con la stessa logica di fine-sorgente gia' usata per i video).

Uso:
    uv run python scripts/run_live.py --player1 Alice --player2 Bob
    uv run python scripts/run_live.py --device-index 1
"""

from __future__ import annotations

import argparse
import signal
import threading

import numpy as np

from dartvision.calibration.homography import Calibrator
from dartvision.config import load_config
from dartvision.detection.frame_diff_detector import FrameDiffDetector
from dartvision.detection.interface import ImpactDetector
from dartvision.detection.ml_tip_detector import MLTipDetector
from dartvision.input.frame_source import FrameSource, WebcamSource
from dartvision.persistence.db import connect
from dartvision.persistence.repository import DartRepository
from dartvision.pipeline.runner import run_pipeline
from dartvision.scoring.game_501 import STARTING_SCORE_DEFAULT
from dartvision.scoring.match_501 import MatchTurnResult


class _InterruptibleSource(FrameSource):
    """Interrompe ``source`` in modo pulito quando ``stop_event`` scatta.

    ``pipeline.runner.run_pipeline`` non sa nulla di webcam o segnali:
    vede semplicemente "la sorgente e' finita" e chiude l'eventuale
    turno parziale con la logica gia' esistente (``flush_turn()``).
    """

    def __init__(self, source: FrameSource, stop_event: threading.Event) -> None:
        self._source = source
        self._stop_event = stop_event

    @property
    def fps(self) -> float:
        return self._source.fps

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._stop_event.is_set():
            return False, None
        return self._source.read()

    def release(self) -> None:
        self._source.release()


def _build_detector(name: str, config) -> ImpactDetector:
    if name == "frame-diff":
        return FrameDiffDetector(config.detection, config.rectified_plane)
    if name == "ml":
        return MLTipDetector(config.ml_detection, config.detection, config.rectified_plane)
    raise ValueError(f"Detector sconosciuto: {name}")


def _report_turn(player_name: str, result: MatchTurnResult) -> None:
    outcome = result.outcome
    if outcome.is_bust:
        status = "BUST"
    elif outcome.is_checkout:
        status = "CHECKOUT!"
    else:
        status = "ok"
    dart_summary = ", ".join(f"{d.scored.ring}({d.scored.points})" for d in outcome.darts)
    print(
        f"{player_name} - turno: [{dart_summary}] -> {outcome.turn_points} punti, "
        f"residuo {outcome.remaining_after} [{status}]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/board_config.yaml")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--detector", choices=["frame-diff", "ml"], default="frame-diff")
    parser.add_argument("--db", default="data/dartvision.db")
    parser.add_argument("--starting-score", type=int, default=STARTING_SCORE_DEFAULT)
    parser.add_argument("--player1", default="Giocatore 1")
    parser.add_argument("--player2", default="Giocatore 2")
    args = parser.parse_args()

    config = load_config(args.config)
    calibrator = Calibrator(config)
    detector = _build_detector(args.detector, config)

    conn = connect(args.db)
    repository = DartRepository(conn)
    player_names = {1: args.player1, 2: args.player2}

    stop_event = threading.Event()

    def _handle_sigint(signum, frame) -> None:
        print("\nInterrotto: chiudo l'eventuale turno in corso e salvo...")
        stop_event.set()

    previous_handler = signal.signal(signal.SIGINT, _handle_sigint)

    try:
        with WebcamSource(device_index=args.device_index) as webcam:
            source = _InterruptibleSource(webcam, stop_event)
            result = run_pipeline(
                source, config, calibrator, detector, repository,
                player1_name=args.player1, player2_name=args.player2,
                starting_score=args.starting_score,
                on_turn=lambda r: _report_turn(player_names[r.player_number], r),
            )
    finally:
        signal.signal(signal.SIGINT, previous_handler)

    print(f"\nPartita #{result.game_id} salvata in {args.db}: {len(result.turns)} turni giocati.")
    conn.close()


if __name__ == "__main__":
    main()
