"""Elabora un video registrato attraverso l'intera pipeline (calibrazione
-> detection -> scoring -> persistenza) e salva la partita nel database.

Uso:
    uv run python scripts/process_video.py --video data/videos/synthetic_calibration.mp4 \
        --player1 Alice --player2 Bob
"""

from __future__ import annotations

import argparse

from dartvision.calibration.homography import Calibrator
from dartvision.config import load_config
from dartvision.detection.frame_diff_detector import FrameDiffDetector
from dartvision.input.frame_source import VideoFileSource
from dartvision.persistence.db import connect
from dartvision.persistence.repository import DartRepository
from dartvision.pipeline.runner import run_pipeline
from dartvision.scoring.game_501 import STARTING_SCORE_DEFAULT
from dartvision.scoring.match_501 import MatchTurnResult


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
    parser.add_argument("--video", required=True)
    parser.add_argument("--db", default="data/dartvision.db")
    parser.add_argument("--starting-score", type=int, default=STARTING_SCORE_DEFAULT)
    parser.add_argument("--player1", default="Giocatore 1")
    parser.add_argument("--player2", default="Giocatore 2")
    args = parser.parse_args()

    config = load_config(args.config)
    calibrator = Calibrator(config)
    detector = FrameDiffDetector(config.detection, config.rectified_plane)

    conn = connect(args.db)
    repository = DartRepository(conn)

    player_names = {1: args.player1, 2: args.player2}

    with VideoFileSource(args.video) as source:
        result = run_pipeline(
            source, config, calibrator, detector, repository,
            player1_name=args.player1, player2_name=args.player2,
            starting_score=args.starting_score,
            on_turn=lambda r: _report_turn(player_names[r.player_number], r),
        )

    print(f"\nPartita #{result.game_id} salvata in {args.db}: {len(result.turns)} turni giocati.")
    conn.close()


if __name__ == "__main__":
    main()
