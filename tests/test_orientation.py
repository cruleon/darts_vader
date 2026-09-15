"""Automatic recognition of the 20 from the ring of printed numbers."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from darts_vader.board import BoardState, geometry as g  # noqa: E402
from darts_vader.board.orientation import match_orientation, number_ring  # noqa: E402
from darts_vader.board.synthetic import SyntheticScene  # noqa: E402
from darts_vader.server.engine import CALIBRATING, LOCK_FRAMES, PLAYING, SEARCHING, GameEngine  # noqa: E402


def truth_board(H: np.ndarray) -> BoardState:
    return BoardState(H / H[2, 2], 1.0, 0.0, 0, False)


def twenty_on_top(board: BoardState, truth: BoardState) -> bool:
    top = truth.to_image([(0.0, -130.0)])
    return g.sector_index(g.polar(*board.to_model(top)[0])[1]) == 0


def lock(engine: GameEngine, frame: np.ndarray) -> None:
    for i in range(LOCK_FRAMES + 10):
        engine.process_frame(frame, i * 0.1)
        if engine.mode != SEARCHING:
            return


def test_template_recognises_every_rotation():
    scene = SyntheticScene(seed=1)
    frame_a, H_a = scene.render(yaw=20, pitch=10)
    frame_b, H_b = scene.render(yaw=-25, pitch=-5, roll=40)
    template = number_ring(frame_a, truth_board(H_a))
    truth = truth_board(H_b)
    for k in (0, 3, 11, 17):
        match = match_orientation(frame_b, truth.rotated(k), template)
        assert match.confident
        assert twenty_on_top(truth.rotated(k).rotated(match.sectors), truth)


def test_engine_skips_manual_calibration_once_the_board_is_known(tmp_path):
    scene = SyntheticScene(seed=2)
    calibration = tmp_path / "calibration.json"

    def new_engine():
        return GameEngine(None, "camera_canon", "cpu", ["A"], 301, False, tmp_path / "labels", calibration_file=calibration)

    frame_a, H_a = scene.render(yaw=15, pitch=8)
    first = new_engine()
    lock(first, frame_a)
    assert first.mode == CALIBRATING  # first run: the 20 must be confirmed by hand
    first.candidate = first.candidate.aligned_to(truth_board(H_a))
    first.command({"type": "confirm_orientation"})
    assert first.mode == PLAYING and (tmp_path / "board_numbers.npy").exists()

    # camera moved and rolled: the saved orientation alone would pick the wrong sector
    frame_b, H_b = scene.render(yaw=-20, pitch=-6, roll=50)
    second = new_engine()
    lock(second, frame_b)
    assert second.mode == PLAYING and twenty_on_top(second.board, truth_board(H_b))

    # R forces the manual confirmation once
    second.command({"type": "recalibrate"})
    lock(second, frame_b)
    assert second.mode == CALIBRATING
