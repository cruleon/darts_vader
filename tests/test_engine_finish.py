"""Double-out distance information exposed by the game engine."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from darts_vader.board import BoardState, geometry as g  # noqa: E402
from darts_vader.server.engine import PLAYING, GameEngine  # noqa: E402


def make_engine(tmp_path, remaining: int, double_out: bool) -> GameEngine:
    engine = GameEngine(None, "camera_canon", "cpu", ["A"], 101, double_out, tmp_path, calibration_file=None)
    engine.mode, engine.board = PLAYING, BoardState(np.eye(3), 1.0, 0.0, 0, False)
    engine.game.scores[0] = remaining
    return engine


def add_dart(engine: GameEngine, radius: float, angle: float) -> None:
    x, y = g.model_points(radius, angle)[0]
    engine.command({"type": "add_sim", "x_mm": float(x), "y_mm": float(y)})


def test_distance_to_finishing_double(tmp_path):
    engine = make_engine(tmp_path, 70, double_out=True)
    add_dart(engine, 103.0, 108.0)  # T10: 40 left
    add_dart(engine, 150.0, 3.0)  # S20, 12 mm short of D20
    state = engine.state()
    first, second = state["turn"]
    assert first["label"] == "T10" and first["finish"] is None  # 70 cannot be finished with one dart
    assert second["label"] == "S20" and second["finish"]["target"] == "D20"
    assert np.isclose(second["finish"]["distance_mm"], 12.0)
    assert np.allclose(second["finish"]["point_mm"], g.model_points(g.R_DOUBLE_IN, 3.0)[0], atol=0.01)
    assert state["finish_target"] == "D10"  # 20 left after S20


def test_bull_and_bust(tmp_path):
    engine = make_engine(tmp_path, 50, double_out=True)
    assert engine.state()["finish_target"] == "BULL"
    add_dart(engine, 103.0, 0.0)  # T20: bust
    state = engine.state()
    assert state["turn"][0]["finish"]["target"] == "BULL" and state["bust"]
    assert state["finish_target"] is None


def test_no_distance_without_double_out(tmp_path):
    engine = make_engine(tmp_path, 40, double_out=False)
    add_dart(engine, 150.0, 3.0)
    state = engine.state()
    assert state["turn"][0]["finish"] is None and state["finish_target"] is None
