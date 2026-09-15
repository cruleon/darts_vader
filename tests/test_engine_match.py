"""Match flow of the game engine: legs, game over, statistics, rematch and player photos."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from darts_vader.board import BoardState, geometry as g  # noqa: E402
from darts_vader.players import PhotoStore  # noqa: E402
from darts_vader.server.engine import FINISHED, PLAYING, GameEngine  # noqa: E402


def make_engine(tmp_path, players, start, double_out, legs_to_win):
    engine = GameEngine(None, "camera_canon", "cpu", players, start, double_out, tmp_path / "labels",
                        calibration_file=None, legs_to_win=legs_to_win, photos=PhotoStore(tmp_path / "photos"))
    engine.mode, engine.board = PLAYING, BoardState(np.eye(3), 1.0, 0.0, 0, False)
    engine.latest_frame = np.zeros((16, 16, 3), np.uint8)
    return engine


def play_turn(engine, *darts):
    """Enter darts (radius mm, angle deg) on the mini board, open the review and confirm the turn."""
    for radius, angle in darts:
        x, y = g.model_points(radius, angle)[0]
        engine.command({"type": "add_sim", "x_mm": float(x), "y_mm": float(y)})
    assert engine.start_review("manual")
    engine.confirm(save=False)


T20, S1, S20_INNER, S20_OUTER, D20 = (103.0, 0.0), (60.0, 18.0), (60.0, 0.0), (150.0, 3.0), (166.0, 0.0)


def test_single_leg_match_ends_with_statistics(tmp_path):
    engine = make_engine(tmp_path, ["Leo", "Marco"], 101, True, 1)
    play_turn(engine, T20, S1)  # Leo: 61 -> 40 left
    play_turn(engine, S20_INNER)  # Marco: 81 left
    play_turn(engine, S20_OUTER, D20)  # Leo: 20 left, then D20 busts -> back to 40
    assert engine.mode == PLAYING and engine.game.scores[0] == 40
    play_turn(engine, S20_INNER)  # Marco: 61 left
    play_turn(engine, D20)  # Leo checks out

    state = engine.state()
    assert state["mode"] == FINISHED and state["finish_target"] is None
    assert state["events"][-1]["kind"] == "game_over"
    summary = state["summary"]
    assert summary["winner"] == 0 and summary["legs_to_win"] == 1 and summary["legs_played"] == 1
    leo, marco = summary["players"]
    assert (leo["legs_won"], leo["highest_checkout"], leo["busts"]) == (1, 40, 1)
    assert (leo["darts_at_double"], leo["checkout_pct"]) == (3, 33.3)  # at 40, at 20 (bust dart) and the winning D20
    assert len(leo["darts"]) == 5 and len(marco["darts"]) == 2
    assert leo["best_finish_distance_mm"] == 0.0

    engine.command({"type": "add_sim", "x_mm": 0.0, "y_mm": 0.0})  # ignored once the match is over
    assert engine.state()["turn"] == []
    engine.command({"type": "new_game", "players": ["Leo", "Marco"], "start": 301, "double_out": False, "legs_to_win": 3})
    state = engine.state()
    assert state["mode"] == PLAYING and state["summary"] is None and state["game"]["legs_to_win"] == 3


def test_match_needs_every_leg(tmp_path):
    engine = make_engine(tmp_path, ["Solo"], 101, False, 2)
    engine.game.scores[0] = 40
    play_turn(engine, D20)  # first leg
    assert engine.mode == PLAYING and engine.game.legs == [1] and engine.game.scores == [101]
    engine.game.scores[0] = 40
    play_turn(engine, D20)  # second leg: match over
    state = engine.state()
    assert state["mode"] == FINISHED and state["summary"]["legs_played"] == 2
    assert state["summary"]["players"][0]["legs_won"] == 2


def test_invalid_legs_setting_is_rejected(tmp_path):
    engine = make_engine(tmp_path, ["Leo"], 301, False, 1)
    engine.command({"type": "new_game", "players": ["Leo"], "start": 301, "legs_to_win": 9})
    assert engine.legs_to_win == 1 and engine.state()["events"][-1]["text"] == "Invalid game settings"


def test_player_photos_in_state(tmp_path):
    engine = make_engine(tmp_path, ["Leo"], 301, False, 1)
    assert engine.state()["game"]["players"][0]["photo"] is None
    engine.photos.save("Leo", cv2.imencode(".png", np.zeros((20, 30, 3), np.uint8))[1].tobytes())
    assert engine.state()["game"]["players"][0]["photo"].startswith("/api/player-photo/")
