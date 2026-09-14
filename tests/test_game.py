"""Scoring geometry and x01 rules."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from darts_vader.board import geometry as g  # noqa: E402
from darts_vader.board.geometry import Hit  # noqa: E402
from darts_vader.game.x01 import BUST, OK, WIN, X01, finishing_double, suggest_checkout  # noqa: E402


def test_score_labels():
    assert Hit(20, 3).label == "T20" and Hit(20, 3).score == 60
    assert Hit(25, 2).label == "BULL" and Hit(25, 1).label == "25"
    assert Hit(0, 0).label == "MISS" and Hit(0, 0).score == 0


def test_score_model_point():
    assert g.score_model_point(0.0, 0.0) == Hit(25, 2)
    assert g.score_model_point(0.0, -10.0) == Hit(25, 1)
    assert g.score_model_point(*g.model_points(103.0, 0.0)[0]) == Hit(20, 3)
    assert g.score_model_point(*g.model_points(166.0, 18.0)[0]) == Hit(1, 2)
    assert g.score_model_point(*g.model_points(60.0, 180.0)[0]) == Hit(3, 1)
    assert g.score_model_point(*g.model_points(200.0, 0.0)[0]) == Hit(0, 0)


def test_nearest_point_in_segment():
    d20 = Hit(20, 2)
    inside = g.model_points(166.0, 2.0)[0]
    point, distance = g.nearest_point_in_segment(*inside, d20)
    assert distance == 0.0 and np.allclose(point, inside)
    point, distance = g.nearest_point_in_segment(0.0, 0.0, d20)  # centre: every point of the inner wire ties
    assert np.isclose(distance, g.R_DOUBLE_IN) and g.score_model_point(*(point * 1.001)) == d20
    _, distance = g.nearest_point_in_segment(*g.model_points(200.0, 0.0)[0], d20)  # beyond the outer wire
    assert np.isclose(distance, 200.0 - g.R_DOUBLE_OUT)
    _, distance = g.nearest_point_in_segment(*g.model_points(166.0, 18.0)[0], d20)  # D1: across the side wire
    assert np.isclose(distance, 166.0 * np.sin(np.radians(9.0)))
    _, distance = g.nearest_point_in_segment(0.0, -30.0, Hit(25, 2))
    assert np.isclose(distance, 30.0 - g.R_BULLSEYE)


def test_finishing_double():
    assert finishing_double(40) == Hit(20, 2) and finishing_double(2) == Hit(1, 2)
    assert finishing_double(50) == Hit(25, 2)
    assert all(finishing_double(v) is None for v in (41, 42, 60, 1, 0, -4))


def test_regular_turn():
    game = X01(["A", "B"], 301)
    result = game.apply_turn([Hit(20, 3)] * 3)
    assert (result.outcome, result.points, result.remaining) == (OK, 180, 121)
    assert game.current == 1 and game.average(0) == 180.0


def test_bust_keeps_score():
    game = X01(["A", "B"], 101, double_out=True)
    result = game.apply_turn([Hit(20, 3), Hit(20, 3)])
    assert (result.outcome, result.remaining, result.darts) == (BUST, 101, ["T20", "T20"])
    assert game.scores[0] == 101 and game.current == 1


def test_double_out_rules():
    assert X01(["A"], 40, double_out=True).apply_turn([Hit(20, 2)]).outcome == WIN
    assert X01(["A"], 40, double_out=True).apply_turn([Hit(20, 1), Hit(20, 1)]).outcome == BUST
    assert X01(["A"], 3, double_out=True).apply_turn([Hit(2, 1)]).outcome == BUST  # leaves 1
    assert X01(["A"], 40).apply_turn([Hit(20, 1), Hit(20, 1)]).outcome == WIN


def test_win_resets_leg():
    game = X01(["A", "B"], 50)
    game.apply_turn([Hit(25, 2)])
    assert game.legs == [1, 0] and game.scores == [50, 50]


def test_checkout_suggestions():
    assert suggest_checkout(180) == "T20 T20 T20"
    assert suggest_checkout(170, double_out=True) == "T20 T20 BULL"
    assert suggest_checkout(40, double_out=True) == "D20"
    assert suggest_checkout(50, 1, double_out=True) == "BULL"
    assert suggest_checkout(1, double_out=True) is None
    assert suggest_checkout(171, double_out=True) is None
