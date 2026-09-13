from __future__ import annotations

import pytest

from dartvision.scoring.game_501 import Game501
from dartvision.scoring.score import (
    RING_BULLSEYE,
    RING_DOUBLE,
    RING_SINGLE_OUTER,
    RING_TRIPLE,
    ScoredThrow,
)


def _throw(points: int, ring: str = RING_SINGLE_OUTER, sector: int | None = 20, multiplier: int = 1):
    """Freccetta finta: le coordinate non contano per la logica di bust,
    solo sector/ring/multiplier/points (coerenti col ring scelto)."""
    return ScoredThrow(
        x_mm=0.0, y_mm=0.0, angle_deg=0.0, radius_mm=0.0,
        sector=sector, ring=ring, multiplier=multiplier, points=points,
    )


def test_normal_turn_subtracts_points(app_config):
    game = Game501(app_config.board)

    outcome = game.play_turn([_throw(20), _throw(20), _throw(20)])

    assert outcome.turn_points == 60
    assert outcome.remaining_after == 441
    assert not outcome.is_bust
    assert not outcome.is_checkout
    assert game.remaining == 441
    assert not game.finished


def test_bust_when_going_below_zero(app_config):
    game = Game501(app_config.board, starting_score=10)

    outcome = game.play_turn([_throw(20)])  # 10 - 20 = -10

    assert outcome.is_bust
    assert outcome.turn_points == 0
    assert outcome.remaining_after == 10
    assert game.remaining == 10
    assert all(d.is_bust and d.points_scored == 0 for d in outcome.darts)


def test_bust_when_landing_exactly_on_one(app_config):
    game = Game501(app_config.board, starting_score=41)

    outcome = game.play_turn([_throw(40, ring=RING_TRIPLE, multiplier=2, sector=20)])  # 41-40=1

    assert outcome.is_bust
    assert game.remaining == 41


def test_bust_when_reaching_zero_without_double_or_bullseye(app_config):
    game = Game501(app_config.board, starting_score=20)

    outcome = game.play_turn([_throw(20, ring=RING_SINGLE_OUTER, multiplier=1, sector=20)])

    assert outcome.is_bust
    assert game.remaining == 20
    assert not game.finished


def test_valid_checkout_on_double(app_config):
    game = Game501(app_config.board, starting_score=40)

    outcome = game.play_turn([_throw(40, ring=RING_DOUBLE, multiplier=2, sector=20)])

    assert not outcome.is_bust
    assert outcome.is_checkout
    assert outcome.remaining_after == 0
    assert game.remaining == 0
    assert game.finished


def test_valid_checkout_on_inner_bullseye(app_config):
    game = Game501(app_config.board, starting_score=50)

    outcome = game.play_turn([_throw(50, ring=RING_BULLSEYE, multiplier=1, sector=None)])

    assert not outcome.is_bust
    assert outcome.is_checkout
    assert game.finished


def test_entire_turn_reverted_on_late_bust(app_config):
    """Le prime due freccette 'valide' del turno vengono annullate se la
    terza manda in bust: il residuo torna a quello di inizio turno."""
    game = Game501(app_config.board, starting_score=100)

    outcome = game.play_turn(
        [
            _throw(20, ring=RING_SINGLE_OUTER, multiplier=1, sector=20),  # 100 -> 80 (provvisorio)
            _throw(20, ring=RING_SINGLE_OUTER, multiplier=1, sector=20),  # 80 -> 60 (provvisorio)
            _throw(60, ring=RING_TRIPLE, multiplier=3, sector=20),  # 60-60=0 ma non e' un doppio -> bust
        ]
    )

    assert outcome.is_bust
    assert outcome.turn_points == 0
    assert outcome.remaining_after == 100
    assert game.remaining == 100
    for dart in outcome.darts:
        assert dart.is_bust
        assert dart.points_scored == 0
        assert dart.remaining_before == 100
        assert dart.remaining_after == 100


def test_darts_after_checkout_are_not_evaluated(app_config):
    game = Game501(app_config.board, starting_score=40)

    outcome = game.play_turn(
        [
            _throw(40, ring=RING_DOUBLE, multiplier=2, sector=20),
            _throw(20, ring=RING_SINGLE_OUTER, multiplier=1, sector=20),
        ]
    )

    assert len(outcome.darts) == 1
    assert outcome.is_checkout


def test_cannot_play_turn_after_game_finished(app_config):
    game = Game501(app_config.board, starting_score=40)
    game.play_turn([_throw(40, ring=RING_DOUBLE, multiplier=2, sector=20)])

    with pytest.raises(ValueError):
        game.play_turn([_throw(1)])


@pytest.mark.parametrize("n_throws", [0, 4])
def test_play_turn_requires_one_to_three_throws(app_config, n_throws):
    game = Game501(app_config.board)
    with pytest.raises(ValueError):
        game.play_turn([_throw(1)] * n_throws)


def test_closing_distance_is_none_when_not_attempting_a_checkout(app_config):
    game = Game501(app_config.board, starting_score=501)
    outcome = game.play_turn([_throw(20, ring=RING_SINGLE_OUTER, multiplier=1, sector=20)])
    assert outcome.darts[0].closing_distance_mm is None


def test_closing_distance_is_populated_when_attempting_checkout(app_config):
    game = Game501(app_config.board, starting_score=40)  # richiede doppio 20
    outcome = game.play_turn([_throw(1, ring=RING_SINGLE_OUTER, multiplier=1, sector=1)])
    assert outcome.darts[0].closing_distance_mm is not None
