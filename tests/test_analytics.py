from __future__ import annotations

from datetime import datetime

import pytest

from dartvision.analytics import compute_stats
from dartvision.persistence.models import ThrowRecord


def _row(
    turn_number,
    points,
    is_bust=False,
    closing_distance_mm=None,
    game_id=1,
    throw_number=1,
    player_number=1,
):
    return ThrowRecord(
        id=None,
        game_id=game_id,
        player_number=player_number,
        turn_number=turn_number,
        throw_number=throw_number,
        timestamp=datetime(2026, 1, 1),
        x_mm=0.0,
        y_mm=0.0,
        angle_deg=0.0,
        radius_mm=0.0,
        sector=20,
        ring="single_outer",
        multiplier=1,
        points=points,
        is_bust=is_bust,
        is_checkout=False,
        closing_distance_mm=closing_distance_mm,
        remaining_before=0,
        remaining_after=0,
    )


def test_empty_throws_returns_zeroed_stats():
    stats = compute_stats([])
    assert stats.num_darts == 0
    assert stats.average_per_dart == 0.0
    assert stats.best_turn_number is None


def test_average_per_dart_and_per_turn():
    throws = [_row(1, 60), _row(1, 60), _row(1, 60), _row(2, 20), _row(2, 20)]
    stats = compute_stats(throws)

    assert stats.num_darts == 5
    assert stats.num_turns == 2
    assert stats.total_points_scored == 220
    assert stats.average_per_dart == pytest.approx(44.0)
    assert stats.average_per_turn == pytest.approx(110.0)


def test_bust_turn_excluded_from_points_but_counted_as_bust():
    throws = [
        _row(1, 60), _row(1, 60), _row(1, 60),  # turno normale: 180
        _row(2, 60, is_bust=True), _row(2, 60, is_bust=True), _row(2, 60, is_bust=True),  # bust
    ]
    stats = compute_stats(throws)

    assert stats.bust_count == 1
    assert stats.total_points_scored == 180  # il turno bust non contribuisce
    assert stats.average_per_turn == pytest.approx(90.0)  # 180 / 2 turni


def test_best_turn_ignores_bust_turns():
    throws = [
        _row(1, 20), _row(1, 20), _row(1, 20),  # turno1: 60
        _row(2, 60, is_bust=True), _row(2, 60, is_bust=True),  # bust, punteggio piu' alto ma non conta
        _row(3, 30), _row(3, 30),  # turno3: 60... uguale, ma testiamo un caso piu' alto sotto
    ]
    stats = compute_stats(throws)
    assert stats.best_turn_number in (1, 3)
    assert stats.best_turn_points == 60


def test_average_closing_distance_ignores_none():
    throws = [
        _row(1, 40, closing_distance_mm=5.0),
        _row(1, 0, closing_distance_mm=None),
        _row(2, 36, closing_distance_mm=15.0),
    ]
    stats = compute_stats(throws)

    assert stats.checkout_attempts == 2
    assert stats.average_closing_distance_mm == pytest.approx(10.0)


def test_no_closing_attempts_gives_none_average():
    throws = [_row(1, 60)]
    stats = compute_stats(throws)
    assert stats.checkout_attempts == 0
    assert stats.average_closing_distance_mm is None
