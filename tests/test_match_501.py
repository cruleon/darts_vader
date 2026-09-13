from __future__ import annotations

import pytest

from dartvision.scoring.match_501 import Match501
from dartvision.scoring.score import RING_DOUBLE, RING_SINGLE_OUTER, ScoredThrow


def _throw(points: int, ring: str = RING_SINGLE_OUTER, sector: int | None = 20, multiplier: int = 1):
    return ScoredThrow(
        x_mm=0.0, y_mm=0.0, angle_deg=0.0, radius_mm=0.0,
        sector=sector, ring=ring, multiplier=multiplier, points=points,
    )


def test_turns_alternate_between_players(app_config):
    match = Match501(app_config.board, "Alice", "Bob")

    r1 = match.play_turn([_throw(20)])
    r2 = match.play_turn([_throw(20)])
    r3 = match.play_turn([_throw(20)])

    assert [r1.player_number, r2.player_number, r3.player_number] == [1, 2, 1]


def test_each_player_has_independent_score(app_config):
    match = Match501(app_config.board, "Alice", "Bob")

    match.play_turn([_throw(60, ring="triple", multiplier=3)])  # Alice: 501 -> 441
    match.play_turn([_throw(20)])  # Bob: 501 -> 481

    assert match.games[1].remaining == 441
    assert match.games[2].remaining == 481


def test_bust_only_affects_the_player_who_busted(app_config):
    match = Match501(app_config.board, "Alice", "Bob", starting_score=10)

    match.play_turn([_throw(20)])  # Alice busts: 10 - 20 < 0
    match.play_turn([_throw(5)])  # Bob: 10 -> 5

    assert match.games[1].remaining == 10  # invariato, turno annullato
    assert match.games[2].remaining == 5


def test_checkout_declares_winner_and_ends_match(app_config):
    match = Match501(app_config.board, "Alice", "Bob", starting_score=40)

    match.play_turn([_throw(20)])  # Alice: 40 -> 20, non chiude
    result = match.play_turn([_throw(40, ring=RING_DOUBLE, multiplier=2, sector=20)])  # Bob chiude

    assert result.player_number == 2
    assert result.outcome.is_checkout
    assert match.winner == 2
    assert match.finished


def test_cannot_play_after_match_finished(app_config):
    match = Match501(app_config.board, "Alice", "Bob", starting_score=40)
    match.play_turn([_throw(40, ring=RING_DOUBLE, multiplier=2, sector=20)])  # Alice chiude

    with pytest.raises(ValueError):
        match.play_turn([_throw(1)])


def test_player_name_lookup(app_config):
    match = Match501(app_config.board, "Alice", "Bob")
    assert match.player_name(1) == "Alice"
    assert match.player_name(2) == "Bob"


@pytest.mark.parametrize("p1,p2", [("", "Bob"), ("Alice", ""), ("   ", "Bob")])
def test_rejects_empty_player_names(app_config, p1, p2):
    with pytest.raises(ValueError):
        Match501(app_config.board, p1, p2)
