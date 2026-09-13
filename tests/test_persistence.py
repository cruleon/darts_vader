from __future__ import annotations

from datetime import datetime

import pytest

from dartvision.config import load_config
from dartvision.persistence.db import connect
from dartvision.persistence.repository import DartRepository
from dartvision.scoring.game_501 import Game501
from dartvision.scoring.score import RING_DOUBLE, RING_SINGLE_OUTER, ScoredThrow


@pytest.fixture
def repo(tmp_path):
    conn = connect(tmp_path / "test.db")
    yield DartRepository(conn)
    conn.close()


@pytest.fixture
def board():
    return load_config("config/board_config.yaml").board


def _throw(points: int, ring: str = RING_SINGLE_OUTER, sector: int | None = 20, multiplier: int = 1):
    return ScoredThrow(
        x_mm=12.3, y_mm=-45.6, angle_deg=15.0, radius_mm=47.4,
        sector=sector, ring=ring, multiplier=multiplier, points=points,
    )


def test_create_and_get_game(repo):
    game_id = repo.create_game(
        starting_score=501, player1_name="Alice", player2_name="Bob",
        started_at=datetime(2026, 1, 1, 10, 0, 0),
    )

    game = repo.get_game(game_id)

    assert game is not None
    assert game.starting_score == 501
    assert game.player1_name == "Alice"
    assert game.player2_name == "Bob"
    assert game.winner_player_number is None
    assert game.finished_at is None


def test_finish_game_records_winner(repo):
    game_id = repo.create_game(starting_score=501, player1_name="Alice", player2_name="Bob")
    repo.finish_game(game_id, winner_player_number=2, finished_at=datetime(2026, 1, 1, 10, 30, 0))

    game = repo.get_game(game_id)

    assert game.finished_at == datetime(2026, 1, 1, 10, 30, 0)
    assert game.winner_player_number == 2


def test_list_games_returns_all(repo):
    repo.create_game(starting_score=501, player1_name="Alice", player2_name="Bob")
    repo.create_game(starting_score=301, player1_name="Carla", player2_name="Dario")

    games = repo.list_games()

    assert len(games) == 2
    assert {g.starting_score for g in games} == {501, 301}


def test_add_turn_outcome_persists_all_darts_for_correct_player(repo, board):
    game_id = repo.create_game(starting_score=501, player1_name="Alice", player2_name="Bob")
    game = Game501(board, starting_score=501)
    outcome = game.play_turn([_throw(20), _throw(20), _throw(20)])

    saved = repo.add_turn_outcome(game_id, player_number=1, turn_number=1, outcome=outcome)

    assert len(saved) == 3
    assert all(r.id is not None for r in saved)
    assert all(r.player_number == 1 for r in saved)
    assert [r.throw_number for r in saved] == [1, 2, 3]

    reloaded = repo.list_throws_for_game(game_id)
    assert len(reloaded) == 3
    assert reloaded[0].points == 20
    assert reloaded[0].remaining_after == 481
    assert reloaded[2].remaining_after == 441
    assert all(not r.is_bust for r in reloaded)


def test_add_turn_outcome_finishes_game_with_winner_on_checkout(repo, board):
    game_id = repo.create_game(starting_score=40, player1_name="Alice", player2_name="Bob")
    game = Game501(board, starting_score=40)

    outcome = game.play_turn(
        [ScoredThrow(0.0, -166.0, 0.0, 166.0, 20, RING_DOUBLE, 2, 40)]
    )
    repo.add_turn_outcome(game_id, player_number=2, turn_number=1, outcome=outcome)

    reloaded_game = repo.get_game(game_id)
    assert reloaded_game.finished_at is not None
    assert reloaded_game.winner_player_number == 2
    throws = repo.list_throws_for_game(game_id)
    assert throws[0].is_checkout
    assert throws[0].player_number == 2


def test_bust_turn_persists_zero_points_and_bust_flag(repo, board):
    game_id = repo.create_game(starting_score=10, player1_name="Alice", player2_name="Bob")
    game = Game501(board, starting_score=10)

    outcome = game.play_turn([_throw(20)])  # 10-20 < 0 -> bust
    repo.add_turn_outcome(game_id, player_number=1, turn_number=1, outcome=outcome)

    throws = repo.list_throws_for_game(game_id)
    assert len(throws) == 1
    assert throws[0].is_bust
    assert throws[0].remaining_after == 10


def test_closing_distance_round_trips_as_none(repo, board):
    game_id = repo.create_game(starting_score=501, player1_name="Alice", player2_name="Bob")
    game = Game501(board, starting_score=501)

    outcome = game.play_turn([_throw(20)])
    repo.add_turn_outcome(game_id, player_number=1, turn_number=1, outcome=outcome)

    throws = repo.list_throws_for_game(game_id)
    assert throws[0].closing_distance_mm is None


def test_list_all_throws_spans_multiple_games(repo, board):
    game1_id = repo.create_game(starting_score=501, player1_name="Alice", player2_name="Bob")
    game2_id = repo.create_game(starting_score=501, player1_name="Carla", player2_name="Dario")

    game1 = Game501(board, starting_score=501)
    game2 = Game501(board, starting_score=501)

    repo.add_turn_outcome(game1_id, 1, 1, game1.play_turn([_throw(20)]))
    repo.add_turn_outcome(game2_id, 1, 1, game2.play_turn([_throw(19, sector=19)]))

    all_throws = repo.list_all_throws()
    assert len(all_throws) == 2
    assert {t.game_id for t in all_throws} == {game1_id, game2_id}


def test_throws_for_both_players_are_stored_and_retrievable(repo, board):
    game_id = repo.create_game(starting_score=501, player1_name="Alice", player2_name="Bob")
    game1 = Game501(board, starting_score=501)
    game2 = Game501(board, starting_score=501)

    repo.add_turn_outcome(game_id, 1, 1, game1.play_turn([_throw(20)]))
    repo.add_turn_outcome(game_id, 2, 1, game2.play_turn([_throw(19, sector=19)]))

    throws = repo.list_throws_for_game(game_id)
    assert {t.player_number for t in throws} == {1, 2}
