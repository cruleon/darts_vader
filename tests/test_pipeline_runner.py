from __future__ import annotations

import numpy as np
import pytest

from dartvision.calibration.homography import CalibrationResult
from dartvision.detection.types import Impact
from dartvision.persistence.db import connect
from dartvision.persistence.repository import DartRepository
from dartvision.pipeline.runner import run_pipeline


class _FakeSource:
    def __init__(self, n_frames: int):
        self._n_frames = n_frames

    def frames(self):
        for _ in range(self._n_frames):
            yield np.zeros((4, 4, 3), dtype=np.uint8)


class _FakeCalibrator:
    def calibrate(self, frame):
        return CalibrationResult(
            success=True, homography=np.eye(3), marker_ids_used=(0, 1, 2, 3), num_markers_detected=4
        )


class _ScriptedDetector:
    """Ritorna un Impact alle chiamate indicate (1-based), None altrimenti."""

    def __init__(self, impacts_by_call: dict[int, Impact]):
        self._impacts_by_call = impacts_by_call
        self._call_count = 0

    def detect(self, prev_frame, curr_frame, homography):
        self._call_count += 1
        return self._impacts_by_call.get(self._call_count)


@pytest.fixture
def repo(tmp_path):
    conn = connect(tmp_path / "test.db")
    yield DartRepository(conn)
    conn.close()


def _impact(x_mm: float, y_mm: float) -> Impact:
    return Impact(x_mm=x_mm, y_mm=y_mm, x_px=0.0, y_px=0.0, area_px=100.0)


def _run(source, config, calibrator, detector, repo, **kwargs):
    kwargs.setdefault("player1_name", "Alice")
    kwargs.setdefault("player2_name", "Bob")
    return run_pipeline(source, config, calibrator, detector, repo, **kwargs)


def test_groups_three_darts_into_one_turn(app_config, repo):
    source = _FakeSource(n_frames=5)
    detector = _ScriptedDetector({1: _impact(0, -166), 2: _impact(0, -166), 3: _impact(0, -166)})

    result = _run(source, app_config, _FakeCalibrator(), detector, repo, starting_score=501)

    assert len(result.turns) == 1
    assert result.turns[0].player_number == 1
    assert len(result.turns[0].outcome.darts) == 3
    assert len(repo.list_throws_for_game(result.game_id)) == 3


def test_flushes_incomplete_final_turn_at_end_of_stream(app_config, repo):
    source = _FakeSource(n_frames=3)
    detector = _ScriptedDetector({1: _impact(0.0, 0.0)})

    result = _run(source, app_config, _FakeCalibrator(), detector, repo, starting_score=501)

    assert len(result.turns) == 1
    assert len(result.turns[0].outcome.darts) == 1


def test_stops_after_checkout(app_config, repo):
    source = _FakeSource(n_frames=10)
    # (0,-166)mm e' il centro del doppio 20: 40 punti, chiude una partita a 40.
    detector = _ScriptedDetector({1: _impact(0.0, -166.0)})

    result = _run(source, app_config, _FakeCalibrator(), detector, repo, starting_score=40)

    assert len(result.turns) == 1
    assert result.turns[0].outcome.is_checkout
    game = repo.get_game(result.game_id)
    assert game.finished_at is not None
    assert game.winner_player_number == 1


def test_skips_frames_when_calibration_fails(app_config, repo):
    class _FlakyCalibrator:
        def __init__(self):
            self._calls = 0

        def calibrate(self, frame):
            self._calls += 1
            success = self._calls > 1  # il primo frame "fallisce" la calibrazione
            return CalibrationResult(
                success=success,
                homography=np.eye(3) if success else None,
                marker_ids_used=(),
                num_markers_detected=0,
                message="" if success else "marker insufficienti",
            )

    source = _FakeSource(n_frames=3)
    detector = _ScriptedDetector({1: _impact(0.0, 0.0)})

    result = _run(source, app_config, _FlakyCalibrator(), detector, repo, starting_score=501)

    # nessuna eccezione: il frame non calibrato viene semplicemente saltato
    assert len(result.turns) == 1


def test_on_turn_callback_is_invoked_per_turn(app_config, repo):
    source = _FakeSource(n_frames=3)
    detector = _ScriptedDetector({1: _impact(0.0, 0.0)})
    seen = []

    _run(
        source, app_config, _FakeCalibrator(), detector, repo,
        starting_score=501, on_turn=seen.append,
    )

    assert len(seen) == 1


def test_alternates_turns_between_two_players(app_config, repo):
    # 6 impatti -> 2 turni da 3 freccette ciascuno: uno per giocatore.
    source = _FakeSource(n_frames=7)
    detector = _ScriptedDetector({i: _impact(0.0, 0.0) for i in range(1, 7)})

    result = _run(source, app_config, _FakeCalibrator(), detector, repo, starting_score=501)

    assert [t.player_number for t in result.turns] == [1, 2]

    throws = repo.list_throws_for_game(result.game_id)
    assert {t.player_number for t in throws} == {1, 2}
    # ciascun giocatore e' al proprio primo turno
    assert all(t.turn_number == 1 for t in throws)


def test_game_record_stores_player_names(app_config, repo):
    source = _FakeSource(n_frames=1)
    detector = _ScriptedDetector({})

    result = _run(
        source, app_config, _FakeCalibrator(), detector, repo,
        player1_name="Alice", player2_name="Bob",
    )

    game = repo.get_game(result.game_id)
    assert game.player1_name == "Alice"
    assert game.player2_name == "Bob"
