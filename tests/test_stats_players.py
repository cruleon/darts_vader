"""Match statistics and player photo storage."""
import base64
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from darts_vader.game.stats import DartRecord, TurnRecord, match_summary  # noqa: E402
from darts_vader.game.x01 import BUST, OK, WIN  # noqa: E402
from darts_vader.players import PHOTO_SIZE, PhotoStore, player_slug  # noqa: E402


def dart(label, score, number, multiplier, at_double=False, distance=None, x=0.0, y=0.0):
    return DartRecord(label, score, number, multiplier, x, y, at_double, distance)


def sample_match():
    turns = [
        TurnRecord(0, 1, 101, 62, OK, 3, [dart("T20", 60, 20, 3, x=0, y=-103), dart("S1", 1, 1, 1, x=20, y=-60),
                                          dart("S1", 1, 1, 1, x=22, y=-62)]),
        TurnRecord(1, 1, 101, 60, OK, 3, [dart("S20", 20, 20, 1)] * 3),
        TurnRecord(0, 1, 39, 39, WIN, 3, [dart("S19", 19, 19, 1, x=-40, y=120),
                                          dart("S10", 10, 10, 1, at_double=True, distance=9.0, x=90, y=-20),
                                          dart("D5", 10, 5, 2, at_double=True, distance=0.0, x=60, y=-150)]),
        TurnRecord(1, 2, 101, 0, BUST, 3, [dart("MISS", 0, 0, 0)]),
    ]
    return match_summary(["Leo", "Marco"], turns, [1, 0], 101, True, 1, 0)


def test_player_statistics():
    summary = sample_match()
    assert summary["winner"] == 0 and summary["legs_played"] == 2
    leo, marco = summary["players"]
    assert (leo["points"], leo["darts_thrown"], leo["average"]) == (101, 6, 50.5)
    assert leo["first9_average"] == 50.5 and leo["highest_turn"] == 62 and leo["scores_60"] == 1
    assert (leo["highest_checkout"], leo["best_leg_darts"]) == (39, 6)
    assert (leo["darts_at_double"], leo["checkout_pct"]) == (2, 50.0)
    assert (leo["trebles"], leo["doubles"], leo["singles"], leo["misses"]) == (1, 1, 4, 0)
    assert (leo["favourite"], leo["favourite_hits"]) == ("1", 2)
    assert (leo["avg_finish_distance_mm"], leo["best_finish_distance_mm"]) == (4.5, 0.0)
    assert len(leo["darts"]) == 6 and leo["grouping_mm"] > 0
    assert (marco["average"], marco["busts"], marco["misses"]) == (30.0, 1, 1)
    assert marco["highest_checkout"] is None and marco["checkout_pct"] is None


def test_straight_out_has_no_checkout_percentage():
    turns = [TurnRecord(0, 1, 40, 40, WIN, 1, [dart("D20", 40, 20, 2, at_double=True)])]
    stats = match_summary(["Solo"], turns, [1], 40, False, 1, 0)["players"][0]
    assert stats["checkout_pct"] is None and stats["darts_at_double"] is None
    assert stats["average"] == 120.0 and stats["centroid_mm"] is None


def test_player_slug_is_safe_and_stable():
    assert player_slug("Leo") == player_slug(" leo ")
    assert player_slug("Leo") != player_slug("Marco")
    assert all(c.isalnum() or c == "-" for c in player_slug("../../evil name/..."))


def test_photo_store(tmp_path):
    store = PhotoStore(tmp_path / "photos")
    assert store.url("Leo") is None
    image = np.zeros((300, 500, 3), np.uint8)
    ok, png = cv2.imencode(".png", image)
    store.save("Leo", "data:image/png;base64," + base64.b64encode(png.tobytes()).decode())
    url = store.url("Leo")
    assert url.startswith("/api/player-photo/") and store.path("Leo").exists()
    assert cv2.imread(str(store.path("Leo"))).shape == (PHOTO_SIZE, PHOTO_SIZE, 3)
    assert store.file(store.path("Leo").name) == store.path("Leo")
    assert store.file("../secret.jpg") is None and store.file("missing-1234.jpg") is None
    with pytest.raises(ValueError):
        store.save("Leo", "data:image/png;base64,bm90IGFuIGltYWdl")
    store.delete("Leo")
    assert store.url("Leo") is None
