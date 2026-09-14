"""Tip model views (homographies) and the label file format."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from darts_vader.board import BoardState, geometry as g  # noqa: E402
from darts_vader.board.synthetic import SyntheticScene  # noqa: E402
from darts_vader.labels import LABELLED, NO_TIPS, PENDING, LabelSession, tip_record  # noqa: E402
from darts_vader.tips.views import VIEWS, render_view, view_geometry  # noqa: E402


@pytest.mark.parametrize("view", VIEWS)
def test_view_homography_points_at_the_right_colours(view):
    frame, H_true = SyntheticScene(seed=2).render(yaw=35, pitch=15, roll=5)
    state = BoardState(H_true, 1.0, 0.0, 0, False)
    image, px_to_mm = render_view(frame, state, view)
    to_px = np.linalg.inv(px_to_mm)

    def pixel(r, deg):
        u, v = np.round(g.apply_homography(to_px, g.model_points(r, deg))[0]).astype(int)
        return image[v, u].astype(int)

    b, gr, r = pixel(103.0, 0.0)  # treble 20: red
    assert r > 120 and gr < 100 and b < 100
    b, gr, r = pixel(60.0, 18.0)  # single 1: cream
    assert min(b, gr, r) > 140


def test_view_tilt():
    frontal, H_frontal = SyntheticScene(seed=3).render()
    steep, H_steep = SyntheticScene(seed=3).render(yaw=45)
    assert view_geometry(H_frontal)[1] < 8.0
    assert 35.0 < view_geometry(H_steep)[1] < 55.0


def test_label_session_roundtrip(tmp_path):
    board = BoardState(np.eye(3), 1.0, 0.0, 0, False)
    session = LabelSession(tmp_path / "session", {"game": 301})
    assert not session.dir.exists()  # nothing is written before the first photo
    frame = np.zeros((8, 8, 3), np.uint8)
    entry = session.add_photo(frame, board, [tip_record([10.0, -20.0], board, "S20", "click")], LABELLED)
    session.add_photo(frame, board, [], NO_TIPS)

    reloaded = LabelSession(tmp_path / "session")
    assert reloaded.count(LABELLED) == 1 and reloaded.count(NO_TIPS) == 1
    assert reloaded.photos[0]["tips"][0]["tip_mm"] == [10.0, -20.0]
    assert (reloaded.dir / entry["image"]).exists()
    assert np.allclose(reloaded.board(reloaded.photos[0]).H, np.eye(3))


def test_legacy_session_is_converted(tmp_path):
    legacy = {"camera": {}, "photos": [
        {"index": 1, "image": "foto_0001.jpg", "turn": 1, "board_H": np.eye(3).tolist(), "board_refit": True,
         "tips_present": [], "new_tips": [{"tip_mm": [1, 2], "score": "S20", "origin": "proposta"}],
         "status": "etichettata"},
        {"index": 2, "image": "foto_0002.jpg", "board_H": np.eye(3).tolist(), "tips_present": [], "new_tips": [],
         "status": "da etichettare"},
    ]}
    (tmp_path / "labels.json").write_text(json.dumps(legacy), encoding="utf-8")
    session = LabelSession(tmp_path)
    first, second = session.photos
    assert first["status"] == LABELLED and first["tips"][0]["origin"] == "proposal"
    assert second["status"] == PENDING and second["tips"] == []
    assert "new_tips" not in first and "turn" not in first
    assert session.board(second).rings == g.RING_RADII
