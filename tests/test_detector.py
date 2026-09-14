"""Board detector accuracy on synthetic scenes with ground truth.

    python -m pytest tests                  # run the tests
    python tests/test_detector.py           # detailed accuracy report
"""
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from darts_vader.board import DartboardDetector, geometry as g  # noqa: E402
from darts_vader.board.synthetic import SyntheticScene, SyntheticVideo  # noqa: E402

POSES = [
    dict(yaw=0, pitch=0, roll=0),
    dict(yaw=30, pitch=0, roll=0),
    dict(yaw=0, pitch=35, roll=0),
    dict(yaw=-45, pitch=20, roll=5),
    dict(yaw=55, pitch=0, roll=-10),
    dict(yaw=-40, pitch=-30, roll=10),
    dict(yaw=60, pitch=15, roll=0),
    dict(yaw=20, pitch=50, roll=-5),
    dict(yaw=0, pitch=0, roll=0, distance=1600),
    dict(yaw=35, pitch=-20, roll=0, distance=500, offset=(-120, 60)),
]


def board_errors(H_est, H_true, rng, n=2000):
    """Error (mm) and sector agreement for random points on the board."""
    r = g.R_DOUBLE_OUT * np.sqrt(rng.uniform(0, 1, n))
    pts = g.model_points(r, rng.uniform(0, 360, n))
    img = g.apply_homography(np.linalg.inv(H_true), pts)
    est = g.apply_homography(H_est, img)
    err = np.linalg.norm(est - pts, axis=1)
    same = np.mean([g.score_model_point(*a) == g.score_model_point(*b) for a, b in zip(pts, est)])
    return err, same


@pytest.mark.parametrize("pose", POSES, ids=lambda p: "_".join(f"{k}{v}" for k, v in p.items()))
def test_single_frame(pose):
    frame, H_true = SyntheticScene(seed=1).render(**pose)
    state = DartboardDetector().process(frame)
    assert state is not None, "board not found"
    err, same = board_errors(state.H, H_true, np.random.default_rng(0))
    assert err.mean() < 1.5 and err.max() < 4.0, f"mean error {err.mean():.2f} mm, max {err.max():.2f} mm"
    # about 6.6 m of boundaries over ~908 cm²: with errors up to 1.5 mm a few % of points change sector
    assert same > 0.95


def test_video_tracking():
    video, detector = SyntheticVideo(), DartboardDetector()
    rng = np.random.default_rng(0)
    found, errs = 0, []
    for _ in range(90):
        _, frame = video.read()
        state = detector.process(frame)
        if state is not None:
            found += 1
            errs.append(board_errors(state.H, video.last_truth, rng, 300)[0].mean())
    assert found >= 85
    assert np.mean(errs) < 1.5


def test_orientation_helpers():
    frame, H_true = SyntheticScene(seed=1).render(yaw=25, pitch=10)
    state = DartboardDetector().process(frame)
    assert state is not None
    rotated = state.rotated(3)
    assert rotated.aligned_to(state).hit(state.to_image([(0.0, -130.0)])[0]).number == 20
    half = state.scaled(0.5)
    p = state.to_image([(40.0, -80.0)])[0]
    assert np.allclose(half.to_image([(40.0, -80.0)])[0], p * 0.5)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    scene = SyntheticScene(seed=1)
    print(f"{'pose':<55} {'mean mm':>8} {'max mm':>7} {'sectors':>8} {'conf':>5} {'ms':>6}")
    for pose in POSES:
        frame, H_true = scene.render(**pose)
        detector = DartboardDetector()
        t0 = time.perf_counter()
        state = detector.process(frame)
        ms = (time.perf_counter() - t0) * 1000
        if state is None:
            print(f"{str(pose):<55} {'NOT FOUND':>8}")
            continue
        err, same = board_errors(state.H, H_true, rng)
        print(f"{str(pose):<55} {err.mean():8.2f} {err.max():7.2f} {same:8.3f} {state.confidence:5.2f} {ms:6.1f}")

    video, detector = SyntheticVideo(), DartboardDetector()
    found, errs, times = 0, [], []
    for _ in range(300):
        _, frame = video.read()
        t0 = time.perf_counter()
        state = detector.process(frame)
        times.append(time.perf_counter() - t0)
        if state is not None:
            found += 1
            errs.append(board_errors(state.H, video.last_truth, rng, 300)[0].mean())
    print(f"\nvideo: board found in {found}/300 frames, mean error {np.mean(errs):.2f} mm, "
          f"{1000 * np.mean(times):.1f} ms/frame ({1 / np.mean(times):.0f} fps)")
