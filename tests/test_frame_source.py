from __future__ import annotations

import cv2
import numpy as np
import pytest

from dartvision.input.frame_source import WebcamSource


class _FakeCapture:
    def __init__(self, opened: bool, fps: float, frames: list[np.ndarray]):
        self._opened = opened
        self._fps = fps
        self._frames = list(frames)
        self.released = False

    def isOpened(self) -> bool:
        return self._opened

    def get(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_FPS:
            return self._fps
        return 0.0

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self._frames:
            return False, None
        return True, self._frames.pop(0)

    def release(self) -> None:
        self.released = True


def test_raises_oserror_when_capture_not_opened():
    factory = lambda index: _FakeCapture(opened=False, fps=30.0, frames=[])

    with pytest.raises(OSError, match="webcam"):
        WebcamSource(device_index=0, capture_factory=factory)


def test_uses_reported_fps_when_valid():
    factory = lambda index: _FakeCapture(opened=True, fps=60.0, frames=[])

    source = WebcamSource(device_index=0, capture_factory=factory)

    assert source.fps == 60.0


def test_falls_back_to_default_fps_when_reported_is_zero():
    factory = lambda index: _FakeCapture(opened=True, fps=0.0, frames=[])

    source = WebcamSource(device_index=0, fallback_fps=24.0, capture_factory=factory)

    assert source.fps == 24.0


def test_reads_frames_until_exhausted_then_release():
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    fake_capture = _FakeCapture(opened=True, fps=30.0, frames=[frame])
    source = WebcamSource(device_index=0, capture_factory=lambda index: fake_capture)

    frames = list(source.frames())
    source.release()

    assert len(frames) == 1
    assert fake_capture.released is True
