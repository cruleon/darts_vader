"""Frame sources: webcams, network streams, video files and still images."""
from __future__ import annotations

import glob
import threading
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

BACKENDS = {"msmf": cv2.CAP_MSMF, "dshow": cv2.CAP_DSHOW, "any": cv2.CAP_ANY}
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def open_camera(source: str, backend: str = "msmf", capture: str | None = None,
                fourcc: str | None = None) -> cv2.VideoCapture:
    """Open a webcam (index) or a network stream (URL), optionally requesting a resolution
    (``"1920x1080"``) and a pixel format (``"MJPG"``)."""
    if source.isdigit():
        cap = cv2.VideoCapture(int(source), BACKENDS[backend])
    elif "://" in source:
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera {source!r}")
    # Resolution first, then format: with DirectShow the opposite order leaves the camera on
    # uncompressed YUY2 at 1-5 fps.
    if capture:
        w, h = (int(v) for v in capture.lower().split("x"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def is_image_source(source: str) -> bool:
    return any(ch in source for ch in "*?") or source.lower().endswith(IMAGE_EXTENSIONS) or Path(source).is_dir()


class FrameSource:
    """Reads a source on a background thread and keeps only the newest frame, so slow
    processing skips frames instead of accumulating latency.

    ``source`` can be a webcam index (``"0"``), a stream URL, a video file (played at its nominal
    frame rate and looped), an image, a folder or a glob of images (each shown for ``hold``
    seconds), or any object with a ``cv2.VideoCapture``-like ``read()`` method.
    """

    def __init__(self, source, backend: str = "msmf", capture: str | None = None, fourcc: str | None = None,
                 hold: float = 6.0, image_fps: float = 15.0):
        self.source, self.backend, self.capture, self.fourcc = source, backend, capture, fourcc
        self.hold, self.image_fps = hold, image_fps
        self.error: str | None = None
        self.loops = 0  # how many times a video file has been rewound
        self._frame: np.ndarray | None = None
        self._seq = 0
        self._returned = 0
        self._running = True
        self._ended = False
        self._cond = threading.Condition()
        self._thread = threading.Thread(target=self._run, name="frame-source", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ consumers

    def latest(self, after_seq: int = 0, timeout: float = 1.0) -> tuple[np.ndarray, int] | None:
        """The newest frame with a sequence number greater than `after_seq`, waiting up to `timeout`."""
        with self._cond:
            self._cond.wait_for(lambda: self._seq > after_seq or not self._running, timeout=timeout)
            if self._seq <= after_seq or self._frame is None:
                return None
            return self._frame, self._seq

    def read(self, timeout: float = 5.0) -> tuple[bool, np.ndarray | None]:
        """``cv2.VideoCapture``-style read of the newest frame not returned yet."""
        with self._cond:
            self._cond.wait_for(lambda: self._seq > self._returned or self._ended or not self._running, timeout=timeout)
            if self._seq <= self._returned or self._frame is None:
                return False, None
            self._returned = self._seq
            return True, self._frame

    def stop(self) -> None:
        with self._cond:
            self._running = False
            self._cond.notify_all()
        self._thread.join(timeout=2.0)

    # ------------------------------------------------------------------ producer

    def _publish(self, frame: np.ndarray) -> None:
        with self._cond:
            self._frame = frame
            self._seq += 1
            self._cond.notify_all()

    def _run(self) -> None:
        try:
            src = self.source
            if not isinstance(src, str):
                self._run_capture(src)
            elif src.isdigit() or "://" in src:
                self._run_capture(open_camera(src, self.backend, self.capture, self.fourcc))
            elif is_image_source(src):
                self._run_images(src)
            else:
                self._run_video(src)
        except Exception as exc:  # reported to consumers through `error`
            self.error = str(exc)
            traceback.print_exc()
        finally:
            with self._cond:
                self._ended = True
                self._cond.notify_all()

    def _run_capture(self, cap) -> None:
        failures = 0
        try:
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    failures += 1
                    if failures > 100:
                        raise RuntimeError("the camera stopped sending frames")
                    time.sleep(0.02)
                    continue
                failures = 0
                self._publish(frame)
        finally:
            cap.release()

    def _run_video(self, path: str) -> None:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video {path!r}")
        period = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 30.0)
        next_t, read_any = time.perf_counter(), False
        try:
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    if not read_any:
                        raise RuntimeError(f"cannot read video {path!r}")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.loops += 1
                    read_any = False
                    continue
                read_any = True
                self._publish(frame)
                next_t += period
                delay = next_t - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -0.25:
                    next_t = time.perf_counter()  # decoding is slower than the video itself
        finally:
            cap.release()

    def _run_images(self, spec: str) -> None:
        files = sorted(glob.glob(str(Path(spec) / "*"))) if Path(spec).is_dir() else sorted(glob.glob(spec))
        images = [img for img in (cv2.imread(f) for f in files if f.lower().endswith(IMAGE_EXTENSIONS)) if img is not None]
        if not images:
            raise RuntimeError(f"no readable images in {spec!r}")
        while self._running:
            for img in images:
                t_end = time.perf_counter() + self.hold
                while self._running and time.perf_counter() < t_end:
                    self._publish(img)
                    time.sleep(1.0 / self.image_fps)
