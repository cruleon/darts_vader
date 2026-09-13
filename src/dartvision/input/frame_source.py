"""Sorgenti di frame (video registrato oggi, webcam live domani).

``FrameSource`` e' l'unica interfaccia che il resto della pipeline
conosce: calibrazione e detection non sanno mai se i frame arrivano da
un file mp4 o da una webcam. Questo rende l'estensione a webcam live
un semplice "aggiungi una nuova implementazione", senza toccare nessun
altro modulo.
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Self

import cv2
import numpy as np


class FrameSource(ABC):
    """Interfaccia comune per qualunque sorgente di frame video."""

    @property
    @abstractmethod
    def fps(self) -> float:
        """Frame per secondo della sorgente (stimati, se necessario)."""

    @abstractmethod
    def read(self) -> tuple[bool, np.ndarray | None]:
        """Legge il prossimo frame.

        Ritorna ``(True, frame)`` se disponibile, ``(False, None)``
        quando la sorgente e' esaurita (fine video) o non disponibile.
        Non solleva eccezioni per la normale fine dello stream.
        """

    @abstractmethod
    def release(self) -> None:
        """Rilascia le risorse sottostanti (file handle, device...)."""

    def frames(self) -> Iterator[np.ndarray]:
        """Itera sui frame finche' la sorgente non e' esaurita."""
        while True:
            ok, frame = self.read()
            if not ok or frame is None:
                return
            yield frame

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()


class VideoFileSource(FrameSource):
    """Legge frame da un file video registrato (mp4, avi, ...)."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        if not self._path.is_file():
            raise FileNotFoundError(f"Video non trovato: {self._path}")

        self._capture = cv2.VideoCapture(str(self._path))
        if not self._capture.isOpened():
            raise OSError(f"Impossibile aprire il video: {self._path}")

        self._fps = self._capture.get(cv2.CAP_PROP_FPS) or 0.0

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int:
        return int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def read(self) -> tuple[bool, np.ndarray | None]:
        ok, frame = self._capture.read()
        if not ok:
            return False, None
        return True, frame

    def release(self) -> None:
        self._capture.release()


class WebcamSource(FrameSource):
    """Legge frame live da una webcam (``cv2.VideoCapture``).

    ``capture_factory`` e' iniettabile per poter testare la classe senza
    aprire un device reale (un test passa un doppio che simula una
    ``VideoCapture`` gia' aperta o non disponibile).
    """

    def __init__(
        self,
        device_index: int = 0,
        fallback_fps: float = 30.0,
        capture_factory: Callable[[int], cv2.VideoCapture] | None = None,
    ) -> None:
        if capture_factory is None:
            # Su Windows CAP_DSHOW apre la webcam piu' velocemente e in
            # modo piu' affidabile del backend di default.
            backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
            capture_factory = lambda index: cv2.VideoCapture(index, backend)

        self._capture = capture_factory(device_index)
        if not self._capture.isOpened():
            raise OSError(f"Impossibile aprire la webcam (device_index={device_index})")

        reported_fps = self._capture.get(cv2.CAP_PROP_FPS)
        self._fps = reported_fps if reported_fps and reported_fps > 0 else fallback_fps

    @property
    def fps(self) -> float:
        return self._fps

    def read(self) -> tuple[bool, np.ndarray | None]:
        ok, frame = self._capture.read()
        if not ok:
            return False, None
        return True, frame

    def release(self) -> None:
        self._capture.release()
