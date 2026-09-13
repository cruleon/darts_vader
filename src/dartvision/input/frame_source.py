"""Sorgenti di frame (video registrato oggi, webcam live domani).

``FrameSource`` e' l'unica interfaccia che il resto della pipeline
conosce: calibrazione e detection non sanno mai se i frame arrivano da
un file mp4 o da una webcam. Questo rende l'estensione a webcam live
un semplice "aggiungi una nuova implementazione", senza toccare nessun
altro modulo.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
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
