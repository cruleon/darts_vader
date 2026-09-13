"""Wrapper sottile su ``cv2.aruco`` per il rilevamento dei marker."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from dartvision.config import ArucoConfig


def resolve_dictionary(name: str) -> cv2.aruco.Dictionary:
    """Risolve il nome di un dizionario ArUco (es. 'DICT_4X4_50') nel
    dizionario OpenCV corrispondente."""
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"Dizionario ArUco sconosciuto: '{name}'")
    dict_id = getattr(cv2.aruco, name)
    return cv2.aruco.getPredefinedDictionary(dict_id)


@dataclass(frozen=True)
class DetectedMarkers:
    """Risultato del rilevamento marker su un singolo frame."""

    # id marker -> 4 angoli (ordine: alto-sx, alto-dx, basso-dx, basso-sx),
    # in coordinate immagine (pixel).
    corners_by_id: dict[int, np.ndarray] = field(default_factory=dict)

    @property
    def ids(self) -> list[int]:
        return list(self.corners_by_id.keys())

    def center(self, marker_id: int) -> tuple[float, float]:
        """Centro (pixel immagine) del marker, media dei suoi 4 angoli."""
        corners = self.corners_by_id[marker_id]
        cx, cy = corners.mean(axis=0)
        return float(cx), float(cy)

    def centers(self) -> dict[int, tuple[float, float]]:
        return {marker_id: self.center(marker_id) for marker_id in self.ids}


class ArucoDetector:
    """Rileva marker ArUco in un frame usando l'API moderna di OpenCV."""

    def __init__(self, config: ArucoConfig) -> None:
        dictionary = resolve_dictionary(config.dictionary)
        parameters = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    def detect(self, frame: np.ndarray) -> DetectedMarkers:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        corners, ids, _rejected = self._detector.detectMarkers(gray)

        corners_by_id: dict[int, np.ndarray] = {}
        if ids is not None:
            for marker_corners, marker_id in zip(corners, ids.flatten()):
                # marker_corners ha shape (1, 4, 2)
                corners_by_id[int(marker_id)] = marker_corners.reshape(4, 2)

        return DetectedMarkers(corners_by_id=corners_by_id)
