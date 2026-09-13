"""Interfaccia del rilevatore di impatti.

Qualunque implementazione (frame-differencing oggi, un domani un
modello ML) rispetta questa firma: due frame grezzi della camera + la
homography corrente in ingresso, un ``Impact`` sul piano raddrizzato
(o ``None``) in uscita. Ne' la pipeline ne' lo scoring devono mai
sapere COME l'impatto e' stato trovato.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from dartvision.detection.types import Impact


class ImpactDetector(ABC):
    @abstractmethod
    def detect(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        homography: np.ndarray,
    ) -> Impact | None:
        """Confronta due frame grezzi della camera e cerca un nuovo impatto.

        ``homography`` mappa coordinate immagine -> coordinate del
        piano raddrizzato (la stessa prodotta da ``Calibrator``).
        Ritorna ``None`` se non viene trovato nulla di plausibile: non
        e' un errore, e' l'esito normale della maggior parte dei
        confronti tra frame consecutivi.
        """
