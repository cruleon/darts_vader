"""Euristica per isolare la punta della freccetta da una regione rilevata.

Quando una freccetta si conficca nel bersaglio, la regione che compare
nel frame-diff comprende punta, fusto e alette: non solo il punto di
contatto. L'euristica adottata (indicata nelle specifiche del progetto)
e' scegliere, tra i punti della sagoma rilevata, quello piu' vicino al
centro del bersaglio: fusto e alette sporgono verso l'esterno/il
giocatore, mentre la punta e' l'estremo piu' "interno".

E' isolata in una funzione a se' stante (non un metodo privato del
detector) cosi' da poter essere sostituita con un'euristica diversa
(es. estremo della forma allungata) senza toccare il resto del
detector, e testata in isolamento.
"""

from __future__ import annotations

import numpy as np


def select_tip_point(
    points_plane: np.ndarray, center_plane: tuple[float, float]
) -> tuple[float, float]:
    """Sceglie, tra ``points_plane`` (Nx2, coordinate sul piano
    raddrizzato), quello piu' vicino a ``center_plane``."""
    if len(points_plane) == 0:
        raise ValueError("points_plane non puo' essere vuoto")

    points = np.asarray(points_plane, dtype=np.float64)
    center = np.asarray(center_plane, dtype=np.float64)
    distances = np.hypot(*(points - center).T)
    idx = int(np.argmin(distances))
    return float(points[idx, 0]), float(points[idx, 1])
