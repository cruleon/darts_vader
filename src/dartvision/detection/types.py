"""Tipi di dato scambiati dal modulo di rilevamento impatto."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Impact:
    """Un impatto rilevato, gia' espresso sul piano raddrizzato.

    ``x_mm``/``y_mm`` sono le coordinate fisiche (origine = centro del
    bersaglio) usate dal livello di scoring. ``x_px``/``y_px`` sono le
    stesse coordinate in pixel del piano raddrizzato (utili per disegno
    e heatmap). ``area_px`` e' l'area (px^2 sul piano raddrizzato) della
    regione rilevata: non e' una probabilita', ma e' utile per debug e
    per capire quanto la detection era "solida".
    """

    x_mm: float
    y_mm: float
    x_px: float
    y_px: float
    area_px: float
