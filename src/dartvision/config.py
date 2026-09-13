"""Caricamento tipizzato della configurazione fisica del setup.

Tutta la geometria fisica (bersaglio, marker ArUco, soglie di
detection) vive in un file YAML esterno (di norma ``config/board_config.yaml``).
Questo modulo si limita a leggerlo e validarlo in dataclass tipizzate,
cosi' il resto del codice non contiene mai numeri "magici" di geometria.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


class ConfigError(ValueError):
    """Configurazione fisica mancante, incompleta o incoerente."""


@dataclass(frozen=True)
class BoardConfig:
    inner_bull_radius_mm: float
    outer_bull_radius_mm: float
    triple_inner_radius_mm: float
    triple_outer_radius_mm: float
    double_inner_radius_mm: float
    double_outer_radius_mm: float
    sector_count: int
    sector_order: tuple[int, ...]
    sector0_offset_deg: float


@dataclass(frozen=True)
class ArucoMarkerConfig:
    id: int
    x_mm: float
    y_mm: float


@dataclass(frozen=True)
class ArucoConfig:
    dictionary: str
    marker_length_mm: float
    markers: tuple[ArucoMarkerConfig, ...]


@dataclass(frozen=True)
class RectifiedPlaneConfig:
    size_px: int
    half_extent_mm: float

    @property
    def mm_per_px(self) -> float:
        return (2.0 * self.half_extent_mm) / self.size_px

    def mm_to_px(self, x_mm: float, y_mm: float) -> tuple[float, float]:
        """Converte coordinate fisiche (origine=centro bersaglio) in
        coordinate pixel sul piano raddrizzato (origine=angolo alto-sx)."""
        scale = self.size_px / (2.0 * self.half_extent_mm)
        return (
            self.size_px / 2.0 + x_mm * scale,
            self.size_px / 2.0 + y_mm * scale,
        )

    def px_to_mm(self, x_px: float, y_px: float) -> tuple[float, float]:
        """Inversa di :meth:`mm_to_px`."""
        return (
            (x_px - self.size_px / 2.0) * self.mm_per_px,
            (y_px - self.size_px / 2.0) * self.mm_per_px,
        )


@dataclass(frozen=True)
class CalibrationConfig:
    min_markers_required: int


@dataclass(frozen=True)
class DetectionConfig:
    diff_threshold: int
    morph_kernel_size: int
    min_impact_area_px: float
    max_impact_area_px: float


@dataclass(frozen=True)
class MLDetectionConfig:
    model_path: str
    confidence_threshold: float
    device: str  # "auto" | "cpu" | "cuda"
    roi_padding_px: float
    candidate_min_area_px: float
    candidate_max_area_px: float


@dataclass(frozen=True)
class AppConfig:
    board: BoardConfig
    aruco: ArucoConfig
    rectified_plane: RectifiedPlaneConfig
    calibration: CalibrationConfig
    detection: DetectionConfig
    ml_detection: MLDetectionConfig


def _require(mapping: dict, key: str, context: str) -> object:
    if key not in mapping:
        raise ConfigError(f"Campo mancante '{key}' in '{context}'")
    return mapping[key]


def _build_board(raw: dict) -> BoardConfig:
    sector_order = tuple(int(v) for v in _require(raw, "sector_order", "board"))
    sector_count = int(_require(raw, "sector_count", "board"))
    if len(sector_order) != sector_count:
        raise ConfigError(
            f"board.sector_order ha {len(sector_order)} elementi, "
            f"attesi sector_count={sector_count}"
        )
    if len(set(sector_order)) != len(sector_order):
        raise ConfigError("board.sector_order contiene numeri duplicati")

    board = BoardConfig(
        inner_bull_radius_mm=float(_require(raw, "inner_bull_radius_mm", "board")),
        outer_bull_radius_mm=float(_require(raw, "outer_bull_radius_mm", "board")),
        triple_inner_radius_mm=float(_require(raw, "triple_inner_radius_mm", "board")),
        triple_outer_radius_mm=float(_require(raw, "triple_outer_radius_mm", "board")),
        double_inner_radius_mm=float(_require(raw, "double_inner_radius_mm", "board")),
        double_outer_radius_mm=float(_require(raw, "double_outer_radius_mm", "board")),
        sector_count=sector_count,
        sector_order=sector_order,
        sector0_offset_deg=float(_require(raw, "sector0_offset_deg", "board")),
    )

    radii = [
        board.inner_bull_radius_mm,
        board.outer_bull_radius_mm,
        board.triple_inner_radius_mm,
        board.triple_outer_radius_mm,
        board.double_inner_radius_mm,
        board.double_outer_radius_mm,
    ]
    if radii != sorted(radii):
        raise ConfigError(
            "I raggi degli anelli in 'board' devono essere strettamente "
            "crescenti dal centro verso l'esterno"
        )
    return board


def _build_aruco(raw: dict) -> ArucoConfig:
    markers_raw = _require(raw, "markers", "aruco")
    markers = tuple(
        ArucoMarkerConfig(
            id=int(_require(m, "id", "aruco.markers[]")),
            x_mm=float(_require(m, "x_mm", "aruco.markers[]")),
            y_mm=float(_require(m, "y_mm", "aruco.markers[]")),
        )
        for m in markers_raw
    )
    ids = [m.id for m in markers]
    if len(set(ids)) != len(ids):
        raise ConfigError("aruco.markers contiene id duplicati")
    if len(markers) < 4:
        raise ConfigError(
            f"Servono almeno 4 marker configurati, trovati {len(markers)}"
        )
    return ArucoConfig(
        dictionary=str(_require(raw, "dictionary", "aruco")),
        marker_length_mm=float(_require(raw, "marker_length_mm", "aruco")),
        markers=markers,
    )


def _build_rectified_plane(raw: dict) -> RectifiedPlaneConfig:
    return RectifiedPlaneConfig(
        size_px=int(_require(raw, "size_px", "rectified_plane")),
        half_extent_mm=float(_require(raw, "half_extent_mm", "rectified_plane")),
    )


def _build_calibration(raw: dict) -> CalibrationConfig:
    min_markers = int(_require(raw, "min_markers_required", "calibration"))
    if min_markers < 4:
        raise ConfigError("calibration.min_markers_required deve essere >= 4")
    return CalibrationConfig(min_markers_required=min_markers)


def _build_detection(raw: dict) -> DetectionConfig:
    min_area = float(_require(raw, "min_impact_area_px", "detection"))
    max_area = float(_require(raw, "max_impact_area_px", "detection"))
    if max_area <= min_area:
        raise ConfigError(
            "detection.max_impact_area_px deve essere maggiore di "
            "detection.min_impact_area_px"
        )
    return DetectionConfig(
        diff_threshold=int(_require(raw, "diff_threshold", "detection")),
        morph_kernel_size=int(_require(raw, "morph_kernel_size", "detection")),
        min_impact_area_px=min_area,
        max_impact_area_px=max_area,
    )


_VALID_ML_DEVICES = {"auto", "cpu", "cuda"}


def _build_ml_detection(raw: dict) -> MLDetectionConfig:
    confidence_threshold = float(_require(raw, "confidence_threshold", "ml_detection"))
    if not (0.0 < confidence_threshold <= 1.0):
        raise ConfigError("ml_detection.confidence_threshold deve essere in (0, 1]")

    device = str(_require(raw, "device", "ml_detection"))
    if device not in _VALID_ML_DEVICES:
        raise ConfigError(
            f"ml_detection.device deve essere uno tra {sorted(_VALID_ML_DEVICES)}, trovato '{device}'"
        )

    roi_padding_px = float(_require(raw, "roi_padding_px", "ml_detection"))
    if roi_padding_px < 0:
        raise ConfigError("ml_detection.roi_padding_px deve essere >= 0")

    min_area = float(_require(raw, "candidate_min_area_px", "ml_detection"))
    max_area = float(_require(raw, "candidate_max_area_px", "ml_detection"))
    if max_area <= min_area:
        raise ConfigError(
            "ml_detection.candidate_max_area_px deve essere maggiore di "
            "ml_detection.candidate_min_area_px"
        )

    return MLDetectionConfig(
        model_path=str(_require(raw, "model_path", "ml_detection")),
        confidence_threshold=confidence_threshold,
        device=device,
        roi_padding_px=roi_padding_px,
        candidate_min_area_px=min_area,
        candidate_max_area_px=max_area,
    )


def load_config(path: str | Path) -> AppConfig:
    """Legge e valida il file YAML di configurazione fisica del setup."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"File di configurazione non trovato: {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ConfigError(f"Contenuto YAML non valido in {path}")

    board = _build_board(_require(raw, "board", "root"))
    aruco = _build_aruco(_require(raw, "aruco", "root"))
    rectified_plane = _build_rectified_plane(_require(raw, "rectified_plane", "root"))
    calibration = _build_calibration(_require(raw, "calibration", "root"))
    detection = _build_detection(_require(raw, "detection", "root"))
    ml_detection = _build_ml_detection(_require(raw, "ml_detection", "root"))

    if aruco.markers and len(aruco.markers) < calibration.min_markers_required:
        raise ConfigError(
            f"Configurati solo {len(aruco.markers)} marker ma "
            f"calibration.min_markers_required={calibration.min_markers_required}"
        )

    max_marker_dist = max(
        (m.x_mm**2 + m.y_mm**2) ** 0.5 for m in aruco.markers
    )
    if max_marker_dist > rectified_plane.half_extent_mm:
        raise ConfigError(
            "rectified_plane.half_extent_mm "
            f"({rectified_plane.half_extent_mm}) e' piu' piccolo della "
            f"distanza dal centro del marker piu' lontano ({max_marker_dist:.1f} mm): "
            "i marker cadrebbero fuori dal piano raddrizzato"
        )

    return AppConfig(
        board=board,
        aruco=aruco,
        rectified_plane=rectified_plane,
        calibration=calibration,
        detection=detection,
        ml_detection=ml_detection,
    )
