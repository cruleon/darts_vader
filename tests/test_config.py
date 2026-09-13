from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dartvision.config import ConfigError, load_config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "board_config.yaml"


def test_load_real_config_is_valid():
    config = load_config(CONFIG_PATH)
    assert config.board.sector_count == 20
    assert len(config.aruco.markers) >= 4


def _load_raw() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _write_and_load(tmp_path: Path, raw: dict):
    path = tmp_path / "board_config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_config(path)


def test_missing_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "does_not_exist.yaml")


def test_sector_order_length_mismatch_raises(tmp_path):
    raw = _load_raw()
    raw["board"]["sector_order"] = raw["board"]["sector_order"][:-1]
    with pytest.raises(ConfigError, match="sector_order"):
        _write_and_load(tmp_path, raw)


def test_duplicate_sector_numbers_raise(tmp_path):
    raw = _load_raw()
    raw["board"]["sector_order"][1] = raw["board"]["sector_order"][0]
    with pytest.raises(ConfigError, match="duplicati"):
        _write_and_load(tmp_path, raw)


def test_non_increasing_radii_raise(tmp_path):
    raw = _load_raw()
    raw["board"]["triple_outer_radius_mm"] = raw["board"]["double_outer_radius_mm"] + 1
    with pytest.raises(ConfigError, match="crescenti"):
        _write_and_load(tmp_path, raw)


def test_too_few_markers_raise(tmp_path):
    raw = _load_raw()
    raw["aruco"]["markers"] = raw["aruco"]["markers"][:3]
    with pytest.raises(ConfigError, match="almeno 4 marker"):
        _write_and_load(tmp_path, raw)


def test_duplicate_marker_ids_raise(tmp_path):
    raw = _load_raw()
    raw["aruco"]["markers"][1]["id"] = raw["aruco"]["markers"][0]["id"]
    with pytest.raises(ConfigError, match="id duplicati"):
        _write_and_load(tmp_path, raw)


def test_marker_outside_plane_extent_raises(tmp_path):
    raw = _load_raw()
    raw["rectified_plane"]["half_extent_mm"] = 10.0
    with pytest.raises(ConfigError, match="half_extent_mm"):
        _write_and_load(tmp_path, raw)


def test_real_config_has_ml_detection_section():
    config = load_config(CONFIG_PATH)
    assert config.ml_detection.device in {"auto", "cpu", "cuda"}
    assert 0.0 < config.ml_detection.confidence_threshold <= 1.0


def test_missing_ml_detection_section_raises(tmp_path):
    raw = _load_raw()
    del raw["ml_detection"]
    with pytest.raises(ConfigError, match="ml_detection"):
        _write_and_load(tmp_path, raw)


def test_ml_detection_invalid_device_raises(tmp_path):
    raw = _load_raw()
    raw["ml_detection"]["device"] = "tpu"
    with pytest.raises(ConfigError, match="device"):
        _write_and_load(tmp_path, raw)


def test_ml_detection_confidence_threshold_out_of_range_raises(tmp_path):
    raw = _load_raw()
    raw["ml_detection"]["confidence_threshold"] = 1.5
    with pytest.raises(ConfigError, match="confidence_threshold"):
        _write_and_load(tmp_path, raw)


def test_ml_detection_non_increasing_candidate_areas_raise(tmp_path):
    raw = _load_raw()
    raw["ml_detection"]["candidate_max_area_px"] = raw["ml_detection"]["candidate_min_area_px"]
    with pytest.raises(ConfigError, match="candidate_max_area_px"):
        _write_and_load(tmp_path, raw)


def test_ml_detection_negative_roi_padding_raises(tmp_path):
    raw = _load_raw()
    raw["ml_detection"]["roi_padding_px"] = -1.0
    with pytest.raises(ConfigError, match="roi_padding_px"):
        _write_and_load(tmp_path, raw)
