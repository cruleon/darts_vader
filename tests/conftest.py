from __future__ import annotations

from pathlib import Path

import pytest

from dartvision.config import load_config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "board_config.yaml"


@pytest.fixture
def app_config():
    return load_config(CONFIG_PATH)
