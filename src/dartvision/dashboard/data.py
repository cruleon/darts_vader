"""Accesso ai dati per la dashboard: solo lettura dal database.

Non conosce nulla della pipeline CV: legge esclusivamente cio' che
``persistence`` ha gia' salvato. La connessione e' cache-ata per
processo Streamlit; le query hanno un TTL breve cosi' l'aggiornamento
"live" (``st.fragment(run_every=...)``) vede i nuovi tiri scritti da
un processo esterno (es. ``process_video.py`` o ``seed_demo_data.py``).
"""

from __future__ import annotations

import sqlite3

import streamlit as st

from dartvision.config import AppConfig, load_config
from dartvision.persistence.db import connect
from dartvision.persistence.models import GameRecord, ThrowRecord
from dartvision.persistence.repository import DartRepository

DEFAULT_CONFIG_PATH = "config/board_config.yaml"
DEFAULT_DB_PATH = "data/dartvision.db"


@st.cache_resource
def get_config(config_path: str = DEFAULT_CONFIG_PATH) -> AppConfig:
    return load_config(config_path)


@st.cache_resource
def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    return connect(db_path)


def get_repository(db_path: str = DEFAULT_DB_PATH) -> DartRepository:
    return DartRepository(get_connection(db_path))


@st.cache_data(ttl=1.0)
def list_games(db_path: str = DEFAULT_DB_PATH) -> list[GameRecord]:
    return get_repository(db_path).list_games()


@st.cache_data(ttl=1.0)
def list_throws_for_game(game_id: int, db_path: str = DEFAULT_DB_PATH) -> list[ThrowRecord]:
    return get_repository(db_path).list_throws_for_game(game_id)


@st.cache_data(ttl=1.0)
def list_all_throws(db_path: str = DEFAULT_DB_PATH) -> list[ThrowRecord]:
    return get_repository(db_path).list_all_throws()
