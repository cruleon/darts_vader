"""Connessione e schema SQLite."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    starting_score INTEGER NOT NULL,
    player1_name TEXT NOT NULL,
    player2_name TEXT NOT NULL,
    winner_player_number INTEGER,
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS throws (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id INTEGER NOT NULL REFERENCES games(id),
    player_number INTEGER NOT NULL,
    turn_number INTEGER NOT NULL,
    throw_number INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    x_mm REAL NOT NULL,
    y_mm REAL NOT NULL,
    angle_deg REAL NOT NULL,
    radius_mm REAL NOT NULL,
    sector INTEGER,
    ring TEXT NOT NULL,
    multiplier INTEGER NOT NULL,
    points INTEGER NOT NULL,
    is_bust INTEGER NOT NULL,
    is_checkout INTEGER NOT NULL,
    closing_distance_mm REAL,
    remaining_before INTEGER NOT NULL,
    remaining_after INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_throws_game_id ON throws(game_id);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    """Apre (creandolo se serve) il database SQLite e assicura lo schema.

    ``check_same_thread=False``: la dashboard Streamlit riusa la stessa
    connessione cache-ata da thread diversi (``st.fragment`` gira sul
    proprio thread di rerun). L'app e' locale, single-user, con accessi
    sostanzialmente sequenziali: nessun bisogno di un pool o di un lock
    esplicito.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn
