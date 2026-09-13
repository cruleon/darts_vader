"""Repository CRUD per partite e tiri.

Nessuna logica di gioco qui: quella vive nel Livello 3
(``scoring.game_501`` / ``scoring.match_501``). Questo modulo si limita a
salvare e rileggere cio' che quel livello produce.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime

from dartvision.persistence.models import (
    GameRecord,
    ThrowRecord,
    throw_record_from_outcome,
)
from dartvision.scoring.game_501 import TurnOutcome


def _row_to_game(row: sqlite3.Row) -> GameRecord:
    return GameRecord(
        id=row["id"],
        started_at=datetime.fromisoformat(row["started_at"]),
        starting_score=row["starting_score"],
        player1_name=row["player1_name"],
        player2_name=row["player2_name"],
        winner_player_number=row["winner_player_number"],
        finished_at=datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None,
    )


def _row_to_throw(row: sqlite3.Row) -> ThrowRecord:
    return ThrowRecord(
        id=row["id"],
        game_id=row["game_id"],
        player_number=row["player_number"],
        turn_number=row["turn_number"],
        throw_number=row["throw_number"],
        timestamp=datetime.fromisoformat(row["timestamp"]),
        x_mm=row["x_mm"],
        y_mm=row["y_mm"],
        angle_deg=row["angle_deg"],
        radius_mm=row["radius_mm"],
        sector=row["sector"],
        ring=row["ring"],
        multiplier=row["multiplier"],
        points=row["points"],
        is_bust=bool(row["is_bust"]),
        is_checkout=bool(row["is_checkout"]),
        closing_distance_mm=row["closing_distance_mm"],
        remaining_before=row["remaining_before"],
        remaining_after=row["remaining_after"],
    )


class DartRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def create_game(
        self,
        starting_score: int,
        player1_name: str,
        player2_name: str,
        started_at: datetime | None = None,
    ) -> int:
        started_at = started_at or datetime.now()
        cur = self._conn.execute(
            """
            INSERT INTO games (started_at, starting_score, player1_name, player2_name, winner_player_number, finished_at)
            VALUES (?, ?, ?, ?, NULL, NULL)
            """,
            (started_at.isoformat(), starting_score, player1_name, player2_name),
        )
        self._conn.commit()
        assert cur.lastrowid is not None
        return cur.lastrowid

    def finish_game(
        self, game_id: int, winner_player_number: int, finished_at: datetime | None = None
    ) -> None:
        finished_at = finished_at or datetime.now()
        self._conn.execute(
            "UPDATE games SET finished_at = ?, winner_player_number = ? WHERE id = ?",
            (finished_at.isoformat(), winner_player_number, game_id),
        )
        self._conn.commit()

    def get_game(self, game_id: int) -> GameRecord | None:
        row = self._conn.execute("SELECT * FROM games WHERE id = ?", (game_id,)).fetchone()
        return _row_to_game(row) if row is not None else None

    def list_games(self) -> list[GameRecord]:
        rows = self._conn.execute("SELECT * FROM games ORDER BY started_at").fetchall()
        return [_row_to_game(row) for row in rows]

    def add_throw(self, record: ThrowRecord) -> int:
        cur = self._conn.execute(
            """
            INSERT INTO throws (
                game_id, player_number, turn_number, throw_number, timestamp,
                x_mm, y_mm, angle_deg, radius_mm,
                sector, ring, multiplier, points,
                is_bust, is_checkout, closing_distance_mm,
                remaining_before, remaining_after
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.game_id,
                record.player_number,
                record.turn_number,
                record.throw_number,
                record.timestamp.isoformat(),
                record.x_mm,
                record.y_mm,
                record.angle_deg,
                record.radius_mm,
                record.sector,
                record.ring,
                record.multiplier,
                record.points,
                int(record.is_bust),
                int(record.is_checkout),
                record.closing_distance_mm,
                record.remaining_before,
                record.remaining_after,
            ),
        )
        self._conn.commit()
        assert cur.lastrowid is not None
        return cur.lastrowid

    def add_turn_outcome(
        self,
        game_id: int,
        player_number: int,
        turn_number: int,
        outcome: TurnOutcome,
        timestamp: datetime | None = None,
    ) -> list[ThrowRecord]:
        """Persiste in un colpo solo tutte le freccette di un ``TurnOutcome``
        (l'output naturale di ``Game501.play_turn`` / ``Match501.play_turn``)."""
        timestamp = timestamp or datetime.now()
        saved: list[ThrowRecord] = []
        for i, dart in enumerate(outcome.darts, start=1):
            record = throw_record_from_outcome(
                game_id, player_number, turn_number, i, dart, timestamp
            )
            throw_id = self.add_throw(record)
            saved.append(ThrowRecord(**{**record.__dict__, "id": throw_id}))

        if outcome.is_checkout:
            self.finish_game(game_id, winner_player_number=player_number, finished_at=timestamp)
        return saved

    def list_throws_for_game(self, game_id: int) -> list[ThrowRecord]:
        rows = self._conn.execute(
            "SELECT * FROM throws WHERE game_id = ? ORDER BY timestamp, throw_number",
            (game_id,),
        ).fetchall()
        return [_row_to_throw(row) for row in rows]

    def list_all_throws(self) -> list[ThrowRecord]:
        rows = self._conn.execute("SELECT * FROM throws ORDER BY timestamp").fetchall()
        return [_row_to_throw(row) for row in rows]
