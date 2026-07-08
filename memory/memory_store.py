# memory/memory_store.py
# Long-term memory layer (spec.md §12): a local SQLite database memory.db
# at the repo root, keyed by session_id, storing short LLM-generated
# conversation summaries instead of full transcripts.

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).resolve().parent.parent / "memory.db"

SUMMARIZE_EVERY_N_TURNS = 6


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memory (
            session_id   TEXT PRIMARY KEY,
            created_at   TEXT NOT NULL,
            summary_text TEXT NOT NULL,
            updated_at   TEXT NOT NULL
        )
        """
    )
    return conn


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def get_summary(session_id: str) -> Optional[str]:
    """Return the stored summary for a session_id, or None if absent."""
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT summary_text FROM memory WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def save_summary(session_id: str, summary_text: str) -> None:
    """Insert or update the summary for a session_id."""
    conn = _get_connection()
    try:
        conn.execute(
            """
            INSERT INTO memory (session_id, created_at, summary_text, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                summary_text = excluded.summary_text,
                updated_at   = excluded.updated_at
            """,
            (session_id, _now(), summary_text, _now()),
        )
        conn.commit()
    finally:
        conn.close()


def should_summarize(turn_count: int) -> bool:
    """True when the conversation has hit a multiple of the 6-turn threshold."""
    return turn_count > 0 and turn_count % SUMMARIZE_EVERY_N_TURNS == 0
