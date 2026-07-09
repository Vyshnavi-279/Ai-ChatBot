"""
observability/logger.py — SQLite-backed request logging.

Exports a single function ``log_event(**fields)`` that other modules call
to record every RAG or tool-call request.

Database: ``observability.db`` at the repo root (see .gitignore).
Table schema matches spec.md section 13.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# DB path — always at the repo root, next to this package.
# ---------------------------------------------------------------------------
_DB_PATH = str(Path(__file__).resolve().parent.parent / "observability.db")

# ---------------------------------------------------------------------------
# Thread-local connection so Streamlit's hot-reload doesn't share state.
# ---------------------------------------------------------------------------
_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    """Return a thread-local connection, creating tables on first use."""
    if not hasattr(_local, "conn") or _local.conn is None:
        conn = sqlite3.connect(_DB_PATH)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=3000")
        _init_table(conn)
        _local.conn = conn
    return _local.conn


def _init_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS observability (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT    NOT NULL,
            session_id      TEXT    NOT NULL,
            question        TEXT    NOT NULL,
            dimension_or_tool TEXT  NOT NULL,
            retrieved_chunk_ids TEXT,
            model_name      TEXT    NOT NULL,
            input_tokens    INTEGER DEFAULT 0,
            output_tokens   INTEGER DEFAULT 0,
            latency_seconds REAL    DEFAULT 0.0,
            refused         INTEGER DEFAULT 0,
            error           TEXT
        )
        """
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def log_event(**fields: Any) -> None:
    """Insert one observability row.

    Accepted keyword arguments (all snake_case versions of the column names):

    ====================  ==========  ========================================
    Key                   Required    Description
    ====================  ==========  ========================================
    session_id            yes         UUID or other session identifier
    question              yes         Already-redacted question text
    dimension_or_tool     yes         "retrieval" | "generation" | tool name
    retrieved_chunk_ids   no          List[str] — stored as JSON
    model_name            yes         Model identifier (e.g. "gpt-4o-mini")
    input_tokens          no          int
    output_tokens         no          int
    latency_seconds       no          float
    refused               no          bool (stored as 0/1)
    error                 no          str or None
    ====================  ==========  ========================================

    Returns ``None``. Errors are silently swallowed so logging never
    breaks the main flow.
    """
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": fields.get("session_id", ""),
        "question": fields.get("question", ""),
        "dimension_or_tool": fields.get("dimension_or_tool", ""),
        "retrieved_chunk_ids": json.dumps(fields.get("retrieved_chunk_ids", [])),
        "model_name": fields.get("model_name", ""),
        "input_tokens": int(fields.get("input_tokens", 0)),
        "output_tokens": int(fields.get("output_tokens", 0)),
        "latency_seconds": float(fields.get("latency_seconds", 0.0)),
        "refused": 1 if fields.get("refused", False) else 0,
        "error": fields.get("error"),  # None → NULL in DB
    }

    try:
        conn = _get_conn()
        conn.execute(
            """
            INSERT INTO observability (
                timestamp, session_id, question, dimension_or_tool,
                retrieved_chunk_ids, model_name, input_tokens, output_tokens,
                latency_seconds, refused, error
            ) VALUES (
                :timestamp, :session_id, :question, :dimension_or_tool,
                :retrieved_chunk_ids, :model_name, :input_tokens, :output_tokens,
                :latency_seconds, :refused, :error
            )
            """,
            row,
        )
        conn.commit()
    except Exception:
        # Logging must never crash the app — silently ignore DB errors.
        pass


# ---------------------------------------------------------------------------
# Convenience: fetch records for the dashboard.
# ---------------------------------------------------------------------------

def fetch_all() -> list[dict]:
    """Return all rows as a list of dicts (used by the dashboard)."""
    try:
        conn = _get_conn()
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM observability ORDER BY timestamp DESC"
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


def fetch_recent_errors(limit: int = 20) -> list[dict]:
    """Return the most recent rows where error is not null."""
    try:
        conn = _get_conn()
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM observability WHERE error IS NOT NULL "
            "ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []