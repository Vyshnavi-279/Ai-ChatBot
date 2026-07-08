# observability/logger.py
# TODO: Full observability module (spec.md §13) — SQLite observability.db with
# timestamp, session_id, redacted question, dimension/tool used, chunk IDs,
# model name, token counts, latency, refused flag, error — plus the
# pages/2_Observability_Dashboard.py Streamlit page. Built in a later phase.
# For now log_event is a minimal stub so tool handlers have a stable call site.

from __future__ import annotations

import json
import time
from pathlib import Path

_LOG_PATH = Path(__file__).resolve().parent / "events.log"


def log_event(event_type: str, payload: dict) -> None:
    """Append a structured event to a plain-text log file.

    TODO: replace with a row insert into observability.db (spec.md §13).
    """
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "event_type": event_type,
        "payload": payload,
    }
    try:
        with _LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass  # logging must never break the chatbot
