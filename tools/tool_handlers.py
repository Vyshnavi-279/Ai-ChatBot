# tools/tool_handlers.py
# Python implementations behind the schemas in tools/tool_definitions.py.
# Handlers return structured JSON-serialisable dicts, never free text, and
# never guess: unknown branches/programs get a clear "not found" object.
# Every call is logged via observability.logger.log_event (spec.md §11 & §13).

from __future__ import annotations

import json
from pathlib import Path

from observability.logger import log_event

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_FEES_PATH = _DATA_DIR / "fees.json"
_DEADLINES_PATH = _DATA_DIR / "deadlines.json"

# Official contact details from data/college_info.docx — Contact section.
_CONTACT_INFO = {
    "campus_address": (
        "Plot No. 8-5/4, Rajiv Gandhi Nagar Colony, Nizampet Rd, "
        "Bachupally, Hyderabad-500090"
    ),
    "campus_phone": "+91 40 4241 7773",
    "emails": ["info@bvrithyderabad.edu.in", "principal@bvrithyderabad.edu.in"],
    "city_office_address": (
        "Anjani Vishnu Centre, Plot No.7 & 8, Ist Floor, Nagarjuna Hills, "
        "Punjagutta, Hyderabad - 500 082"
    ),
    "city_office_phone": "+91 40 40334848",
    "city_office_fax": "+91 40 40334848",
}


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalise(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def get_fee_by_branch(branch: str) -> dict:
    """Exact fee lookup from data/fees.json. Returns not-found on no match."""
    try:
        data = _load_json(_FEES_PATH)
    except (OSError, json.JSONDecodeError) as exc:
        result = {"found": False, "error": f"fees data unavailable: {exc}"}
        log_event("tool_call", {"tool": "get_fee_by_branch", "branch": branch, "result": result})
        return result

    wanted = _normalise(branch)
    for entry in data.get("tuition_by_branch", []):
        if wanted in (_normalise(entry.get("branch", "")), _normalise(entry.get("program", ""))):
            result = {"found": True, "currency": data.get("currency"), **entry}
            log_event("tool_call", {"tool": "get_fee_by_branch", "branch": branch, "result": result})
            return result

    result = {
        "found": False,
        "branch": branch,
        "message": (
            "No fee entry found for this branch. Known branches: "
            + ", ".join(e.get("branch", "") for e in data.get("tuition_by_branch", []))
        ),
    }
    log_event("tool_call", {"tool": "get_fee_by_branch", "branch": branch, "result": result})
    return result


def get_admission_deadline(program: str) -> dict:
    """Exact deadline lookup from data/deadlines.json. Returns not-found on no match."""
    try:
        data = _load_json(_DEADLINES_PATH)
    except (OSError, json.JSONDecodeError) as exc:
        result = {"found": False, "error": f"deadlines data unavailable: {exc}"}
        log_event("tool_call", {"tool": "get_admission_deadline", "program": program, "result": result})
        return result

    wanted = _normalise(program)
    for entry in data.get("deadlines", []):
        entry_key = _normalise(entry.get("program", ""))
        if wanted == entry_key or wanted in entry_key or entry_key in wanted:
            result = {"found": True, **entry}
            log_event("tool_call", {"tool": "get_admission_deadline", "program": program, "result": result})
            return result

    result = {
        "found": False,
        "program": program,
        "message": (
            "No deadline entry found for this program. Known programs: "
            + ", ".join(e.get("program", "") for e in data.get("deadlines", []))
        ),
    }
    log_event("tool_call", {"tool": "get_admission_deadline", "program": program, "result": result})
    return result


def log_unanswered_question(question: str) -> dict:
    """Record a knowledge-base gap in the observability log."""
    log_event("unanswered_question", {"question": question})
    return {"logged": True, "question": question}


def escalate_to_contact(reason: str) -> dict:
    """Return official contact info and log the escalation."""
    log_event("escalation", {"reason": reason})
    return {"escalated": True, "reason": reason, "contact": _CONTACT_INFO}


TOOL_HANDLERS = {
    "get_fee_by_branch": get_fee_by_branch,
    "get_admission_deadline": get_admission_deadline,
    "log_unanswered_question": log_unanswered_question,
    "escalate_to_contact": escalate_to_contact,
}
