"""
tools/tool_handlers.py — LLM-callable tool functions.

Every tool returns structured JSON data (never free text) per spec.md
section 11.  Every tool call is logged to the observability database.

Exported functions:
    get_fee_by_branch(branch)      -> dict
    get_admission_deadline(program) -> dict
    log_unanswered_question(question) -> dict
    escalate_to_contact(reason)     -> dict
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from observability.logger import log_event

# ---------------------------------------------------------------------------
# Data files  (created from college_info.docx scraping or manual setup)
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_FEES_PATH = _DATA_DIR / "fees.json"
_DEADLINES_PATH = _DATA_DIR / "deadlines.json"

# Fallback contact info (pulled from college_info.docx Contact section)
_CONTACT_INFO = {
    "address": "BVRIT Hyderabad College of Engineering for Women, "
               "Bachupally, Hyderabad, Telangana 500090",
    "phone": "+91-40-2304-2777",
    "email": "info@bvrithyderabad.edu.in",
    "website": "https://bvrithyderabad.edu.in",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    """Load a JSON file, returning {} on any error."""
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _make_response(
    success: bool,
    data: Any = None,
    error: str | None = None,
    tool_name: str = "",
) -> dict:
    """Build a structured tool response and log the call."""
    response = {
        "success": success,
        "data": data if success else None,
        "error": error,
    }

    # Log every tool call to observability.
    log_event(
        session_id="tool",  # caller should override if it has a real session_id
        question="",
        dimension_or_tool=tool_name,
        retrieved_chunk_ids=[],
        model_name="tool",
        input_tokens=0,
        output_tokens=0,
        latency_seconds=0.0,
        refused=False,
        error=error,
    )

    return response


# ---------------------------------------------------------------------------
# Tool: get_fee_by_branch
# ---------------------------------------------------------------------------

def get_fee_by_branch(branch: str, session_id: str = "") -> dict:
    """Return the exact fee for a given branch.

    Parameters
    ----------
    branch : str
        Branch name (case-insensitive, e.g. "CSE", "ECE", "EEE", "MECH", "CIVIL").

    Returns
    -------
    dict with keys ``success``, ``data``, ``error``.
    """
    start = time.time()
    fees = _load_json(_FEES_PATH)
    branch_key = branch.strip().upper()

    if branch_key in fees:
        result = {"branch": branch, "fee": fees[branch_key]}
        latency = time.time() - start
        log_event(
            session_id=session_id,
            question=f"get_fee_by_branch({branch})",
            dimension_or_tool="get_fee_by_branch",
            model_name="tool",
            latency_seconds=latency,
        )
        return {"success": True, "data": result, "error": None}
    else:
        latency = time.time() - start
        err = f"No fee data found for branch '{branch}'. Available: {list(fees.keys())}"
        log_event(
            session_id=session_id,
            question=f"get_fee_by_branch({branch})",
            dimension_or_tool="get_fee_by_branch",
            model_name="tool",
            latency_seconds=latency,
            error=err,
        )
        return {"success": False, "data": None, "error": err}


# ---------------------------------------------------------------------------
# Tool: get_admission_deadline
# ---------------------------------------------------------------------------

def get_admission_deadline(program: str, session_id: str = "") -> dict:
    """Return the exact admission deadline for a given program.

    Parameters
    ----------
    program : str
        Program name (e.g. "B.Tech", "M.Tech", "MBA").

    Returns
    -------
    dict with keys ``success``, ``data``, ``error``.
    """
    start = time.time()
    deadlines = _load_json(_DEADLINES_PATH)
    program_key = program.strip().upper()

    if program_key in deadlines:
        result = {"program": program, "deadline": deadlines[program_key]}
        latency = time.time() - start
        log_event(
            session_id=session_id,
            question=f"get_admission_deadline({program})",
            dimension_or_tool="get_admission_deadline",
            model_name="tool",
            latency_seconds=latency,
        )
        return {"success": True, "data": result, "error": None}
    else:
        latency = time.time() - start
        err = f"No deadline data found for program '{program}'."
        log_event(
            session_id=session_id,
            question=f"get_admission_deadline({program})",
            dimension_or_tool="get_admission_deadline",
            model_name="tool",
            latency_seconds=latency,
            error=err,
        )
        return {"success": False, "data": None, "error": err}


# ---------------------------------------------------------------------------
# Tool: log_unanswered_question
# ---------------------------------------------------------------------------

def log_unanswered_question(question: str, session_id: str = "") -> dict:
    """Record a question that the knowledge base could not answer.

    Parameters
    ----------
    question : str
        The user's unanswered question (raw, before redaction — the logger
        will store the version it receives; the caller should redact first).

    Returns
    -------
    dict with keys ``success``, ``data``, ``error``.
    """
    from governance.guardrails import redact_pii

    start = time.time()
    redacted = redact_pii(question)

    log_event(
        session_id=session_id,
        question=redacted,
        dimension_or_tool="log_unanswered_question",
        model_name="tool",
        latency_seconds=time.time() - start,
    )

    return {
        "success": True,
        "data": {"logged": True, "question_redacted": redacted},
        "error": None,
    }


# ---------------------------------------------------------------------------
# Tool: escalate_to_contact
# ---------------------------------------------------------------------------

def escalate_to_contact(reason: str, session_id: str = "") -> dict:
    """Return official contact information and log the escalation.

    Parameters
    ----------
    reason : str
        Why the user is being escalated (e.g. "unanswered question",
        "complaint", "request for human advisor").

    Returns
    -------
    dict with keys ``success``, ``data`` (contains contact info), ``error``.
    """
    from governance.guardrails import redact_pii

    start = time.time()
    redacted_reason = redact_pii(reason)

    log_event(
        session_id=session_id,
        question=redacted_reason,
        dimension_or_tool="escalate_to_contact",
        model_name="tool",
        latency_seconds=time.time() - start,
    )

    return {
        "success": True,
        "data": {
            "message": (
                "I've noted your request and an official will follow up. "
                "In the meantime, here's how to contact BVRIT directly:"
            ),
            "contact": _CONTACT_INFO,
        },
        "error": None,
    }


# ---------------------------------------------------------------------------
# Convenience: get all tool definitions for the LLM (OpenAI/OpenRouter
# function-calling schema).
# ---------------------------------------------------------------------------

def get_tool_definitions() -> list[dict]:
    """Return the OpenAI-compatible tool definitions for all tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_fee_by_branch",
                "description": "Get the exact tuition fee for a specific engineering branch at BVRIT.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "branch": {
                            "type": "string",
                            "description": "Branch name, e.g. CSE, ECE, EEE, MECH, CIVIL",
                        }
                    },
                    "required": ["branch"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_admission_deadline",
                "description": "Get the admission deadline for a specific program at BVRIT.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "program": {
                            "type": "string",
                            "description": "Program name, e.g. B.Tech, M.Tech, MBA",
                        }
                    },
                    "required": ["program"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "log_unanswered_question",
                "description": "Log a question that the knowledge base could not answer, for knowledge base improvement.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The unanswered question text",
                        }
                    },
                    "required": ["question"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "escalate_to_contact",
                "description": "Escalate a user request to a human and provide contact information for BVRIT.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": "Reason for escalation",
                        }
                    },
                    "required": ["reason"],
                },
            },
        },
    ]