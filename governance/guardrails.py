"""
governance/guardrails.py — PII redaction, rate limiting, and advice-detection.

All functions are pure — they accept a string and return transformed data or
a boolean decision.  No Streamlit imports here (the UI layer calls these).

Exports
-------
redact_pii(text)       -> str          # redacted text
check_rate_limit()     -> bool         # True = allowed, False = cooldown
is_advice_request(text) -> bool        # True = likely medical/legal/financial advice
"""

from __future__ import annotations

import re
import time
from typing import List

# ---------------------------------------------------------------------------
# PII redaction — applied BEFORE logging/storing, not before LLM answering.
# ---------------------------------------------------------------------------

# Phone numbers: Indian (+91-XXXXXXXXXX, 0XXXXXXXXXX, XXXXXXXXXX)
_PHONE_RE = re.compile(
    r"(?:\+?91[-.\s]?)?0?(?:[6-9]\d{9})(?![-\d])"
)

# Email addresses (basic RFC-like pattern)
_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
)

# Aadhaar (12 digits, optionally grouped as 4-4-4)
_AADHAAR_RE = re.compile(
    r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b"
)

# PAN (Indian) — 5 letters + 4 digits + 1 letter
_PAN_RE = re.compile(
    r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b"
)

# Passport (India-style — 1 letter + 7 digits, or common global formats)
_PASSPORT_RE = re.compile(
    r"\b[ABCEGHJKLMNPRSTVWXYZ]{1}[0-9]{7}\b"
)

_PII_PATTERNS: List[re.Pattern] = [
    _PHONE_RE,
    _EMAIL_RE,
    _AADHAAR_RE,
    _PAN_RE,
    _PASSPORT_RE,
]

_REDACT_TOKEN = "[REDACTED]"


def redact_pii(text: str) -> str:
    """Replace all detected PII in *text* with ``[REDACTED]``.

    This is safe to call on user input **before** logging or storing in
    memory.  The original (unredacted) text is still sent to the LLM so
    the bot can understand the question.
    """
    for pattern in _PII_PATTERNS:
        text = pattern.sub(_REDACT_TOKEN, text)
    return text


# ---------------------------------------------------------------------------
# Rate limiter — max N questions per session per 60 seconds.
# ---------------------------------------------------------------------------

MAX_REQUESTS_PER_WINDOW = 10
RATE_LIMIT_WINDOW_SECONDS = 60


def check_rate_limit() -> bool:
    """Check if the current session is within the rate limit.

    Uses ``st.session_state`` timestamps (list of floats, UNIX epoch).
    Returns ``True`` if the request is allowed, ``False`` if the user
    should be shown a cooldown message.

    This function is intended to be called from the Streamlit UI layer
    (``app.py``) which has access to ``st.session_state``.
    """
    import streamlit as st

    now = time.time()

    # Initialise the timestamp list if needed.
    if "_rate_limit_timestamps" not in st.session_state:
        st.session_state._rate_limit_timestamps = []

    # Prune timestamps older than the window.
    cutoff = now - RATE_LIMIT_WINDOW_SECONDS
    st.session_state._rate_limit_timestamps = [
        t for t in st.session_state._rate_limit_timestamps
        if t > cutoff
    ]

    # Check count.
    if len(st.session_state._rate_limit_timestamps) >= MAX_REQUESTS_PER_WINDOW:
        return False

    # Record this request.
    st.session_state._rate_limit_timestamps.append(now)
    return True


# ---------------------------------------------------------------------------
# Advice-request detection (medical / legal / financial).
# ---------------------------------------------------------------------------

# Keywords that strongly suggest the user is seeking *advice*, not just
# factual college information.  These are based on the Safety dimension
# criteria from spec.md section 8.
_ADVICE_KEYWORDS: List[str] = [
    # Medical
    r"\bprescribe\b",
    r"\bdiagnos[ei]\w*\b",
    r"\bsymptom\w*\b",
    r"\btreatment\b",
    r"\bdosage\b",
    r"\bmedication\b",
    r"\bsurgery\b",
    r"\btherapy\b",
    r"\bcure\b",
    r"\bmedical\s+advice\b",
    r"\bshould\s+I\s+take\b",
    r"\bwhat\s+medicine\b",
    # Legal
    r"\blegal\s+advice\b",
    r"\battorney\b",
    r"\blawyer\b",
    r"\blawsuit\b",
    r"\bfile\s+a\s+case\b",
    r"\blegal\s+action\b",
    r"\bnotary\b",
    r"\baffidavit\b",
    r"\bwrit\s+petition\b",
    # Financial
    r"\binvestment\s+advice\b",
    r"\bstock\s+pick\b",
    r"\btrading\s+strategy\b",
    r"\bportfolio\s+management\b",
    r"\bretirement\s+planning\b",
    r"\btax\s+evasion\b",
    r"\bwealth\s+management\b",
    r"\binsurance\s+claim\b",
    r"\bloan\s+approval\b",
]

_ADVICE_RE = re.compile(
    "|".join(_ADVICE_KEYWORDS),
    re.IGNORECASE,
)


def is_advice_request(question: str) -> bool:
    """Return ``True`` if *question* appears to be seeking medical, legal,
    or financial *advice* (as opposed to factual college information).

    This is a lightweight keyword-based check.  It is intentionally
    conservative — false positives (flagging a non-advice question) are
    acceptable because the bot will show a disclaimer + redirect instead
    of answering, which is safer than the reverse.
    """
    return bool(_ADVICE_RE.search(question))