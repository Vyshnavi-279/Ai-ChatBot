"""
app.py — BVRIT RAG Chatbot · Streamlit UI
Light theme per spec.md section 7. Reuses rag_core.retriever + rag_core.generation.

Includes governance (PII redaction, rate limiting, advice detection), observability
logging, and an admin panel with maintenance mode.

Run:  streamlit run app.py
"""

from __future__ import annotations

import csv
import io
import os
import sys
import time
import uuid
from pathlib import Path

import streamlit as st

# Page config — must be the FIRST Streamlit call.
st.set_page_config(
    page_title="BVRIT Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS — spec.md section 7 color tokens, full light theme.
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
/* ── Color tokens (spec.md §7) ── */
:root {
  --bg:          #F5F1EA;
  --sidebar-bg:  #EDE7DC;
  --surface:     #FFFFFF;
  --primary:     #3D5A80;
  --accent:      #98C1D9;
  --accent-tint: #EAF4FA;
  --success:     #6A994E;
  --warning:     #E09F3E;
  --error:       #BC4749;
  --text:        #293241;
  --border:      #E4E0D8;
}

/* ── App background ── */
.stApp { background-color: var(--bg) !important; color: var(--text) !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background-color: var(--sidebar-bg) !important;
}
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
  color: var(--text) !important;
}

/* ── Block container (main column) ── */
.block-container {
  padding-top: 1.4rem !important;
  max-width: 860px;
}

/* ── Chat bubbles ── */
[data-testid="stChatMessage"] {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 0.85rem 1.1rem;
  margin-bottom: 0.55rem;
  box-shadow: 0 1px 5px rgba(0,0,0,0.06);
}

/* User messages — accent-tinted bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
  background: var(--accent-tint) !important;
  border-color: var(--accent) !important;
}

/* ── Typography ── */
h1, h2, h3 { color: var(--primary) !important; }

/* ── Primary action buttons ── */
.stButton > button {
  background-color: var(--primary) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 8px !important;
  padding: 0.38rem 1rem !important;
  font-size: 0.88rem !important;
  transition: opacity 0.15s;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* ── Example-question chip buttons ── */
.chip-btn > button {
  background-color: var(--accent-tint) !important;
  color: var(--primary) !important;
  border: 1px solid var(--accent) !important;
  border-radius: 20px !important;
  font-size: 0.82rem !important;
  padding: 0.28rem 0.85rem !important;
  white-space: normal !important;
  height: auto !important;
  line-height: 1.4 !important;
}
.chip-btn > button:hover {
  background-color: var(--accent) !important;
  color: #fff !important;
}

/* ── Citation pill badges ── */
.citation-pill {
  display: inline-block;
  background: var(--accent-tint);
  color: var(--primary);
  border: 1px solid var(--accent);
  border-radius: 20px;
  font-size: 0.74rem;
  font-weight: 600;
  padding: 0.14rem 0.62rem;
  margin: 0.18rem 0.22rem 0 0;
  white-space: nowrap;
}

/* ── Refusal badge ── */
.refusal-pill {
  display: inline-block;
  background: #fce8e8;
  color: var(--error);
  border: 1px solid var(--error);
  border-radius: 20px;
  font-size: 0.74rem;
  font-weight: 700;
  padding: 0.14rem 0.62rem;
  margin-top: 0.18rem;
  white-space: nowrap;
}

/* ── KB status card ── */
.kb-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 0.7rem 0.9rem;
  margin-bottom: 0.75rem;
  font-size: 0.82rem;
  line-height: 1.8;
  box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.kb-card b { color: var(--primary); }

/* ── Welcome banner image wrapper ── */
.welcome-img-wrap img {
  width: 100%;
  max-height: 220px;
  object-fit: cover;
  border-radius: 12px;
  margin-bottom: 0.5rem;
  box-shadow: 0 2px 10px rgba(0,0,0,0.10);
}

/* ── Selectbox, chat input ── */
[data-testid="stSelectbox"] > div { border-radius: 8px !important; }
[data-testid="stChatInput"] textarea { border-radius: 10px !important; }

/* ── Cooldown message (rate limiting) ── */
.cooldown-msg {
  background: #fef5e7;
  border: 1px solid var(--warning);
  border-left: 4px solid var(--warning);
  border-radius: 8px;
  padding: 0.75rem 1rem;
  margin: 0.5rem 0;
  font-size: 0.9rem;
  color: var(--text);
}

/* ── Maintenance mode message ── */
.maintenance-msg {
  background: #fce8e8;
  border: 1px solid var(--error);
  border-left: 4px solid var(--error);
  border-radius: 8px;
  padding: 2rem;
  margin: 2rem 0;
  font-size: 1.1rem;
  text-align: center;
  color: var(--text);
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
hr { border-color: var(--border) !important; margin: 0.5rem 0 !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Asset paths
# ---------------------------------------------------------------------------
_ASSETS = Path(__file__).parent / "assets" / "images"
LOGO_PATH   = _ASSETS / "logo.png"
CAMPUS_PATH = _ASSETS / "campus.jpg"

# Maintenance mode flag file path.
_MAINTENANCE_FLAG = Path(__file__).parent / "maintenance.flag"

# ---------------------------------------------------------------------------
# Load rag_core (cached).
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading knowledge base…")
def _load_rag():
    try:
        from rag_core.retriever import retrieve, _get_collection
        from rag_core.generation import generate
        import rag_core.config as cfg

        try:
            col = _get_collection()
            n = col.count()
            status = "✅ Ready" if n > 0 else "⚠️ Empty — run ingest.py"
        except Exception:
            n, status = 0, "❌ ChromaDB unreachable"

        return retrieve, generate, cfg, n, status

    except FileNotFoundError as e:
        st.error(f"Configuration error:\n\n{e}")
        st.stop()
    except EnvironmentError as e:
        st.error(f"Environment error:\n\n{e}")
        st.stop()
    except ModuleNotFoundError as e:
        st.error(
            f"**Missing dependency:** `{e}`\n\n"
            "Run `pip install -r requirements.txt` then restart Streamlit."
        )
        st.stop()


retrieve_fn, generate_fn, cfg, chunk_count, index_status = _load_rag()

# ---------------------------------------------------------------------------
# Load admin password from env (never hardcoded in code, never logged).
# ---------------------------------------------------------------------------
_ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")

# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history: list[dict] = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question: str = ""

if "session_id" not in st.session_state:
    st.session_state.session_id: str = str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Helper: render badges
# ---------------------------------------------------------------------------
def _render_badges(citations: list[str], refused: bool) -> None:
    if refused:
        st.markdown(
            '<span class="refusal-pill">⛔ REFUSED — not in knowledge base</span>',
            unsafe_allow_html=True,
        )
    elif citations:
        pills = "".join(
            f'<span class="citation-pill">📎 {c}</span>' for c in citations
        )
        st.markdown(pills, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper: check maintenance mode
# ---------------------------------------------------------------------------
def _is_maintenance_mode() -> bool:
    return _MAINTENANCE_FLAG.exists()


def _set_maintenance_mode(enabled: bool) -> None:
    if enabled:
        _MAINTENANCE_FLAG.touch()
    else:
        _MAINTENANCE_FLAG.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=120)
    else:
        st.markdown(
            "<h2 style='color:var(--primary);margin:0 0 0.4rem 0;font-size:1.15rem;'>"
            "🎓 BVRIT</h2>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Knowledge-base status card
    doc_name = Path(cfg.DOCX_PATH).name
    st.markdown(
        f"""
        <div class="kb-card">
          <b>Knowledge Base</b><br>
          📄 {doc_name}<br>
          🗂️ Chunks indexed: <b>{chunk_count}</b><br>
          Status: {index_status}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Retrieval settings
    with st.expander("⚙️ Retrieval settings", expanded=False):
        st.markdown(
            f"""
            <div style="font-size:0.82rem;line-height:1.95;color:var(--text);">
              <b>Chunk size:</b>&nbsp; {cfg.CHUNK_SIZE} chars<br>
              <b>Overlap:</b>&nbsp; {cfg.CHUNK_OVERLAP} chars<br>
              <b>Top-K:</b>&nbsp; {cfg.TOP_K} chunks<br>
              <b>Embed:</b>&nbsp; {cfg.EMBEDDING_MODEL}<br>
              <b>Generate:</b>&nbsp; {cfg.GENERATION_MODEL}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Section filter
    SECTION_OPTIONS = [
        "All", "About BVRIT", "Departments", "Admissions",
        "Fee Structure", "Placements", "Campus & Facilities",
        "Faculty", "Contact",
    ]
    section_filter: str = st.selectbox(
        "🔍 Filter by section",
        options=SECTION_OPTIONS,
        index=0,
        help="Restrict retrieval to a single document section.",
    )

    st.markdown("---")

    # Clear conversation
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.session_state.pending_question = ""
        st.rerun()

    # -----------------------------------------------------------------------
    # ADMIN PANEL — password-gated
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🔐 Admin Panel")

    admin_password_input = st.text_input(
        "Admin password",
        type="password",
        placeholder="Enter password",
        label_visibility="collapsed",
    )

    if admin_password_input:
        if _ADMIN_PASSWORD and admin_password_input == _ADMIN_PASSWORD:
            st.success("Authenticated")

            # Download observability CSV
            if st.button("📥 Download observability log as CSV", use_container_width=True):
                try:
                    from observability.logger import fetch_all
                    rows = fetch_all()
                    if rows:
                        import csv as _csv
                        import io as _io
                        output = _io.StringIO()
                        writer = _csv.writer(output)
                        if rows:
                            writer.writerow(rows[0].keys())
                            for r in rows:
                                writer.writerow(r.values())
                        st.download_button(
                            label="📄 Click to download CSV",
                            data=output.getvalue(),
                            file_name=f"observability_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                        )
                    else:
                        st.info("No data in observability log yet.")
                except Exception as e:
                    st.warning(f"Could not export CSV: {e}")

            # Maintenance mode toggle
            maintenance_enabled = _is_maintenance_mode()
            if st.checkbox(
                "🔧 Maintenance mode",
                value=maintenance_enabled,
                help="When enabled, the chat shows a static 'unavailable' message.",
            ):
                if not maintenance_enabled:
                    _set_maintenance_mode(True)
                    st.rerun()
            else:
                if maintenance_enabled:
                    _set_maintenance_mode(False)
                    st.rerun()

            if _is_maintenance_mode():
                st.markdown(
                    '<p style="color:var(--error);font-size:0.85rem;">'
                    "⚠️ Maintenance mode is **ON** — chat is disabled.</p>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<p style="color:var(--error);font-size:0.85rem;">'
                "❌ Incorrect password</p>",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# CHECK: Maintenance mode — show static message and stop.
# ---------------------------------------------------------------------------
if _is_maintenance_mode():
    # Clear any existing UI content
    st.markdown(
        "<div class='maintenance-msg'>"
        "🛠️ **Temporarily Unavailable**<br><br>"
        "The BVRIT chatbot is currently undergoing maintenance. "
        "Please check back soon.<br><br>"
        "For urgent inquiries, contact BVRIT Hyderabad directly at "
        "<b>info@bvrithyderabad.edu.in</b> or <b>+91-40-2304-2777</b>."
        "</div>",
        unsafe_allow_html=True,
    )
    st.stop()


# ---------------------------------------------------------------------------
# MAIN AREA — welcome screen
# ---------------------------------------------------------------------------
EXAMPLE_QUESTIONS = [
    "What departments does BVRIT offer?",
    "What are the hostel facilities?",
    "When was BVRIT Hyderabad established?",
    "How do I apply for admission?",
]

if not st.session_state.history:
    if CAMPUS_PATH.exists():
        st.markdown('<div class="welcome-img-wrap">', unsafe_allow_html=True)
        st.image(str(CAMPUS_PATH), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        "<h1 style='color:var(--primary);margin-bottom:0.2rem;'>"
        "Ask me anything about BVRIT</h1>"
        "<p style='color:var(--text);margin-top:0;font-size:0.95rem;'>"
        "Every answer is grounded in the official knowledge base "
        "with section & page citations.</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<p style='font-size:0.85rem;color:var(--text);margin-bottom:0.35rem;'>"
        "<b>Try asking:</b></p>",
        unsafe_allow_html=True,
    )
    chip_cols = st.columns(len(EXAMPLE_QUESTIONS))
    for col, question in zip(chip_cols, EXAMPLE_QUESTIONS):
        with col:
            st.markdown('<div class="chip-btn">', unsafe_allow_html=True)
            if st.button(question, key=f"chip_{question}"):
                st.session_state.pending_question = question
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# CHAT HISTORY — render prior turns
# ---------------------------------------------------------------------------
for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn["role"] == "assistant":
            _render_badges(turn.get("citations", []), turn.get("refused", False))


# ---------------------------------------------------------------------------
# CHAT INPUT
# ---------------------------------------------------------------------------
user_input: str = st.chat_input("Ask a question about BVRIT…") or ""

if st.session_state.pending_question and not user_input:
    user_input = st.session_state.pending_question
    st.session_state.pending_question = ""


# ---------------------------------------------------------------------------
# PROCESS NEW QUESTION
# ---------------------------------------------------------------------------
if user_input.strip():
    query = user_input.strip()

    # --- RATE LIMITING CHECK ---
    from governance.guardrails import check_rate_limit, redact_pii

    if not check_rate_limit():
        st.markdown(
            "<div class='cooldown-msg'>"
            "⏳ **Slow down!** You've reached the maximum of 10 questions "
            "per 60 seconds. Please wait a moment before asking your next "
            "question."
            "</div>",
            unsafe_allow_html=True,
        )
        st.stop()

    # --- LOG the query to observability (before retrieval) ---
    from observability.logger import log_event

    session_id = st.session_state.session_id

    # Immediately show the user's message
    with st.chat_message("user"):
        st.markdown(query)

    # Retrieve then generate
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base…"):
            filter_value = None if section_filter == "All" else section_filter
            try:
                chunks = retrieve_fn(query, section_filter=filter_value)
            except Exception as exc:
                st.error(f"Retrieval failed: {exc}")
                st.stop()

        with st.spinner("Generating answer…"):
            try:
                result = generate_fn(
                    query=query,
                    chunks=chunks,
                    history=st.session_state.history,
                    session_id=session_id,
                )
            except Exception as exc:
                # Log the error
                log_event(
                    session_id=session_id,
                    question=redact_pii(query),
                    dimension_or_tool="generation",
                    model_name=cfg.GENERATION_MODEL,
                    error=str(exc),
                )
                st.error(f"Generation failed: {exc}")
                st.stop()

        answer    = result["answer"]
        citations = result["citations"]
        refused   = result["refused"]

        st.markdown(answer)
        _render_badges(citations, refused)

    # Persist both turns to history (redact PII before storing)
    redacted_query = redact_pii(query)
    st.session_state.history.append({"role": "user", "content": redacted_query})
    st.session_state.history.append(
        {
            "role": "assistant",
            "content": answer,
            "citations": citations,
            "refused": refused,
        }
    )
