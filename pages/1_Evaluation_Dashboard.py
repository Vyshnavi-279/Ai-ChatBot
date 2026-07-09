"""
pages/1_Evaluation_Dashboard.py — Streamlit evaluation dashboard.

Loads evaluation_report.json and displays pass/fail per dimension, RAGAS
scores, weakest dimension, and fix recommendation — all styled with the
spec.md section 7 light theme.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
import pandas as pd

# Must be the first Streamlit command.
st.set_page_config(
    page_title="Evaluation Dashboard",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Inject light-theme CSS (spec.md §7) — 8px grid, hover states, custom alerts.
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
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
.stApp {
  background-color: var(--bg) !important;
  color: var(--text) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background-color: var(--sidebar-bg) !important;
  padding: 24px 16px !important;
}

/* ── 8px grid spacing ── */
.block-container {
  padding: 32px 24px !important;
  max-width: 960px;
}

/* ── Typography ── */
h1, h2, h3 { color: var(--primary) !important; }

/* ── Buttons — hover states ── */
.stButton > button {
  background-color: var(--primary) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 8px !important;
  padding: 8px 16px !important;
  font-size: 0.88rem !important;
  transition: all 0.2s ease-in-out !important;
}
.stButton > button:hover {
  background-color: #293241 !important;
  transform: translateY(-1px);
  box-shadow: 0 4px 8px rgba(0,0,0,0.10);
}

/* ── Cards ── */
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px;
  margin-bottom: 16px;
  box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}

/* ── Metric cards ── */
.metric-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px;
  text-align: center;
  box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  transition: all 0.2s ease;
}
.metric-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0,0,0,0.10);
}

/* ── Custom alert boxes (replace st.error, st.warning) ── */
.custom-error {
  background: #fce8e8;
  border: 1px solid var(--error);
  border-left: 4px solid var(--error);
  border-radius: 8px;
  padding: 12px 16px;
  margin: 16px 0;
  font-size: 0.9rem;
  color: var(--text);
}
.custom-warning {
  background: #fef5e7;
  border: 1px solid var(--warning);
  border-left: 4px solid var(--warning);
  border-radius: 8px;
  padding: 12px 16px;
  margin: 16px 0;
  font-size: 0.9rem;
  color: var(--text);
}
.custom-success {
  background: #eaf7e6;
  border: 1px solid var(--success);
  border-left: 4px solid var(--success);
  border-radius: 8px;
  padding: 12px 16px;
  margin: 16px 0;
  font-size: 0.9rem;
  color: var(--text);
}

/* ── Dataframe styling ── */
[data-testid="stDataFrame"] {
  background: var(--surface) !important;
  border-radius: 8px !important;
  border: 1px solid var(--border) !important;
}

/* ── Sidebar hover items ── */
.sidebar-section {
  padding: 8px 12px;
  margin: 4px 0;
  border-radius: 8px;
  transition: all 0.2s ease;
  cursor: default;
}
.sidebar-section:hover {
  background: rgba(152, 193, 217, 0.15);
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
hr { border-color: var(--border) !important; margin: 16px 0 !important; }

/* ── Responsive narrow width ── */
@media (max-width: 768px) {
  .block-container { padding: 16px !important; }
  .metric-card { padding: 12px !important; }
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load evaluation report
# ---------------------------------------------------------------------------
_REPORT_PATH = Path(__file__).resolve().parent.parent / "evaluation_report.json"

def load_report() -> dict | None:
    try:
        if _REPORT_PATH.exists():
            with open(_REPORT_PATH, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return None


report = load_report()

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
st.title("📋 Evaluation Dashboard")
st.markdown(
    "<p style='color:var(--text);font-size:0.95rem;margin-bottom:24px;'>"
    "Automated 8-dimension test results + RAGAS metrics.</p>",
    unsafe_allow_html=True,
)

if report is None:
    st.markdown(
        "<div class='custom-warning'>"
        "⚠️ **No evaluation report found.** Run the evaluation pipeline first:"
        "<br><code>python evaluation/report.py</code>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.stop()

# ---------------------------------------------------------------------------
# Summary metrics row
# ---------------------------------------------------------------------------
summary = report.get("summary", {})
total = summary.get("total", 0)
passed = summary.get("passed", 0)
failed = summary.get("failed", 0)
warnings = summary.get("warning", 0)
pass_rate = summary.get("pass_rate", "0%")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        f"<div class='metric-card'><h3 style='margin:0;font-size:1.8rem;'>{total}</h3>"
        f"<p style='margin:0;font-size:0.85rem;color:var(--text);'>Total Tests</p></div>",
        unsafe_allow_html=True,
    )
with col2:
    color = "var(--success)" if int(pass_rate.strip('%')) >= 80 else "var(--warning)"
    st.markdown(
        f"<div class='metric-card'><h3 style='margin:0;font-size:1.8rem;color:{color};'>{pass_rate}</h3>"
        f"<p style='margin:0;font-size:0.85rem;color:var(--text);'>Pass Rate</p></div>",
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        f"<div class='metric-card'><h3 style='margin:0;font-size:1.8rem;color:var(--success);'>{passed}</h3>"
        f"<p style='margin:0;font-size:0.85rem;color:var(--text);'>Passed</p></div>",
        unsafe_allow_html=True,
    )
with col4:
    fail_color = "var(--error)" if failed > 0 else "var(--text)"
    st.markdown(
        f"<div class='metric-card'><h3 style='margin:0;font-size:1.8rem;color:{fail_color};'>{failed}</h3>"
        f"<p style='margin:0;font-size:0.85rem;color:var(--text);'>Failed</p></div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ---------------------------------------------------------------------------
# Per-dimension breakdown
# ---------------------------------------------------------------------------
st.header("📊 Per-Dimension Breakdown")

per_dim = report.get("per_dimension", {})
if per_dim:
    dim_data = []
    for key, value in per_dim.items():
        # Parse "3/3" or "0/0"
        parts = value.split("/")
        dim_passed = int(parts[0]) if len(parts) == 2 and parts[0].isdigit() else 0
        dim_total = int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 0
        dim_rate = f"{int((dim_passed / dim_total) * 100)}%" if dim_total > 0 else "N/A"
        dim_data.append({
            "Dimension": key,
            "Passed": dim_passed,
            "Total": dim_total,
            "Rate": dim_rate,
        })

    if dim_data:
        df = pd.DataFrame(dim_data)
        st.dataframe(df, hide_index=True, use_container_width=True)
    else:
        st.markdown(
            "<div class='custom-warning'>No dimension data available.</div>",
            unsafe_allow_html=True,
        )
else:
    st.markdown(
        "<div class='custom-warning'>No dimension breakdown available.</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ---------------------------------------------------------------------------
# Weakest dimension & fix
# ---------------------------------------------------------------------------
st.header("🔍 Weakest Dimension & Recommended Fix")

weakest = report.get("weakest_dimension", "unknown")
fix = report.get("recommended_fix", "No fix recommendation available.")

st.markdown(
    f"<div class='card'>"
    f"<p><b>Weakest Dimension:</b> "
    f"<span style='color:var(--error);'>{weakest}</span></p>"
    f"<p><b>Recommended Fix:</b><br>{fix}</p>"
    f"</div>",
    unsafe_allow_html=True,
)

st.markdown("---")

# ---------------------------------------------------------------------------
# RAGAS scores
# ---------------------------------------------------------------------------
st.header("📈 RAGAS Metrics")

ragas = report.get("ragas_scores", {})
diagnosis = report.get("ragas_diagnosis", "N/A")

if ragas:
    ragas_cols = st.columns(len(ragas))
    for i, (metric, score) in enumerate(ragas.items()):
        with ragas_cols[i]:
            score_color = (
                "var(--success)" if score >= 0.8
                else "var(--warning)" if score >= 0.5
                else "var(--error)"
            )
            st.markdown(
                f"<div class='metric-card'><h3 style='margin:0;font-size:1.6rem;color:{score_color};'>"
                f"{score:.2f}</h3>"
                f"<p style='margin:0;font-size:0.78rem;color:var(--text);text-transform:capitalize;'>"
                f"{metric.replace('_', ' ')}</p></div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='card'><b>Diagnosis:</b> {diagnosis}</div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        "<div class='custom-warning'>No RAGAS scores available.</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    "<p style='text-align:center;font-size:0.8rem;color:var(--text);opacity:0.6;'>"
    "BVRIT Chatbot Evaluation Suite · 8 Dimensions · RAGAS Metrics</p>",
    unsafe_allow_html=True,
    )
