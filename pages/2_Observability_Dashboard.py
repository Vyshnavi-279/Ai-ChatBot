"""
pages/2_Observability_Dashboard.py — Streamlit multipage dashboard.

Shows requests-over-time, latency metrics, cost calculator, refusal rate,
top-10 questions, and recent errors — all styled with the spec.md section 7
light theme (8px grid, hover states, custom alerts).
"""

from __future__ import annotations

import csv
import io
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.express as px

# Must be the first Streamlit command.
st.set_page_config(
    page_title="Observability Dashboard",
    page_icon="📊",
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

/* ── Sidebar — 8px grid ── */
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
h1 { margin-bottom: 8px !important; }
h2 { margin-bottom: 8px !important; }
h3 { margin-bottom: 8px !important; }

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

/* ── Metric cards ── */
.metric-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px;
  text-align: center;
  box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  transition: all 0.2s ease;
  margin-bottom: 16px;
}
.metric-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0,0,0,0.10);
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

/* ── Custom alert boxes (replace st.error, st.warning, st.info) ── */
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
hr {
  border-color: var(--border) !important;
  margin: 16px 0 !important;
}

/* ── Responsive narrow width ── */
@media (max-width: 768px) {
  .block-container { padding: 16px !important; }
  [data-testid="stSidebar"] { padding: 16px 12px !important; }
  .metric-card { padding: 12px !important; }
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Per-model token-cost constants (editable via sidebar).
# ---------------------------------------------------------------------------
DEFAULT_COST_PER_1K_INPUT = {
    "openai/gpt-4o-mini": 0.00015,
    "openai/gpt-4o": 0.0025,
    "deepseek/deepseek-chat": 0.00014,
    "default": 0.00015,
}
DEFAULT_COST_PER_1K_OUTPUT = {
    "openai/gpt-4o-mini": 0.0006,
    "openai/gpt-4o": 0.01,
    "deepseek/deepseek-chat": 0.00028,
    "default": 0.0006,
}

# ---------------------------------------------------------------------------
# Load data from observability logger.
# ---------------------------------------------------------------------------
def _load_data() -> pd.DataFrame:
    """Fetch all rows from observability.db and return as a DataFrame."""
    try:
        from observability.logger import fetch_all
        rows = fetch_all()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        # Parse timestamp column.
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["latency_seconds"] = pd.to_numeric(df["latency_seconds"], errors="coerce").fillna(0)
        df["input_tokens"] = pd.to_numeric(df["input_tokens"], errors="coerce").fillna(0).astype(int)
        df["output_tokens"] = pd.to_numeric(df["output_tokens"], errors="coerce").fillna(0).astype(int)
        df["refused"] = pd.to_numeric(df["refused"], errors="coerce").fillna(0).astype(int)
        return df
    except Exception as e:
        st.markdown(
            f"<div class='custom-error'>Could not load observability data: {e}</div>",
            unsafe_allow_html=True,
        )
        return pd.DataFrame()


df = _load_data()

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
st.title("📊 Observability Dashboard")
st.markdown(
    "<p style='color:var(--text);font-size:0.95rem;margin-bottom:24px;'>"
    "Request-level metrics, latency, costs, and errors.</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ---------------------------------------------------------------------------
# Sidebar — cost configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 💰 Token Cost Rates ($/1K tokens)")

    cost_per_1k_input = {}
    cost_per_1k_output = {}
    models_in_data = set(df["model_name"].unique()) if not df.empty else {"default"}

    for model in sorted(models_in_data):
        default_in = DEFAULT_COST_PER_1K_INPUT.get(model, DEFAULT_COST_PER_1K_INPUT["default"])
        default_out = DEFAULT_COST_PER_1K_OUTPUT.get(model, DEFAULT_COST_PER_1K_OUTPUT["default"])
        cost_per_1k_input[model] = st.number_input(
            f"Input — {model}",
            value=default_in,
            format="%.6f",
            key=f"cost_in_{model}",
        )
        cost_per_1k_output[model] = st.number_input(
            f"Output — {model}",
            value=default_out,
            format="%.6f",
            key=f"cost_out_{model}",
        )

    st.markdown("---")
    st.markdown(
        "<p style='font-size:0.8rem;color:var(--text);'>"
        "Data refreshes on page reload.</p>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Compute derived metrics
# ---------------------------------------------------------------------------
if df.empty:
    st.markdown(
        "<div class='custom-warning'>No observability data yet. Start chatting to generate logs.</div>",
        unsafe_allow_html=True,
    )
    st.stop()

# Cost calculations.
def _compute_cost(row: pd.Series) -> float:
    model = row.get("model_name", "default")
    rate_in = cost_per_1k_input.get(model, DEFAULT_COST_PER_1K_INPUT["default"])
    rate_out = cost_per_1k_output.get(model, DEFAULT_COST_PER_1K_OUTPUT["default"])
    return (row["input_tokens"] / 1000 * rate_in) + (row["output_tokens"] / 1000 * rate_out)

df["cost"] = df.apply(_compute_cost, axis=1)

total_cost_today = df[df["timestamp"].dt.date == datetime.now().date()]["cost"].sum()
total_cost_this_week = df[df["timestamp"] >= pd.Timestamp.now() - pd.Timedelta(days=7)]["cost"].sum()

avg_latency = df["latency_seconds"].mean()
p95_latency = df["latency_seconds"].quantile(0.95)
total_requests = len(df)
refusal_rate = df["refused"].mean() * 100 if total_requests > 0 else 0.0

# ---------------------------------------------------------------------------
# Row 1 — Key metrics
# ---------------------------------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(
        f"<div class='metric-card'><h3 style='margin:0;font-size:1.5rem;'>{total_requests:,}</h3>"
        f"<p style='margin:0;font-size:0.8rem;color:var(--text);'>Total Requests</p></div>",
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f"<div class='metric-card'><h3 style='margin:0;font-size:1.5rem;'>{avg_latency:.2f}s</h3>"
        f"<p style='margin:0;font-size:0.8rem;color:var(--text);'>Avg Latency</p></div>",
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        f"<div class='metric-card'><h3 style='margin:0;font-size:1.5rem;'>{p95_latency:.2f}s</h3>"
        f"<p style='margin:0;font-size:0.8rem;color:var(--text);'>P95 Latency</p></div>",
        unsafe_allow_html=True,
    )
with col4:
    st.markdown(
        f"<div class='metric-card'><h3 style='margin:0;font-size:1.5rem;color:var(--error);'>{refusal_rate:.1f}%</h3>"
        f"<p style='margin:0;font-size:0.8rem;color:var(--text);'>Refusal Rate</p></div>",
        unsafe_allow_html=True,
    )
with col5:
    st.markdown(
        f"<div class='metric-card'><h3 style='margin:0;font-size:1.5rem;'>${total_cost_today:.4f}</h3>"
        f"<p style='margin:0;font-size:0.8rem;color:var(--text);'>Cost Today</p></div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ---------------------------------------------------------------------------
# Row 2 — Requests over time (line chart)
# ---------------------------------------------------------------------------
st.subheader("📈 Requests Over Time")
df_time = df.set_index("timestamp").sort_index()
if not df_time.empty:
    if len(df_time) > 50:
        requests_per_hour = df_time.resample("1h").size()
    else:
        requests_per_hour = df_time.resample("1min").size()

    fig_reqs = px.line(
        x=requests_per_hour.index,
        y=requests_per_hour.values,
        labels={"x": "Time", "y": "Requests"},
        markers=True,
    )
    fig_reqs.update_traces(line_color="#3D5A80")
    fig_reqs.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#293241",
        margin=dict(l=0, r=0, t=0, b=0),
    )
    st.plotly_chart(fig_reqs, use_container_width=True)
else:
    st.markdown(
        "<div class='custom-warning'>Not enough data to plot requests over time.</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ---------------------------------------------------------------------------
# Row 3 — Refusal rate over time + Cost summary
# ---------------------------------------------------------------------------
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("🚫 Refusal Rate Over Time")
    if not df_time.empty and len(df_time) > 10:
        refusal_over_time = df_time["refused"].resample("1h" if len(df_time) > 50 else "1min").mean() * 100
        fig_refusal = px.line(
            x=refusal_over_time.index,
            y=refusal_over_time.values,
            labels={"x": "Time", "y": "Refusal Rate (%)"},
            markers=True,
        )
        fig_refusal.update_traces(line_color="#BC4749")
        fig_refusal.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#293241",
            margin=dict(l=0, r=0, t=0, b=0),
        )
        st.plotly_chart(fig_refusal, use_container_width=True)
    else:
        st.markdown(
            "<div class='custom-warning'>Not enough data to plot refusal rate.</div>",
            unsafe_allow_html=True,
        )

with col_right:
    st.subheader("💰 Estimated Token Costs")
    st.markdown(
        f"<div class='card'><b>Cost Today:</b> ${total_cost_today:.4f}<br>"
        f"<b>Cost This Week:</b> ${total_cost_this_week:.4f}<br>"
        f"<b>Avg Cost Per Request:</b> ${(df['cost'].mean()):.6f}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Latency metrics as a small table.
    st.subheader("⚡ Latency Breakdown")
    lat_df = pd.DataFrame({
        "Metric": ["Average", "P50 (Median)", "P95", "P99", "Max"],
        "Seconds": [
            f"{avg_latency:.2f}",
            f"{df['latency_seconds'].median():.2f}",
            f"{p95_latency:.2f}",
            f"{df['latency_seconds'].quantile(0.99):.2f}",
            f"{df['latency_seconds'].max():.2f}",
        ],
    })
    st.dataframe(lat_df, hide_index=True, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Row 4 — Top 10 most-asked questions
# ---------------------------------------------------------------------------
st.subheader("🔝 Top 10 Most-Asked Questions")
if not df.empty and "question" in df.columns:
    top_questions = (
        df.groupby("question")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(10)
    )
    st.dataframe(top_questions, hide_index=True, use_container_width=True)
else:
    st.markdown(
        "<div class='custom-warning'>No questions logged yet.</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ---------------------------------------------------------------------------
# Row 5 — Recent errors table
# ---------------------------------------------------------------------------
st.subheader("❌ Recent Errors (Last 20)")
try:
    from observability.logger import fetch_recent_errors
    errors = fetch_recent_errors(limit=20)
    if errors:
        err_df = pd.DataFrame(errors)
        cols = [c for c in ["timestamp", "session_id", "question", "dimension_or_tool", "error"] if c in err_df.columns]
        st.dataframe(
            err_df[cols],
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.markdown(
            "<div class='custom-success'>No errors recorded!</div>",
            unsafe_allow_html=True,
        )
except Exception:
    # Fallback: filter the main DataFrame.
    err_df = df[df["error"].notna() & (df["error"] != "")].head(20)
    if not err_df.empty:
        cols = [c for c in ["timestamp", "session_id", "question", "dimension_or_tool", "error"] if c in err_df.columns]
        st.dataframe(err_df[cols], hide_index=True, use_container_width=True)
    else:
        st.markdown(
            "<div class='custom-success'>No errors recorded!</div>",
            unsafe_allow_html=True,
        )

st.markdown("---")

# ---------------------------------------------------------------------------
# Footer — download as CSV
# ---------------------------------------------------------------------------
if st.button("📥 Download Full Log as CSV"):
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(df.columns.tolist())
    for _, row in df.iterrows():
        writer.writerow(row.tolist())
    st.download_button(
        label="📄 Download CSV",
        data=csv_buffer.getvalue(),
        file_name=f"observability_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )
