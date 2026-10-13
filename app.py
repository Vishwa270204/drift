"""Streamlit UI for the Model Drift Detection Dashboard.

Run with: streamlit run app.py
"""

import os

import pandas as pd
import plotly.express as px
import streamlit as st

from drift import (
    CONT, DEFAULT_CSV_PATH, INCR, LEARNERS, ONLINE, PALETTE, REF_PCT,
    SCRATCH, STATIC, TARGET_COL, TRANS, detect_task, prepare_df, run_experiment,
    split_columns, style_fig, timeline_fig,
)





# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------



st.set_page_config(page_title="Model Drift Dashboard", page_icon="📉", layout="wide")

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
<style>
.block-container {padding-top: 1.6rem; max-width: 1280px;}
.hero {
    background: linear-gradient(120deg, #4F46E5 0%, #7C3AED 50%, #06B6D4 100%);
    padding: 1.8rem 2rem; border-radius: 18px; color: #fff; margin-bottom: 1.2rem;
    box-shadow: 0 10px 30px rgba(79,70,229,.25);
}
.hero h1 {margin: 0; font-size: 2rem; font-weight: 750; color: #fff;}
.hero p {margin: .35rem 0 0 0; opacity: .92; font-size: 1.02rem;}
.chip {
    display: inline-block; padding: .28rem .8rem; margin: .5rem .4rem 0 0; border-radius: 999px;
    background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.35);
    font-size: .82rem; font-weight: 600; color: #fff;
}
div[data-testid="stMetric"] {
    background: rgba(99,102,241,.07); border: 1px solid rgba(99,102,241,.22);
    border-left: 5px solid #6366F1; padding: .9rem 1.1rem; border-radius: 14px;
}
div[data-testid="stMetricLabel"] {font-weight: 600; opacity: .8;}
div[data-testid="stMetricValue"] {font-weight: 750;}
.stTabs [data-baseweb="tab-list"] {gap: .5rem;}
.stTabs [data-baseweb="tab"] {
    background: rgba(99,102,241,.08); border-radius: 10px 10px 0 0; padding: .55rem 1.2rem; font-weight: 600;
}
.stTabs [aria-selected="true"] {background: rgba(99,102,241,.22);}
.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #4F46E5, #7C3AED); border: 0; border-radius: 12px;
    padding: .65rem 2rem; font-weight: 700; box-shadow: 0 6px 16px rgba(79,70,229,.3);
}
.stButton > button[kind="primary"]:hover {filter: brightness(1.08);}
.section-note {
    background: rgba(6,182,212,.08); border-left: 4px solid #06B6D4;
    padding: .7rem 1rem; border-radius: 8px; margin-bottom: .8rem; font-size: .92rem;
}
.best-badge {
    background: linear-gradient(90deg, #10B981, #06B6D4); color: #fff; padding: .6rem 1.1rem;
    border-radius: 12px; font-weight: 650; display: inline-block; margin: .4rem 0 1rem 0;
}
</style>
""",
    unsafe_allow_html=True,
)




# ----------------------------------------------------------------------------
# Data preparation
# ----------------------------------------------------------------------------








# ----------------------------------------------------------------------------
# Models and metrics
# ----------------------------------------------------------------------------












# ----------------------------------------------------------------------------
# The experiment
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------------
HERO_TITLE = '''<h1>📉 Model Drift Detection Dashboard</h1>
  <p>ADWIN drift detection on prequential errors, with adaptation that starts only after a drift signal.</p>'''
hero_ph = st.empty()
hero_ph.markdown(f'<div class="hero">{HERO_TITLE}</div>', unsafe_allow_html=True)  # title first, chips added later

uploaded_file = st.file_uploader("Upload CSV (optional, otherwise data.csv is used)", type=["csv"])

try:
    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)
        source_name = uploaded_file.name
    elif os.path.exists(DEFAULT_CSV_PATH):
        df_raw = pd.read_csv(DEFAULT_CSV_PATH)
        source_name = DEFAULT_CSV_PATH
    else:
        st.warning("Upload a CSV file or add data.csv to the project.")
        st.stop()
except Exception as exc:
    st.error(f"Could not read the CSV file: {exc}")
    st.stop()

if df_raw.shape[0] < 100 or df_raw.shape[1] < 2:
    st.error("The CSV needs at least 100 rows and 2 columns.")
    st.stop()
df_raw.columns = [str(c) for c in df_raw.columns]

# ----------------------------------------------------------------------------
# Fixed target + automatic task
# ----------------------------------------------------------------------------
cols = list(df_raw.columns)
target = TARGET_COL if TARGET_COL in cols else cols[-1]

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
with st.expander("⚙️ Settings", expanded=True):
    a1, a2, a3 = st.columns(3)
    order_choice = a1.selectbox("Order rows by (optional)", ["(keep file order)"] + [c for c in cols if c != target],
                                help="Drift analysis is chronological. Pick a timestamp column if the file is not already in time order.")
    batch = a2.slider("Window / batch size (rows)", 20, 2000, 200, step=10)
    adwin_delta = a3.number_input("ADWIN delta", min_value=0.000001, max_value=0.5, value=0.002,
                                  step=0.001, format="%.4f")
    b1, b2 = st.columns(2)
    adapt_windows = b1.slider("Windows to adapt after a drift signal (k)", 1, 10, 3)
    buffer_size = b2.slider("Continual replay buffer size", 100, 5000, 500, step=100)
    d1, d2 = st.columns(2)
    strategies = d1.multiselect("Learning strategies", LEARNERS, default=LEARNERS)
    onset = d2.number_input("Known drift onset row (0 = unknown)", min_value=0, max_value=int(len(df_raw)),
                            value=0, step=100, help="Optional. Enables detection-delay and false-alarm counts.")
time_col = None if order_choice == "(keep file order)" else order_choice

# ----------------------------------------------------------------------------
# Dataset preparation
# ----------------------------------------------------------------------------
df = prepare_df(df_raw, target, time_col)
num_cols, cat_cols, roles = split_columns(df, target, time_col)
if len(num_cols) + len(cat_cols) == 0:
    st.error("No usable feature columns were found after excluding the target and invalid columns.")
    st.stop()

task = detect_task(df[target])
if task == "classification" and df[target].nunique() > 50:
    st.error("The target (last column) has more than 50 distinct values for classification. Use a different dataset.")
    st.stop()
model_name = "SGDClassifier" if task == "classification" else "SGDRegressor"

# ----------------------------------------------------------------------------
# Hero header
# ----------------------------------------------------------------------------
hero_ph.markdown(
    f"""
<div class="hero">
  {HERO_TITLE}
  <span class="chip">🎯 Target: {target}</span>
  <span class="chip">🤖 {task.capitalize()} · {model_name}</span>
  <span class="chip">📚 Training data: first {REF_PCT}% of rows</span>
  <span class="chip">⚡ Adapt after drift only</span>
</div>
""",
    unsafe_allow_html=True,
)

run_clicked = st.button("▶ Run Analysis", type="primary")

# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------
cfg = {"strategies": strategies, "batch": batch, "adwin_delta": float(adwin_delta),
       "adapt_windows": adapt_windows, "buffer": buffer_size, "onset": int(onset)}
cfg_key = repr((source_name, target, time_col, task, sorted(cfg.items(), key=lambda kv: kv[0])))

if run_clicked:
    if not strategies:
        st.error("Select at least one learning strategy.")
    else:
        bar = st.progress(0.0, text="Starting...")
        try:
            with st.spinner("Running ADWIN and the learning strategies..."):
                st.session_state["result"] = run_experiment(
                    df, num_cols, cat_cols, target, task, cfg, lambda f, t: bar.progress(f, text=t))
            st.session_state["result_key"] = cfg_key
        except ValueError as exc:
            st.session_state.pop("result", None)
            st.error(str(exc))
        except Exception as exc:
            st.session_state.pop("result", None)
            st.error(f"The analysis failed: {exc}")
        bar.empty()

res = st.session_state.get("result")

# ----------------------------------------------------------------------------
# KPI cards
# ----------------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("📄 Records", f"{len(df):,}")
k2.metric("🧩 Features used", len(num_cols) + len(cat_cols))
if res is not None:
    k3.metric("🚨 Drift events (ADWIN)", len(res["adwin_events"]))
    mdf = res["metrics"]
    latest = mdf[(mdf.strategy == res["best"]) & (mdf.window == mdf.window.max())].iloc[0]
    label = "F1 (macro)" if res["primary"] == "f1" else "MAE"
    k4.metric(f"🏁 Latest window {label}", f"{latest[res['primary']]:.3f}", help=f"Best strategy: {res['best']}")
else:
    k3.metric("🚨 Drift events (ADWIN)", "-", help="Press Run Analysis.")
    k4.metric("🏁 Latest model metric", "-", help="Press Run Analysis.")

with st.expander("Column roles"):
    st.dataframe(roles, use_container_width=True, hide_index=True)

if res is not None and st.session_state.get("result_key") != cfg_key:
    st.info("Settings changed since the last run. Press Run Analysis to refresh the results below.")

if res is None:
    st.info("Adjust the settings above and press **Run Analysis**.")
    st.stop()

windows = res["windows"]
first_w = res["first_flag_window"]
prim, higher = res["primary"], res["higher_better"]
plabel = "F1 (macro)" if prim == "f1" else "MAE"

st.markdown(f'<div class="best-badge">🏆 Best strategy: {res["best"]}</div>', unsafe_allow_html=True)

tab_a, tab_b = st.tabs(["🔍 Drift detection (ADWIN)", "📊 Learning strategy comparison"])

# ----------------------------------------------------------------------------
# A. Drift detection
# ----------------------------------------------------------------------------
with tab_a:
    st.caption(f"Reference: rows 0-{res['n_ref'] - 1}. Stream: {len(windows)} windows of about "
               f"{res['cfg']['batch']} rows. Adaptation was active in {len(res['active_windows'])} of {len(windows)} windows.")
    st.markdown('<div class="section-note">ADWIN watches the prequential error stream of the never-updated static model '
                '(0/1 errors for classification, clipped standardised absolute error for regression). It signals that the '
                'monitored signal changed. It cannot say which input feature caused the change.</div>',
                unsafe_allow_html=True)
    st.plotly_chart(timeline_fig(res), use_container_width=True)
    if res["adwin_events"]:
        ev = pd.DataFrame(res["adwin_events"]).rename(columns={"row": "stream row index", "window": "window #"})
        st.dataframe(ev, use_container_width=True, hide_index=True, height=min(250, 40 + 35 * len(ev)))
    else:
        st.success("No ADWIN drift events were detected.")
    if res["delays"] is not None:
        st.subheader("Detection delay (known onset)")
        cols_d = st.columns(len(res["delays"]))
        for c_, (k_, v_) in zip(cols_d, res["delays"].items()):
            c_.metric(k_, "not detected" if v_ is None else v_)

# ----------------------------------------------------------------------------
# B. Learning strategy comparison
# ----------------------------------------------------------------------------
with tab_b:
    summary = res["summary"]
    st.caption("Metrics are pooled over all stream rows. 'Post-drift' uses rows after the first window with an ADWIN event. "
               "'Recovery' counts windows from the first degraded window until the metric is within 5% of its "
               "pre-drift level. All models predict before they are updated.")
    st.dataframe(summary.style.format({c: "{:.4f}" for c in summary.columns
                                       if c in ("Accuracy", "F1 (macro)", "Post-drift F1", "MAE", "RMSE",
                                                "Post-drift MAE", "Update time (s)")}),
                 use_container_width=True, hide_index=True)

    smooth = st.checkbox("Smooth with a 3-window rolling mean", value=True)
    mdf = res["metrics"].sort_values(["strategy", "window"]).copy()
    mdf["value"] = mdf.groupby("strategy")[prim].transform(
        lambda s: s.rolling(3, min_periods=1).mean() if smooth else s)
    mdf["row"] = (mdf["start"] + mdf["end"]) / 2
    line = px.line(mdf, x="row", y="value", color="strategy", labels={"row": "Row index", "value": plabel})
    line.update_traces(line=dict(width=3))
    line.add_vline(x=res["n_ref"], line_dash="dot", line_color="gray")
    if first_w is not None:
        line.add_vline(x=windows[first_w][1], line_dash="dash", line_color="#EF4444", annotation_text="first drift signal")
    st.plotly_chart(style_fig(line, 420), use_container_width=True)

    c1, c2 = st.columns(2)
    basis = "post-drift" if res["rank_uses_post_drift"] else "overall"
    rank_df = pd.DataFrame({"strategy": list(res["rank_basis"].keys()), plabel: list(res["rank_basis"].values())})
    c1.plotly_chart(style_fig(px.bar(rank_df, x="strategy", y=plabel, color="strategy",
                                     title=f"{plabel} ({basis})").update_layout(showlegend=False), 360),
                    use_container_width=True)
    c2.plotly_chart(style_fig(px.bar(summary, x="Strategy", y="Update time (s)", color="Strategy",
                                     title="Total model update time (computational cost)").update_layout(showlegend=False), 360),
                    use_container_width=True)
