"""Streamlit dashboard: detect model drift on streaming sensor data and test recovery strategies."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import streamlit as st

from core import data as D
from core import drift as R
from core import modeling as M
from core import plots as P
from core import simulation as S

st.set_page_config(page_title="Model Drift Dashboard", page_icon=":chart_with_downwards_trend:", layout="wide")
st.markdown("<style>.block-container{padding-top:2rem;max-width:1250px}</style>", unsafe_allow_html=True)

SAMPLE, UPLOAD = "Sample data (generated)", "Upload CSV files"
OLD, NEW, BOTH, INCR = "Old data only", "New data only", "Old + New data", "Incremental update"
STRATEGY_HELP = {
    OLD: "**Full retrain** - fresh scaler + fresh SGDRegressor on baseline training rows only.",
    NEW: "**Full retrain** - fresh scaler + fresh SGDRegressor on labeled post-drift adaptation rows only.",
    BOTH: "**Full retrain** - fresh scaler + fresh SGDRegressor on baseline training rows + post-drift adaptation rows.",
    INCR: "**Incremental update** - copies the original fitted model, keeps its scaler and calls `partial_fit()` "
          "on the post-drift adaptation rows. Coefficients are continued, not reset.",
}
DEFAULT_PASSES = 5


# --------------------------------------------------------------------------- cached loading
@st.cache_data(show_spinner=False)
def load_csv(raw: bytes) -> pd.DataFrame:
    return D.read_csv_bytes(raw)


@st.cache_data(show_spinner=False)
def load_sample():
    return D.make_sample_data()


def df_hash(df: pd.DataFrame) -> bytes:
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes()).digest()


def fmt(v, d=3):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:,.{d}f}"


# --------------------------------------------------------------------------- sidebar: data
ss = st.session_state
sb = st.sidebar
sb.title("Controls")
sb.subheader("1. Data")
source = sb.radio("Data source", [SAMPLE, UPLOAD], key="source", label_visibility="collapsed")

if source == SAMPLE:
    base_raw, cur_raw = load_sample()
    sb.caption("Synthetic sensor data generated in-app (not your data). Switch to *Upload CSV files* to use your own.")
else:
    f_base = sb.file_uploader("Baseline (pre-drift) CSV", type="csv", key="up_base")
    f_cur = sb.file_uploader("Post-drift CSV", type="csv", key="up_cur")
    if f_base is None or f_cur is None:
        st.title("Model Drift Detection & Recovery")
        st.info("Upload both CSV files in the sidebar (or switch to the generated sample data) to begin.")
        st.stop()
    try:
        base_raw = load_csv(f_base.getvalue())
    except D.DataError as e:
        st.error(f"Baseline file: {e}")
        st.stop()
    try:
        cur_raw = load_csv(f_cur.getvalue())
    except D.DataError as e:
        st.error(f"Post-drift file: {e}")
        st.stop()

common = [c for c in base_raw.columns if c in cur_raw.columns]
col_sig = hashlib.md5(("|".join(base_raw.columns) + "#" + "|".join(cur_raw.columns)).encode()).hexdigest()[:8]
is_num = lambda s: pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)

sb.subheader("2. Columns")
ts_guess = next((c for c in common if any(k in c.lower() for k in ("time", "date"))), "(none)")
ts_opts = ["(none)"] + common
ts_sel = sb.selectbox("Timestamp column (optional)", ts_opts, index=ts_opts.index(ts_guess), key=f"ts_{col_sig}")
ts_col = None if ts_sel == "(none)" else ts_sel

target_opts = [c for c in base_raw.columns if c != ts_col and is_num(base_raw[c])]
if not target_opts:
    st.error("The baseline file has no numeric column that can be used as a target.")
    st.stop()
target = sb.selectbox("Target column", target_opts, index=len(target_opts) - 1, key=f"tg_{col_sig}_{ts_sel}")

feat_opts = [c for c in common if c not in (ts_col, target)]
feat_default = [c for c in feat_opts if is_num(base_raw[c]) or base_raw[c].nunique() <= 20]
features = sb.multiselect("Input feature columns", feat_opts, default=feat_default,
                          key=f"ft_{col_sig}_{ts_sel}_{target}")

# --------------------------------------------------------------------------- sidebar: thresholds
sb.subheader("3. Monitoring")
kind = sb.radio("Rolling error metric", ["MAE", "RMSE"], horizontal=True, key="kind")
window_in = sb.slider("Rolling window (observations)", 5, 200, 48, key="window")
factor = sb.slider("Performance threshold (x baseline error)", 1.1, 5.0, 1.5, 0.1, key="factor",
                   help="Threshold = this multiple of the original model's error on the held-out baseline rows.")
psi_warn = sb.number_input("PSI warning level", 0.01, 1.0, 0.10, 0.01, key="psi_warn")
psi_alert = sb.number_input("PSI alert level", 0.02, 2.0, 0.25, 0.01, key="psi_alert")
eval_pct = sb.slider("Held-out evaluation share of post-drift data (%)", 20, 50, 40, 5, key="eval_pct",
                     help="The LAST part of the post-drift data is never used for training or adaptation.")
min_change = sb.slider("Meaningful change in error (%)", 1, 20, 5, key="min_change",
                       help="MAE and RMSE must both move by at least this much to call it improved / worsened.")

sb.subheader("4. Training strategy")
sb.radio("Strategy", [OLD, NEW, BOTH, INCR], key="strategy", label_visibility="collapsed")
sb.caption("Applied when you press **Train / Update Model** in the main page.")

# --------------------------------------------------------------------------- prepare data + original model
try:
    prep = D.prepare(base_raw, cur_raw, ts_col, features, target)
    splits = D.split_data(prep, eval_pct / 100)
except D.DataError as e:
    st.title("Model Drift Detection & Recovery")
    st.error(str(e))
    st.stop()

X = D.X_COL
xtitle = "Time" if prep.x_is_time else ("Time (numeric)" if ts_col else "Observation #")
sig = hashlib.md5(df_hash(base_raw) + df_hash(cur_raw) +
                  repr((ts_col, features, target, eval_pct)).encode()).hexdigest()

if ss.get("sig") != sig:  # data or column selection changed -> reset everything model-related
    try:
        orig0 = M.train_full(splits.base_train, prep.features, prep.num_cols, prep.cat_cols, target,
                             "Original (baseline training rows)", n_new=0)
    except M.ModelError as e:
        st.title("Model Drift Detection & Recovery")
        st.error(str(e))
        st.stop()
    ss.update(sig=sig, orig=orig0, deployed=orig0, result=None, train_error=None, sim=S.new_sim())
orig: M.ModelBundle = ss["orig"]

# --------------------------------------------------------------------------- core computations
timeline = pd.concat([splits.base_val, prep.cur], ignore_index=True)  # held-out baseline rows, then all post-drift rows
pred_tl = M.predict(orig, timeline)
tx = timeline[X]
boundary_x = prep.cur[X].iloc[0]
eval_start_x = splits.cur_eval[X].iloc[0]
window = int(min(window_in, max(3, len(timeline) // 4)))
labels = prep.has_labels

if labels:
    actual_tl = timeline[target].to_numpy(float)
    err_tl = actual_tl - pred_tl
    n_val = len(splits.base_val)
    ref_err = M.regression_metrics(actual_tl[:n_val], pred_tl[:n_val])[kind.lower()]
    threshold = max(factor * ref_err, 1e-12)
    roll_tl = R.rolling_error(err_tl, window, kind)
    drift_i = R.detect_degradation(roll_tl, threshold)
    drift_x = tx.iloc[drift_i] if drift_i is not None else None
else:
    actual_tl = None

eval_df = splits.cur_eval
pred_eval_orig = M.predict(orig, eval_df)
m0 = M.regression_metrics(eval_df[target], pred_eval_orig) if labels else None
result = ss.get("result")


# --------------------------------------------------------------------------- training callback
def run_training():
    strategy = ss["strategy"]
    try:
        if strategy == OLD:
            b = M.train_full(splits.base_train, prep.features, prep.num_cols, prep.cat_cols, target, strategy, 0)
        elif strategy == NEW:
            b = M.train_full(splits.cur_adapt, prep.features, prep.num_cols, prep.cat_cols, target, strategy,
                             len(splits.cur_adapt))
        elif strategy == BOTH:
            both = pd.concat([splits.base_train, splits.cur_adapt], ignore_index=True)
            b = M.train_full(both, prep.features, prep.num_cols, prep.cat_cols, target, strategy,
                             len(splits.cur_adapt))
        else:
            b = M.incremental_update(ss["orig"], splits.cur_adapt, ss.get("passes", DEFAULT_PASSES), strategy)
        p = M.predict(b, splits.cur_eval)
        ss["result"] = {"bundle": b, "pred_eval": p, "strategy": strategy, "passes": ss.get("passes", DEFAULT_PASSES),
                        "m1": M.regression_metrics(splits.cur_eval[target], p)}
        ss["train_error"] = None
    except (M.ModelError, D.DataError) as e:
        ss["train_error"] = str(e)


# --------------------------------------------------------------------------- header
st.title("Model Drift Detection & Recovery")
if source == SAMPLE:
    st.info("Showing **generated sample data** (synthetic hourly sensor readings with built-in drift), not your data.", icon=":material/science:")
if prep.notes:
    with st.expander(f"Data notes ({len(prep.notes)})"):
        for n in prep.notes:
            st.write("- " + n)
st.caption(
    f"Original model: SGDRegressor trained on the first {len(splits.base_train)} baseline rows "
    f"({len(splits.base_val)} baseline rows held out). Post-drift data: {len(splits.cur_adapt)} adaptation rows "
    f"+ {len(splits.cur_eval)} held-out evaluation rows (chronological split)."
)

# --------------------------------------------------------------------------- 1. KPI cards
def kpi_row(title, m, n_train, n_new, ref=None):
    st.markdown(f"**{title}**")
    c = st.columns(5)
    for col, key, label in ((c[0], "mae", "MAE"), (c[1], "rmse", "RMSE")):
        delta = None if ref is None else f"{m[key] - ref[key]:+.3f}"
        col.metric(label, fmt(m[key]), delta, delta_color="inverse")
    r2 = m["r2"]
    delta = None if ref is None or r2 is None or ref["r2"] is None else f"{r2 - ref['r2']:+.3f}"
    c[2].metric("R²", fmt(r2), delta)
    c[3].metric("Training observations", f"{n_train:,}")
    c[4].metric("New observations used", f"{n_new:,}")


if labels:
    st.caption(f"KPIs are computed on the same held-out evaluation segment (n = {m0['n']}). Deltas compare the updated model with the original.")
    kpi_row("Original model", m0, orig.n_train, 0)
    if result:
        kpi_row(f"Updated model - {result['strategy']}", result["m1"], result["bundle"].n_train,
                result["bundle"].n_new, ref=m0)
    else:
        st.caption("Train / update a model in the *Retraining comparison* section to see the updated model here.")
else:
    st.warning("Actual target values are unavailable for the post-drift data, so MAE / RMSE / R² are not computed "
               "and model accuracy is unknown. Only predictions and feature drift (PSI) are shown.")

# --------------------------------------------------------------------------- 2. actual vs predicted
st.subheader("Actual vs predicted")
st.plotly_chart(P.fig_actual_vs_pred(
    tx, actual_tl if labels else None, pred_tl, boundary_x, eval_start_x, xtitle, target,
    x_eval=eval_df[X] if result else None, pred_upd=result["pred_eval"] if result else None), width="stretch")

# --------------------------------------------------------------------------- 3. degradation
st.subheader("Model degradation")
with st.expander("What does each signal monitor?"):
    st.markdown(
        "- **Rolling MAE/RMSE vs threshold** - *predictive performance*. Needs actual target values.\n"
        "- **PSI** - *input feature distribution* shift vs the baseline. Needs no labels, and a shifted feature "
        "does not by itself mean the model got worse.\n"
        "- **ADWIN** (change detection on an error stream) is *not* included in this demo."
    )
if labels:
    if drift_i is None:
        st.success(f"No degradation detected: rolling {kind} stayed at or below the threshold {threshold:.3g} "
                   f"(= {factor:.1f} x baseline hold-out {kind} {ref_err:.3g}).")
    else:
        st.warning(f"Degradation detected: rolling {kind} first exceeded the threshold {threshold:.3g} "
                   f"(= {factor:.1f} x baseline hold-out {kind} {ref_err:.3g}) at {tx.iloc[drift_i]} "
                   f"(observation {drift_i + 1} of {len(tx)} in the monitored timeline).", icon=":material/warning:")
    if window < window_in:
        st.caption(f"Window reduced to {window} because the timeline is short.")
    st.plotly_chart(P.fig_rolling(tx, roll_tl, threshold, boundary_x, drift_x, kind, window, xtitle), width="stretch")
    st.plotly_chart(P.fig_abs_error(tx, np.abs(err_tl), drift_x, boundary_x, xtitle), width="stretch")
else:
    st.info("Performance monitoring needs actual target values for the post-drift period.")

# --------------------------------------------------------------------------- 4. feature drift
st.subheader("Feature drift")
psi_df = R.compute_psi(prep.base, prep.cur, prep.features, prep.num_cols, psi_warn, psi_alert)
feat_sel = st.selectbox("Feature", prep.features, index=prep.features.index(psi_df["feature"].iloc[0]),
                        key=f"fsel_{sig}")
row = psi_df.loc[psi_df.feature == feat_sel].iloc[0]
c1, c2 = st.columns([1, 3])
with c1:
    st.metric("PSI (baseline vs post-drift)", fmt(row["psi"]))
    msg = f"{row['status']} (warn >= {psi_warn:.2f}, alert >= {psi_alert:.2f})"
    (st.error if row["status"] == "significant shift" else st.warning if row["status"] == "moderate shift" else st.success)(msg)
    st.caption("Feature drift does not necessarily mean predictive performance has degraded - check the error charts above.")
with c2:
    st.plotly_chart(P.fig_feature(prep.base[feat_sel], prep.cur[feat_sel], feat_sel in prep.num_cols, feat_sel),
                    width="stretch")
with st.expander("PSI for all features"):
    st.dataframe(psi_df.assign(psi=psi_df["psi"].round(4)), hide_index=True, width="stretch")

# --------------------------------------------------------------------------- 5. retraining comparison
st.subheader("Retraining comparison")
strategy = ss["strategy"]
st.markdown(STRATEGY_HELP[strategy])
if strategy == INCR:
    st.number_input("Passes over the new data (partial_fit calls)", 1, 50, DEFAULT_PASSES, key="passes")
    st.caption(f"Will use {len(splits.cur_adapt)} new labeled observations.")
elif strategy in (NEW, BOTH):
    st.caption(f"Will use {len(splits.cur_adapt)} post-drift labeled observations"
               + (f" + {len(splits.base_train)} baseline rows." if strategy == BOTH else "."))
else:
    st.caption(f"Will use {len(splits.base_train)} baseline rows (no post-drift data).")

st.button("Train / Update Model", type="primary", on_click=run_training, disabled=not labels,
          help=None if labels else "Needs actual target values in the post-drift data.")
if ss.get("train_error"):
    st.error(ss["train_error"])

if labels and result:
    m1 = result["m1"]
    tol = min_change / 100

    def rel(a, b):
        return 0.0 if a == b else (b - a) / a if a > 0 else np.inf

    dm, dr = rel(m0["mae"], m1["mae"]), rel(m0["rmse"], m1["rmse"])
    if dm <= -tol and dr <= -tol:
        st.success(f"**Performance improved** on held-out post-drift data (MAE {dm:+.1%}, RMSE {dr:+.1%}).")
    elif dm >= tol and dr >= tol:
        st.error(f"**Performance worsened** on held-out post-drift data (MAE {dm:+.1%}, RMSE {dr:+.1%}).")
    else:
        st.info(f"**No meaningful improvement** (MAE {dm:+.1%}, RMSE {dr:+.1%}; threshold +/-{min_change}% on both).")

    rows = []
    for name, k in (("MAE", "mae"), ("RMSE", "rmse"), ("R²", "r2")):
        a, b = m0[k], m1[k]
        both = a is not None and b is not None
        rows.append({"Metric": name, "Original": fmt(a, 4), "Updated": fmt(b, 4),
                     "Difference (updated - original)": f"{b - a:+.4f}" if both else "n/a",
                     "Change %": f"{(b - a) / abs(a):+.1%}" if both and name != "R²" and a else "-"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Lower MAE / RMSE and higher R² are better. Evaluated on the held-out final post-drift rows that "
               "neither model saw during training or adaptation.")

    xe = eval_df[X]
    actual_e = eval_df[target].to_numpy(float)
    w_eval = int(min(window, max(3, len(eval_df) // 3)))
    r0 = R.rolling_error(actual_e - pred_eval_orig, w_eval, kind)
    r1 = R.rolling_error(actual_e - result["pred_eval"], w_eval, kind)
    cc1, cc2 = st.columns(2)
    cc1.plotly_chart(P.fig_compare_pred(xe, actual_e, pred_eval_orig, result["pred_eval"], xtitle, target), width="stretch")
    cc2.plotly_chart(P.fig_compare_rolling(xe, r0, r1, kind, w_eval, xtitle), width="stretch")

    b1, b2 = st.columns([1, 3])
    if b1.button("Deploy this model to the simulation"):
        ss["deployed"] = result["bundle"]
        ss["sim"] = S.new_sim()
        st.toast("Deployed. Simulation was reset.")
    b2.caption(f"Currently deployed in the simulation: **{ss['deployed'].strategy}**")

# --------------------------------------------------------------------------- 6. live simulation
st.subheader("Live-data simulation (optional)")
st.caption("Replays the post-drift CSV in chronological batches. Predictions use features only; each actual value becomes "
           "'known' only after the label delay. The model is never updated automatically.")
sim = ss["sim"]
n_cur = len(prep.cur)
k1, k2, k3, k4 = st.columns([1, 1, 1, 1])
batch = k3.number_input("Batch size", 1, 200, 10, key="sim_batch")
delay = k4.number_input("Label delay (observations)", 0, 500, 20, key="sim_delay",
                        disabled=not labels, help="Observation i's actual value is available once i + delay observations have been processed.")
if not labels:
    delay = 0
done = sim["pos"] >= n_cur
if k1.button("Pause" if sim["running"] else "Start", type="primary", disabled=done and not sim["running"]):
    sim["running"] = not sim["running"]
    st.rerun()
if k2.button("Reset simulation"):
    ss["sim"] = S.new_sim()
    st.rerun()

u1, u2 = st.columns([1, 3])
n_avail = S.labeled_count(sim, delay)
if u1.button("Update deployed model", disabled=not labels or n_avail - sim["last_update"] < 1,
             help="partial_fit on labeled observations received since the last update."):
    try:
        new_obs = prep.cur.iloc[sim["last_update"]:n_avail]
        ss["deployed"] = M.incremental_update(ss["deployed"], new_obs, ss.get("passes", DEFAULT_PASSES),
                                              "Incremental update (simulation)")
        sim["last_update"] = n_avail
        sim["update_marks"].append(prep.cur[X].iloc[n_avail - 1])
        st.toast(f"Deployed model updated with {len(new_obs)} labeled observations.")
    except M.ModelError as e:
        st.warning(str(e))
u2.caption(f"Deployed model: **{ss['deployed'].strategy}** - {ss['deployed'].n_new} post-drift labeled observations "
           f"used so far. Observations awaiting an update: {max(0, n_avail - sim['last_update'])}.")


def sim_body():
    sm = ss["sim"]
    if sm["running"]:
        S.step(sm, ss["deployed"], prep.cur, batch)
        if not sm["running"]:
            st.rerun()  # finished: stop the timer
    if sm["pos"] == 0:
        st.info("Press **Start** to begin streaming.")
        return
    v = S.view(sm, prep.cur, target, delay, labels)
    lab = v[v.label_available]
    m = st.columns(4)
    m[0].metric("Processed", f"{sm['pos']} / {n_cur}")
    m[1].metric("Labels received", len(lab))
    m[2].metric("Awaiting label", sm["pos"] - len(lab))
    roll = None
    if len(lab):
        mp = min(window, max(3, window // 4))
        roll = R.rolling_error(lab["actual"] - lab["prediction"], window, kind, min_periods=mp)
        last = roll.dropna()
        m[3].metric(f"Rolling {kind}", fmt(last.iloc[-1]) if len(last) else "n/a")
    else:
        m[3].metric(f"Rolling {kind}", "n/a")
    g1, g2 = st.columns(2)
    g1.plotly_chart(P.fig_sim_stream(v, xtitle, target), width="stretch")
    if roll is not None and roll.notna().any():
        g2.plotly_chart(P.fig_sim_rolling(lab["x"], roll, threshold, kind, sm["update_marks"], xtitle), width="stretch")
    else:
        g2.info("Performance chart appears once enough labels have arrived.")
    shown = v.tail(8).copy()
    shown["actual"] = shown["actual"].where(shown["label_available"]).map(lambda a: "pending" if pd.isna(a) else f"{a:.3f}")
    shown["prediction"] = shown["prediction"].round(3)
    st.dataframe(shown[["obs", "x", "prediction", "actual"]].iloc[::-1], hide_index=True, width="stretch")
    st.download_button("Download stored observations + predictions (CSV)",
                       pd.concat([prep.cur[prep.features].iloc[:sm["pos"]].reset_index(drop=True),
                                  v[["prediction", "actual"]]], axis=1).to_csv(index=False),
                       "simulation_log.csv", "text/csv")


st.fragment(run_every=0.7 if sim["running"] else None)(sim_body)()