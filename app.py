import copy
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Model Drift Monitor", layout="wide")
NONE = "(row order)"
METRICS = ["MAE", "RMSE", "R²"]
STRATS = ["New data only", "Old data + new data", "Incremental update"]
ss = st.session_state


# ---------------------------------------------------------------- helpers
def score(y, p, m):
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) == 0:
        return np.nan
    if m == "MAE":
        return float(mean_absolute_error(y, p))
    if m == "RMSE":
        return float(np.sqrt(mean_squared_error(y, p)))
    return float(r2_score(y, p)) if len(y) > 1 and np.var(y) > 0 else np.nan


def roll_metric(y, p, w, m):
    e = pd.Series(np.asarray(y, float) - np.asarray(p, float))
    if m == "MAE":
        return e.abs().rolling(w).mean().values
    if m == "RMSE":
        return np.sqrt((e ** 2).rolling(w).mean()).values
    var = pd.Series(np.asarray(y, float)).rolling(w).var(ddof=0).replace(0, np.nan)
    return (1 - (e ** 2).rolling(w).mean() / var).values


def threshold(ref, t, m):
    return ref - t * max(abs(ref), 1e-9) if m == "R²" else ref * (1 + t)


def is_degraded(val, ref, t, m):
    return val < threshold(ref, t, m) if m == "R²" else val > threshold(ref, t, m)


def improvement(cand, base, m):  # positive = better
    if m == "R²":
        return (cand - base) / max(abs(base), 1e-9)
    return (base - cand) / max(abs(base), 1e-9)


def psi(b, c):
    b, c = b.dropna(), c.dropna()
    if len(b) == 0 or len(c) == 0:
        return np.nan
    if pd.api.types.is_numeric_dtype(b):
        cuts = np.unique(np.quantile(b, np.linspace(0.1, 0.9, 9)))
        pb = np.bincount(np.searchsorted(cuts, b.values, side="right"), minlength=len(cuts) + 1) / len(b)
        pc = np.bincount(np.searchsorted(cuts, c.values, side="right"), minlength=len(cuts) + 1) / len(c)
    else:
        b, c = b.astype(str), c.astype(str)
        cats = sorted(set(b) | set(c))
        pb = b.value_counts(normalize=True).reindex(cats, fill_value=0).values
        pc = c.value_counts(normalize=True).reindex(cats, fill_value=0).values
    pb, pc = np.clip(pb, 1e-4, None), np.clip(pc, 1e-4, None)
    return float(np.sum((pc - pb) * np.log(pc / pb)))


def raw_features(df, p):
    X = (df[p["num"]].apply(pd.to_numeric, errors="coerce")
         .replace([np.inf, -np.inf], np.nan).fillna(p["med"]))
    if p["cat"]:
        cat = df[p["cat"]].astype(object).fillna("__na__").astype(str)
        D = pd.get_dummies(cat, dtype=float).reindex(columns=p["dcols"], fill_value=0.0)
        X = pd.concat([X, D], axis=1)
    return X


def make_X(df):  # scaler is fitted once on the baseline training rows and never refitted
    p = ss.S["p"]
    return p["sc"].transform(raw_features(df, p).values)


def predict(model, df):
    p = ss.S["p"]
    return model.predict(make_X(df)) * p["ys"] + p["ym"]


def scaled_y(y):
    p = ss.S["p"]
    return (np.asarray(y, float) - p["ym"]) / p["ys"]


def shade(fig, x0, x1, color, row=None):
    kw = dict(row=row, col=1) if row else {}
    fig.add_vrect(x0=x0, x1=x1, fillcolor=color, opacity=0.12, line_width=0, layer="below", **kw)


# ---------------------------------------------------------------- baseline
def init_baseline(df, target, feats, tcol, frac, lag):
    d = df.copy()
    if tcol != NONE:
        d[tcol] = pd.to_datetime(d[tcol], errors="coerce")
        d = d.dropna(subset=[tcol]).sort_values(tcol, kind="stable")
    d[target] = pd.to_numeric(d[target], errors="coerce")
    n_bad = int(d[target].isna().sum())
    d = d.dropna(subset=[target]).reset_index(drop=True)  # labels are never invented
    if n_bad:
        st.warning(f"Dropped {n_bad} rows with a missing/non-numeric target.")
    d["_x"] = d[tcol] if tcol != NONE else np.arange(len(d))
    feats = list(feats)
    if lag:  # previous actual: known at prediction time (one-step-ahead)
        d["_lag1"] = d[target].shift(1)
        feats.append("_lag1")
        d = d.iloc[1:].reset_index(drop=True)
    n = len(d)
    n_tr = int(n * frac)
    if n_tr < 30 or n - n_tr < 30:
        st.error(f"Not enough rows for a reliable split (total {n}, train {n_tr}, stream {n - n_tr}). Need ≥30 each.")
        return
    fit_n = int(n_tr * 0.85)  # last 15% of training = baseline reference (not fitted on)
    fit = d.iloc[:fit_n]
    num = [c for c in feats if pd.api.types.is_numeric_dtype(d[c])]
    cat = [c for c in feats if c not in num]
    wide = [c for c in cat if d[c].nunique() > 30]
    if wide:
        st.warning(f"Skipped high-cardinality columns: {wide}")
    cat = [c for c in cat if c not in wide]
    if not num and not cat:
        st.error("No usable feature columns.")
        return
    p = dict(num=num, cat=cat, med=fit[num].apply(pd.to_numeric, errors="coerce").median().fillna(0))
    p["dcols"] = list(pd.get_dummies(fit[cat].astype(object).fillna("__na__").astype(str), dtype=float).columns) if cat else []
    p["ym"], p["ys"] = float(fit[target].mean()), float(fit[target].std() or 1.0)
    p["sc"] = StandardScaler().fit(raw_features(fit, p).values)
    ss.S = dict(d=d, feats=num + cat, target=target, n_tr=n_tr, fit_n=fit_n, p=p)
    model = SGDRegressor(random_state=0).fit(make_X(fit), scaled_y(fit[target]))
    ss.S["model"] = model
    ss.S["pred"] = predict(model, d)  # features only; no stream labels are used
    ss.pos, ss.cands, ss.split_key, ss.drift = 0, {}, None, None
    ss.active, ss.amodel = "Original model", None


# ---------------------------------------------------------------- sidebar
sb = st.sidebar
sb.header("Controls")
up = sb.file_uploader("CSV file", type="csv")
path = sb.text_input("…or path to existing CSV", "")
raw = None
if up is not None:
    raw = pd.read_csv(up)
elif path and os.path.exists(path):
    raw = pd.read_csv(path)
elif path:
    sb.error("Path not found.")

st.title("Model Drift, Performance Monitoring & Retraining")
st.caption("CSV-based live-stream simulation — rows are replayed in order; this is not a real-time production stream.")

if raw is None:
    st.info("Upload a CSV (or enter its path) in the sidebar.")
    st.stop()

with st.expander("Dataset inspection", expanded=False):
    st.write(f"{raw.shape[0]} rows × {raw.shape[1]} columns")
    st.dataframe(pd.DataFrame({"dtype": raw.dtypes.astype(str), "missing": raw.isna().sum(),
                               "unique": raw.nunique()}))
    st.dataframe(raw.head())

cols = list(raw.columns)
num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(raw[c])]
if not num_cols:
    st.error("No numeric column found, so there is no suitable regression target.")
    st.stop()
target = sb.selectbox("Target column", num_cols, index=len(num_cols) - 1)
tcol = sb.selectbox("Timestamp column", [NONE] + cols)
feats = sb.multiselect("Feature columns", [c for c in cols if c not in (target, tcol)],
                       default=[c for c in num_cols if c != target and c != tcol])
lag = sb.checkbox("Add previous target (lag-1) as feature", value=False)
frac = sb.slider("Initial training proportion", 0.3, 0.9, 0.7, 0.05)
metric = sb.selectbox("Performance metric", METRICS)
win = sb.number_input("Rolling window (rows)", 5, 1000, 30)
thr_pct = sb.slider("Degradation threshold (% worse than baseline)", 5, 200, 30) / 100
delay = sb.number_input("Label delay (rows)", 0, 500, 0, help="Actuals arrive this many rows after the prediction.")
if sb.button("Initialize / train baseline", type="primary"):
    if not feats and not lag:
        sb.error("Select at least one feature.")
    else:
        init_baseline(raw, target, feats, tcol, frac, lag)

if "S" not in ss:
    st.info("Configure the sidebar and click **Initialize / train baseline**.")
    st.stop()

S = ss.S
d, n_tr, fit_n, tgt = S["d"], S["n_tr"], S["fit_n"], S["target"]
n_stream = len(d) - n_tr
y_all, p_all, xs = d[tgt].values, S["pred"], d["_x"]

# ---------------------------------------------------------------- stream controls
c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
batch = c1.number_input("Batch size", 1, max(n_stream, 1), min(50, n_stream))
if c2.button("Process next batch"):
    ss.pos = min(ss.pos + batch, n_stream)
if c3.button("Process all"):
    ss.pos = n_stream
if c4.button("Reset stream"):
    ss.pos, ss.cands, ss.split_key, ss.drift = 0, {}, None, None
k = ss.pos
st.progress(k / n_stream, text=f"Processed {k} / {n_stream} stream rows · active model: {ss.active}")
if k == 0:
    st.info("Process a batch to start the simulation.")
    st.stop()

# ---------------------------------------------------------------- monitoring
lab = max(k - delay, 0)  # rows whose actuals have arrived
ref = score(y_all[fit_n:n_tr], p_all[fit_n:n_tr], metric)
roll = roll_metric(y_all[n_tr:n_tr + lab], p_all[n_tr:n_tr + lab], win, metric)
bad = np.where(~np.isnan(roll) & is_degraded(roll, ref, thr_pct, metric))[0] if lab >= win else []
d_rel = int(bad[0]) if len(bad) else None
ss.drift = d_rel
onset = max(d_rel - win + 1, 0) if d_rel is not None else None
x_stream = xs.iloc[n_tr:n_tr + k]

if delay and k - lab:
    st.caption(f"⏳ {k - lab} most recent rows are pending labels (delay = {delay}); metrics use labeled rows only.")
if lab < win:
    st.warning(f"Performance metric pending: need {win} labeled rows (have {lab}).")
elif d_rel is None:
    st.success(f"No performance degradation detected so far (reference {metric} = {ref:.4g}).")
else:
    st.error(f"Performance degradation detected at stream row {d_rel + delay} "
             f"(approx. onset: row {onset}). Data drift alone does not confirm concept drift.")

# Graph 1
st.subheader("1 · Actual vs Predicted")
n_show = n_tr + k
g1 = go.Figure()
g1.add_scatter(x=xs[:n_show], y=y_all[:n_show], name="Actual", line=dict(color="#444", width=1))
g1.add_scatter(x=xs[:n_show], y=p_all[:n_show], name="Original model", line=dict(color="#1f77b4", width=1))
if ss.amodel is not None:
    g1.add_scatter(x=x_stream, y=predict(ss.amodel, d.iloc[n_tr:n_tr + k]), name=f"Active: {ss.active}",
                   line=dict(color="#2ca02c", width=1))
shade(g1, xs.iloc[0], xs.iloc[n_tr - 1], "#1f77b4")
if d_rel is not None:
    shade(g1, xs.iloc[n_tr + onset], xs.iloc[n_show - 1], "#d62728")
    g1.add_scatter(x=[xs.iloc[n_tr + d_rel]], y=[y_all[n_tr + d_rel]], mode="markers", name="Degradation detected",
                   marker=dict(color="red", size=11, symbol="x"))
g1.update_layout(height=380, margin=dict(t=20), legend=dict(orientation="h"),
                 xaxis_title="Time / index", yaxis_title=tgt)
st.plotly_chart(g1, width="stretch")
st.caption("Blue shading: baseline training period · Red shading: post-drift period (from approximate onset).")

# Graph 2
st.subheader("2 · Model Performance / Degradation")
g2 = go.Figure()
if metric != "R²":
    g2.add_scatter(x=x_stream.iloc[:lab], y=np.abs(y_all[n_tr:n_tr + lab] - p_all[n_tr:n_tr + lab]),
                   name="Absolute error", line=dict(color="#bbb", width=1))
g2.add_scatter(x=x_stream.iloc[:lab], y=roll, name=f"Rolling {metric}", line=dict(color="#1f77b4"))
g2.add_hline(y=threshold(ref, thr_pct, metric), line_dash="dash", line_color="red", annotation_text="Threshold")
g2.add_hline(y=ref, line_dash="dot", line_color="green", annotation_text="Baseline reference")
if lab:
    shade(g2, x_stream.iloc[0], x_stream.iloc[(onset if d_rel is not None else lab - 1)], "#2ca02c")
if d_rel is not None:
    shade(g2, x_stream.iloc[onset], x_stream.iloc[lab - 1], "#d62728")
    g2.add_scatter(x=[x_stream.iloc[d_rel]], y=[roll[d_rel]], mode="markers", name="Degradation detected",
                   marker=dict(color="red", size=12, symbol="x"))
g2.update_layout(height=340, margin=dict(t=20), legend=dict(orientation="h"), xaxis_title="Time / index",
                 yaxis_title=metric)
st.plotly_chart(g2, width="stretch")
st.caption("Green: pre-drift · Red: post-drift. The baseline reference uses the last 15% of the training segment, which the model was not fitted on.")

# Graph 3
st.subheader("3 · Data Drift (PSI)")
pc1, pc2, pc3 = st.columns(3)
base_n = pc1.slider("Baseline window (first N training rows)", 20, n_tr, min(n_tr, 200))
cur_n = pc2.slider("Current window (latest N processed rows)", 10, max(k, 11), min(max(k, 11), 100))
psi_t = pc3.slider("PSI threshold", 0.05, 0.5, 0.2, 0.05)
bw, cw = d.iloc[:base_n], d.iloc[n_tr + max(k - cur_n, 0):n_tr + k]
ps = pd.Series({f: psi(bw[f], cw[f]) for f in S["feats"]})
g3 = go.Figure(go.Bar(x=ps.index, y=ps.values, marker_color=["#d62728" if v > psi_t else "#1f77b4" for v in ps.fillna(0)]))
g3.add_hline(y=psi_t, line_dash="dash", line_color="red")
g3.update_layout(height=320, margin=dict(t=20), yaxis_title="PSI")
st.plotly_chart(g3, width="stretch")
over = ps[ps > psi_t].index.tolist()
st.caption(f"Features above threshold: {over if over else 'none'}. PSI flags input distribution change only; it does not prove concept drift.")

# ---------------------------------------------------------------- retraining
st.header("Model Update / Retraining")
if d_rel is None:
    st.info("Available once performance degradation is detected.")
    st.stop()

ev_frac = st.slider("Held-out evaluation share of post-drift labeled rows", 0.2, 0.5, 0.3, 0.05)
gap = st.number_input("Purge gap between training and evaluation rows", 0, 100, 1 if lag else 0,
                      help="Rows skipped so overlapping windows/lag features can't leak into evaluation.")
m_pool = lab - onset
n_ev = int(m_pool * ev_frac)
tr_end, ev_s = onset + m_pool - n_ev - gap, onset + m_pool - n_ev  # stream-relative
if tr_end - onset < 10 or n_ev < 10:
    st.warning(f"Not enough labeled post-drift rows yet (pool {m_pool}). Process more data.")
    st.stop()
key = (onset, tr_end, ev_s, lab)
if ss.split_key != key:
    if ss.cands:
        st.warning("Evaluation split changed with new data — previous candidates cleared; retrain them.")
    ss.cands, ss.split_key = {}, key
pool_n = tr_end - onset
ev = d.iloc[n_tr + ev_s:n_tr + lab]
st.caption(f"Post-drift labeled pool: {pool_n} training rows (stream {onset}–{tr_end - 1}) · "
           f"{n_ev} held-out evaluation rows (stream {ev_s}–{lab - 1}) · scaler/target scaling stay fixed from the baseline.")

tabs = st.tabs(STRATS)
for strat, tab in zip(STRATS, tabs):
    with tab:
        n_new = st.slider(f"New rows to use ({strat})", min(10, pool_n), pool_n, pool_n)
        epochs = st.slider(f"Passes ({strat})", 1, 20, 3) if strat == "Incremental update" else 1
        new = d.iloc[n_tr + tr_end - n_new:n_tr + tr_end]  # most recent labeled rows before the gap
        n_rows = n_new + (n_tr if strat == "Old data + new data" else 0)
        st.write(f"Training rows: **{n_rows}**")
        if st.button("Run update", key=f"run_{strat}"):
            if strat == "Incremental update":
                m = copy.deepcopy(S["model"])
                Xn, yn = make_X(new), scaled_y(new[tgt])
                for _ in range(epochs):
                    m.partial_fit(Xn, yn)
            else:
                tr = pd.concat([d.iloc[:n_tr], new]) if strat == "Old data + new data" else new
                m = SGDRegressor(random_state=0).fit(make_X(tr), scaled_y(tr[tgt]))
            ss.cands[strat] = dict(model=m, rows=n_rows)

# ---------------------------------------------------------------- evaluation
st.subheader("Evaluation on held-out post-drift rows")
y_ev = ev[tgt].values
p_base = p_all[n_tr + ev_s:n_tr + lab]
rows = [dict(Model="Original model", Rows=fit_n, **{m: score(y_ev, p_base, m) for m in METRICS}, Change="—", Accept="—")]
preds = {}
accepted = {}
min_gain = st.slider("Acceptance: minimum improvement on selected metric (%)", 0, 50, 5) / 100
b_val = score(y_ev, p_base, metric)
for s, c in ss.cands.items():
    preds[s] = predict(c["model"], ev)
    vals = {m: score(y_ev, preds[s], m) for m in METRICS}
    gain = improvement(vals[metric], b_val, metric)
    accepted[s] = bool(gain >= min_gain)
    rows.append(dict(Model=s, Rows=c["rows"], **vals, Change=f"{gain * 100:+.1f}%",
                     Accept="✅" if accepted[s] else "❌"))
if not ss.cands:
    st.info("Run at least one update to compare against the original model.")
else:
    st.dataframe(pd.DataFrame(rows).style.format({m: "{:.4g}" for m in METRICS}), width="stretch", hide_index=True)
    st.caption(f"Change = improvement ({metric}) vs the original model; positive is better.")

    st.subheader("4 · Model Performance Comparison")
    names = ["Original (before)"] + [f"{s} (after)" for s in ss.cands]
    g4 = go.Figure(go.Bar(x=names, y=[b_val] + [r[metric] for r in rows[1:]],
                          marker_color=["#888"] + ["#2ca02c" if accepted[s] else "#d62728" for s in ss.cands]))
    g4.update_layout(height=320, margin=dict(t=20), yaxis_title=metric)
    st.plotly_chart(g4, width="stretch")

    fe = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.06,
                       subplot_titles=("Predictions", "Absolute error"))
    fe.add_scatter(x=ev["_x"], y=y_ev, name="Actual", line=dict(color="#444"), row=1, col=1)
    fe.add_scatter(x=ev["_x"], y=p_base, name="Original", line=dict(color="#1f77b4"), row=1, col=1)
    fe.add_scatter(x=ev["_x"], y=np.abs(y_ev - p_base), name="Original", line=dict(color="#1f77b4"),
                   showlegend=False, row=2, col=1)
    for s, pr in preds.items():
        fe.add_scatter(x=ev["_x"], y=pr, name=s, row=1, col=1)
        fe.add_scatter(x=ev["_x"], y=np.abs(y_ev - pr), name=s, showlegend=False, row=2, col=1)
    fe.update_layout(height=450, margin=dict(t=30), legend=dict(orientation="h"))
    st.plotly_chart(fe, width="stretch")

    st.subheader("Promote")
    pick = st.selectbox("Candidate", list(ss.cands))
    if not accepted[pick]:
        st.warning("Candidate does not meet the acceptance criterion — the original model stays active.")
    if st.button("Promote Candidate Model", disabled=not accepted[pick]):
        ss.active, ss.amodel = pick, copy.deepcopy(ss.cands[pick]["model"])
        st.success(f"Promoted: {pick}. It is now shown as the active model in Graph 1.")
        st.rerun()