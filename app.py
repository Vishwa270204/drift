import copy
import glob
import io
import os
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Model Drift Monitor", layout="wide")
ss = st.session_state
HERE = os.path.dirname(os.path.abspath(__file__))

# ---- sensible defaults (no manual settings) ----
TRAIN_FRAC = 0.70      # first 70% of rows = initial training data
DEGRADE_PCT = 0.30     # degraded when recent error is 30% above baseline error
PSI_LIMIT = 0.20       # common rule of thumb for a significant input shift
EVAL_FRAC = 0.30       # share of recent labeled rows kept unseen for evaluation
MIN_GAIN = 0.05        # updated model must cut MAE by at least 5% (and not worsen RMSE)
OPTIONS = ["Use New Data Only", "Use Old + New Data", "Update Existing Model"]
TARGET_WORDS = ["target", "label", "y", "output", "response", "price", "sales", "demand", "value",
                "load", "close", "revenue", "count", "temperature", "consumption", "energy", "power", "amount"]


# ---------------------------------------------------------------- loading & auto-detection
@st.cache_data
def read_path(path, mtime):
    return pd.read_csv(path)


@st.cache_data
def read_bytes(b):
    return pd.read_csv(io.BytesIO(b))


def to_dt(s):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return pd.to_datetime(s, errors="coerce", format="mixed")
        except (TypeError, ValueError):
            return pd.to_datetime(s, errors="coerce")


def is_num(s):
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def find_time(df):
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    for c in df.columns:
        if is_num(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            continue
        smp = df[c].dropna().head(200)
        if len(smp) and to_dt(smp).notna().mean() >= 0.9:
            return c
    return None


def find_target(df, tcol):
    nums = [c for c in df.columns if c != tcol and is_num(df[c]) and df[c].nunique() > 5]
    low = {c: c.lower().strip() for c in nums}
    exact = [c for c in nums if low[c] in TARGET_WORDS]
    if len(exact) == 1:
        return exact[0], nums
    part = [c for c in nums if any(w in low[c] for w in TARGET_WORDS if len(w) > 3)]
    if not exact and len(part) == 1:
        return part[0], nums
    return None, nums  # not confident -> ask


def pick_features(df, target, tcol):
    out = []
    for c in df.columns:
        s = df[c]
        if c in (target, tcol) or s.isna().mean() > 0.5 or s.nunique() <= 1:
            continue
        if c.lower() in ("id", "index") or c.lower().endswith("_id"):
            continue
        if not is_num(s) and s.nunique() > 30:
            continue
        if s.nunique() == len(s) and not pd.api.types.is_float_dtype(s):
            continue  # looks like an ID / row counter
        out.append(c)
    return out


# ---------------------------------------------------------------- model helpers
def score(y, p, m):
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) == 0:
        return np.nan
    return float(mean_absolute_error(y, p)) if m == "MAE" else float(np.sqrt(mean_squared_error(y, p)))


def psi(b, c):
    b, c = b.dropna(), c.dropna()
    if len(b) == 0 or len(c) == 0:
        return np.nan
    if is_num(b):
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
        X = pd.concat([X, pd.get_dummies(cat, dtype=float).reindex(columns=p["dcols"], fill_value=0.0)], axis=1)
    return X


def make_X(df):  # scaler fitted once on baseline training rows; never refitted
    p = ss.S["p"]
    return p["sc"].transform(raw_features(df, p).values)


def scaled_y(y):
    p = ss.S["p"]
    return (np.asarray(y, float) - p["ym"]) / p["ys"]


def predict(model, df):
    p = ss.S["p"]
    return model.predict(make_X(df)) * p["ys"] + p["ym"]


def init_baseline(df, target, feats, tcol, lag):
    d = df.copy()
    if tcol:
        d[tcol] = to_dt(d[tcol])
        d = d.dropna(subset=[tcol]).sort_values(tcol, kind="stable")
    d[target] = pd.to_numeric(d[target], errors="coerce")  # missing targets stay missing (never invented)
    d = d.reset_index(drop=True)
    d["_x"] = d[tcol] if tcol else np.arange(len(d))
    feats = list(feats)
    if lag:  # no usable feature columns: use the previous actual value (known at prediction time)
        d["_lag1"] = d[target].shift(1)
        feats.append("_lag1")
        d = d.iloc[1:].reset_index(drop=True)
    n = len(d)
    n_tr = int(n * TRAIN_FRAC)
    fit_n = int(n_tr * 0.85)  # last 15% of training = baseline reference, not fitted on
    lab = d[target].notna().values
    if lab.sum() == 0:
        st.error("This CSV has no usable target values, so performance can't be measured. "
                 "Supervised model updating requires labeled data.")
        return False
    if n_tr < 30 or n - n_tr < 30 or lab[:fit_n].sum() < 20 or lab[fit_n:n_tr].sum() < 5:
        st.error(f"Not enough labeled rows for a reliable split ({n} rows, {int(lab.sum())} with a target value).")
        return False
    fit = d.iloc[:fit_n]
    fit = fit[fit[target].notna()]
    num = [c for c in feats if is_num(d[c]) or pd.api.types.is_bool_dtype(d[c])]
    cat = [c for c in feats if c not in num]
    p = dict(num=num, cat=cat, med=fit[num].apply(pd.to_numeric, errors="coerce").median().fillna(0))
    p["dcols"] = (list(pd.get_dummies(fit[cat].astype(object).fillna("__na__").astype(str), dtype=float).columns)
                  if cat else [])
    p["ym"], p["ys"] = float(fit[target].mean()), float(fit[target].std() or 1.0)
    p["sc"] = StandardScaler().fit(raw_features(fit, p).values)
    ss.S = dict(d=d, feats=num + cat, target=target, n_tr=n_tr, fit_n=fit_n, p=p, lag=lag)
    model = SGDRegressor(random_state=0).fit(make_X(fit), scaled_y(fit[target]))
    ss.S["model"] = model
    ss.S["pred"] = predict(model, d)  # predictions use features only
    ss.result, ss.active, ss.amodel = None, "Original model", None
    return True


# ---------------------------------------------------------------- 1. automatic setup
st.title("Model Drift Monitor")
st.caption("Simulated live data: CSV rows are replayed in time order (not a real production stream).")
sb = st.sidebar
sb.header("Data")
up = sb.file_uploader("Upload a CSV (optional)", type="csv")
files = sorted({os.path.abspath(f) for f in glob.glob(os.path.join(HERE, "*.csv")) + glob.glob("*.csv")})
if up is not None:
    raw, name = read_bytes(up.getvalue()), up.name
elif files:
    f = files[0] if len(files) == 1 else sb.selectbox("CSV file", files, format_func=os.path.basename)
    raw, name = read_path(f, os.path.getmtime(f)), os.path.basename(f)
else:
    st.info("Place a CSV next to app.py, or upload one in the sidebar.")
    st.stop()

tcol = find_time(raw)
target, nums = find_target(raw, tcol)
if not nums:
    st.error("No numeric column found to predict.")
    st.stop()
if target is None:
    target = sb.selectbox("Which column should be predicted?", nums, index=len(nums) - 1)
feats = pick_features(raw, target, tcol)
lag = not feats
sb.caption(f"**File:** {name} ({len(raw):,} rows)  \n**Predicting:** {target}  \n"
           f"**Time column:** {tcol or 'row order'}  \n**Input columns:** {len(feats) if feats else 'previous value'}")

sig = (name, len(raw), target, tcol)
if ss.get("sig") != sig:
    ss.pop("S", None)
    ss.sig = sig if init_baseline(raw, target, feats, tcol, lag) else None
    if "S" in ss:
        ss.pos = 0
if "S" not in ss:
    st.stop()

S = ss.S
d, n_tr, fit_n, tgt = S["d"], S["n_tr"], S["fit_n"], S["target"]
n_stream = len(d) - n_tr
batch = max(10, n_stream // 10)
win = int(np.clip(n_stream // 10, 10, 50))
if ss.pos == 0:
    ss.pos = min(batch, n_stream)

# ---------------------------------------------------------------- 4. simulate incoming data
sb.header("Live simulation")
if sb.button("Process next batch", type="primary"):
    ss.pos = min(ss.pos + batch, n_stream)
if sb.button("Process all rows"):
    ss.pos = n_stream
if sb.button("Restart"):
    ss.pos, ss.result, ss.active, ss.amodel = min(batch, n_stream), None, "Original model", None
k = ss.pos
y_all, p_all, xs = d[tgt].values, S["pred"], d["_x"]

# ---------------------------------------------------------------- monitoring numbers
stream = d.iloc[n_tr:n_tr + k]
ys, ps = stream[tgt].values, p_all[n_tr:n_tr + k]
L = np.where(~np.isnan(ys))[0]  # processed rows whose actual value is known
yl, pl, xl, lab = ys[L], ps[L], stream["_x"].iloc[L], len(L)
vm = ~np.isnan(y_all[fit_n:n_tr])
ref_mae = score(y_all[fit_n:n_tr][vm], p_all[fit_n:n_tr][vm], "MAE")
limit = ref_mae * (1 + DEGRADE_PCT)
roll = pd.Series(np.abs(yl - pl)).rolling(win).mean().values
bad = np.where(roll > limit)[0]
d_rel = int(bad[0]) if len(bad) else None
onset = max(d_rel - win + 1, 0) if d_rel is not None else None

if lab == 0:
    status, note = "Not measurable", "No actual values yet — performance can't be measured. Updating requires labeled data."
elif lab < win:
    status, note = "Collecting data", f"Need {win} rows with actual values (have {lab})."
elif d_rel is not None:
    status, note = "⚠️ Degraded", "Error rose above the allowed limit."
else:
    status, note = "✅ Healthy", "Error is within the allowed limit."

cur = score(yl[-win:], pl[-win:], "MAE") if lab else np.nan
r = ss.get("result")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Records processed", f"{k:,} / {n_stream:,}")
c2.metric("Current error (MAE)", "—" if lab == 0 else f"{cur:.3g}",
          delta=None if lab == 0 else f"{(cur / ref_mae - 1) * 100:+.0f}% vs baseline", delta_color="inverse")
c3.metric("Model status", status)
c4.metric("Original → Updated (MAE)", "No update yet" if not r else f"{r['base'][0]:.3g} → {r['new'][0]:.3g}")
(st.warning if lab == 0 else st.info)(note)
if k - lab and lab:
    st.caption(f"{k - lab} processed rows have no actual value; they are predicted but not scored.")

# ---------------------------------------------------------------- graphs
st.subheader("1 · Actual vs Predicted")
n_show = n_tr + k
g1 = go.Figure()
g1.add_scatter(x=xs[:n_show], y=y_all[:n_show], name="Actual", line=dict(color="#444", width=1))
g1.add_scatter(x=xs[:n_show], y=p_all[:n_show], name="Predicted", line=dict(color="#1f77b4", width=1))
if ss.amodel is not None:
    g1.add_scatter(x=stream["_x"], y=predict(ss.amodel, stream), name="Updated model",
                   line=dict(color="#2ca02c", width=1))
g1.add_vrect(x0=xs.iloc[0], x1=xs.iloc[n_tr - 1], fillcolor="#1f77b4", opacity=0.10, line_width=0, layer="below")
if d_rel is not None:
    g1.add_vrect(x0=xl.iloc[onset], x1=xs.iloc[n_show - 1], fillcolor="#d62728", opacity=0.10, line_width=0, layer="below")
    g1.add_scatter(x=[xl.iloc[d_rel]], y=[yl[d_rel]], mode="markers", name="Degradation detected",
                   marker=dict(color="red", size=11, symbol="x"))
g1.update_layout(height=340, margin=dict(t=10, b=10), legend=dict(orientation="h"), yaxis_title=tgt)
st.plotly_chart(g1, width="stretch")
st.caption("Blue = initial training period · Red = after degradation.")

st.subheader("2 · Model Performance Over Time")
g2 = go.Figure()
g2.add_scatter(x=xl, y=roll, name="Recent error (MAE)", line=dict(color="#1f77b4"))
g2.add_hline(y=limit, line_dash="dash", line_color="red", annotation_text="Allowed limit")
g2.add_hline(y=ref_mae, line_dash="dot", line_color="green", annotation_text="Baseline")
if d_rel is not None:
    g2.add_vrect(x0=xl.iloc[onset], x1=xl.iloc[-1], fillcolor="#d62728", opacity=0.10, line_width=0, layer="below")
    g2.add_scatter(x=[xl.iloc[d_rel]], y=[roll[d_rel]], mode="markers", name="Degradation detected",
                   marker=dict(color="red", size=12, symbol="x"))
g2.update_layout(height=300, margin=dict(t=10, b=10), legend=dict(orientation="h"), yaxis_title="MAE")
st.plotly_chart(g2, width="stretch")

st.subheader("3 · Data Drift")
if k < 20:
    st.info("Process at least 20 records to check for data drift.")
else:
    base, recent = d.iloc[:n_tr], d.iloc[n_tr + max(k - 100, 0):n_tr + k]
    ps_ = pd.Series({f: psi(base[f], recent[f]) for f in S["feats"]}).dropna().sort_values(ascending=False).head(12)
    g3 = go.Figure(go.Bar(x=ps_.index, y=ps_.values,
                          marker_color=["#d62728" if v > PSI_LIMIT else "#1f77b4" for v in ps_.values]))
    g3.add_hline(y=PSI_LIMIT, line_dash="dash", line_color="red")
    g3.update_layout(height=280, margin=dict(t=10, b=10), yaxis_title="Shift score (PSI)")
    st.plotly_chart(g3, width="stretch")
    shifted = ps_[ps_ > PSI_LIMIT].index.tolist()
    st.caption(("Incoming data differs from baseline for: " + ", ".join(shifted)) if shifted
               else "Incoming data looks similar to baseline.")
    st.caption("A data shift alone does not prove the model is worse — check graph 2.")

# ---------------------------------------------------------------- 3. update model
st.header("Update Model")
if lab == 0:
    st.info("Model updating needs actual target values (labels). None are available yet.")
    st.stop()
opt = st.radio("Training data", OPTIONS, captions=[
    "Fresh model on recent labeled data", "Fresh model on original + recent labeled data",
    "Continue training the current model on recent data"])

# recent labeled data: from the degradation point if detected, otherwise the latest rows
start = onset if d_rel is not None else max(lab - 4 * win, 0)
n_ev = int((lab - start) * EVAL_FRAC)
gap = 1 if lag else 0  # skip a row when the lag feature links neighbours
tr_end, ev_s = lab - n_ev - gap, lab - n_ev
dl = d.iloc[n_tr + L]  # labeled stream rows
ready = tr_end - start >= 20 and n_ev >= 10
if ready:
    new, ev = dl.iloc[start:tr_end], dl.iloc[ev_s:lab]
    n_train = len(new) + (int(d.iloc[:n_tr][tgt].notna().sum()) if opt == "Use Old + New Data" else 0)
    st.caption(f"Training rows: {n_train} · Unseen evaluation rows: {len(ev)}")
else:
    st.info("Not enough recent labeled data yet — keep processing records.")

if st.button("Update and Compare Model", type="primary", disabled=not ready):
    if opt == "Update Existing Model":
        model = copy.deepcopy(S["model"])  # original stays untouched
        Xn, yn = make_X(new), scaled_y(new[tgt])
        for _ in range(3):
            model.partial_fit(Xn, yn)
    else:
        tr = new
        if opt == "Use Old + New Data":
            old = d.iloc[:n_tr]
            tr = pd.concat([old[old[tgt].notna()], new])
        model = SGDRegressor(random_state=0).fit(make_X(tr), scaled_y(tr[tgt]))
    y_ev, p_old, p_new = ev[tgt].values, pl[ev_s:lab], predict(model, ev)
    base_s = (score(y_ev, p_old, "MAE"), score(y_ev, p_old, "RMSE"))
    new_s = (score(y_ev, p_new, "MAE"), score(y_ev, p_new, "RMSE"))
    passed = new_s[0] <= base_s[0] * (1 - MIN_GAIN) and new_s[1] <= base_s[1]
    ss.result = dict(opt=opt, base=base_s, new=new_s, passed=passed, n_eval=len(ev))
    if passed:  # promote only after passing evaluation
        ss.active, ss.amodel = opt, model
    st.rerun()

r = ss.get("result")
if r:
    b, n_ = r["base"], r["new"]
    if r["passed"]:
        st.success(f"Improved — the updated model ({r['opt']}) is now in use.")
    elif n_[0] > b[0]:
        st.error("Worse — the original model is kept.")
    else:
        st.warning("No meaningful improvement — the original model is kept.")
    m1, m2 = st.columns(2)
    m1.metric("MAE (updated)", f"{n_[0]:.3g}", delta=f"{n_[0] - b[0]:+.3g} vs original", delta_color="inverse")
    m2.metric("RMSE (updated)", f"{n_[1]:.3g}", delta=f"{n_[1] - b[1]:+.3g} vs original", delta_color="inverse")
    st.subheader("4 · Model Comparison")
    g4 = go.Figure()
    g4.add_bar(x=["MAE", "RMSE"], y=list(b), name="Original", marker_color="#888")
    g4.add_bar(x=["MAE", "RMSE"], y=list(n_), name="Updated", marker_color="#2ca02c" if r["passed"] else "#d62728")
    g4.update_layout(barmode="group", height=300, margin=dict(t=10, b=10), legend=dict(orientation="h"))
    st.plotly_chart(g4, width="stretch")
    st.caption(f"Both models scored on the same {r['n_eval']} unseen rows that were never used for training.")
