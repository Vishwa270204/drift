"""Model Drift Detection Dashboard (single file, ADWIN only).

One model family per run, chosen from the target column:
    classification -> SGDClassifier,  regression -> SGDRegressor
Drift is detected with ADWIN on the prequential prediction errors of that model.
Learning strategies that adapt the model are compared:

    Static (baseline), Incremental, Online, Continual, Transfer (+ a scratch baseline)

Protocol (identical for every strategy, strictly chronological):
    1. The first N% of rows are the REFERENCE period. A source model is trained on it.
    2. The remaining rows are replayed as consecutive WINDOWS.
    3. For every window: every model PREDICTS first (nothing has been trained on
       this window yet), then labels "arrive", ADWIN runs, and models may update.
    4. An ADWIN signal at the end of window t switches adaptation on for the next
       k windows (or adaptation can run on every window). Detection and adaptation
       are separate steps: a signal never retrains anything by itself.

Run:  streamlit run app.py
"""

import copy
import os
import time
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from river.drift import ADWIN
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

DEFAULT_CSV_PATH = "data.csv"

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
MAX_CAT_UNIQUE = 50     # categorical columns with more levels are treated as ID-like and excluded
TARGET_CAP = 2000       # most recent labelled target-period rows kept for Transfer / Scratch
TRANSFER_EPOCHS = 3     # passes over the target buffer per Transfer / Scratch update
SEED = 42

STATIC = "Static (no adaptation)"
INCR = "Incremental"
ONLINE = "Online"
CONT = "Continual"
TRANS = "Transfer"
SCRATCH = "Scratch (target-only baseline)"
LEARNERS = [INCR, ONLINE, CONT, TRANS]

st.set_page_config(page_title="Model Drift Dashboard", layout="wide")


# ----------------------------------------------------------------------------
# Data preparation
# ----------------------------------------------------------------------------
def prepare_df(df_raw, target, time_col):
    """Replace inf, drop rows without a target, optionally sort chronologically."""
    df = df_raw.replace([np.inf, -np.inf], np.nan).dropna(subset=[target])
    if time_col is not None:
        key = df[time_col]
        if not pd.api.types.is_numeric_dtype(key):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(key, errors="coerce")
            if parsed.notna().mean() > 0.5:
                key = parsed
        df = df.assign(_sort_key=key).sort_values("_sort_key", kind="stable").drop(columns="_sort_key")
    return df.reset_index(drop=True)


def split_columns(df, target, time_col):
    """Return numeric columns, categorical columns and a role table (with exclusion reasons)."""
    num_cols, cat_cols, roles = [], [], []
    for c in df.columns:
        s = df[c]
        if c == target:
            roles.append((c, str(s.dtype), "target", ""))
        elif c == time_col:
            roles.append((c, str(s.dtype), "ordering column", "used for sorting only"))
        elif pd.api.types.is_datetime64_any_dtype(s):
            roles.append((c, str(s.dtype), "excluded", "datetime column"))
        elif s.isna().all():
            roles.append((c, str(s.dtype), "excluded", "all values missing"))
        elif s.nunique(dropna=True) <= 1:
            roles.append((c, str(s.dtype), "excluded", "constant column"))
        elif (pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
              and s.notna().all() and s.is_unique and (s.is_monotonic_increasing or s.is_monotonic_decreasing)):
            roles.append((c, str(s.dtype), "excluded", "monotonic counter / index-like (use 'Order rows by' for time)"))
        elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            num_cols.append(c)
            roles.append((c, str(s.dtype), "numeric feature", ""))
        elif s.nunique(dropna=True) > MAX_CAT_UNIQUE:
            roles.append((c, str(s.dtype), "excluded", f"more than {MAX_CAT_UNIQUE} categories (ID-like / free text)"))
        else:
            cat_cols.append(c)
            roles.append((c, str(s.dtype), "categorical feature", ""))
    return num_cols, cat_cols, pd.DataFrame(roles, columns=["column", "dtype", "role", "note"])


def detect_task(y):
    """Classification for text/bool targets and integer-valued targets with <= 10 values."""
    y = y.dropna()
    if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
        return "classification"
    if y.nunique() <= 10 and bool((y % 1 == 0).all()):
        return "classification"
    return "regression"


def feature_frame(df, num_cols, cat_cols):
    """Feature columns with categoricals as strings (missing stays NaN)."""
    X = df[num_cols + cat_cols].copy()
    for c in cat_cols:
        col = X[c].astype(object)
        X[c] = col.where(col.isna(), col.astype(str))
    return X


# ----------------------------------------------------------------------------
# Models and metrics
# ----------------------------------------------------------------------------
def make_model(task):
    if task == "classification":
        return SGDClassifier(loss="log_loss", alpha=1e-4, random_state=SEED)
    return SGDRegressor(alpha=1e-4, random_state=SEED)


def fit_epochs(model, X, y, task, classes, epochs=1, rng=None):
    """Run `partial_fit` for a number of shuffled passes (works for fresh and trained models)."""
    rng = rng if rng is not None else np.random.default_rng(SEED)
    for _ in range(epochs):
        order = rng.permutation(len(y))
        if task == "classification":
            model.partial_fit(X[order], y[order], classes=classes)
        else:
            model.partial_fit(X[order], y[order])


def score(task, y_true, y_pred):
    if task == "classification":
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        }
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def reservoir_add(Xb, yb, filled, seen, Xn, yn, rng):
    """Bounded replay buffer: reservoir sampling keeps a uniform sample of everything seen."""
    cap = len(yb)
    for i in range(len(yn)):
        seen += 1
        if filled < cap:
            Xb[filled], yb[filled] = Xn[i], yn[i]
            filled += 1
        else:
            j = int(rng.integers(0, seen))
            if j < cap:
                Xb[j], yb[j] = Xn[i], yn[i]
    return filled, seen


def make_windows(n_ref, n, batch):
    """Consecutive (start, end) windows; a short last chunk is merged into the previous window."""
    bounds = list(range(n_ref, n, batch))
    windows = [(s, min(s + batch, n)) for s in bounds]
    if len(windows) > 1 and windows[-1][1] - windows[-1][0] < batch // 2:
        last = windows.pop()
        windows[-1] = (windows[-1][0], last[1])
    return windows


def recovery_text(scores, first_w, higher_better, tol=0.05):
    """Windows between the first degraded window after the drift signal and recovery (within 5% of pre-drift level)."""
    if first_w is None:
        return "n/a (no drift signal)"
    if first_w == 0:
        return "n/a (no pre-drift windows)"
    scores = np.asarray(scores, float)
    pre = scores[:first_w].mean()
    ok = scores >= pre * (1 - tol) if higher_better else scores <= pre * (1 + tol)
    after = ok[first_w:]
    bad = np.where(~after)[0]
    if len(bad) == 0:
        return "0 (no degradation)"
    rest = np.where(after[bad[0]:])[0]
    if len(rest) == 0:
        return "not recovered"
    return f"{int(rest[0])} windows"


# ----------------------------------------------------------------------------
# The experiment
# ----------------------------------------------------------------------------
def run_experiment(df, num_cols, cat_cols, target, task, cfg, progress):
    n = len(df)
    n_ref = int(n * cfg["ref_pct"] / 100)
    if n_ref < 50:
        raise ValueError("The reference period has fewer than 50 rows. Increase the reference percentage.")
    windows = make_windows(n_ref, n, cfg["batch"])
    if len(windows) < 3:
        raise ValueError("Fewer than 3 evaluation windows. Reduce the window size or the reference percentage.")
    always = cfg["trigger"] == "Every window"
    X_df = feature_frame(df, num_cols, cat_cols)
    rng = np.random.default_rng(SEED)

    # --- preprocessing is fitted on the reference period only ---
    steps = []
    if num_cols:
        steps.append(("num", Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
                                       ("sc", StandardScaler())]), num_cols))
    if cat_cols:
        steps.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                                       ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols))
    pre = ColumnTransformer(steps).fit(X_df.iloc[:n_ref])
    X_all = np.asarray(pre.transform(X_df), dtype=float)

    classes = None
    if task == "classification":
        le = LabelEncoder().fit(df[target].astype(str))  # only the label vocabulary comes from all rows
        y_all = le.transform(df[target].astype(str))
        classes = np.arange(len(le.classes_))
        y_true_all = y_all
        mu, sd = 0.0, 1.0
        if len(np.unique(y_all[:n_ref])) < 2:
            raise ValueError("The reference period contains a single class. Increase the reference percentage.")
    else:
        y_raw = df[target].astype(float).to_numpy()
        mu = float(y_raw[:n_ref].mean())
        sd = float(y_raw[:n_ref].std()) or 1.0
        y_all = (y_raw - mu) / sd  # target scaling uses reference statistics only
        y_true_all = y_raw

    # --- one source model, trained on the reference period ---
    base = make_model(task)
    fit_epochs(base, X_all[:n_ref], y_all[:n_ref], task, classes, epochs=10, rng=rng)
    models = {STATIC: base}
    for name in cfg["strategies"]:
        models[name] = copy.deepcopy(base)
    if TRANS in cfg["strategies"]:
        models[SCRATCH] = None  # fitted at the first adaptation; until then the source model predicts

    d = X_all.shape[1]
    cap = cfg["buffer"]
    Xb, yb = np.zeros((cap, d)), np.zeros(cap, dtype=y_all.dtype)
    filled = seen = 0
    if CONT in models:
        filled, seen = reservoir_add(Xb, yb, filled, seen, X_all[:n_ref], y_all[:n_ref], rng)
    Xt, yt = np.zeros((0, d)), np.zeros(0, dtype=y_all.dtype)

    def predict(name, X):
        m = models[name]
        return (base if m is None else m).predict(X)

    def to_orig(p):
        return p if task == "classification" else p * sd + mu

    # --- state ---
    adwin = ADWIN(delta=cfg["adwin_delta"])
    adwin_events, flagged_windows, active_windows = [], [], []
    metric_rows, preds_store = [], {name: [] for name in models}
    upd_time = {name: 0.0 for name in models}
    upd_count = {name: 0 for name in models}
    remaining = 0

    for w, (s, e) in enumerate(windows):
        progress((w + 1) / len(windows), f"Window {w + 1} / {len(windows)}")
        active = always or remaining > 0
        if active:
            active_windows.append(w)
        Xw, yw = X_all[s:e], y_all[s:e]

        # 1) PREDICT with every model before anything is trained on this window
        raw = {}
        for name in models:
            if name == ONLINE and active:
                # true prequential: predict row i, then learn row i
                out, t_up = [], 0.0
                for i in range(len(yw)):
                    xi = Xw[i:i + 1]
                    out.append(models[ONLINE].predict(xi)[0])
                    t0 = time.perf_counter()
                    fit_epochs(models[ONLINE], xi, yw[i:i + 1], task, classes, rng=rng)
                    t_up += time.perf_counter() - t0
                raw[name] = np.array(out)
                upd_time[ONLINE] += t_up
                upd_count[ONLINE] += len(yw)
            else:
                raw[name] = predict(name, Xw)
        for name, p in raw.items():
            preds_store[name].append(p)
            metric_rows.append({"strategy": name, "window": w, "start": s, "end": e,
                                **score(task, y_true_all[s:e], to_orig(p))})

        # 2) DETECT: ADWIN on the errors of the NEVER-updated static model (independent of adaptation)
        if task == "classification":
            errors = (raw[STATIC] != yw).astype(float)
        else:
            errors = np.clip(np.abs(raw[STATIC] - yw), 0, 5)  # standardised absolute error, clipped for stability
        flagged = False
        for i, err in enumerate(errors):
            adwin.update(float(err))
            if adwin.drift_detected:
                adwin_events.append({"row": s + i, "window": w})
                flagged = True
        if flagged:
            flagged_windows.append(w)

        # 3) ADAPT (labels are available now; only models that are allowed to adapt are updated)
        if CONT in models:
            if active and filled > 0:
                k = min(len(yw), filled)
                idx = rng.choice(filled, size=k, replace=False)
                t0 = time.perf_counter()
                fit_epochs(models[CONT], np.vstack([Xw, Xb[idx]]), np.concatenate([yw, yb[idx]]),
                           task, classes, rng=rng)
                upd_time[CONT] += time.perf_counter() - t0
                upd_count[CONT] += 1
            filled, seen = reservoir_add(Xb, yb, filled, seen, Xw, yw, rng)
        if INCR in models and active:
            t0 = time.perf_counter()
            fit_epochs(models[INCR], Xw, yw, task, classes, rng=rng)
            upd_time[INCR] += time.perf_counter() - t0
            upd_count[INCR] += 1
        if TRANS in models:
            Xt, yt = np.vstack([Xt, Xw])[-TARGET_CAP:], np.concatenate([yt, yw])[-TARGET_CAP:]
            if active:
                t0 = time.perf_counter()
                tm = copy.deepcopy(base)  # start from the frozen source model
                fit_epochs(tm, Xt, yt, task, classes, epochs=TRANSFER_EPOCHS, rng=rng)
                upd_time[TRANS] += time.perf_counter() - t0
                models[TRANS] = tm
                upd_count[TRANS] += 1
                t0 = time.perf_counter()
                sm = make_model(task)  # same data and budget, random start
                fit_epochs(sm, Xt, yt, task, classes, epochs=TRANSFER_EPOCHS, rng=rng)
                upd_time[SCRATCH] += time.perf_counter() - t0
                models[SCRATCH] = sm
                upd_count[SCRATCH] += 1

        # 4) update the adaptation switch for the NEXT windows
        if active and not always:
            remaining -= 1
        if flagged:
            remaining = cfg["adapt_windows"]

    # --- summaries ---
    higher = task == "classification"
    prim = "f1" if higher else "mae"
    first_w = flagged_windows[0] if flagged_windows else None
    res = {"cfg": cfg, "task": task, "n_ref": n_ref, "windows": windows, "adwin_events": adwin_events,
           "flagged_windows": flagged_windows, "active_windows": active_windows, "first_flag_window": first_w,
           "metrics": pd.DataFrame(metric_rows), "primary": prim, "higher_better": higher}

    onset = cfg["onset"]
    res["delays"] = None
    if onset > 0:
        a_rows = [ev["row"] for ev in adwin_events]
        after = [r for r in a_rows if r >= onset]
        res["delays"] = {"ADWIN delay (rows)": (after[0] - onset) if after else None,
                         "False alarms before onset": sum(r < onset for r in a_rows)}

    stream_end = windows[-1][1]
    y_stream_true = y_true_all[n_ref:stream_end]
    post_mask = None
    if first_w is not None:
        post_mask = np.arange(n_ref, stream_end) >= windows[first_w][1]
        if post_mask.sum() == 0:
            post_mask = None
    mdf = res["metrics"]
    rows, rank = [], {}
    for name in models:
        p = to_orig(np.concatenate(preds_store[name]))
        overall = score(task, y_stream_true, p)
        post = score(task, y_stream_true[post_mask], p[post_mask]) if post_mask is not None else None
        series = mdf[mdf.strategy == name].sort_values("window")[prim].to_numpy()
        row = {"Strategy": name}
        if task == "classification":
            row.update({"Accuracy": overall["accuracy"], "F1 (macro)": overall["f1"],
                        "Post-drift F1": post["f1"] if post else np.nan})
        else:
            row.update({"MAE": overall["mae"], "RMSE": overall["rmse"],
                        "Post-drift MAE": post["mae"] if post else np.nan})
        row.update({"Recovery": recovery_text(series, first_w, higher),
                    "Updates": upd_count[name],
                    "Update time (s)": upd_time[name]})
        rows.append(row)
        rank[name] = post[prim] if post else overall[prim]
    res["summary"], res["rank_basis"] = pd.DataFrame(rows), rank
    res["rank_uses_post_drift"] = post_mask is not None
    cands = [s_ for s_ in cfg["strategies"] if s_ in rank]
    res["best"] = max(cands, key=lambda k: rank[k]) if higher else min(cands, key=lambda k: rank[k])
    return res


# ----------------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------------
def timeline_fig(res):
    windows = res["windows"]
    mid = [(s + e) / 2 for s, e in windows]
    prim = res["primary"]
    ylabel = "F1 (macro)" if prim == "f1" else "MAE"
    sdf = res["metrics"][res["metrics"].strategy == STATIC].sort_values("window")
    y = sdf[prim].to_numpy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mid, y=y, mode="lines+markers", name=f"Static model {ylabel}"))
    lo, hi = float(np.min(y)), float(np.max(y))
    pad = (hi - lo) * 0.12 or 0.05
    if res["adwin_events"]:
        fig.add_trace(go.Scatter(x=[ev["row"] for ev in res["adwin_events"]], y=[lo - pad] * len(res["adwin_events"]),
                                 mode="markers", name="ADWIN drift event",
                                 marker=dict(symbol="triangle-up", color="red", size=10)))
    fig.add_vline(x=res["n_ref"], line_dash="dot", line_color="gray", annotation_text="end of reference")
    if res["cfg"]["onset"] > 0:
        fig.add_vline(x=res["cfg"]["onset"], line_dash="dash", line_color="green", annotation_text="known onset")
    fig.update_layout(xaxis_title="Row index", yaxis_title=ylabel, height=420, margin=dict(t=30),
                      legend=dict(orientation="h", y=-0.2))
    return fig


# ----------------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------------
st.title("Model Drift Detection Dashboard")
st.sidebar.header("Configuration")

uploaded_file = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])

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
except Exception as exc:  # unreadable / malformed file
    st.error(f"Could not read the CSV file: {exc}")
    st.stop()

if df_raw.shape[0] < 100 or df_raw.shape[1] < 2:
    st.error("The CSV needs at least 100 rows and 2 columns.")
    st.stop()
df_raw.columns = [str(c) for c in df_raw.columns]

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
cols = list(df_raw.columns)
target = st.sidebar.selectbox("Target column", cols, index=len(cols) - 1)
order_choice = st.sidebar.selectbox("Order rows by (optional)", ["(keep file order)"] + cols,
                                    help="Drift analysis is chronological. Pick a timestamp column if the file is not already in time order.")
time_col = None if order_choice == "(keep file order)" else order_choice
task_choice = st.sidebar.radio("Model type", ["Auto", "Classification", "Regression"], horizontal=True,
                               help="One model per run. Auto picks classification for text or few-valued integer targets, otherwise regression.")
ref_pct = st.sidebar.slider("Reference data (% of rows)", 10, 70, 30)
strategies = st.sidebar.multiselect("Learning strategies", LEARNERS, default=LEARNERS)
batch = st.sidebar.slider("Window / batch size (rows)", 20, 2000, 200, step=10)
adwin_delta = st.sidebar.number_input("ADWIN delta", min_value=0.000001, max_value=0.5, value=0.002,
                                      step=0.001, format="%.4f")
trigger = st.sidebar.radio("Adapt models", ["On drift signal", "Every window"], horizontal=True,
                           help="'On drift signal': after ADWIN fires, models adapt for the next k windows.")
adapt_windows = st.sidebar.slider("Windows to adapt after a drift signal (k)", 1, 10, 3,
                                  disabled=(trigger == "Every window"))
buffer_size = st.sidebar.slider("Continual replay buffer size", 100, 5000, 500, step=100)
onset = st.sidebar.number_input("Known drift onset row (0 = unknown)", min_value=0, max_value=int(len(df_raw)),
                                value=0, step=100, help="Optional. Enables detection-delay and false-alarm counts.")
run_clicked = st.sidebar.button("Run Analysis", type="primary", use_container_width=True)

# ----------------------------------------------------------------------------
# Dataset preparation
# ----------------------------------------------------------------------------
df = prepare_df(df_raw, target, time_col)
num_cols, cat_cols, roles = split_columns(df, target, time_col)
if len(num_cols) + len(cat_cols) == 0:
    st.error("No usable feature columns were found after excluding the target and invalid columns.")
    st.stop()

task = detect_task(df[target]) if task_choice == "Auto" else task_choice.lower()
if task == "regression" and not pd.api.types.is_numeric_dtype(df[target]):
    st.error("Regression needs a numeric target column.")
    st.stop()
if task == "classification" and df[target].nunique() > 50:
    st.error("The target has more than 50 distinct values. Choose Regression or a different target.")
    st.stop()

# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------
cfg = {"ref_pct": ref_pct, "strategies": strategies, "batch": batch, "adwin_delta": float(adwin_delta),
       "trigger": trigger, "adapt_windows": adapt_windows, "buffer": buffer_size, "onset": int(onset)}
cfg_key = repr((source_name, target, time_col, task, sorted(cfg.items(), key=lambda kv: kv[0])))

if run_clicked:
    if not strategies:
        st.sidebar.error("Select at least one learning strategy.")
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
        except Exception as exc:  # unexpected failure: show it instead of a stack trace
            st.session_state.pop("result", None)
            st.error(f"The analysis failed: {exc}")
        bar.empty()

res = st.session_state.get("result")

# ----------------------------------------------------------------------------
# KPI cards
# ----------------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Records", f"{len(df):,}")
k2.metric("Features used", len(num_cols) + len(cat_cols))
if res is not None:
    k3.metric("Drift events (ADWIN)", len(res["adwin_events"]))
    mdf = res["metrics"]
    latest = mdf[(mdf.strategy == res["best"]) & (mdf.window == mdf.window.max())].iloc[0]
    label = "F1 (macro)" if res["primary"] == "f1" else "MAE"
    k4.metric(f"Latest window {label}", f"{latest[res['primary']]:.3f}", help=f"Best strategy: {res['best']}")
else:
    k3.metric("Drift events (ADWIN)", "-", help="Press Run Analysis.")
    k4.metric("Latest model metric", "-", help="Press Run Analysis.")

if res is not None and st.session_state.get("result_key") != cfg_key:
    st.info("Settings changed since the last run. Press Run Analysis to refresh the results below.")

# ----------------------------------------------------------------------------
# A. Dataset overview
# ----------------------------------------------------------------------------
st.header("A. Dataset overview")
st.caption(f"Source: {source_name} | rows after dropping missing targets: {len(df):,} | model: "
           + ("SGDClassifier" if task == "classification" else "SGDRegressor"))
st.write(f"**Features used:** {len(num_cols)} numeric ({', '.join(num_cols) or 'none'}), "
         f"{len(cat_cols)} categorical ({', '.join(cat_cols) or 'none'}).")
excluded = roles[roles.role == "excluded"]
if len(excluded):
    st.write("**Excluded:** " + "; ".join(f"{r.column} ({r.note})" for r in excluded.itertuples()) + ".")
miss = df.isna().sum()
miss = miss[miss > 0]
st.write("**Missing values:** " + (", ".join(f"{c} ({100 * v / len(df):.1f}%)" for c, v in miss.items())
                                   if len(miss) else "none") + ". Imputed using reference-period statistics.")
if task == "classification":
    st.subheader("Class distribution")
    dist = df[target].astype(str).value_counts().reset_index()
    dist.columns = ["class", "count"]
    st.plotly_chart(px.bar(dist, x="class", y="count", height=300), use_container_width=True)

if res is None:
    st.info("Configure the analysis in the sidebar and press **Run Analysis**.")
    st.stop()

windows = res["windows"]
first_w = res["first_flag_window"]

# ----------------------------------------------------------------------------
# B. Drift detection (ADWIN)
# ----------------------------------------------------------------------------
st.header("B. Drift detection (ADWIN)")
st.caption(f"Reference: rows 0-{res['n_ref'] - 1}. Stream: {len(windows)} windows of about "
           f"{res['cfg']['batch']} rows. Adaptation was active in {len(res['active_windows'])} of {len(windows)} windows.")
st.caption("ADWIN watches the prequential error stream of the never-updated static model (0/1 errors for "
           "classification, clipped standardised absolute error for regression). It signals that the monitored "
           "signal changed. It cannot say which input feature caused the change.")
st.plotly_chart(timeline_fig(res), use_container_width=True)
if res["adwin_events"]:
    ev = pd.DataFrame(res["adwin_events"]).rename(columns={"row": "stream row index", "window": "window #"})
    st.dataframe(ev, use_container_width=True, hide_index=True, height=min(250, 40 + 35 * len(ev)))
else:
    st.write("No ADWIN drift events were detected.")
if res["delays"] is not None:
    st.subheader("Detection delay (known onset)")
    cols_d = st.columns(len(res["delays"]))
    for c_, (k_, v_) in zip(cols_d, res["delays"].items()):
        c_.metric(k_, "not detected" if v_ is None else v_)

# ----------------------------------------------------------------------------
# C. Learning strategy comparison
# ----------------------------------------------------------------------------
st.header("C. Learning strategy comparison")
summary = res["summary"]
prim, higher = res["primary"], res["higher_better"]
plabel = "F1 (macro)" if prim == "f1" else "MAE"
st.caption("Metrics are pooled over all stream rows. 'Post-drift' uses rows after the first window with an ADWIN event. "
           "'Recovery' counts windows from the first degraded window until the metric is within 5% of its "
           "pre-drift level. All models predict before they are updated.")
st.dataframe(summary.style.format({c: "{:.4f}" for c in summary.columns
                                   if c in ("Accuracy", "F1 (macro)", "Post-drift F1", "MAE", "RMSE", "Post-drift MAE", "Update time (s)")}),
             use_container_width=True, hide_index=True)

smooth = st.checkbox("Smooth with a 3-window rolling mean", value=True)
mdf = res["metrics"].sort_values(["strategy", "window"]).copy()
mdf["value"] = mdf.groupby("strategy")[prim].transform(lambda s: s.rolling(3, min_periods=1).mean() if smooth else s)
mdf["row"] = (mdf["start"] + mdf["end"]) / 2
line = px.line(mdf, x="row", y="value", color="strategy", height=420, labels={"row": "Row index", "value": plabel})
line.add_vline(x=res["n_ref"], line_dash="dot", line_color="gray")
if first_w is not None:
    line.add_vline(x=windows[first_w][1], line_dash="dash", line_color="red", annotation_text="first drift signal")
st.plotly_chart(line, use_container_width=True)

c1, c2 = st.columns(2)
basis = "post-drift" if res["rank_uses_post_drift"] else "overall"
rank_df = pd.DataFrame({"strategy": list(res["rank_basis"].keys()), plabel: list(res["rank_basis"].values())})
c1.plotly_chart(px.bar(rank_df, x="strategy", y=plabel, title=f"{plabel} ({basis})", height=360),
                use_container_width=True)
c2.plotly_chart(px.bar(summary, x="Strategy", y="Update time (s)", title="Total model update time (computational cost)",
                       height=360), use_container_width=True)

# ----------------------------------------------------------------------------
# D. Final summary
# ----------------------------------------------------------------------------
st.header("D. Final summary")
lines = []
if res["adwin_events"]:
    rows_ = [ev["row"] for ev in res["adwin_events"]]
    lines.append(f"**ADWIN** signalled {len(rows_)} change(s) in the error stream, first at row {rows_[0]}. "
                 "ADWIN does not identify which feature is responsible, so no feature-level drift is claimed.")
else:
    lines.append("**ADWIN** found no change in the error stream of the static model on this data and these settings.")
if res["delays"] is not None:
    lines.append("**Known onset:** " + "; ".join(f"{k} = {'not detected' if v is None else v}" for k, v in res["delays"].items()) + ".")
st.markdown("\n\n".join(lines))

s_df = res["summary"].set_index("Strategy")
best, rank = res["best"], res["rank_basis"]
basis = "post-drift" if res["rank_uses_post_drift"] else "overall (no drift signal, so the whole stream is used)"
st.markdown(f"**Strategy comparison ({basis} {plabel}):**")
for name in sorted(rank, key=lambda k: rank[k], reverse=higher):
    st.markdown(f"- {name}: {rank[name]:.4f}  (update time {s_df.loc[name, 'Update time (s)']:.2f}s, "
                f"{int(s_df.loc[name, 'Updates'])} updates)")
gap = rank[best] - rank[STATIC] if higher else rank[STATIC] - rank[best]
material = gap > (0.01 if higher else 0.02 * abs(rank[STATIC]) + 1e-12)
if material:
    st.success(f"**Best strategy for this dataset and setup: {best}.** It improved {plabel} over the static model "
               f"by {abs(gap):.4f} ({basis}). Its update cost was {s_df.loc[best, 'Update time (s)']:.2f}s in total.")
else:
    st.info(f"**{best}** scored best among the selected strategies, but its advantage over the static model "
            f"({abs(gap):.4f} {plabel}) is small. On this data and setup, adaptation did not clearly help.")
st.caption("This is one chronological run with one linear SGD model. Treat differences as indicative, not "
           "statistically tested. Rankings can change with the window size, trigger policy and ADWIN delta.")
