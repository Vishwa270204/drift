"""Core logic and plotting helpers for the Model Drift Detection Dashboard."""

import copy
import time
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from river.drift import ADWIN
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


DEFAULT_CSV_PATH = "data.csv"


TARGET_COL = None       # None -> last column of the CSV; or put a fixed column name here


REF_PCT = 30            # FIXED size of the training (reference) data, in % of rows


MAX_CAT_UNIQUE = 50


TARGET_CAP = 2000


TRANSFER_EPOCHS = 3


SEED = 42


STATIC = "Static (no adaptation)"


INCR = "Incremental"


ONLINE = "Online"


CONT = "Continual"


TRANS = "Transfer"


SCRATCH = "Scratch (target-only baseline)"


LEARNERS = [INCR, ONLINE, CONT, TRANS]


PALETTE = ["#6366F1", "#06B6D4", "#F59E0B", "#10B981", "#EF4444", "#8B5CF6"]


def style_fig(fig, height=420):
    fig.update_layout(
        template="plotly_white", height=height, colorway=PALETTE,
        margin=dict(t=50, l=10, r=10, b=10), font=dict(size=13),
        legend=dict(orientation="h", y=-0.22, title_text=""),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(gridcolor="rgba(150,150,150,.2)")
    fig.update_yaxes(gridcolor="rgba(150,150,150,.2)")
    return fig


def prepare_df(df_raw, target, time_col):
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
    X = df[num_cols + cat_cols].copy()
    for c in cat_cols:
        col = X[c].astype(object)
        X[c] = col.where(col.isna(), col.astype(str))
    return X


def make_model(task):
    if task == "classification":
        return SGDClassifier(loss="log_loss", alpha=1e-4, random_state=SEED)
    return SGDRegressor(alpha=1e-4, random_state=SEED)


def fit_epochs(model, X, y, task, classes, epochs=1, rng=None):
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
    bounds = list(range(n_ref, n, batch))
    windows = [(s, min(s + batch, n)) for s in bounds]
    if len(windows) > 1 and windows[-1][1] - windows[-1][0] < batch // 2:
        last = windows.pop()
        windows[-1] = (windows[-1][0], last[1])
    return windows


def recovery_text(scores, first_w, higher_better, tol=0.05):
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


def run_experiment(df, num_cols, cat_cols, target, task, cfg, progress):
    n = len(df)
    n_ref = int(n * REF_PCT / 100)
    if n_ref < 50:
        raise ValueError("The fixed training (reference) period has fewer than 50 rows. Use a larger dataset.")
    windows = make_windows(n_ref, n, cfg["batch"])
    if len(windows) < 3:
        raise ValueError("Fewer than 3 evaluation windows. Reduce the window size.")
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
        le = LabelEncoder().fit(df[target].astype(str))
        y_all = le.transform(df[target].astype(str))
        classes = np.arange(len(le.classes_))
        y_true_all = y_all
        mu, sd = 0.0, 1.0
        if len(np.unique(y_all[:n_ref])) < 2:
            raise ValueError("The reference period contains a single class. Use a different dataset.")
    else:
        y_raw = df[target].astype(float).to_numpy()
        mu = float(y_raw[:n_ref].mean())
        sd = float(y_raw[:n_ref].std()) or 1.0
        y_all = (y_raw - mu) / sd
        y_true_all = y_raw

    # --- one source model, trained on the reference period ---
    base = make_model(task)
    fit_epochs(base, X_all[:n_ref], y_all[:n_ref], task, classes, epochs=10, rng=rng)
    models = {STATIC: base}
    for name in cfg["strategies"]:
        models[name] = copy.deepcopy(base)
    if TRANS in cfg["strategies"]:
        models[SCRATCH] = None

    d = X_all.shape[1]
    cap = cfg["buffer"]
    Xb, yb = np.zeros((cap, d)), np.zeros(cap, dtype=y_all.dtype)
    filled = seen = 0
    if CONT in models:
        filled, seen = reservoir_add(Xb, yb, filled, seen, X_all[:n_ref], y_all[:n_ref], rng)
    Xt, yt = np.zeros((0, d)), np.zeros(0, dtype=y_all.dtype)

    # Select adaptation data only from the reference period and rows already observed.
    # The default preserves the historical + current-window behavior for batch learners.
    retrain_data = cfg.get("retrain_data", "old_new")
    X_ref, y_ref = X_all[:n_ref], y_all[:n_ref]

    def adaptation_data(X_new, y_new):
        if retrain_data == "old":
            return X_ref, y_ref
        if retrain_data == "new":
            return X_new, y_new
        # old_new (default): retain the original reference examples and add current data.
        return np.vstack([X_ref, X_new]), np.concatenate([y_ref, y_new])

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
    remaining = 0  # windows of adaptation left (only set by an ADWIN drift signal)

    for w, (s, e) in enumerate(windows):
        progress((w + 1) / len(windows), f"Window {w + 1} / {len(windows)}")
        active = remaining > 0  # adaptation is ON only after a drift signal
        if active:
            active_windows.append(w)
        Xw, yw = X_all[s:e], y_all[s:e]

        # 1) PREDICT with every model before anything is trained on this window
        raw = {}
        for name in models:
            if name == ONLINE and active:
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

        # 2) DETECT: ADWIN on the errors of the never-updated static model
        if task == "classification":
            errors = (raw[STATIC] != yw).astype(float)
        else:
            errors = np.clip(np.abs(raw[STATIC] - yw), 0, 5)
        flagged = False
        for i, err in enumerate(errors):
            adwin.update(float(err))
            if adwin.drift_detected:
                adwin_events.append({"row": s + i, "window": w})
                flagged = True
        if flagged:
            flagged_windows.append(w)

        # 3) ADAPT (only while adaptation is active)
        if CONT in models:
            if active:
                if retrain_data == "old":
                    X_cont, y_cont = X_ref, y_ref
                elif retrain_data == "new" or filled == 0:
                    X_cont, y_cont = Xw, yw
                else:
                    # Preserve Continual's replay-buffer behavior for the combined option.
                    k = min(len(yw), filled)
                    idx = rng.choice(filled, size=k, replace=False)
                    X_cont = np.vstack([Xw, Xb[idx]])
                    y_cont = np.concatenate([yw, yb[idx]])
                t0 = time.perf_counter()
                fit_epochs(models[CONT], X_cont, y_cont, task, classes, rng=rng)
                upd_time[CONT] += time.perf_counter() - t0
                upd_count[CONT] += 1
            filled, seen = reservoir_add(Xb, yb, filled, seen, Xw, yw, rng)
        if INCR in models and active:
            X_adapt, y_adapt = adaptation_data(Xw, yw)
            t0 = time.perf_counter()
            fit_epochs(models[INCR], X_adapt, y_adapt, task, classes, rng=rng)
            upd_time[INCR] += time.perf_counter() - t0
            upd_count[INCR] += 1
        if TRANS in models:
            Xt, yt = np.vstack([Xt, Xw])[-TARGET_CAP:], np.concatenate([yt, yw])[-TARGET_CAP:]
            if active:
                X_adapt, y_adapt = adaptation_data(Xt, yt)
                t0 = time.perf_counter()
                tm = copy.deepcopy(base)
                fit_epochs(tm, X_adapt, y_adapt, task, classes, epochs=TRANSFER_EPOCHS, rng=rng)
                upd_time[TRANS] += time.perf_counter() - t0
                models[TRANS] = tm
                upd_count[TRANS] += 1
                # Keep Scratch as the target-only comparison baseline, independent of selector.
                t0 = time.perf_counter()
                sm = make_model(task)
                fit_epochs(sm, Xt, yt, task, classes, epochs=TRANSFER_EPOCHS, rng=rng)
                upd_time[SCRATCH] += time.perf_counter() - t0
                models[SCRATCH] = sm

        # 4) adaptation switch for the NEXT windows
        if active:
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


def timeline_fig(res):
    windows = res["windows"]
    mid = [(s + e) / 2 for s, e in windows]
    prim = res["primary"]
    ylabel = "F1 (macro)" if prim == "f1" else "MAE"
    sdf = res["metrics"][res["metrics"].strategy == STATIC].sort_values("window")
    y = sdf[prim].to_numpy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mid, y=y, mode="lines+markers", name=f"Static model {ylabel}",
                             line=dict(width=3, color=PALETTE[0]), marker=dict(size=7),
                             fill="tozeroy", fillcolor="rgba(99,102,241,.08)"))
    lo, hi = float(np.min(y)), float(np.max(y))
    pad = (hi - lo) * 0.12 or 0.05
    if res["adwin_events"]:
        fig.add_trace(go.Scatter(x=[ev["row"] for ev in res["adwin_events"]], y=[lo - pad] * len(res["adwin_events"]),
                                 mode="markers", name="ADWIN drift event",
                                 marker=dict(symbol="triangle-up", color="#EF4444", size=12)))
    fig.add_vline(x=res["n_ref"], line_dash="dot", line_color="gray", annotation_text="end of reference")
    if res["cfg"]["onset"] > 0:
        fig.add_vline(x=res["cfg"]["onset"], line_dash="dash", line_color="#10B981", annotation_text="known onset")
    fig.update_layout(xaxis_title="Row index", yaxis_title=ylabel)
    return style_fig(fig, 420)


