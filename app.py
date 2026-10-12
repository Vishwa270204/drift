"""
Model Drift Dashboard — Hoeffding Adaptive Tree edition

Install:
    pip install streamlit river pandas numpy plotly scikit-learn
Run:
    streamlit run app.py

Rows are processed in their supplied order unless a time column is selected.
Updates are manual by default. The candidate is a deep copy of the deployed
River model and is evaluated on recent held-out labelled rows before deployment.
"""
import copy
import io
import warnings
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix

try:
    from river import drift as river_drift
    from river import tree as river_tree
    RIVER_OK = True
except ImportError:
    river_tree, river_drift, RIVER_OK = None, None, False

warnings.filterwarnings("ignore", category=UserWarning, module="river")

SENS = {"Low (fewer alerts)": 4.0, "Medium": 3.0, "High (more alerts)": 2.0}
MIN_FIT = 30
DELTAS = [1e-3, 1e-5, 1e-7, 1e-9]
ADWIN_DELTAS = [0.1, 0.01, 0.002, 0.0005]
INC, KEEP = "Continue Learning (new data only)", "Keep Current Model"
METRICS = {
    "classification": ["Error rate", "1 - macro F1", "Fault-class miss rate (1 - recall)"],
    "regression": ["MAE", "RMSE"],
}
ROW_WISE = {"Error rate", "MAE"}
PALETTE = ["#1f6fb2", "#e08a00", "#7a4fb5", "#00897b", "#a0522d", "#c2185b"]


def make_tree(task, cat, hp, seed):
    kw = dict(
        grace_period=int(hp["grace"]), delta=float(hp["delta"]),
        tau=float(hp["tau"]), nominal_attributes=list(cat) or None,
        drift_window_threshold=int(hp["win_thr"]),
        drift_detector=river_drift.ADWIN(delta=float(hp["adwin"])),
        seed=int(seed),
    )
    if task == "regression":
        return river_tree.HoeffdingAdaptiveTreeRegressor(**kw)
    return river_tree.HoeffdingAdaptiveTreeClassifier(**kw)


class Bundle:
    """Online tree plus numeric median imputation."""
    def __init__(self, task, num, cat, hp, seed):
        self.task, self.num, self.cat = task, list(num), list(cat)
        self.hp, self.seed = hp, seed
        self.cols = self.num + self.cat
        self.med, self.major, self.tree = {}, None, None

    def _rows(self, X):
        Z = X[self.cols]
        if self.num:
            Z = Z.copy()
            Z[self.num] = Z[self.num].fillna(self.med)
        return Z.to_dict("records")

    def _learn(self, rows, ys):
        reg = self.task == "regression"
        for x, y in zip(rows, ys):
            self.tree.learn_one(x, float(y) if reg else str(y))

    def fit(self, X, y):
        y = np.asarray(y)
        if len(y) != len(X):
            raise ValueError("Features and target have different lengths.")
        if len(y) < MIN_FIT:
            raise ValueError(f"Need at least {MIN_FIT} labelled rows to fit a tree.")
        if self.task == "regression":
            if not np.all(np.isfinite(y.astype(float))):
                raise ValueError("Regression target contains non-finite values.")
        else:
            self.major = str(pd.Series(y).mode().iloc[0])
        self.med = {
            c: (float(X[c].median()) if X[c].notna().any() else 0.0)
            for c in self.num
        }
        self.tree = make_tree(self.task, self.cat, self.hp, self.seed)
        self._learn(self._rows(X), y)
        return self

    def learn(self, X, y):
        y = np.asarray(y)
        if not len(y):
            raise ValueError("No labelled rows to learn from.")
        self._learn(self._rows(X), y)
        return self

    def predict(self, X):
        reg, out = self.task == "regression", []
        for x in self._rows(X):
            pred = self.tree.predict_one(x)
            if reg:
                out.append(0.0 if pred is None else float(pred))
            else:
                out.append(self.major if pred is None else str(pred))
        return np.asarray(out, float) if reg else np.asarray(out).astype(str)


def row_err(task, y, p):
    if task == "regression":
        return np.abs(np.asarray(y, float) - np.asarray(p, float))
    return (np.asarray(y) != np.asarray(p)).astype(float)


def ref_stats(task, errors):
    mean, std = float(np.mean(errors)), float(np.std(errors))
    if task == "classification":
        std = max(std, 0.1)
    return mean, max(std, 0.1 * mean, 1e-9)


def evaluate(task, y, p, pos=None):
    y, p = np.asarray(y), np.asarray(p)
    if task == "regression":
        e = np.abs(y.astype(float) - p.astype(float))
        return {"mae": float(e.mean()), "rmse": float(np.sqrt(np.mean(e ** 2)))}
    labels = np.unique(np.concatenate([y, p]))
    out = {
        "err": float(np.mean(y != p)),
        "macro_f1": float(f1_score(y, p, labels=labels, average="macro", zero_division=0)),
        "precision": np.nan, "recall": np.nan, "f1": np.nan,
        "n_pos": int(np.sum(y == pos)) if pos is not None else 0,
    }
    if pos is not None and pos in labels:
        pr, rc, f1, _ = precision_recall_fscore_support(
            y, p, labels=[pos], average=None, zero_division=0
        )
        out.update(precision=float(pr[0]), f1=float(f1[0]))
        if out["n_pos"]:
            out["recall"] = float(rc[0])
    return out


def selection_score(metric, m):
    if metric == "Error rate":
        return m["err"]
    if metric == "1 - macro F1":
        return 1 - m["macro_f1"]
    if metric.startswith("Fault-class"):
        return 1 - m["recall"] if np.isfinite(m["recall"]) else np.nan
    return m["mae"] if metric == "MAE" else m["rmse"]


@st.cache_data(show_spinner="Reading CSV...")
def read_csv_bytes(data):
    for enc in ("utf-8-sig", "latin-1"):
        try:
            frame = pd.read_csv(io.BytesIO(data), encoding=enc)
            if frame.shape[1] <= 1:
                frame = pd.read_csv(io.BytesIO(data), encoding=enc, sep=None, engine="python")
            return frame
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not read this file as CSV.")


def label_str(s):
    s = pd.Series(s)
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        values = s.to_numpy(float)
        if len(values) and np.all(np.isfinite(values)) and np.all(np.mod(values, 1) == 0):
            return values.astype(np.int64).astype(str)
    return s.astype(str).to_numpy()


def pick_features(df, exclude):
    numeric, categorical = [], []
    for col in df.columns:
        if col in exclude:
            continue
        s, n = df[col], df[col].notna().sum()
        if n == 0 or s.nunique() <= 1:
            continue
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            is_id = s.nunique() / n > 0.99 and bool(np.all(np.mod(s.dropna(), 1) == 0))
            if not is_id:
                numeric.append(col)
        elif s.nunique() <= 50:
            categorical.append(col)
    return numeric, categorical


def build_frame(df, numeric, categorical):
    parts = {}
    for col in numeric:
        values = np.array(pd.to_numeric(df[col], errors="coerce"), dtype=float)
        values[~np.isfinite(values)] = np.nan
        parts[col] = values
    for col in categorical:
        parts[col] = np.where(df[col].notna(), df[col].astype(str), "(missing)")
    return pd.DataFrame(parts, index=df.index)


def init_session(df, target, time_col, task, cfg):
    if not RIVER_OK:
        raise ValueError("Install River with: pip install river")
    d = df.copy()
    if time_col not in (None, "(row order)"):
        if pd.api.types.is_numeric_dtype(d[time_col]):
            tv = pd.to_numeric(d[time_col], errors="coerce")
        else:
            tv = pd.to_datetime(d[time_col], errors="coerce", utc=True)
        d = d.loc[tv.notna()].assign(_time=tv[tv.notna()])
        d = d.sort_values("_time", kind="stable").drop(columns="_time")
    d = d[d[target].notna()]
    if task == "regression":
        yv = pd.to_numeric(d[target], errors="coerce")
        d = d.loc[np.isfinite(yv)].copy()
        y = pd.to_numeric(d[target]).to_numpy(float)
    d = d.reset_index(drop=True)
    if task == "classification":
        y = label_str(d[target])
    B = int(cfg["B"])
    n_train = int(len(d) * cfg["train_pct"] / 100)
    n_fit = int(n_train * 0.8)
    if len(d) < 400:
        raise ValueError("Please provide at least 400 labelled rows.")
    if n_fit < 50 or len(d) - n_train < 2 * B:
        raise ValueError("Training data is too small or the stream has fewer than two batches. Reduce batch size or adjust the training percentage.")
    numeric, categorical = pick_features(d.iloc[:n_train], {target, time_col})
    if not (numeric or categorical):
        raise ValueError("No usable feature columns were found.")
    X = build_frame(d, numeric, categorical)
    pos = cfg.get("pos")
    if task == "classification" and pos is None:
        values, counts = np.unique(y, return_counts=True)
        pos = str(values[np.argmin(counts)])
    hp = {k: cfg[k] for k in ("grace", "delta", "tau", "adwin", "win_thr")}
    S = {
        "X": X, "y": y, "task": task, "num": numeric, "cat": categorical,
        "hp": hp, "seed": int(cfg["seed"]), "n_tr": n_train, "B": B,
        "N": len(y) - n_train, "nb": int(np.ceil((len(y) - n_train) / B)),
        "z": cfg["z"], "delay": int(cfg["delay"]), "min_rows": int(cfg["min_rows"]),
        "snooze": int(cfg["snooze"]), "hold_frac": float(cfg["hold_frac"]),
        "metric": cfg["metric"], "pos": pos, "version": 0, "k": 0,
        "errs": [], "thr": [], "ver": [], "bad": [], "good": 0,
        "alert": None, "snooze_until": 0, "cand": None, "events": [],
        "retrains": [], "flash": None,
    }
    S["model"] = Bundle(task, numeric, categorical, hp, S["seed"]).fit(
        X.iloc[:n_fit], y[:n_fit]
    )
    errors = row_err(task, y[n_fit:n_train], S["model"].predict(X.iloc[n_fit:n_train]))
    S["ref_m"], S["ref_s"] = ref_stats(task, errors)
    return S


def log_event(S, event, detail):
    S["events"].append({"Batch": S["k"], "Event": event, "Detail": detail})


def new_rows(S):
    if S["alert"] is None:
        return 0
    known = S["k"] - S["delay"]
    return max(0, min(known * S["B"], S["N"]) - S["alert"]["start"] * S["B"])


def step(S):
    k, B, n_train = S["k"], S["B"], S["n_tr"]
    a = n_train + k * B
    b = min(n_train + (k + 1) * B, len(S["y"]))
    if b <= a:
        S["k"] = S["nb"]
        return
    errors = row_err(S["task"], S["y"][a:b], S["model"].predict(S["X"].iloc[a:b]))
    S["errs"].append(float(errors.mean()))
    S["thr"].append(S["ref_m"] + S["z"] * S["ref_s"] / np.sqrt(b - a))
    S["ver"].append(S["version"])
    S["k"] = k + 1
    j = k - S["delay"]
    if j >= 0 and S["ver"][j] == S["version"]:
        flagged = S["errs"][j] > S["thr"][j]
        if S["alert"] is None:
            S["bad"] = S["bad"] + [j] if flagged else []
            if len(S["bad"]) >= 2 and S["k"] > S["snooze_until"]:
                S["alert"] = {"start": S["bad"][0], "status": "waiting"}
                S["cand"], S["good"], S["bad"] = None, 0, []
                log_event(S, "Drift alert", f"Errors above threshold since batch {S['alert']['start'] + 1}")
        elif S["alert"]["status"] == "waiting":
            S["good"] = 0 if flagged else S["good"] + 1
            if S["good"] >= 3:
                S["alert"] = None
                log_event(S, "Alert cleared", "Error returned to normal before retraining")
    if S["alert"] and S["alert"]["status"] == "waiting" and new_rows(S) >= S["min_rows"]:
        S["alert"]["status"] = "ready"


def ready(S):
    return S["alert"] is not None and S["alert"]["status"] == "ready"


def run_batches(S, count=None):
    i = 0
    while S["k"] < S["nb"] and not ready(S) and (count is None or i < count):
        step(S)
        i += 1


def get_data(S):
    start = S["n_tr"] + S["alert"]["start"] * S["B"]
    end = max(start, S["n_tr"] + min((S["k"] - S["delay"]) * S["B"], S["N"]))
    return (
        S["X"].iloc[:start], S["y"][:start],
        S["X"].iloc[start:end], S["y"][start:end],
    )


@dataclass
class Plan:
    kind: str
    X: Optional[pd.DataFrame] = None
    y: Optional[np.ndarray] = None
    note: str = ""


@dataclass
class Strategy:
    name: str
    plan: Callable
    blurb: str


def plan_inc(S, X_old, y_old, X_new, y_new):
    return Plan("continue", X_new, y_new, f"Learn from {len(y_new):,} new rows")


def plan_keep(S, X_old, y_old, X_new, y_new):
    return Plan("keep", note="Leave the deployed model unchanged")


STRATEGIES = [
    Strategy(INC, plan_inc, "Keep existing knowledge and learn from recent labelled rows."),
    Strategy(KEEP, plan_keep, "Make no changes to the deployed model."),
]
REGISTRY = {strategy.name: strategy for strategy in STRATEGIES}


def execute(S, plan):
    if plan.kind == "keep":
        return S["model"]
    if plan.kind == "continue":
        return copy.deepcopy(S["model"]).learn(plan.X, plan.y)
    raise ValueError(f"Unknown plan kind: {plan.kind}")


def build_candidate(S, name, data):
    plan = REGISTRY[name].plan(S, *data)
    return execute(S, plan), plan.note


def compute_candidates(S):
    X_old, y_old, X_new, y_new = get_data(S)
    n = len(y_new)
    if n < 2 * MIN_FIT:
        return {"error": f"Only {n} new labelled rows. At least {2 * MIN_FIT} are needed to train and evaluate an update."}
    hold = int(np.clip(round(n * S["hold_frac"]), MIN_FIT, n - MIN_FIT))
    X_hold, y_hold = X_new.iloc[-hold:], y_new[-hold:]
    X_train, y_train = X_new.iloc[:-hold], y_new[:-hold]
    metrics, errors, predictions, failures = {}, {}, {}, []
    for strategy in STRATEGIES:
        try:
            model, _ = build_candidate(S, strategy.name, (X_old, y_old, X_train, y_train))
            pred = model.predict(X_hold)
            metrics[strategy.name] = evaluate(S["task"], y_hold, pred, S["pos"])
            errors[strategy.name] = row_err(S["task"], y_hold, pred)
            predictions[strategy.name] = pred
        except Exception as exc:
            failures.append(f"{strategy.name}: {type(exc).__name__}: {exc}")
    if KEEP not in metrics:
        return {"error": "The current model could not be evaluated on held-out rows."}
    metric = S["metric"]
    notes = []
    if metric.startswith("Fault-class") and not np.isfinite(metrics[KEEP]["recall"]):
        metric = "Error rate"
        notes.append("No positive-class examples were present in the holdout; showing error rate instead.")
    rows = []
    for strategy in STRATEGIES:
        if strategy.name not in metrics:
            continue
        m = metrics[strategy.name]
        row = {"Strategy": strategy.name, "Selection score": selection_score(metric, m)}
        if S["task"] == "regression":
            row.update(MAE=m["mae"], RMSE=m["rmse"])
        else:
            row.update({
                "Error rate": m["err"], "Macro F1": m["macro_f1"],
                "Fault precision": m["precision"], "Fault recall": m["recall"],
                "Fault F1": m["f1"],
            })
        rows.append(row)
    table = pd.DataFrame(rows)
    keep_score = float(table.loc[table["Strategy"] == KEEP, "Selection score"].iloc[0])
    new_score = (
        float(table.loc[table["Strategy"] == INC, "Selection score"].iloc[0])
        if INC in metrics else np.nan
    )
    verdict = "Not measurable"
    clear_win = False
    if np.isfinite(new_score):
        if metric in ROW_WISE:
            difference = errors[KEEP] - errors[INC]
            se = difference.std(ddof=1) / np.sqrt(len(difference)) if len(difference) > 1 else np.inf
            z_score = difference.mean() / se if np.isfinite(se) and se > 0 else 0.0
            verdict = (
                "Updated model is clearly better" if z_score > 2 else
                "Updated model is clearly worse" if z_score < -2 else
                "Difference is within noise"
            )
            clear_win = z_score > 2
        else:
            verdict = (
                "Updated model scores better" if new_score < keep_score else
                "Updated model scores worse" if new_score > keep_score else
                "No measurable difference"
            )
        if hold < 100:
            notes.append(f"Only {hold} held-out rows; the score difference may be noisy.")
        if S["task"] == "classification" and metrics[KEEP]["n_pos"] < 10:
            notes.append("Fewer than 10 fault-class examples: fault-class metrics may be unreliable.")
    return {
        "table": table, "errs": errors, "preds": predictions, "yh": y_hold,
        "hold": hold, "n_old": len(y_old), "n_new": n,
        "n_train": len(y_train), "metric": metric, "warnings": notes,
        "failed": failures, "k_s": keep_score, "r_s": new_score,
        "verdict": verdict, "clear_win": clear_win,
    }


def apply_choice(S, name):
    candidate, alert = S["cand"], S["alert"]
    if name == KEEP:
        S["snooze_until"] = S["k"] + S["snooze"]
        log_event(S, "Kept current model", f"No alerts for {S['snooze']} batches")
        message = f"Kept current model v{S['version']}."
    else:
        try:
            if not candidate or name not in candidate["errs"]:
                raise ValueError("This strategy has no valid evaluation.")
            data = get_data(S)
            model, _ = build_candidate(S, name, data)
            # Smoke test before swapping the deployed model.
            model.predict(data[2].iloc[:1])
            score_table = candidate["table"].set_index("Strategy")["Selection score"]
            old_version = S["version"]
            S["model"] = model
            S["version"] += 1
            # This is a provisional baseline based on held-out candidate errors.
            S["ref_m"], S["ref_s"] = ref_stats(S["task"], candidate["errs"][name])
            S["retrains"].append({
                "version": S["version"], "strategy": name, "applied": S["k"],
                "start": alert["start"], "score_new": float(score_table[name]),
                "score_keep": float(score_table[KEEP]), "metric": candidate["metric"],
            })
            log_event(S, "Retrained", f"v{S['version']} deployed; previous version v{old_version}")
            message = f"Deployed model v{S['version']}."
        except Exception as exc:
            log_event(S, "Retraining failed", f"{name}: {exc}. Deployed model unchanged.")
            return False, f"Update failed: {exc}. The deployed model was not changed."
    S["alert"], S["cand"], S["bad"], S["good"] = None, None, [], 0
    return True, message


def show_drift_chart(S):
    n = min(S["k"], len(S["ver"]), len(S["errs"]), len(S["thr"]))
    if not n:
        return
    x = np.arange(1, n + 1)
    versions = np.asarray(S["ver"])[:n]
    errors, thresholds = np.asarray(S["errs"])[:n], np.asarray(S["thr"])[:n]
    fig = go.Figure()
    for i, version in enumerate(sorted(np.unique(versions))):
        idx = np.flatnonzero(versions == version)
        fig.add_trace(go.Scatter(
            x=x[idx], y=errors[idx], mode="lines+markers",
            name=f"Model v{version}", line=dict(color=PALETTE[i % len(PALETTE)])
        ))
    fig.add_trace(go.Scatter(
        x=x, y=thresholds, mode="lines", name="Drift threshold",
        line=dict(dash="dash", color="#c62828")
    ))
    if S["alert"]:
        fig.add_vline(x=S["alert"]["start"] + 1, line_dash="dot", line_color="#c62828")
    fig.update_layout(
        title="Prediction error by incoming batch",
        xaxis_title="Batch",
        yaxis_title="MAE (target units)" if S["task"] == "regression" else "Error rate",
        height=350, margin=dict(t=50, b=35, l=45, r=15),
        legend=dict(orientation="h", y=-0.22),
    )
    st.plotly_chart(fig, use_container_width=True)


def decision_panel(S):
    c, alert = S["cand"], S["alert"]
    st.subheader("Drift detected — choose a strategy")
    if c.get("error"):
        st.warning(c["error"])
        if st.button("Keep current model", key="keep_unavailable"):
            S["snooze_until"] = S["k"] + S["snooze"]
            log_event(S, "Kept current model", "Insufficient data to evaluate update")
            S["alert"], S["cand"], S["bad"], S["good"] = None, None, [], 0
            S["flash"] = ("info", "Kept the current model.")
            st.rerun()
        return

    st.caption(
        f"Drift started at batch {alert['start'] + 1} · "
        f"{c['hold']:,} held-out rows · {c['n_new']:,} new labelled rows"
    )
    st.info(c["verdict"])
    left, right = st.columns(2)
    can_update = INC in c["errs"]

    with left:
        with st.container(border=True):
            st.markdown("### Keep current model")
            st.metric(c["metric"], f"{c['k_s']:.4f}")
            st.caption(f"Deployed version v{S['version']}")
            st.write("No changes.")
            if st.button("Keep model", key=f"keep_{S['k']}", use_container_width=True):
                _, msg = apply_choice(S, KEEP)
                S["flash"] = ("info", msg)
                st.rerun()

    with right:
        with st.container(border=True):
            st.markdown("### Update model")
            if can_update:
                st.metric(
                    c["metric"], f"{c['r_s']:.4f}",
                    delta=f"{c['r_s'] - c['k_s']:+.4f}",
                    delta_color="inverse",
                )
            else:
                st.metric(c["metric"], "Unavailable")
            st.write("Learn from new labelled rows only.")
            if st.button(
                "Update model", type="primary", disabled=not can_update,
                key=f"update_{S['k']}", use_container_width=True,
            ):
                ok, msg = apply_choice(S, INC)
                S["flash"] = ("success" if ok else "error", msg)
                st.rerun()

    st.caption("Both scores use the same held-out rows. Lower is better; future performance may differ.")
    if c.get("warnings") or c.get("failed"):
        with st.expander("Evaluation notes"):
            for note in c.get("warnings", []):
                st.write(f"• {note}")
            for failure in c.get("failed", []):
                st.write(f"• {failure}")


def show_history(S):
    st.subheader("Model history")
    rows = []
    for r in S["retrains"]:
        rows.append({
            "Version": f"v{r['version']}",
            "Strategy": r["strategy"],
            "Applied at batch": r["applied"] + 1,
            "Metric": r["metric"],
            "Updated score": r["score_new"],
            "Previous score": r["score_keep"],
        })
    if rows:
        st.dataframe(pd.DataFrame(rows).round(4), hide_index=True, use_container_width=True)


def sidebar_config(S):
    with st.sidebar:
        st.header("Configuration")
        upload = st.file_uploader("Upload labelled CSV", type=["csv"])
        if upload is None:
            return
        try:
            df = read_csv_bytes(upload.getvalue())
        except Exception as exc:
            st.error(str(exc))
            return
        if len(df.columns) < 2:
            st.error("CSV must contain a target and at least one feature column.")
            return

        target = st.selectbox("Target column", df.columns, index=len(df.columns) - 1)
        time_col = st.selectbox(
            "Time column (optional)", ["(row order)"] + [col for col in df.columns if col != target]
        )
        guess = "regression" if (
            pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique() > 10
        ) else "classification"
        task = st.radio(
            "Task", ["classification", "regression"],
            index=["classification", "regression"].index(guess), horizontal=True,
        )
        train_pct = st.slider("Initial training (%)", 20, 70, 40, 5)
        batch_size = int(st.number_input("Batch size", 20, 5000, 100, step=10))
        sensitivity = st.selectbox("Drift sensitivity", list(SENS), index=1)
        delay = int(st.number_input("Label delay (batches)", 0, 20, 0))
        min_rows = int(st.number_input("New labelled rows before decision", 100, 100000, 300, step=50))
        snooze = int(st.number_input("Snooze after keeping (batches)", 0, 200, 10))
        hold_frac = st.slider("Holdout share", 0.10, 0.40, 0.25, 0.05)
        metric = st.selectbox("Evaluation metric (lower is better)", METRICS[task])
        pos = None
        if task == "classification":
            labels = sorted(set(label_str(df[target].dropna())))
            if labels:
                default = labels.index("1") if "1" in labels else 0
                pos = st.selectbox("Fault class", labels, index=default)

        with st.expander("Tree settings"):
            grace = int(st.number_input("Grace period", 10, 5000, 200, step=10))
            delta = st.select_slider("Split confidence (delta)", DELTAS, value=1e-7,
                                     format_func=lambda v: f"{v:.0e}")
            tau = st.number_input("Tie threshold", 0.001, 0.5, 0.05, step=0.01)
            adwin = st.select_slider("ADWIN sensitivity", ADWIN_DELTAS, value=0.002)
            win_thr = int(st.number_input("Alternate-tree window", 50, 5000, 300, step=50))
            seed = int(st.number_input("Random seed", 0, 10000, 42))

        if S is not None and S["k"] > 0:
            st.caption("Changing settings requires restarting the session.")
        if st.button("Restart with settings" if S else "Start monitoring", type="primary"):
            cfg = {
                "train_pct": train_pct, "B": batch_size, "z": SENS[sensitivity],
                "delay": delay, "min_rows": min_rows, "snooze": snooze,
                "hold_frac": hold_frac, "metric": metric, "pos": pos, "seed": seed,
                "grace": grace, "delta": delta, "tau": float(tau),
                "adwin": adwin, "win_thr": win_thr,
            }
            try:
                with st.spinner("Training initial Hoeffding Adaptive Tree..."):
                    st.session_state["S"] = init_session(df, target, time_col, task, cfg)
            except Exception as exc:
                st.error(str(exc))
                return
            st.rerun()


def main():
    st.set_page_config(page_title="Model Drift Dashboard", page_icon="📈", layout="wide")
    st.markdown("""
    <style>
      .block-container {padding-top: 1.3rem; padding-bottom: 1.5rem;}
      [data-testid="stMetric"] {
        background: rgba(128,128,128,.08); border-radius: 10px;
        padding: .6rem .8rem;
      }
    </style>
    """, unsafe_allow_html=True)
    st.title("📈 Model Drift Dashboard")
    st.caption("Monitor one deployed online model and review measured evidence before updating.")
    if not RIVER_OK:
        st.error("River is required. Install it with `pip install river` and restart the app.")
        st.stop()

    S = st.session_state.get("S")
    sidebar_config(S)
    S = st.session_state.get("S")
    if S is None:
        st.info("Upload a CSV in the sidebar, select the target and click Start monitoring.")
        st.stop()

    finished = S["k"] >= S["nb"]
    blocked = ready(S) or finished
    with st.container(border=True):
        controls = st.columns([2.2, 1, 1, 1])
        if controls[0].button("▶ Run until decision", type="primary", disabled=blocked, use_container_width=True):
            run_batches(S)
            st.rerun()
        if controls[1].button("Next batch", disabled=blocked, use_container_width=True):
            run_batches(S, 1)
            st.rerun()
        if controls[2].button("Next 10", disabled=blocked, use_container_width=True):
            run_batches(S, 10)
            st.rerun()
        if controls[3].button("Reset session", use_container_width=True):
            st.session_state.pop("S", None)
            st.rerun()

    alert = S["alert"]
    status = (
        "Finished" if finished and not ready(S) else
        "Decision needed" if ready(S) else
        "Collecting labels" if alert else "Monitoring"
    )
    cols = st.columns(4)
    cols[0].metric("Batches processed", f"{S['k']} / {S['nb']}")
    cols[1].metric("Deployed model", f"v{S['version']}")
    cols[2].metric("Updates", len(S["retrains"]))
    cols[3].metric("Status", status)

    if S.get("flash"):
        level, message = S["flash"]
        S["flash"] = None
        getattr(st, level)(message)

    show_drift_chart(S)
    if alert and alert["status"] == "waiting":
        st.info(
            f"Drift suspected from batch {alert['start'] + 1}. "
            f"{new_rows(S):,} of {S['min_rows']:,} new labelled rows are available. "
            "Monitoring continues until enough labels arrive or the alert clears."
        )

    if ready(S):
        if S["cand"] is None:
            with st.spinner("Evaluating the candidate on held-out recent rows..."):
                try:
                    S["cand"] = compute_candidates(S)
                except Exception as exc:
                    S["cand"] = {"error": f"Could not evaluate candidate: {exc}"}
        decision_panel(S)

    if S["retrains"]:
        show_history(S)
    if S["events"]:
        with st.expander("Event log"):
            st.dataframe(pd.DataFrame(S["events"]), hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
