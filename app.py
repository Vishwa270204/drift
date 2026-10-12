"""Model Drift Dashboard (Hoeffding Adaptive Tree edition).

Streams a labelled CSV through ONE deployed model, a Hoeffding Adaptive Tree from `river` (pip install river),
detects drift from per-batch error, and when drift is confirmed shows the user measured evidence comparing the
current model with a copy that keeps learning on the new rows. The user then decides: update or keep.

    Strategy (registry)  ->  plan function (data selection)  ->  execute() (training procedure)

Drift threshold (one-sided z-test on the batch mean error):
    reference mean/std of per-row error come from the last 20% of the initial training rows;
    threshold = ref_mean + z * ref_std / sqrt(batch_size); two flagged batches in a row raise an alert.

The candidate is trained WITHOUT the newest labelled rows and scored on those same held-out rows.
The app does NOT tell the user what to do; it reports the measurement and a neutral evidence statement.
The deployed model only changes after Update succeeds.
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
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

try:
    from river import drift as river_drift
    from river import tree as river_tree
    RIVER_OK = True
except ImportError:                                   # reported in main()
    river_tree, river_drift, RIVER_OK = None, None, False

warnings.filterwarnings("ignore", category=UserWarning, module="river")

# ------------------------------------------------------------------ constants
SENS = {"Low (fewer alerts)": 4.0, "Medium": 3.0, "High (more alerts)": 2.0}
MIN_FIT = 30       # minimum usable labelled rows to train a tree
DELTAS = [1e-3, 1e-5, 1e-7, 1e-9]
ADWIN_DELTAS = [0.1, 0.01, 0.002, 0.0005]

INC, KEEP = "Continue Learning (new data only)", "Keep Current Model"

METRICS = {
    "classification": ["Error rate", "1 - macro F1", "Fault-class miss rate (1 - recall)"],
    "regression": ["MAE", "RMSE"],
}
ROW_WISE = {"Error rate", "MAE"}   # metrics with a per-row loss (paired noise check possible)
PALETTE = ["#1f6fb2", "#e08a00", "#7a4fb5", "#00897b", "#a0522d", "#c2185b"]


# ------------------------------------------------------------------ model
def make_tree(task, cat, hp, seed):
    kw = dict(grace_period=int(hp["grace"]), delta=float(hp["delta"]), tau=float(hp["tau"]),
              nominal_attributes=list(cat) or None, drift_window_threshold=int(hp["win_thr"]),
              drift_detector=river_drift.ADWIN(delta=float(hp["adwin"])), seed=int(seed))
    if task == "regression":
        return river_tree.HoeffdingAdaptiveTreeRegressor(**kw)
    return river_tree.HoeffdingAdaptiveTreeClassifier(**kw)


class Bundle:
    """A Hoeffding Adaptive Tree plus the small amount of preprocessing it needs (median imputation for numeric
    columns; categorical columns are passed as nominal attributes)."""

    def __init__(self, task, num, cat, hp, seed):
        self.task, self.num, self.cat, self.hp, self.seed = task, list(num), list(cat), hp, seed
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
        for x, t in zip(rows, ys):
            self.tree.learn_one(x, float(t) if reg else str(t))

    def fit(self, X, y):
        """Initial tree trained in one pass, in the given (time) order."""
        y = np.asarray(y)
        if len(y) != len(X):
            raise ValueError("features and target have different lengths")
        if len(y) < MIN_FIT:
            raise ValueError(f"only {len(y)} usable labelled rows (need at least {MIN_FIT})")
        if self.task == "regression":
            if not np.all(np.isfinite(y.astype(float))):
                raise ValueError("target contains non-finite values")
        else:
            self.major = str(pd.Series(y).mode().iloc[0])
        self.med = {c: (float(X[c].median()) if X[c].notna().any() else 0.0) for c in self.num}
        self.tree = make_tree(self.task, self.cat, self.hp, self.seed)
        self._learn(self._rows(X), y)
        return self

    def learn(self, X, y):
        """Keep learning incrementally on the existing tree, one row at a time."""
        y = np.asarray(y)
        if len(y) == 0:
            raise ValueError("no labelled rows to learn from")
        self._learn(self._rows(X), y)
        return self

    def predict(self, X):
        reg = self.task == "regression"
        out = []
        for x in self._rows(X):
            p = self.tree.predict_one(x)
            out.append((0.0 if p is None else float(p)) if reg else (self.major if p is None else str(p)))
        return np.asarray(out, float) if reg else np.asarray(out).astype(str)


def row_err(task, y, p):
    if task == "regression":
        return np.abs(np.asarray(y, float) - np.asarray(p, float))
    return (np.asarray(y) != np.asarray(p)).astype(float)


def ref_stats(task, e):
    m, s = float(e.mean()), float(e.std())
    s = max(s, 0.1) if task == "classification" else s
    return m, max(s, 0.1 * m, 1e-9)


# ------------------------------------------------------------------ metrics
def evaluate(task, y, p, pos=None):
    y, p = np.asarray(y), np.asarray(p)
    if task == "regression":
        e = np.abs(y.astype(float) - p.astype(float))
        return dict(mae=float(e.mean()), rmse=float(np.sqrt((e ** 2).mean())))
    labels = np.unique(np.concatenate([y, p]))
    out = dict(err=float((y != p).mean()),
               macro_f1=float(f1_score(y, p, labels=labels, average="macro", zero_division=0)),
               precision=np.nan, recall=np.nan, f1=np.nan, n_pos=int((y == pos).sum()) if pos is not None else 0)
    if pos is not None and pos in labels:
        pr, rc, f1, _ = precision_recall_fscore_support(y, p, labels=[pos], average=None, zero_division=0)
        out["precision"], out["f1"] = float(pr[0]), float(f1[0])
        if out["n_pos"] > 0:
            out["recall"] = float(rc[0])
    return out


def selection_score(metric, m):
    if metric == "Error rate":
        return m["err"]
    if metric == "1 - macro F1":
        return 1.0 - m["macro_f1"]
    if metric.startswith("Fault-class miss rate"):
        return 1.0 - m["recall"] if np.isfinite(m["recall"]) else np.nan
    return m["mae"] if metric == "MAE" else m["rmse"]


# ------------------------------------------------------------------ data prep
@st.cache_data(show_spinner="Reading CSV...")
def read(data):
    for enc in ("utf-8-sig", "latin-1"):
        try:
            out = pd.read_csv(io.BytesIO(data), encoding=enc)
            if out.shape[1] <= 1:
                out = pd.read_csv(io.BytesIO(data), encoding=enc, sep=None, engine="python")
            return out
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not read the file as CSV.")


def label_str(s):
    """Stable string labels: integral numbers become '0','1' (not '0.0','1.0'). Expects no missing values."""
    s = pd.Series(s)
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        v = s.to_numpy(float)
        if len(v) and np.all(np.isfinite(v)) and np.all(np.mod(v, 1) == 0):
            return v.astype(np.int64).astype(str)
    return s.astype(str).to_numpy()


def build_frame(df, num, cat):
    parts = {}
    for c in num:
        v = np.array(pd.to_numeric(df[c], errors="coerce"), dtype=float)
        v[~np.isfinite(v)] = np.nan
        parts[c] = v
    for c in cat:
        parts[c] = np.where(df[c].notna(), df[c].astype(str), "(missing)")
    return pd.DataFrame(parts, index=df.index)


def pick_features(df, exclude):
    num, cat = [], []
    for c in df.columns:
        if c in exclude:
            continue
        s, n = df[c], df[c].notna().sum()
        if n == 0 or s.nunique() <= 1:
            continue
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            if not (s.nunique() / n > 0.99 and bool(np.all(np.mod(s.dropna(), 1) == 0))):  # skip id-like
                num.append(c)
        elif s.nunique() <= 50:
            cat.append(c)
    return num, cat


# ------------------------------------------------------------------ session setup (UI independent)
def init_session(df, target, tcol, task, cfg):
    """Build the monitoring session dict. Raises ValueError with a readable message on bad input."""
    if not RIVER_OK:
        raise ValueError("The `river` package is not installed (pip install river).")
    d = df.copy()
    if tcol not in (None, "(row order)"):
        tv = pd.to_numeric(d[tcol], errors="coerce") if pd.api.types.is_numeric_dtype(d[tcol]) \
            else pd.to_datetime(d[tcol], errors="coerce", utc=True)
        d = d.loc[tv.notna()].assign(_t=tv[tv.notna()]).sort_values("_t", kind="stable").drop(columns="_t")
    d = d[d[target].notna()]
    if task == "regression":
        yv = pd.to_numeric(d[target], errors="coerce")
        d = d.loc[np.isfinite(yv)]
        y = yv[np.isfinite(yv)].to_numpy(float)
    d = d.reset_index(drop=True)
    if task == "classification":
        y = label_str(d[target])
    B = int(cfg["B"])
    n_tr = int(len(d) * cfg["train_pct"] / 100)
    n_fit = int(n_tr * 0.8)
    if len(d) < 400:
        raise ValueError("Need at least 400 labelled rows.")
    if n_fit < 50 or len(d) - n_tr < 2 * B:
        raise ValueError("Initial training part is too small, or the streaming part has fewer than 2 batches. "
                         "Lower the batch size or adjust the training percentage.")
    num, cat = pick_features(d.iloc[:n_tr], {target, tcol})   # chosen from the initial training rows only
    if not (num or cat):
        raise ValueError("No usable feature columns were found.")
    X = build_frame(d, num, cat)
    pos = cfg.get("pos")
    if task == "classification" and pos is None:
        vals, cnt = np.unique(y, return_counts=True)
        pos = str(vals[np.argmin(cnt)])
    hp = dict(grace=int(cfg["grace"]), delta=float(cfg["delta"]), tau=float(cfg["tau"]),
              adwin=float(cfg["adwin"]), win_thr=int(cfg["win_thr"]))
    S = dict(X=X, y=y, task=task, num=num, cat=cat, hp=hp, seed=int(cfg["seed"]), n_tr=n_tr, B=B,
             N=len(y) - n_tr, nb=int(np.ceil((len(y) - n_tr) / B)), z=cfg["z"], delay=int(cfg["delay"]),
             min_rows=int(cfg["min_rows"]), snooze=int(cfg["snooze"]), auto=bool(cfg.get("auto", True)), hold_frac=float(cfg["hold_frac"]),
             metric=cfg.get("metric") or METRICS[task][0], pos=pos,
             version=0, k=0, errs=[], thr=[], ver=[], bad=[], good=0, alert=None, snooze_until=0,
             cand=None, events=[], retrains=[], flash=None)
    try:
        S["model"] = Bundle(task, num, cat, hp, S["seed"]).fit(X.iloc[:n_fit], y[:n_fit])
    except Exception as exc:
        raise ValueError(f"Could not train the first model: {exc}")
    e = row_err(task, y[n_fit:n_tr], S["model"].predict(X.iloc[n_fit:n_tr]))
    S["ref_m"], S["ref_s"] = ref_stats(task, e)
    return S


# ------------------------------------------------------------------ streaming engine
def log(S, event, detail):
    S["events"].append({"Batch": S["k"], "Event": event, "Detail": detail})


def new_rows(S):
    """Labelled rows available since the suspected drift start (respects label delay)."""
    al = S["alert"]
    if al is None:
        return 0
    known = S["k"] - S["delay"]
    return max(0, min(known * S["B"], S["N"]) - al["start"] * S["B"])


def step(S):
    """Receive one batch: predict with the deployed model, then judge the batch whose labels just arrived."""
    k, B, n_tr = S["k"], S["B"], S["n_tr"]
    a, b = n_tr + k * B, min(n_tr + (k + 1) * B, len(S["y"]))
    if b <= a:
        S["k"] = S["nb"]
        return
    e = row_err(S["task"], S["y"][a:b], S["model"].predict(S["X"].iloc[a:b]))
    S["errs"].append(float(e.mean()))
    S["thr"].append(S["ref_m"] + S["z"] * S["ref_s"] / np.sqrt(b - a))
    S["ver"].append(S["version"])
    S["k"] = k + 1
    j = k - S["delay"]                                   # batch whose labels arrive now
    if j >= 0 and S["ver"][j] == S["version"]:           # only judge batches scored by the deployed model
        flagged = S["errs"][j] > S["thr"][j]
        if S["alert"] is None:
            S["bad"] = S["bad"] + [j] if flagged else []
            if len(S["bad"]) >= 2 and S["k"] > S["snooze_until"]:     # 2 bad batches in a row = alert
                S["alert"], S["cand"], S["good"] = dict(start=S["bad"][0], status="waiting"), None, 0
                S["bad"] = []
                log(S, "Drift alert", f"errors above threshold since batch {S['alert']['start'] + 1}")
        elif S["alert"]["status"] == "waiting":
            S["good"] = 0 if flagged else S["good"] + 1
            if S["good"] >= 3:                                         # it went away by itself
                S["alert"] = None
                log(S, "Alert cleared", "error returned to normal before retraining was needed")
    al = S["alert"]
    if al and al["status"] == "waiting" and new_rows(S) >= S["min_rows"]:
        al["status"] = "ready"


def ready(S):
    return S["alert"] is not None and S["alert"]["status"] == "ready"


def run(S, n):
    i = 0
    while S["k"] < S["nb"] and not ready(S) and (n is None or i < n):
        step(S)
        i += 1


def get_data(S):
    """(old X, old y, new X, new y): old = labelled rows before the drift start, new = labelled rows since."""
    n_tr, B = S["n_tr"], S["B"]
    s0 = n_tr + S["alert"]["start"] * B
    s1 = max(s0, n_tr + min((S["k"] - S["delay"]) * B, S["N"]))
    X, y = S["X"], S["y"]
    return X.iloc[:s0], y[:s0], X.iloc[s0:s1], y[s0:s1]


# ------------------------------------------------------------------ strategies: data selection -> training
@dataclass
class Plan:
    kind: str                       # keep | continue
    X: Optional[pd.DataFrame] = None
    y: Optional[np.ndarray] = None
    note: str = ""


@dataclass
class Strategy:
    name: str
    plan: Callable
    blurb: str


def plan_inc(S, Xo, yo, Xn, yn):
    return Plan("continue", Xn, yn, note=f"the model learns from the {len(yn):,} new rows")


def plan_keep(S, Xo, yo, Xn, yn):
    return Plan("keep", note="deployed model unchanged")


STRATEGIES = [
    Strategy(INC, plan_inc, "The model keeps what it already knows and learns from the recent data. "
                            "Parts that no longer work are replaced automatically."),
    Strategy(KEEP, plan_keep, "No changes. The model stays as it is."),
]
REG = {s.name: s for s in STRATEGIES}


def execute(S, plan):
    """Training procedure for a plan. Never mutates the deployed model (continue plans use a deep copy)."""
    if plan.kind == "keep":
        return S["model"]
    if plan.kind == "continue":
        return copy.deepcopy(S["model"]).learn(plan.X, plan.y)
    raise ValueError(f"unknown plan kind {plan.kind}")


def build(S, name, data):
    plan = REG[name].plan(S, *data)
    return execute(S, plan), plan.note


# ------------------------------------------------------------------ candidate comparison (evidence only)
def compute_cands(S):
    """Train the candidate WITHOUT the newest labelled rows and score it against the current model on those rows.
    Returns measurements and a neutral evidence statement; it does not recommend an action."""
    Xo, yo, Xn, yn = get_data(S)
    n = len(yn)
    if n < 2 * MIN_FIT:
        return dict(error=f"Only {n} new labelled rows: not enough to hold out an evaluation set and still "
                          f"train (need at least {2 * MIN_FIT}). Keep the current model for now, or lower "
                          "'Min new labelled rows'.")
    hold = int(np.clip(round(n * S["hold_frac"]), MIN_FIT, n - MIN_FIT))
    Xh, yh, Xt, yt = Xn.iloc[-hold:], yn[-hold:], Xn.iloc[:-hold], yn[:-hold]
    task, pos = S["task"], S["pos"]
    mets, errs, preds, notes, failed = {}, {}, {}, {}, []
    for sp in STRATEGIES:
        try:
            model, note = build(S, sp.name, (Xo, yo, Xt, yt))
            p = model.predict(Xh)
            if task == "regression" and not np.all(np.isfinite(np.asarray(p, float))):
                raise ValueError("non-finite predictions")
            mets[sp.name] = evaluate(task, yh, p, pos)
            errs[sp.name], preds[sp.name], notes[sp.name] = row_err(task, yh, p), np.asarray(p), note
        except Exception as exc:
            failed.append((sp.name, f"{type(exc).__name__}: {exc}"))
    if KEEP not in mets:
        return dict(error="The current model could not be evaluated on the held-out rows.")
    metric, warns = S["metric"], []
    if metric.startswith("Fault-class") and not np.isfinite(mets[KEEP]["recall"]):
        metric = "Error rate"
        warns.append(f"The held-out rows contain no '{pos}' (fault) examples, so recall is undefined. "
                     "Error rate is used instead.")
    rows = []
    for sp in STRATEGIES:
        if sp.name not in mets:
            continue
        m = mets[sp.name]
        row = {"Strategy": sp.name, "Selection score": selection_score(metric, m)}
        if task == "regression":
            row.update({"MAE": m["mae"], "RMSE": m["rmse"]})
        else:
            row.update({"Error rate": m["err"], "Precision (fault class)": m["precision"],
                        "Recall (fault class)": m["recall"], "F1 (fault class)": m["f1"], "Macro F1": m["macro_f1"]})
        rows.append(row)
    t = pd.DataFrame(rows)
    k_s = float(t.loc[t["Strategy"] == KEEP, "Selection score"].iloc[0])
    t["Better than keeping (%)"] = (k_s - t["Selection score"]) / k_s * 100 if k_s > 1e-12 else np.nan
    r_s = float(t.loc[t["Strategy"] == INC, "Selection score"].iloc[0]) if INC in mets else np.nan

    # reliability notes
    if hold < 100:
        warns.append(f"Only {hold} held-out rows: differences between the models may be noise.")
    if task == "classification":
        npos = mets[KEEP]["n_pos"]
        if npos < 10:
            warns.append(f"Only {npos} '{pos}' (fault) rows in the held-out set: recall and F1 are unreliable.")
        share = max(np.mean(yh == v) for v in np.unique(yh))
        if share > 0.95 and metric == "Error rate":
            warns.append(f"The held-out set is {share:.0%} one class; error rate can look good for a model that "
                         "never predicts the fault class. Consider the fault-class miss rate in Advanced settings.")

    # neutral evidence statement (no recommendation)
    verdict = "Not measurable"
    if np.isfinite(r_s) and np.isfinite(k_s):
        if metric in ROW_WISE and INC in errs:
            d = errs[KEEP] - errs[INC]                     # positive = updated model has lower error
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.inf
            zz = d.mean() / se if np.isfinite(se) and se > 0 else 0.0
            verdict = ("Updated model is clearly better (more than 2 standard errors)" if zz > 2 else
                       "Updated model is clearly worse (more than 2 standard errors)" if zz < -2 else
                       "Difference is within noise (less than 2 standard errors)")
        else:
            verdict = ("Updated model scores better (no noise test for this metric)" if r_s < k_s else
                       "Updated model scores worse (no noise test for this metric)" if r_s > k_s else
                       "No difference")
    rec = INC if (np.isfinite(r_s) and r_s < k_s) else KEEP    # used only for chart colour
    if np.isfinite(r_s):
        why = (f"Held-out {metric} on {hold:,} recent rows (lower is better): current model {k_s:.4f}, "
               f"updated model {r_s:.4f}. **Evidence: {verdict}.** This is a measurement on recent data, "
               "not a guarantee for future data.")
    else:
        why = f"The updated model could not be evaluated. Held-out {metric} of the current model: {k_s:.4f}."
    return dict(table=t, errs=errs, preds=preds, yh=yh, notes=notes, hold=hold, n_old=len(yo), n_new=n,
                n_train=len(yt), failed=failed, metric=metric, warnings=warns, rec=rec, why=why,
                verdict=verdict, k_s=k_s, r_s=r_s,
                clear_win=verdict.startswith("Updated model is clearly better"))


def apply_choice(S, name):
    """Apply a decision. Returns (ok, message). The deployed model and version change only on success."""
    c, al = S["cand"], S["alert"]
    if name == KEEP:
        S["snooze_until"] = S["k"] + S["snooze"]
        log(S, "Kept current model", f"no alerts for {S['snooze']} batches")
        msg = f"Kept the current model (v{S['version']}). No new alerts for {S['snooze']} batches."
    else:
        try:
            if name not in c["errs"]:
                raise ValueError("this strategy has no valid evaluation")
            Xo, yo, Xn, yn = get_data(S)
            model, note = build(S, name, (Xo, yo, Xn, yn))     # final build on ALL new labelled rows
            model.predict(Xn.iloc[:1])                          # smoke test before accepting
        except Exception as exc:
            log(S, "Retraining failed", f"{name}: {exc}. Deployed model unchanged.")
            return False, f"Retraining with {name} failed: {exc}. The deployed model was not changed."
        S["model"] = model
        S["version"] += 1
        S["ref_m"], S["ref_s"] = ref_stats(S["task"], c["errs"][name])   # threshold from honest held-out errors
        t = c["table"].set_index("Strategy")["Selection score"]
        S["retrains"].append(dict(version=S["version"], strategy=name, applied=S["k"], start=al["start"],
                                  score_new=float(t[name]), score_keep=float(t[KEEP]), metric=c["metric"]))
        log(S, "Retrained", f"v{S['version']} = {name} (old {c['n_old']:,} rows, new {c['n_new']:,} rows)")
        msg = f"Deployed model v{S['version']} ({name})."
    S["alert"], S["cand"], S["bad"], S["good"] = None, None, [], 0
    return True, msg


# ------------------------------------------------------------------ UI helpers
_NEW_API = tuple(int(x) for x in st.__version__.split(".")[:2] if x.isdigit()) >= (1, 50)


def wide(fn, *a, **kw):
    kw["width" if _NEW_API else "use_container_width"] = "stretch" if _NEW_API else True
    return fn(*a, **kw)


def confirm_box(flag, message, yes_label):
    if not st.session_state.get(flag):
        return False
    st.warning(message)
    a, b = st.columns(2)
    if a.button(yes_label, key=flag + "_yes", type="primary"):
        st.session_state[flag] = False
        return True
    if b.button("Cancel", key=flag + "_no"):
        st.session_state[flag] = False
        st.rerun()
    return False


def drift_chart(S):
    n = min(S["k"], len(S["ver"]), len(S["errs"]), len(S["thr"]))
    xs, ver = np.arange(1, n + 1), np.asarray(S["ver"])[:n]
    errs, thr = np.asarray(S["errs"], float)[:n], np.asarray(S["thr"], float)[:n]
    reg = S["task"] == "regression"
    fig = go.Figure()
    for i, v in enumerate(sorted(np.unique(ver))):
        idx = np.flatnonzero(ver == v)
        fig.add_trace(go.Scatter(x=xs[idx], y=errs[idx], mode="lines+markers", name=f"Model v{v}",
                                 line=dict(color=PALETTE[i % len(PALETTE)])))
    fig.add_trace(go.Scatter(x=xs, y=thr, mode="lines", name="Drift threshold", line=dict(dash="dash", color="#c62828")))
    for r in S["retrains"]:
        if 1 <= r["applied"] + 1 <= n:
            fig.add_vline(x=r["applied"] + 1, line_color="#2e7d32", annotation_text=f"Retrain v{r['version']}",
                          annotation_position="top left")
        if 1 <= r["start"] + 1 <= n:
            fig.add_vline(x=r["start"] + 1, line_color="#c62828", line_dash="dot", opacity=0.5)
    if S["alert"] is not None and 1 <= S["alert"]["start"] + 1 <= n:
        fig.add_vline(x=S["alert"]["start"] + 1, line_color="#c62828", annotation_text="Drift starts",
                      annotation_position="top right")
    fig.update_layout(title="Prediction error per incoming batch", xaxis_title="Batch",
                      yaxis_title="MAE (target units)" if reg else "Error rate (share of wrong predictions)",
                      height=360, margin=dict(t=50, b=40, l=50, r=20), legend=dict(orientation="h", y=-0.2))
    wide(st.plotly_chart, fig)


def confusion_fig(y, p):
    labels = list(np.unique(np.concatenate([y, p])))
    cm = confusion_matrix(y, p, labels=labels)
    fig = go.Figure(go.Heatmap(z=cm, x=[f"pred {l}" for l in labels], y=[f"true {l}" for l in labels], text=cm,
                               texttemplate="%{text}", colorscale="Blues", showscale=False))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=280, margin=dict(t=20, b=20, l=20, r=20))
    return fig


def decision_panel(S):
    c, al = S["cand"], S["alert"]
    with st.container(border=True):
        st.subheader("Drift confirmed: review the evidence and decide")
        if c.get("error"):
            st.error(c["error"])
            if st.button("Keep current model", key="skip_err"):
                S["snooze_until"] = S["k"] + S["snooze"]
                log(S, "Kept current model", "not enough data to evaluate the update")
                S["alert"], S["cand"], S["bad"], S["good"] = None, None, [], 0
                S["flash"] = ("info", "Kept the current model.")
                st.rerun()
            return
        m = st.columns(4)
        m[0].metric("Drift started at batch", al["start"] + 1)
        m[1].metric("Historical labelled rows", f"{c['n_old']:,}")
        m[2].metric("New labelled rows", f"{c['n_new']:,}", help=f"{c['n_train']:,} used for training the "
                    f"candidate, {c['hold']:,} held out")
        m[3].metric("Held-out evaluation rows", f"{c['hold']:,}")

        can_update = INC in c["errs"]
        st.markdown(f"**Test result:** {c['verdict']}")
        k = S["k"]
        o1, o2 = st.columns(2)
        with o1.container(border=True):
            st.markdown("#### Keep current model")
            st.metric(f"{c['metric']} (v{S['version']})", f"{c['k_s']:.4f}")
            st.caption(REG[KEEP].blurb)
            if st.button("Keep current model", key=f"skip_{k}"):
                ok, msg = apply_choice(S, KEEP)
                S["flash"] = ("info", msg)
                st.rerun()
        with o2.container(border=True):
            st.markdown("#### Update model")
            if can_update:
                st.metric(f"{c['metric']} after update", f"{c['r_s']:.4f}", delta=f"{c['r_s'] - c['k_s']:+.4f}",
                          delta_color="inverse")
            else:
                st.metric(f"{c['metric']} after update", "n/a")
            st.caption(REG[INC].blurb)
            if st.button("Update model", type="primary", disabled=not can_update, key=f"apply_{k}"):
                ok, msg = apply_choice(S, INC)
                S["flash"] = ("success" if ok else "error", msg)
                st.rerun()
        st.caption(f"Lower is better. Tested on {c['hold']:,} recent rows kept aside.")
        if c["warnings"]:
            with st.expander(f"Notes ({len(c['warnings'])})"):
                for w in c["warnings"]:
                    st.write("• " + w)
        if c["failed"]:
            st.caption("Failed to build: " + "; ".join(f"{n} ({why})" for n, why in c["failed"]))
        with st.expander("Detailed held-out metrics and confusion matrix"):
            wide(st.dataframe, c["table"].round(4), hide_index=True)
            if S["task"] == "classification":
                shown = INC if INC in c["preds"] else KEEP
                st.caption(f"Fault class = '{S['pos']}'. Precision/recall/F1 are for that class.")
                st.write(f"Confusion matrix: {shown}")
                wide(st.plotly_chart, confusion_fig(c["yh"], c["preds"][shown]))


def retrain_results(S):
    st.subheader("How is each retrained model performing now?")
    ver, errs = np.asarray(S["ver"]), np.asarray(S["errs"], float)
    unit = "MAE" if S["task"] == "regression" else "error rate"
    rows = []
    for r in S["retrains"]:
        idx = np.arange(r["start"], min(r["applied"], len(errs)))
        before = errs[idx][ver[idx] == r["version"] - 1] if len(idx) else np.array([])
        after = errs[ver == r["version"]]
        b, a = (before.mean() if len(before) else np.nan), (after.mean() if len(after) else np.nan)
        rows.append({"Version": f"v{r['version']}", "Strategy": r["strategy"], "Applied at batch": r["applied"] + 1,
                     f"Held-out {r['metric']} at decision": r["score_new"],
                     "Held-out score of old model": r["score_keep"],
                     f"Live {unit}: old model since drift": b, f"Live {unit}: this model": a,
                     "Batches measured": int(len(after)),
                     "Live change (%)": (b - a) / b * 100 if np.isfinite(b) and b > 1e-12 and np.isfinite(a) else np.nan})
    wide(st.dataframe, pd.DataFrame(rows).round(4), hide_index=True)
    st.caption("Held-out columns use the decision's selection metric; live columns use the monitoring metric "
               f"({unit}). Early live numbers cover few batches, so read them with care.")


# ------------------------------------------------------------------ app
CSS = """<style>
.block-container{padding-top:1.4rem;padding-bottom:1.5rem}
[data-testid="stMetric"]{background:rgba(128,128,128,.08);border-radius:8px;padding:.45rem .8rem}
</style>"""


def sidebar_config(S):
    with st.sidebar:
        st.header("Configuration")
        up = st.file_uploader("Upload a CSV (rows in time order, or pick a time column)", type=["csv"])
        if up is None:
            return
        try:
            df = read(up.getvalue())
        except Exception as exc:
            st.error(str(exc))
            return
        target = st.selectbox("Target column", df.columns, index=len(df.columns) - 1)
        tcol = st.selectbox("Time column (optional)", ["(row order)"] + [c for c in df.columns if c != target])
        st.caption("Model: Hoeffding Adaptive Tree (river), learns one row at a time and adapts to drift internally.")
        guess = "regression" if (pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique() > 10) else "classification"
        task = st.radio("Task", ["classification", "regression"], index=["classification", "regression"].index(guess),
                        horizontal=True)
        train_pct = st.slider("Initial training %", 20, 70, 40, 5)
        B = int(st.number_input("Batch size (rows)", 20, 5000, 100))
        sens = st.selectbox("Drift sensitivity", list(SENS), index=1,
                            help="Threshold = reference error + z * reference std / sqrt(batch size). "
                                 "z is 4 (Low), 3 (Medium) or 2 (High). Two batches in a row above it raise an alert.")
        delay = int(st.number_input("Label delay (batches)", 0, 20, 0))
        min_rows = int(st.number_input("Min new labelled rows before asking", 100, 100000, 300,
                                       help="After a drift alert, monitoring continues until this many post-drift "
                                            "labelled rows exist."))
        snooze = int(st.number_input("Snooze after 'keep' (batches)", 0, 200, 10))
        auto = st.checkbox("Auto-update when clearly better", value=True,
                           help="If the test shows the updated model is clearly better, it is deployed "
                                "automatically. Otherwise you are asked.")
        with st.expander("Advanced settings"):
            hold_frac = st.slider("Share of new labelled rows held out for evaluation", 0.10, 0.40, 0.25, 0.05)
            metric = st.selectbox("Evaluation metric (lower is better)", METRICS[task])
            pos = None
            if task == "classification":
                tgt = df[target].dropna()
                labs = sorted(set(label_str(tgt)))[:50]
                vc = pd.Series(label_str(tgt)).value_counts()
                dflt = "1" if "1" in labs else (str(vc.index[-1]) if str(vc.index[-1]) in labs else labs[0])
                pos = st.selectbox("Fault (positive) class", labs, index=labs.index(dflt))
            st.markdown("**Hoeffding Adaptive Tree settings**")
            grace = int(st.number_input("Grace period (rows between split attempts)", 10, 5000, 200, step=10,
                                        help="Lower = the tree splits sooner and adapts faster, but is noisier."))
            delta = st.select_slider("Split confidence (delta, smaller = more cautious)", DELTAS, value=1e-7,
                                     format_func=lambda v: f"{v:.0e}")
            tau = st.number_input("Tie threshold (tau)", 0.001, 0.5, 0.05, step=0.01, format="%.3f")
            adwin = st.select_slider("ADWIN sensitivity (larger = reacts to drift sooner)", ADWIN_DELTAS, value=0.002,
                                     help="Confidence of the drift detector inside each tree node.")
            win_thr = int(st.number_input("Alternate-tree window (rows)", 50, 5000, 300, step=50,
                                          help="Rows an alternate subtree must see before it can replace "
                                               "the original branch."))
            seed = int(st.number_input("Random seed", 0, 10_000, 42))
        running = S is not None and S["k"] > 0
        go_ = False
        if st.button("Restart with these settings" if S is not None else "Start monitoring", type="primary"):
            if running:
                st.session_state["confirm_restart"] = True
            else:
                go_ = True
        if confirm_box("confirm_restart", "This discards the current session (model versions, events, history).",
                       "Yes, restart"):
            go_ = True
        if go_:
            cfg = dict(train_pct=train_pct, B=B, z=SENS[sens], delay=delay, min_rows=min_rows, snooze=snooze,
                       hold_frac=hold_frac, metric=metric, pos=pos, seed=seed, auto=auto,
                       grace=grace, delta=delta, tau=float(tau), adwin=adwin, win_thr=win_thr)
            try:
                with st.spinner("Training the first Hoeffding Adaptive Tree..."):
                    st.session_state["S"] = init_session(df, target, tcol, task, cfg)
            except ValueError as exc:
                st.error(str(exc))
                return
            st.rerun()


def main():
    st.set_page_config(page_title="Model Drift Dashboard", page_icon="📈", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("📈 Model Drift Dashboard")
    if not RIVER_OK:
        st.error("This app needs the `river` package. Install it with `pip install river` and restart.")
        st.stop()
    S = st.session_state.get("S")
    sidebar_config(S)
    S = st.session_state.get("S")
    if S is None:
        st.info("Upload a CSV in the sidebar, choose your columns and click **Start monitoring**.")
        st.stop()

    done = S["k"] >= S["nb"]
    off = ready(S) or done
    with st.container(border=True):
        b0, b1, b2, b3 = st.columns([2.4, 1, 1, 1])
        if b0.button("▶ Run until a decision is needed", type="primary", disabled=off, key="run_all"):
            run(S, None)
        if b1.button("Next batch", disabled=off, key="run_1"):
            run(S, 1)
        if b2.button("Next 10 batches", disabled=off, key="run_10"):
            run(S, 10)
        if b3.button("Reset", key="reset"):
            if S["k"] > 0:
                st.session_state["confirm_reset"] = True
            else:
                st.session_state.pop("S")
                st.rerun()
    if confirm_box("confirm_reset", "Reset the session? Model versions, events and history will be discarded.",
                   "Yes, reset"):
        st.session_state.pop("S")
        st.rerun()

    al = S["alert"]
    status = ("Finished" if done and not ready(S) else "Decision needed" if ready(S)
              else "Drift suspected, collecting data" if al else "Monitoring")
    m = st.columns(4)
    m[0].metric("Batches received", f"{S['k']} / {S['nb']}")
    m[1].metric("Deployed model", f"Adaptive tree v{S['version']}")
    m[2].metric("Retraining events", len(S["retrains"]))
    m[3].metric("Status", status)

    if S.get("flash"):
        lvl, txt = S["flash"]
        S["flash"] = None
        getattr(st, lvl)(txt)

    if S["k"]:
        drift_chart(S)

    if al and al["status"] == "waiting":
        st.info(f"Drift suspected since batch {al['start'] + 1}, but only {new_rows(S):,} of {S['min_rows']:,} new "
                "labelled rows have arrived. Not enough to evaluate an update safely, so monitoring continues. You "
                "will be asked once there is enough data, or the alert clears itself if the error returns to normal.")

    if ready(S):
        if S["cand"] is None:
            with st.spinner("Testing incremental learning on the newest labelled rows..."):
                try:
                    S["cand"] = compute_cands(S)
                except Exception as exc:
                    S["cand"] = dict(error=f"Could not evaluate the update: {exc}")
        c = S["cand"]
        if S.get("auto") and c.get("clear_win"):
            ok, msg = apply_choice(S, INC)
            S["flash"] = ("success" if ok else "error", "Auto-updated: " + msg if ok else msg)
            st.rerun()
        decision_panel(S)

    if S["retrains"]:
        retrain_results(S)
    if S["events"]:
        with st.expander("Event log", expanded=False):
            wide(st.dataframe, pd.DataFrame(S["events"]), hide_index=True)


if __name__ == "__main__":
    main()
