"""Model Drift Dashboard: fully automatic (Hoeffding Adaptive Tree, `river`).

One deployed model streams through the data. When drift is confirmed the app adapts BY ITSELF:
    1. A copy of the model keeps learning from the new rows (never touches the live model).
    2. The copy and the live model are tested on the newest rows kept aside.
    3. The update is deployed unless it is worse. If it is worse, the live model is kept.
    4. After deploying, the old model is held in reserve. If the new one does worse than the old one over the next
       few batches, the app rolls back automatically.
Drift threshold: ref_error + z * ref_std / sqrt(batch size); two flagged batches in a row raise an alert.
"""
import copy
import io
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import f1_score, precision_recall_fscore_support

try:
    from river import drift as river_drift
    from river import tree as river_tree
    RIVER_OK = True
except ImportError:
    river_tree, river_drift, RIVER_OK = None, None, False

warnings.filterwarnings("ignore", category=UserWarning, module="river")

# ------------------------------------------------------------------ constants
SENS = {"Low (fewer alerts)": 4.0, "Medium": 3.0, "High (more alerts)": 2.0}
MIN_FIT = 30        # minimum labelled rows to train / test
WATCH = 3           # batches an updated model is watched before the old one is released
DELTAS = [1e-3, 1e-5, 1e-7, 1e-9]
ADWIN_DELTAS = [0.1, 0.01, 0.002, 0.0005]
METRICS = {
    "classification": ["Error rate", "1 - macro F1", "Fault-class miss rate (1 - recall)"],
    "regression": ["MAE", "RMSE"],
}
ROW_WISE = {"Error rate", "MAE"}   # metrics with a per-row loss, so a noise test is possible
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
    """Hoeffding Adaptive Tree + minimal preprocessing (median imputation; categoricals as nominal)."""

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
               recall=np.nan, n_pos=int((y == pos).sum()) if pos is not None else 0)
    if pos is not None and pos in labels and out["n_pos"] > 0:
        _, rc, _, _ = precision_recall_fscore_support(y, p, labels=[pos], average=None, zero_division=0)
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


# ------------------------------------------------------------------ session setup
def init_session(df, target, tcol, task, cfg):
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
    num, cat = pick_features(d.iloc[:n_tr], {target, tcol})
    if not (num or cat):
        raise ValueError("No usable feature columns were found.")
    X = build_frame(d, num, cat)
    pos = cfg.get("pos")
    if task == "classification" and pos is None:
        vals, cnt = np.unique(y, return_counts=True)
        pos = str(vals[np.argmin(cnt)])
    hp = dict(grace=int(cfg["grace"]), delta=float(cfg["delta"]), tau=float(cfg["tau"]),
              adwin=float(cfg["adwin"]), win_thr=int(cfg["win_thr"]))
    S = dict(X=X, y=y, task=task, hp=hp, seed=int(cfg["seed"]), n_tr=n_tr, B=B,
             N=len(y) - n_tr, nb=int(np.ceil((len(y) - n_tr) / B)), z=cfg["z"], delay=int(cfg["delay"]),
             min_rows=int(cfg["min_rows"]), snooze=int(cfg["snooze"]), hold_frac=float(cfg["hold_frac"]),
             metric=cfg.get("metric") or METRICS[task][0], pos=pos,
             version=0, k=0, errs=[], thr=[], ver=[], bad=[], good=0, alert=None, snooze_until=0,
             watch=None, events=[], updates=[], flash=None)
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
    S["flash"] = (event, detail)


def new_rows(S):
    al = S["alert"]
    if al is None:
        return 0
    return max(0, min((S["k"] - S["delay"]) * S["B"], S["N"]) - al["start"] * S["B"])


def check_rollback(S):
    """After an update, compare the new model with the old one over a few batches; roll back if it is worse."""
    w = S["watch"]
    if not w or S["k"] - w["from_k"] < WATCH:
        return
    S["watch"] = None
    live = float(np.mean(S["errs"][w["from_k"]:S["k"]]))
    if np.isfinite(w["baseline"]) and live > w["baseline"]:
        S["model"] = w["prev_model"]
        S["version"] += 1
        S["ref_m"], S["ref_s"] = w["prev_ref"]
        S["snooze_until"] = S["k"] + S["snooze"]
        S["bad"] = []
        w["record"]["outcome"] = "Rolled back"
        log(S, "Rolled back", f"v{w['record']['version']} was worse than the old model "
                              f"({live:.3f} vs {w['baseline']:.3f}); old model restored as v{S['version']}")
    else:
        w["record"]["outcome"] = "Kept"


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
    check_rollback(S)
    j = k - S["delay"]
    if j >= 0 and S["ver"][j] == S["version"]:
        flagged = S["errs"][j] > S["thr"][j]
        if S["alert"] is None:
            S["bad"] = (S["bad"] + [j] if flagged else []) if not S["watch"] else []
            if len(S["bad"]) >= 2 and S["k"] > S["snooze_until"]:
                S["alert"], S["good"] = dict(start=S["bad"][0], status="waiting"), 0
                S["bad"] = []
                log(S, "Drift detected", f"errors above threshold since batch {S['alert']['start'] + 1}")
        elif S["alert"]["status"] == "waiting":
            S["good"] = 0 if flagged else S["good"] + 1
            if S["good"] >= 3:
                S["alert"] = None
                log(S, "Alert cleared", "error returned to normal on its own")
    al = S["alert"]
    if al and al["status"] == "waiting" and new_rows(S) >= S["min_rows"]:
        al["status"] = "ready"


def ready(S):
    return S["alert"] is not None and S["alert"]["status"] == "ready"


def get_new(S):
    n_tr, B = S["n_tr"], S["B"]
    s0 = n_tr + S["alert"]["start"] * B
    s1 = max(s0, n_tr + min((S["k"] - S["delay"]) * B, S["N"]))
    return S["X"].iloc[s0:s1], S["y"][s0:s1]


def adapt(S):
    """Automatic response to confirmed drift. Deploys the updated model unless it tests worse."""
    al, task = S["alert"], S["task"]
    Xn, yn = get_new(S)
    n = len(yn)

    def skip(why):
        S["snooze_until"] = S["k"] + S["snooze"]
        S["alert"], S["bad"], S["good"] = None, [], 0
        log(S, "Update skipped", why)

    if n < 2 * MIN_FIT:
        return skip(f"only {n} new labelled rows, not enough to test an update")
    hold = int(np.clip(round(n * S["hold_frac"]), MIN_FIT, n - MIN_FIT))
    Xh, yh, Xt, yt = Xn.iloc[-hold:], yn[-hold:], Xn.iloc[:-hold], yn[:-hold]
    try:
        cand = copy.deepcopy(S["model"]).learn(Xt, yt)            # test copy: learns from the older new rows
        p_old, p_new = S["model"].predict(Xh), cand.predict(Xh)
        if task == "regression" and not (np.all(np.isfinite(p_old)) and np.all(np.isfinite(p_new))):
            raise ValueError("non-finite predictions")
    except Exception as exc:
        return skip(f"could not test the update ({exc})")
    metric = S["metric"]
    m_old, m_new = evaluate(task, yh, p_old, S["pos"]), evaluate(task, yh, p_new, S["pos"])
    if metric.startswith("Fault-class") and not np.isfinite(m_old["recall"]):
        metric = "Error rate"
    k_s, r_s = selection_score(metric, m_old), selection_score(metric, m_new)
    if not (np.isfinite(k_s) and np.isfinite(r_s)):
        return skip("test score could not be computed")
    e_old, e_new = row_err(task, yh, p_old), row_err(task, yh, p_new)
    if metric in ROW_WISE:                                    # worse only if clearly worse (> 2 std. errors)
        d = e_old - e_new
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.inf
        worse = np.isfinite(se) and se > 0 and d.mean() / se < -2
    else:
        worse = r_s > k_s
    if worse:
        return skip(f"updated model tested worse ({metric} {r_s:.3f} vs {k_s:.3f}); current model kept")
    try:
        final = copy.deepcopy(S["model"]).learn(Xn, yn)       # deployed model learns from ALL new rows
        final.predict(Xn.iloc[:1])
    except Exception as exc:
        return skip(f"update failed ({exc})")
    ver = np.asarray(S["ver"][al["start"]:S["k"]])
    old_live = np.asarray(S["errs"][al["start"]:S["k"]])[ver == S["version"]]
    baseline = float(old_live.mean()) if len(old_live) else np.nan
    rec = dict(version=S["version"] + 1, applied=S["k"], start=al["start"], metric=metric,
               test_old=k_s, test_new=r_s, live_before=baseline, outcome="Watching")
    S["watch"] = dict(prev_model=S["model"], prev_ref=(S["ref_m"], S["ref_s"]), from_k=S["k"],
                      baseline=baseline, record=rec)
    S["model"], S["version"] = final, S["version"] + 1
    S["ref_m"], S["ref_s"] = ref_stats(task, e_new)
    S["updates"].append(rec)
    S["alert"], S["bad"], S["good"] = None, [], 0
    log(S, "Model updated", f"v{S['version']} deployed ({metric} {r_s:.3f} vs {k_s:.3f} on {hold:,} recent rows)")


def run(S, n):
    i = 0
    while S["k"] < S["nb"] and (n is None or i < n):
        step(S)
        i += 1
        if ready(S):
            adapt(S)


# ------------------------------------------------------------------ UI
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
    fig = go.Figure()
    for i, v in enumerate(sorted(np.unique(ver))):
        idx = np.flatnonzero(ver == v)
        fig.add_trace(go.Scatter(x=xs[idx], y=errs[idx], mode="lines+markers", name=f"Model v{v}",
                                 line=dict(color=PALETTE[i % len(PALETTE)])))
    fig.add_trace(go.Scatter(x=xs, y=thr, mode="lines", name="Drift threshold", line=dict(dash="dash", color="#c62828")))
    for r in S["updates"]:
        if 1 <= r["applied"] + 1 <= n:
            fig.add_vline(x=r["applied"] + 1, line_color="#2e7d32", annotation_text=f"Update v{r['version']}",
                          annotation_position="top left")
    if S["alert"] is not None and 1 <= S["alert"]["start"] + 1 <= n:
        fig.add_vline(x=S["alert"]["start"] + 1, line_color="#c62828", annotation_text="Drift starts",
                      annotation_position="top right")
    fig.update_layout(title="Prediction error per incoming batch", xaxis_title="Batch",
                      yaxis_title="MAE (target units)" if S["task"] == "regression" else "Error rate",
                      height=360, margin=dict(t=50, b=40, l=50, r=20), legend=dict(orientation="h", y=-0.2))
    wide(st.plotly_chart, fig)


def updates_table(S):
    st.subheader("Automatic updates")
    ver, errs = np.asarray(S["ver"]), np.asarray(S["errs"], float)
    unit = "MAE" if S["task"] == "regression" else "Error rate"
    rows = []
    for r in S["updates"]:
        after = errs[ver == r["version"]]
        rows.append({"Version": f"v{r['version']}", "At batch": r["applied"] + 1,
                     f"Test {r['metric']} (old to new)": f"{r['test_old']:.3f} to {r['test_new']:.3f}",
                     f"Live {unit} before": r["live_before"],
                     f"Live {unit} after": float(after.mean()) if len(after) else np.nan,
                     "Result": r["outcome"]})
    wide(st.dataframe, pd.DataFrame(rows).round(4), hide_index=True)


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
        st.caption("Model: Hoeffding Adaptive Tree. Drift is handled automatically.")
        guess = "regression" if (pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique() > 10) else "classification"
        task = st.radio("Task", ["classification", "regression"], index=["classification", "regression"].index(guess),
                        horizontal=True)
        train_pct = st.slider("Initial training %", 20, 70, 40, 5)
        B = int(st.number_input("Batch size (rows)", 20, 5000, 100))
        sens = st.selectbox("Drift sensitivity", list(SENS), index=1,
                            help="Threshold = reference error + z * reference std / sqrt(batch size). "
                                 "z is 4 (Low), 3 (Medium) or 2 (High). Two batches in a row above it count as drift.")
        delay = int(st.number_input("Label delay (batches)", 0, 20, 0))
        min_rows = int(st.number_input("New labelled rows needed before updating", 100, 100000, 300))
        snooze = int(st.number_input("Pause after a skipped or rolled-back update (batches)", 0, 200, 10))
        with st.expander("Advanced settings"):
            hold_frac = st.slider("Share of new rows kept aside for testing", 0.10, 0.40, 0.25, 0.05)
            metric = st.selectbox("Test metric (lower is better)", METRICS[task])
            pos = None
            if task == "classification":
                tgt = df[target].dropna()
                labs = sorted(set(label_str(tgt)))[:50]
                vc = pd.Series(label_str(tgt)).value_counts()
                dflt = "1" if "1" in labs else (str(vc.index[-1]) if str(vc.index[-1]) in labs else labs[0])
                pos = st.selectbox("Fault (positive) class", labs, index=labs.index(dflt))
            st.markdown("**Tree settings**")
            grace = int(st.number_input("Grace period (rows between split attempts)", 10, 5000, 200, step=10))
            delta = st.select_slider("Split confidence (smaller = more cautious)", DELTAS, value=1e-7,
                                     format_func=lambda v: f"{v:.0e}")
            tau = st.number_input("Tie threshold (tau)", 0.001, 0.5, 0.05, step=0.01, format="%.3f")
            adwin = st.select_slider("ADWIN sensitivity (larger = reacts sooner)", ADWIN_DELTAS, value=0.002)
            win_thr = int(st.number_input("Alternate-tree window (rows)", 50, 5000, 300, step=50))
            seed = int(st.number_input("Random seed", 0, 10_000, 42))
        running = S is not None and S["k"] > 0
        go_ = False
        if st.button("Restart with these settings" if S is not None else "Start monitoring", type="primary"):
            if running:
                st.session_state["confirm_restart"] = True
            else:
                go_ = True
        if confirm_box("confirm_restart", "This discards the current session.", "Yes, restart"):
            go_ = True
        if go_:
            cfg = dict(train_pct=train_pct, B=B, z=SENS[sens], delay=delay, min_rows=min_rows, snooze=snooze,
                       hold_frac=hold_frac, metric=metric, pos=pos, seed=seed,
                       grace=grace, delta=delta, tau=float(tau), adwin=adwin, win_thr=win_thr)
            try:
                with st.spinner("Training the first model..."):
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
        st.info("Upload a CSV in the sidebar and click **Start monitoring**.")
        st.stop()

    done = S["k"] >= S["nb"]
    with st.container(border=True):
        b0, b1, b2, b3 = st.columns([2.4, 1, 1, 1])
        if b0.button("▶ Run all", type="primary", disabled=done, key="run_all"):
            run(S, None)
        if b1.button("Next batch", disabled=done, key="run_1"):
            run(S, 1)
        if b2.button("Next 10 batches", disabled=done, key="run_10"):
            run(S, 10)
        if b3.button("Reset", key="reset"):
            if S["k"] > 0:
                st.session_state["confirm_reset"] = True
            else:
                st.session_state.pop("S")
                st.rerun()
    if confirm_box("confirm_reset", "Reset the session? History will be discarded.", "Yes, reset"):
        st.session_state.pop("S")
        st.rerun()

    al = S["alert"]
    status = ("Finished" if done else "Collecting data after drift" if al
              else "Checking updated model" if S["watch"] else "Monitoring")
    m = st.columns(4)
    m[0].metric("Batches received", f"{S['k']} / {S['nb']}")
    m[1].metric("Deployed model", f"Adaptive tree v{S['version']}")
    m[2].metric("Automatic updates", len(S["updates"]))
    m[3].metric("Status", status)

    if S.get("flash"):
        title, detail = S["flash"]
        S["flash"] = None
        (st.success if title == "Model updated" else st.warning if title in ("Rolled back", "Update skipped")
         else st.info)(f"**{title}:** {detail}")

    if S["k"]:
        drift_chart(S)
    if S["updates"]:
        updates_table(S)
    if S["events"]:
        with st.expander("Activity log", expanded=False):
            wide(st.dataframe, pd.DataFrame(S["events"]), hide_index=True)


if __name__ == "__main__":
    main()
    
