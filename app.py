"""Model Drift Dashboard (Hoeffding Adaptive Tree edition).

Prequential (test-then-train) online learning on a stream of labelled rows:

    for every observation:  1) predict_one(x)   2) update metrics with that prediction   3) learn_one(x, y)

Three things are kept strictly apart:
    * Incremental learning : the model learns from each sample (learn_one).
    * Drift detection      : INSIDE HoeffdingAdaptiveTreeClassifier, River runs an ADWIN detector per node. River
                             exposes no per-event signal, so the dashboard shows only River's public counters.
                             An OPTIONAL external ADWIN on the error stream (off by default) gives observable events.
    * Adaptation           : the tree grows alternate subtrees and swaps them in when they do better. This is not a
                             full model reset.
A known injected drift point is a property of the synthetic stream, not a detection by any model.

Scope: classification only. The earlier batch/regression workflow (strategy picker, regression trees) is not part of
this version because the adaptive tree adapts inside the model.
Run:  streamlit run app.py
"""
import io
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from river import drift as river_drift
    from river import tree as river_tree
    RIVER_OK = True
except ImportError:                                   # reported in main()
    river_drift, river_tree, RIVER_OK = None, None, False

# ------------------------------------------------------------------ constants
HAT, HT = "Hoeffding Adaptive Tree", "Hoeffding Tree (standard)"
STAT, ABRUPT, GRAD, RECUR, CSV = ("Synthetic: stationary", "Synthetic: abrupt drift", "Synthetic: gradual drift",
                                  "Synthetic: recurring concepts", "Upload CSV")
N_FEAT = 5            # x0..x4; x4 is irrelevant noise
MAX_POINTS = 1500     # max points per plotted curve
COLORS = {HAT: "#1f6fb2", HT: "#e08a00"}
NONE_LABEL = "(none)"  # prediction of a model that has not seen any label yet; counted as wrong


# ------------------------------------------------------------------ stream generation
def build_synth(kind, n, seed, noise, pos_pct, width):
    """Two concepts over different features, so the old concept is useless after drift:
        concept A: label = x0 + x1 > 1        concept B: label = x2 + x3 > 1
    Abrupt: A -> B at a fixed index. Gradual: P(B) ramps 0 -> 1 over `width` samples. Recurring: A -> B -> A.
    Label noise flips each label with probability `noise`."""
    rng = np.random.default_rng(seed)
    X = rng.random((n, N_FEAT))
    a, b = X[:, 0] + X[:, 1] > 1.0, X[:, 2] + X[:, 3] > 1.0
    t, p = np.arange(n), int(n * pos_pct / 100)
    if kind == STAT:
        use_b, points, w = np.zeros(n, bool), [], 0
    elif kind == ABRUPT:
        use_b, points, w = t >= p, [p], 0
    elif kind == GRAD:
        use_b, points, w = rng.random(n) < np.clip((t - p) / max(width, 1), 0, 1), [p], int(width)
    else:
        p1, p2 = n // 3, 2 * n // 3
        use_b, points, w = (t >= p1) & (t < p2), [p1, p2], 0
    y = np.where(use_b, b, a) ^ (rng.random(n) < noise)
    names = [f"x{i}" for i in range(N_FEAT)]
    return dict(xs=[dict(zip(names, r)) for r in X.tolist()], ys=np.where(y, "1", "0").tolist(),
                points=points, width=w, kind=kind, nominal=None, classes=["0", "1"])


def label_str(s):
    """Stable string labels: integral numbers become '0','1' (not '0.0','1.0')."""
    s = pd.Series(s)
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        v = s.to_numpy(float)
        if len(v) and np.all(np.isfinite(v)) and np.all(np.mod(v, 1) == 0):
            return v.astype(np.int64).astype(str)
    return s.astype(str).to_numpy()


@st.cache_data(show_spinner="Reading CSV...")
def read_csv(data):
    for enc in ("utf-8-sig", "latin-1"):
        try:
            out = pd.read_csv(io.BytesIO(data), encoding=enc)
            if out.shape[1] <= 1:
                out = pd.read_csv(io.BytesIO(data), encoding=enc, sep=None, engine="python")
            return out
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not read the file as CSV.")


def build_csv(df, target, n):
    """Rows are streamed in file order. Numeric NaN -> feature omitted (River handles absent features);
    categorical NaN -> '(missing)'. Rows without a label are dropped."""
    d = df[df[target].notna()].head(n)
    if len(d) < 100:
        raise ValueError(f"Only {len(d)} labelled rows; need at least 100.")
    ys = label_str(d[target]).tolist()
    if len(set(ys)) < 2:
        raise ValueError("The target column has only one class.")
    num, cat = [], []
    for c in d.columns:
        if c == target or d[c].nunique() <= 1:
            continue
        if pd.api.types.is_numeric_dtype(d[c]) and not pd.api.types.is_bool_dtype(d[c]):
            num.append(c)
        elif d[c].nunique() <= 50:
            cat.append(c)
    if not (num or cat):
        raise ValueError("No usable feature columns were found.")
    xs = []
    for rec in d[num + cat].to_dict("records"):
        row = {c: float(rec[c]) for c in num if pd.notna(rec[c])}
        row.update({c: (str(rec[c]) if pd.notna(rec[c]) else "(missing)") for c in cat})
        xs.append(row)
    return dict(xs=xs, ys=ys, points=[], width=0, kind=CSV, nominal=cat or None, classes=sorted(set(ys)))


# ------------------------------------------------------------------ models and prequential tracker
def make_model(name, hp, nominal):
    if name == HAT:
        # drift_detector=None -> River's default per-node ADWIN. Only replaced if the user customises it.
        det = river_drift.ADWIN(delta=hp["adwin"]) if hp["adwin"] else None
        return river_tree.HoeffdingAdaptiveTreeClassifier(
            grace_period=hp["grace"], delta=hp["delta"], drift_window_threshold=hp["win_thr"],
            drift_detector=det, nominal_attributes=nominal, seed=hp["seed"])
    return river_tree.HoeffdingTreeClassifier(grace_period=hp["grace"], delta=hp["delta"], nominal_attributes=nominal)


def tree_stats(model):
    """Public River properties only. Alternate-tree counters are cumulative counts kept by River and exist on the
    adaptive tree only. They do not say WHICH sample triggered an event."""
    out = {"Nodes": getattr(model, "n_nodes", None), "Height": getattr(model, "height", None)}
    for label, attr in (("Alternate trees started", "n_alternate_trees"),
                        ("Alternate trees pruned", "n_pruned_alternate_trees"),
                        ("Alternate trees swapped in", "n_switch_alternate_trees")):
        out[label] = getattr(model, attr, None)
    return out


class Tracker:
    """One model plus its own prequential metrics. Models never share state, only the (read-only) stream rows."""

    def __init__(self, name, model, ext_delta):
        self.name, self.model = name, model
        self.ext = river_drift.ADWIN(delta=ext_delta) if ext_delta else None   # optional external detector
        self.correct = bytearray()                       # 1 if the pre-learning prediction was right
        self.pred = []                                   # pre-learning predictions (for the label plot)
        self.tp = self.fp = self.fn = 0
        self.t_pred = self.t_learn = 0.0
        self.events = []                                 # external ADWIN detections (sample indices)
        self.snaps = []                                  # (sample, started, pruned, swapped) once per chunk

    def step(self, i, x, y, pos):
        t0 = time.perf_counter()
        p = self.model.predict_one(x)                    # 1) PREDICT (model has not seen this label)
        t1 = time.perf_counter()
        p = NONE_LABEL if p is None else str(p)
        ok = p == y                                      # 2) EVALUATE using that prediction
        self.correct.append(1 if ok else 0)
        self.pred.append(p)
        self.tp += (p == pos) and (y == pos)
        self.fp += (p == pos) and (y != pos)
        self.fn += (p != pos) and (y == pos)
        if self.ext is not None:
            self.ext.update(0.0 if ok else 1.0)          # external detector watches the error stream
            if self.ext.drift_detected:
                self.events.append(i)
        t2 = time.perf_counter()
        self.model.learn_one(x, y)                       # 3) LEARN (only now is the label used)
        self.t_pred += t1 - t0
        self.t_learn += time.perf_counter() - t2

    def snapshot(self, i):
        s = tree_stats(self.model)
        if s["Alternate trees started"] is not None:
            self.snaps.append((i, s["Alternate trees started"], s["Alternate trees pruned"],
                               s["Alternate trees swapped in"]))

    def metrics(self, window):
        n = len(self.correct)
        c = sum(self.correct)
        prec = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else None
        rec = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else None
        f1 = 2 * prec * rec / (prec + rec) if prec is not None and rec is not None and (prec + rec) else None
        return dict(n=n, acc=c / n if n else None, err=1 - c / n if n else None,
                    recent=sum(self.correct[-window:]) / min(n, window) if n else None,
                    precision=prec, recall=rec, f1=f1, runtime=self.t_pred + self.t_learn)


def process_chunk(exp, n_steps):
    """Advance the stream by up to n_steps samples, one sample at a time (genuine prequential loop)."""
    s, pos = exp["stream"], exp["pos"]
    i0 = exp["i"]
    i1 = min(i0 + n_steps, len(s["ys"]))
    for i in range(i0, i1):
        for tr in exp["trackers"]:
            tr.step(i, s["xs"][i], s["ys"][i], pos)
    exp["i"] = i1
    for tr in exp["trackers"]:
        tr.snapshot(i1)
    if i1 >= len(s["ys"]):
        exp["running"] = False


def new_experiment(stream, cfg):
    names = [HAT] + ([HT] if cfg["compare"] else []) if cfg["primary"] == HAT else \
            [HT] + ([HAT] if cfg["compare"] else [])
    trackers = [Tracker(nm, make_model(nm, cfg["hp"], stream["nominal"]), cfg["ext_delta"]) for nm in names]
    pos = cfg["pos"] if cfg["pos"] in stream["classes"] else stream["classes"][-1]
    return dict(stream=stream, trackers=trackers, i=0, running=True, window=cfg["window"], pos=pos,
                chunk=cfg["chunk"], delay=cfg["delay"])


# ------------------------------------------------------------------ drift matching
def match_delays(points, events, n):
    """Delay of the first external-detector event inside [drift point, next drift point). Only external ADWIN events
    are matched; River's internal detector exposes no per-event signal."""
    bounds = list(points) + [n]
    out = []
    for k, p in enumerate(points):
        hit = [e for e in events if p <= e < bounds[k + 1]]
        out.append((p, (hit[0] - p) if hit else None))
    return out


# ------------------------------------------------------------------ plotting
_NEW_API = tuple(int(x) for x in st.__version__.split(".")[:2] if x.isdigit()) >= (1, 50)


def wide(fn, *a, **kw):
    kw["width" if _NEW_API else "use_container_width"] = "stretch" if _NEW_API else True
    return fn(*a, **kw)


def curves(correct, window):
    c = np.asarray(correct, float)
    cs = np.concatenate([[0.0], np.cumsum(c)])
    idx = np.arange(1, len(c) + 1)
    cum = cs[1:] / idx
    win = (cs[1:] - cs[np.maximum(0, idx - window)]) / np.minimum(idx, window)
    return cum, win


def thin(n):
    return np.unique(np.linspace(0, n - 1, min(n, MAX_POINTS)).astype(int)) if n else np.array([], int)


def add_drift_marks(fig, s):
    if s["width"]:
        for p in s["points"]:
            fig.add_vrect(x0=p, x1=p + s["width"], fillcolor="#c62828", opacity=0.10, line_width=0)
    for k, p in enumerate(s["points"]):
        fig.add_vline(x=p, line_color="#c62828", line_dash="dot",
                      annotation_text="Injected drift" if k == 0 else None, annotation_position="top left")


def line_chart(exp, title, which, ytitle):
    fig = go.Figure()
    for tr in exp["trackers"]:
        if not tr.correct:
            continue
        cum, win = curves(tr.correct, exp["window"])
        y = cum if which == "cum" else win
        x = thin(len(y))
        if which == "win":
            x = x[x >= exp["window"] - 1]                # hide partial windows
        fig.add_trace(go.Scatter(x=x, y=y[x], mode="lines", name=tr.name, line=dict(color=COLORS[tr.name])))
        if tr.events:
            ev = np.asarray([e for e in tr.events if e < len(y) and (which == "cum" or e >= exp["window"] - 1)], int)
            fig.add_trace(go.Scatter(x=ev, y=y[ev], mode="markers", name=f"External ADWIN event ({tr.name})",
                                     marker=dict(symbol="diamond", size=9, color=COLORS[tr.name],
                                                 line=dict(color="black", width=1))))
    add_drift_marks(fig, exp["stream"])
    fig.update_layout(title=title, xaxis_title="Samples processed", yaxis_title=ytitle, yaxis_range=[0, 1.02],
                      height=320, margin=dict(t=45, b=40, l=50, r=20), legend=dict(orientation="h", y=-0.28))
    wide(st.plotly_chart, fig)


def error_blocks_chart(exp):
    fig = go.Figure()
    w = exp["window"]
    for tr in exp["trackers"]:
        c = np.asarray(tr.correct, float)
        nb = len(c) // w
        if nb:
            err = 1 - c[:nb * w].reshape(nb, w).mean(axis=1)
            fig.add_trace(go.Bar(x=(np.arange(nb) + 0.5) * w, y=err, name=tr.name, marker_color=COLORS[tr.name]))
    add_drift_marks(fig, exp["stream"])
    fig.update_layout(title=f"Prediction error per block of {w} samples", barmode="group", height=320,
                      xaxis_title="Samples processed", yaxis_title="Error rate", margin=dict(t=45, b=40, l=50, r=20),
                      legend=dict(orientation="h", y=-0.28))
    wide(st.plotly_chart, fig)


def true_vs_pred_chart(exp):
    tr, s = exp["trackers"][0], exp["stream"]
    n = len(tr.pred)
    if not n:
        return
    a = max(0, n - 200)
    xs = np.arange(a, n)
    yt, yp = np.asarray(s["ys"][a:n]), np.asarray(tr.pred[a:n])
    wrong = yt != yp
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=yt, mode="markers", name="True label",
                             marker=dict(symbol="circle-open", size=9, color="#1f6fb2")))
    fig.add_trace(go.Scatter(x=xs, y=yp, mode="markers", name="Predicted (before learning)",
                             marker=dict(symbol="x", size=7, color=np.where(wrong, "#c62828", "#2e7d32"))))
    fig.update_yaxes(type="category", categoryorder="array", categoryarray=sorted(set(s["classes"]) | {NONE_LABEL}))
    fig.update_layout(title=f"True vs predicted, last {n - a} samples ({tr.name}; red x = wrong)", height=320,
                      xaxis_title="Sample index", margin=dict(t=45, b=40, l=50, r=20),
                      legend=dict(orientation="h", y=-0.28))
    wide(st.plotly_chart, fig)


def alt_tree_chart(exp):
    fig = go.Figure()
    for tr in exp["trackers"]:
        if tr.snaps:
            a = np.asarray(tr.snaps)
            for col, nm, dash in ((1, "started", "solid"), (2, "pruned", "dot"), (3, "swapped in", "dash")):
                fig.add_trace(go.Scatter(x=a[:, 0], y=a[:, col], mode="lines", name=f"Alternate trees {nm}",
                                         line=dict(dash=dash, shape="hv")))
    if not fig.data:
        return
    add_drift_marks(fig, exp["stream"])
    fig.update_layout(title="Alternate-subtree counters reported by River (cumulative; sampled once per chunk)",
                      height=300, xaxis_title="Samples processed", yaxis_title="Count",
                      margin=dict(t=45, b=40, l=50, r=20), legend=dict(orientation="h", y=-0.28))
    wide(st.plotly_chart, fig)


# ------------------------------------------------------------------ panels
def fmt(v, pct=True):
    return "n/a" if v is None else (f"{v:.2%}" if pct else f"{v:.3f}")


def metrics_panel(exp):
    st.subheader("Live performance")
    tr = exp["trackers"][0]
    m = tr.metrics(exp["window"])
    st.caption(f"Primary model: {tr.name}. Positive class for precision/recall/F1: '{exp['pos']}'. A prediction of "
               f"{NONE_LABEL} (no label seen yet) counts as wrong. 'n/a' = undefined so far.")
    r1 = st.columns(4)
    r1[0].metric("Samples processed", f"{m['n']:,} / {len(exp['stream']['ys']):,}")
    r1[1].metric("Cumulative (prequential) accuracy", fmt(m["acc"]))
    r1[2].metric("Prequential error rate", fmt(m["err"]))
    r1[3].metric(f"Recent-window accuracy (last {exp['window']})", fmt(m["recent"]))
    r2 = st.columns(4)
    r2[0].metric("Precision", fmt(m["precision"]))
    r2[1].metric("Recall", fmt(m["recall"]))
    r2[2].metric("F1", fmt(m["f1"]))
    r2[3].metric("Status", "Running" if exp["running"] else ("Finished" if exp["i"] >= len(exp["stream"]["ys"])
                                                             else "Paused"))
    rows = []
    for t in exp["trackers"]:
        mm = t.metrics(exp["window"])
        rows.append({"Model": t.name, "Accuracy": fmt(mm["acc"]), "Error rate": fmt(mm["err"]),
                     "Recent accuracy": fmt(mm["recent"]), "Precision": fmt(mm["precision"]),
                     "Recall": fmt(mm["recall"]), "F1": fmt(mm["f1"]), "Runtime (s)": f"{mm['runtime']:.2f}",
                     "µs / sample": f"{mm['runtime'] / mm['n'] * 1e6:.0f}" if mm["n"] else "n/a"})
    if len(rows) > 1:
        st.markdown("**Model comparison (same stream, separate model instances)**")
        wide(st.dataframe, pd.DataFrame(rows), hide_index=True)
        st.caption("The adaptive tree's internal drift handling does not guarantee better accuracy on every dataset, "
                   "and it costs extra runtime. Runtime covers predict_one + learn_one only and varies with the "
                   "machine.")


def drift_panel(exp):
    s, n = exp["stream"], exp["i"]
    st.subheader("Drift and adaptation")
    if s["points"]:
        extra = f", gradual transition over {s['width']:,} samples after each point" if s["width"] else ""
        st.markdown(f"**Known injected drift:** yes ({s['kind']}) at sample(s) "
                    f"{', '.join(f'{p:,}' for p in s['points'])}{extra}. These are properties of the generated "
                    "stream, not detections.")
    else:
        st.markdown("**Known injected drift:** none " + ("(stationary stream)." if s["kind"] != CSV else
                                                          "(uploaded CSV; real drift, if any, is unknown)."))
    st.markdown("**What River handles internally.** Each node of the adaptive tree runs its own ADWIN detector on "
                "that node's error. When it signals, an alternate subtree is started; the original branch is replaced "
                "only if the alternate performs significantly better, otherwise the alternate is pruned. River does "
                "not publish individual internal drift events, so none are shown or counted here. Only its public "
                "counters below are observable.")
    rows = []
    for t in exp["trackers"]:
        r = {"Model": t.name}
        r.update({k: ("unavailable" if v is None else str(v)) for k, v in tree_stats(t.model).items()})
        rows.append(r)
    st.markdown("**Tree statistics (public River properties)**")
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("'unavailable' = the standard tree has no alternate subtrees.")
    alt_tree_chart(exp)
    ext = [t for t in exp["trackers"] if t.ext is not None]
    if not ext:
        st.info("External ADWIN monitoring is off, so no confirmed detector events and no detection delay are "
                "available. Enable it in the sidebar to get observable events on the error stream.")
        return
    st.markdown("**External ADWIN events** (dashboard-side detector on each model's error stream; independent of the "
                "tree's internal detector)")
    for t in ext:
        st.write(f"{t.name}: {len(t.events)} event(s)" + (f" at samples {t.events[:30]}" if t.events else ""))
        if s["points"]:
            d = match_delays(s["points"], t.events, max(n, 1))
            wide(st.dataframe, pd.DataFrame({"Injected point": [p for p, _ in d],
                                             "Detection delay (samples)": [("not detected" if x is None else x)
                                                                           for _, x in d]}), hide_index=True)
    st.caption("A detection delay is shown only when an external ADWIN event falls between an injected point and the "
               "next one. A synthetic drift point does not prove that any detector fired at that exact moment, and "
               "falling accuracy alone is not a confirmed detection.")


# ------------------------------------------------------------------ app
CSS = """<style>
.block-container{padding-top:1.4rem;padding-bottom:1.5rem}
[data-testid="stMetric"]{background:rgba(128,128,128,.08);border-radius:8px;padding:.45rem .8rem}
</style>"""


def cb_start(cfg_fn):
    """Start a new experiment, or resume a paused one."""
    exp = st.session_state.get("exp")
    if exp is not None:
        exp["running"] = exp["i"] < len(exp["stream"]["ys"])
        return
    try:
        cfg, stream = cfg_fn()
        st.session_state["exp"] = new_experiment(stream, cfg)
        st.session_state.pop("err", None)
    except Exception as exc:                              # shown in the sidebar, not swallowed
        st.session_state["err"] = str(exc)


def cb_pause():
    if st.session_state.get("exp"):
        st.session_state["exp"]["running"] = False


def cb_reset():
    st.session_state.pop("exp", None)
    st.session_state.pop("err", None)


def sidebar():
    exp = st.session_state.get("exp")
    lock = exp is not None
    with st.sidebar:
        st.header("Configuration")
        if lock:
            st.caption("Settings are locked while an experiment exists. Press Reset to change them.")
        data = st.selectbox("Dataset", [STAT, ABRUPT, GRAD, RECUR, CSV], index=1, disabled=lock)
        df = target = None
        if data == CSV:
            up = st.file_uploader("CSV (rows in time order; classification target)", type=["csv"], disabled=lock)
            if up is not None:
                try:
                    df = read_csv(up.getvalue())
                    target = st.selectbox("Target column", df.columns, index=len(df.columns) - 1, disabled=lock)
                except Exception as exc:
                    st.error(str(exc))
        n = int(st.number_input("Number of samples", 500, 200_000, 10_000, step=500, disabled=lock))
        seed = int(st.number_input("Random seed", 0, 10_000, 42, disabled=lock))
        noise = pos_pct = width = None
        if data != CSV:
            noise = st.slider("Label noise", 0.0, 0.3, 0.05, 0.01, disabled=lock)
        if data in (ABRUPT, GRAD):
            pos_pct = st.slider("Drift position (% of stream)", 10, 90, 50, 5, disabled=lock)
        if data == GRAD:
            width = int(st.number_input("Gradual transition width (samples)", 50, 50_000, 1000, step=50,
                                        disabled=lock))
        if data == RECUR:
            st.caption("Concept A, then B from 1/3 of the stream, then A again from 2/3.")
        st.markdown("**Model**")
        primary = st.radio("Primary model", [HAT, HT], disabled=lock)
        compare = st.checkbox("Also run the other tree on the same stream", value=False, disabled=lock)
        grace = int(st.number_input("Grace period", 10, 5000, 200, step=10, disabled=lock))
        delta = st.select_slider("Split confidence delta", [1e-3, 1e-5, 1e-7, 1e-9], value=1e-7,
                                 format_func=lambda v: f"{v:.0e}", disabled=lock)
        win_thr = int(st.number_input("Alternate-tree window (adaptive tree)", 50, 5000, 300, step=50, disabled=lock,
                                      help="Rows an alternate subtree must see before it can replace the branch."))
        custom = st.checkbox("Customise the adaptive tree's internal ADWIN delta", value=False, disabled=lock,
                             help="Off = River's default detector.")
        adwin = st.select_slider("Internal ADWIN delta", [0.1, 0.01, 0.002, 0.0005], value=0.002,
                                 disabled=lock or not custom) if custom else None
        st.markdown("**Monitoring**")
        window = int(st.number_input("Recent-window size", 20, 5000, 200, step=20, disabled=lock))
        use_ext = st.checkbox("Run an external ADWIN on the error stream", value=False, disabled=lock,
                              help="Dashboard-side detector, independent of the tree. Off by default.")
        ext_delta = st.select_slider("External ADWIN delta", [0.1, 0.01, 0.002, 0.0005], value=0.002,
                                     disabled=lock) if use_ext else None
        pos = "1"
        if data == CSV and df is not None and target:
            labs = sorted(set(label_str(df[target].dropna())))[:50]
            pos = st.selectbox("Positive class (precision/recall/F1)", labs, index=len(labs) - 1, disabled=lock)
        st.markdown("**Stream speed**")
        chunk = int(st.slider("Samples per update", 10, 2000, 200, 10, disabled=lock))
        delay = st.slider("Pause between updates (s)", 0.0, 1.0, 0.05, 0.05, disabled=lock)

        def cfg_fn():
            if data == CSV:
                if df is None or not target:
                    raise ValueError("Upload a CSV and choose a target column first.")
                stream = build_csv(df, target, n)
            else:
                stream = build_synth(data, n, seed, noise, pos_pct or 50, width or 0)
            cfg = dict(primary=primary, compare=compare, window=window, ext_delta=ext_delta, pos=pos, chunk=chunk,
                       delay=delay, hp=dict(grace=grace, delta=delta, win_thr=win_thr, adwin=adwin, seed=seed))
            return cfg, stream

        c1, c2, c3 = st.columns(3)
        started = exp is not None
        done = started and exp["i"] >= len(exp["stream"]["ys"])
        c1.button("Resume" if started else "Start", type="primary", on_click=cb_start, args=(cfg_fn,),
                  disabled=done or (started and exp["running"]), key="btn_start")
        c2.button("Pause", on_click=cb_pause, disabled=not (started and exp["running"]), key="btn_pause")
        c3.button("Reset", on_click=cb_reset, key="btn_reset")
        if st.session_state.get("err"):
            st.error(st.session_state["err"])


def main():
    st.set_page_config(page_title="Model Drift Dashboard", page_icon="📈", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("📈 Model Drift Dashboard: Hoeffding Adaptive Tree")
    if not RIVER_OK:
        st.error("This app needs the `river` package. Install it with `pip install -r requirements.txt` "
                 "(or add it to requirements.txt on Streamlit Community Cloud) and restart.")
        st.stop()
    sidebar()
    exp = st.session_state.get("exp")
    if exp is None:
        st.info("Choose a dataset in the sidebar and press **Start**. Each sample is predicted first, scored, and only "
                "then learned from (prequential evaluation).")
        st.stop()

    # Process one chunk per script run, then rerun while running. Keeps the UI responsive and Pause effective.
    if exp["running"]:
        try:
            process_chunk(exp, exp["chunk"])
        except Exception as exc:
            exp["running"] = False
            st.error(f"Processing stopped: {exc}")
            st.exception(exc)

    metrics_panel(exp)
    if exp["i"]:
        c = st.columns(2)
        with c[0]:
            line_chart(exp, "Cumulative prequential accuracy", "cum", "Accuracy")
        with c[1]:
            line_chart(exp, f"Recent-window accuracy (last {exp['window']} samples)", "win", "Accuracy")
        c = st.columns(2)
        with c[0]:
            error_blocks_chart(exp)
        with c[1]:
            true_vs_pred_chart(exp)
        drift_panel(exp)

    if exp["running"]:
        time.sleep(exp["delay"])
        st.rerun()


if __name__ == "__main__":
    main()
