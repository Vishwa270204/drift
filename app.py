"""Live Drift Dashboard - one Hoeffding Adaptive Tree, built from scratch.

    streamlit run hat_drift_dashboard.py

How it works
------------
* ONE model: river's Hoeffding Adaptive Tree (classifier or regressor). It adapts internally: every node
  watches its own error with ADWIN and grows an alternate subtree when it degrades.
* Training strategy: prequential "test-then-train" with an optional label delay.
    1. every incoming row is PREDICTED first (this is what gets scored),
    2. its label is released to the model `label delay` rows later and the tree learns from it one row at a time.
  The first `warm-up` rows are used for training only (no scoring, no detection).
* Drift detection: a separate ADWIN watches the stream of per-row prediction errors. When its window splits
  into two significantly different halves, a drift event is logged. The tree's own internal reactions
  (alternate subtrees created / swapped in) are tracked on a second chart.
* The charts update by themselves while streaming is running; no clicking needed after Start.
"""
import io
from collections import deque

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from river import drift, tree

MAX_POINTS = 1500          # points drawn per chart (older history is thinned out)
MAX_MARKERS = 40           # drift markers drawn at most
SENS = {"Low (fewer alerts)": 1e-4, "Medium": 2e-3, "High (more alerts)": 2e-2}   # ADWIN delta


# ------------------------------------------------------------------ data
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


def demo_stream(n=12000, seed=0):
    """Built-in demo: 3 numeric features. Data drift at 40% (feature means shift), concept drift at 70%
    (the labelling rule changes)."""
    r = np.random.default_rng(seed)
    a, b, c = r.normal(0, 1, n), r.normal(0, 1, n), r.normal(0, 1, n)
    d1, d2 = int(n * 0.4), int(n * 0.7)
    a[d1:] += 1.5
    b[d1:] *= 2.0
    y = np.zeros(n, dtype=int)
    y[:d1] = (a[:d1] + 0.5 * b[:d1] > 0).astype(int)
    y[d1:d2] = (a[d1:d2] + 0.5 * b[d1:d2] > 1.5).astype(int)
    y[d2:] = (a[d2:] - 0.5 * b[d2:] < 1.5).astype(int)          # rule changed: concept drift
    flip = r.random(n) < 0.03                                    # 3% label noise
    y = np.where(flip, 1 - y, y)
    return pd.DataFrame({"feat_a": a, "feat_b": b, "feat_c": c, "label": y})


def pick_features(df, exclude):
    num, cat = [], []
    for col in df.columns:
        if col in exclude:
            continue
        s, n = df[col], df[col].notna().sum()
        if n == 0 or s.nunique() <= 1:
            continue
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            if not (s.nunique() / n > 0.99 and bool(np.all(np.mod(s.dropna(), 1) == 0))):   # skip id-like
                num.append(col)
        elif s.nunique() <= 50:
            cat.append(col)
    return num, cat


def to_rows(df, num, cat):
    """List of feature dicts (river format). Missing numeric values are simply left out of the dict."""
    parts = {}
    for col in num:
        parts[col] = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
    for col in cat:
        parts[col] = np.where(df[col].notna(), df[col].astype(str), "(missing)")
    frame = pd.DataFrame(parts)
    rows = frame.to_dict("records")
    return [{k: v for k, v in r.items() if not (isinstance(v, float) and v != v)} for r in rows]


def label_str(s):
    s = pd.Series(s)
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        v = s.to_numpy(float)
        if len(v) and np.all(np.isfinite(v)) and np.all(np.mod(v, 1) == 0):
            return v.astype(np.int64).astype(str)
    return s.astype(str).to_numpy()


# ------------------------------------------------------------------ session
def make_model(task, cfg):
    kw = dict(grace_period=cfg["grace"], delta=cfg["tree_delta"], max_depth=cfg["max_depth"] or None,
              seed=cfg["seed"])
    if task == "classification":
        return tree.HoeffdingAdaptiveTreeClassifier(**kw)
    return tree.HoeffdingAdaptiveTreeRegressor(**kw)


def init_session(df, target, tcol, task, cfg):
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
    if len(d) < cfg["warm"] + 100:
        raise ValueError(f"Need at least {cfg['warm'] + 100} labelled rows (warm-up {cfg['warm']} + 100 to stream).")
    num, cat = pick_features(d.iloc[:cfg["warm"]], {target, tcol})
    if not (num or cat):
        raise ValueError("No usable feature columns were found.")
    return dict(
        rows=to_rows(d, num, cat), y=y, N=len(y), task=task, cfg=cfg, features=num + cat,
        model=make_model(task, cfg), adwin=drift.ADWIN(delta=cfg["adwin_delta"]),
        i=0, running=False, pending=deque(), win=deque(), win_sum=0.0, err_sum=0.0, n_scored=0,
        xs=[], roll=[], cum=[], nodes=[], switches=[], events=[], drift_rows=[],
        y_n=0, y_mean=0.0, y_m2=0.0, last_sw=0, last_alt=0, default=None,
    )


def norm_error(S, y, p):
    """Per-row error in roughly [0, 1] so ADWIN sees a comparable signal for both tasks."""
    if S["task"] == "classification":
        return 0.0 if (p is not None and p == y) else 1.0
    S["y_n"] += 1                                     # running std of the target (Welford)
    dlt = y - S["y_mean"]
    S["y_mean"] += dlt / S["y_n"]
    S["y_m2"] += dlt * (y - S["y_mean"])
    sd = (S["y_m2"] / max(S["y_n"] - 1, 1)) ** 0.5 or 1.0
    raw = abs(y - (p if p is not None else S["y_mean"]))
    S["raw_sum"] = S.get("raw_sum", 0.0) + raw
    return min(raw / (3 * sd), 1.0)


def tree_stats(m):
    def g(name, default=0):
        try:
            v = getattr(m, name)
            return v() if callable(v) else v
        except Exception:
            return default
    return dict(nodes=g("n_nodes"), leaves=g("n_leaves"), depth=g("height"),
                alt=g("n_alternate_trees"), sw=g("n_switch_alternate_trees"), pruned=g("n_pruned_alternate_trees"))


def step_rows(S, n):
    """Process up to n rows: test-then-train with label delay, ADWIN on the error stream."""
    cfg, m = S["cfg"], S["model"]
    for _ in range(n):
        i = S["i"]
        if i >= S["N"]:
            S["running"] = False
            return
        x, y = S["rows"][i], S["y"][i]
        S["i"] = i + 1
        if i >= cfg["warm"]:
            p = m.predict_one(x)
            e = norm_error(S, y, p)
            S["win"].append(e)
            S["win_sum"] += e
            if len(S["win"]) > cfg["roll"]:
                S["win_sum"] -= S["win"].popleft()
            S["err_sum"] += e
            S["n_scored"] += 1
            S["adwin"].update(e)
            if S["adwin"].drift_detected:
                st_ = tree_stats(m)
                S["drift_rows"].append(i)
                S["events"].append({"Row": i, "Rolling error": round(S["win_sum"] / len(S["win"]), 4),
                                    "Tree nodes": st_["nodes"], "Alternate trees": st_["alt"],
                                    "Subtree swaps so far": st_["sw"]})
            S["pending"].append((x, y))
            while len(S["pending"]) > cfg["delay"]:       # labels arrive `delay` rows late
                m.learn_one(*S["pending"].popleft())
            if S["n_scored"] % cfg["rec"] == 0:
                st_ = tree_stats(m)
                S["xs"].append(i)
                S["roll"].append(S["win_sum"] / len(S["win"]))
                S["cum"].append(S["err_sum"] / S["n_scored"])
                S["nodes"].append(st_["nodes"])
                S["switches"].append(st_["sw"])
        else:
            m.learn_one(x, y)                              # warm-up: train only


# ------------------------------------------------------------------ charts
def thin(a, k=MAX_POINTS):
    if len(a) <= k:
        return np.asarray(a)
    return np.asarray(a)[np.linspace(0, len(a) - 1, k).astype(int)]


def live_chart(S, true_drifts=()):
    unit = "Error rate" if S["task"] == "classification" else "Normalised abs. error"
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.06,
                        subplot_titles=(f"{unit} of the live predictions", "Tree size and subtree swaps"))
    if S["xs"]:
        xs = thin(S["xs"])
        fig.add_trace(go.Scatter(x=xs, y=thin(S["roll"]), mode="lines", name=f"Rolling ({S['cfg']['roll']} rows)",
                                 line=dict(color="#1f6fb2", width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=xs, y=thin(S["cum"]), mode="lines", name="Cumulative",
                                 line=dict(color="#9e9e9e", dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=xs, y=thin(S["nodes"]), mode="lines", name="Tree nodes",
                                 line=dict(color="#00897b")), row=2, col=1)
        fig.add_trace(go.Scatter(x=xs, y=thin(S["switches"]), mode="lines", name="Subtree swaps (cumulative)",
                                 line=dict(color="#e08a00", shape="hv")), row=2, col=1)
    for r in S["drift_rows"][-MAX_MARKERS:]:
        fig.add_vline(x=r, line_color="#c62828", line_width=1, opacity=0.6)
    for r in true_drifts:
        fig.add_vline(x=r, line_color="#6a1b9a", line_dash="dash", annotation_text="true drift",
                      annotation_position="top left")
    fig.update_xaxes(range=[S["cfg"]["warm"], S["N"]], title_text="Row in stream", row=2, col=1)
    fig.update_layout(height=560, margin=dict(t=40, b=40, l=50, r=20), legend=dict(orientation="h", y=-0.14),
                      uirevision="live")
    st.plotly_chart(fig, use_container_width=True)


def live_panel(S, true_drifts):
    """Runs repeatedly by itself while streaming; processes a chunk of rows, then redraws."""
    if S["running"]:
        step_rows(S, S["cfg"]["speed"])
        if not S["running"]:
            st.rerun()                                     # finished: refresh buttons / status
    scored = S["n_scored"]
    ts = tree_stats(S["model"])
    c = st.columns(5)
    c[0].metric("Rows streamed", f"{S['i']:,} / {S['N']:,}")
    c[1].metric("Rolling error", f"{S['win_sum'] / len(S['win']):.3f}" if S["win"] else "-")
    c[2].metric("Overall error", f"{S['err_sum'] / scored:.3f}" if scored else "-")
    c[3].metric("Drifts detected", len(S["drift_rows"]))
    c[4].metric("Tree nodes / depth", f"{ts['nodes']} / {ts['depth']}")
    st.progress(min(S["i"] / S["N"], 1.0))
    if S["i"] < S["cfg"]["warm"]:
        st.caption(f"Warm-up: training on the first {S['cfg']['warm']:,} rows (no scoring yet).")
    live_chart(S, true_drifts)
    if S["events"]:
        st.subheader("Drift events")
        st.dataframe(pd.DataFrame(S["events"]).iloc[::-1], hide_index=True, use_container_width=True)


# ------------------------------------------------------------------ app
def sidebar():
    with st.sidebar:
        st.header("Stream")
        up = st.file_uploader("Upload a CSV (rows in time order)", type=["csv"])
        true_drifts = ()
        if up is None:
            st.caption("No file: using the built-in demo stream (data drift at 40%, concept drift at 70%).")
            df = demo_stream()
            target, true_drifts = "label", (int(len(df) * 0.4), int(len(df) * 0.7))
            tcol = "(row order)"
        else:
            try:
                df = read_csv(up.getvalue())
            except Exception as exc:
                st.error(str(exc))
                return None
            target = st.selectbox("Target column", df.columns, index=len(df.columns) - 1)
            tcol = st.selectbox("Time column (optional)", ["(row order)"] + [c for c in df.columns if c != target])
        guess = "regression" if (pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique() > 10) \
            else "classification"
        task = st.radio("Task", ["classification", "regression"], horizontal=True,
                        index=["classification", "regression"].index(guess))
        st.header("Streaming")
        speed = int(st.slider("Rows per refresh", 1, 500, 40))
        refresh = st.slider("Refresh interval (seconds)", 0.1, 2.0, 0.3, 0.1)
        st.header("Training strategy")
        warm = int(st.number_input("Warm-up rows (train only)", 50, 100000, 500, step=50))
        delay = int(st.number_input("Label delay (rows)", 0, 5000, 0,
                                    help="Labels reach the model this many rows after the prediction."))
        st.header("Drift detection")
        sens = st.selectbox("Sensitivity", list(SENS), index=1)
        roll = int(st.number_input("Rolling window (rows)", 20, 5000, 200, step=20))
        with st.expander("Tree settings"):
            grace = int(st.number_input("Grace period", 20, 2000, 200, step=20))
            tree_delta = st.select_slider("Split confidence delta", [1e-3, 1e-5, 1e-7, 1e-9], value=1e-7)
            max_depth = int(st.number_input("Max depth (0 = unlimited)", 0, 50, 0))
            seed = int(st.number_input("Seed", 0, 10000, 42))
        cfg = dict(speed=speed, refresh=refresh, warm=warm, delay=delay, adwin_delta=SENS[sens], roll=roll,
                   grace=grace, tree_delta=float(tree_delta), max_depth=max_depth, seed=seed,
                   rec=max(1, speed // 4))
        if st.button("Load stream with these settings", type="primary"):
            try:
                st.session_state["S"] = init_session(df, target, tcol, task, cfg)
                st.session_state["true_drifts"] = true_drifts
            except ValueError as exc:
                st.error(str(exc))
        return cfg


def main():
    st.set_page_config(page_title="Live Drift Dashboard", page_icon="📈", layout="wide")
    st.title("📈 Live Drift Dashboard · Hoeffding Adaptive Tree")
    sidebar()
    S = st.session_state.get("S")
    if S is None:
        st.info("Choose settings in the sidebar and click **Load stream with these settings**, then **Start**.")
        st.stop()

    done = S["i"] >= S["N"]
    b = st.columns([1, 1, 1, 5])
    if b[0].button("▶ Start", type="primary", disabled=S["running"] or done):
        S["running"] = True
        st.rerun()
    if b[1].button("⏸ Pause", disabled=not S["running"]):
        S["running"] = False
        st.rerun()
    if b[2].button("↺ Restart"):
        st.session_state.pop("S")
        st.rerun()
    b[3].markdown(f"**Status:** {'Finished' if done else 'Streaming' if S['running'] else 'Paused'}  ·  "
                  f"features: {', '.join(S['features'][:8])}{'…' if len(S['features']) > 8 else ''}")

    # the fragment re-runs itself every `refresh` seconds while streaming, so the graph fills in automatically
    run_every = S["cfg"]["refresh"] if S["running"] else None
    st.fragment(run_every=run_every)(live_panel)(S, st.session_state.get("true_drifts", ()))


if __name__ == "__main__":
    main()
