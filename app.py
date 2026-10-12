import time
from collections import deque

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from river import drift, metrics, stats, tree

st.set_page_config(page_title="Model Drift Dashboard", layout="wide")

TASKS = ["Regression", "Classification"]
WARMUP, WINDOW, REPLAY, PLOT_WIN = 30, 200, 100, 50
STATUS = {
    "Normal": "🟢 Normal",
    "Drift Detected": "🔴 Drift Detected",
    "Retraining": "🟠 Retraining",
}


def synthetic(task, n=3000):
    rng = np.random.default_rng(7)
    X = rng.normal(size=(n, 5))
    seg = np.minimum(np.arange(n) // 1000, 2)
    if task == "Regression":
        W = np.array([[3, -2, 1, 0, 0], [-3, 2, 0, 2, 0], [0, 1, -3, 0, 3]], float)
        y = (X * W[seg]).sum(axis=1) + 0.3 * rng.normal(size=n)
    else:
        W = np.array([[1, 1, 0, 0, 0], [-1, 1, 0, 0, 0], [0, 0, 1, -1, 0]], float)
        y = ((X * W[seg]).sum(axis=1) > 0).astype(int)
        y = np.where(rng.random(n) < 0.05, 1 - y, y)
    return [f"f{i}" for i in range(5)], X.tolist(), y.tolist()


def load_csv(file, task):
    df = pd.read_csv(file).replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    if df.shape[1] < 2:
        raise ValueError("need at least one feature column and a target column")
    df.columns = df.columns.astype(str)
    target = df.columns[-1]
    raw = df[target]
    if task == "Regression":
        y = pd.to_numeric(raw, errors="coerce")
        if raw.notna().sum() == 0 or y.notna().sum() / raw.notna().sum() < 0.9:
            raise ValueError("target is not numeric; use Classification")
    else:
        y = raw
    X = df.drop(columns=target).apply(pd.to_numeric, errors="coerce")
    X = X.dropna(axis=1, how="all")
    if X.shape[1] == 0:
        raise ValueError("no usable numeric feature columns")
    X = X.fillna(X.median()).fillna(0.0)
    ok = y.notna()
    X, y = X[ok], y[ok]
    if task == "Classification":
        y = y.astype(str)
        if y.nunique() < 2:
            raise ValueError("target needs at least two classes")
        if y.nunique() > 50:
            raise ValueError("target has too many classes; use Regression")
    if len(y) < 50:
        raise ValueError("need at least 50 valid rows")
    return list(X.columns), X.to_numpy(float).tolist(), y.tolist()


def new_model(task):
    if task == "Regression":
        return tree.HoeffdingAdaptiveTreeRegressor(grace_period=50, seed=0)
    return tree.HoeffdingAdaptiveTreeClassifier(grace_period=50, seed=0)


def new_detector():
    return drift.ADWIN(delta=0.002)


def fresh():
    task = st.session_state.get("task", TASKS[0])
    file = st.session_state.get("file")
    msg = ""
    if file is None:
        cols, X, y = synthetic(task)
    else:
        try:
            file.seek(0)
            cols, X, y = load_csv(file, task)
        except Exception as e:
            msg = f"CSV rejected: {e}. Using synthetic stream."
            cols, X, y = synthetic(task)
    metric = metrics.MAE() if task == "Regression" else metrics.Accuracy()
    return dict(
        task=task, cols=cols, X=X, y=y, i=0, msg=msg,
        model=new_model(task), det=new_detector(),
        metric=metrics.Rolling(metric, WINDOW), var=stats.Var(),
        buf=deque(maxlen=REPLAY), win=deque(maxlen=PLOT_WIN),
        xs=[], ys=[], drifts=[], since=0, status="Normal",
        pending=False, running=False,
    )


def restart():
    st.session_state.s = fresh()


def toggle():
    s = st.session_state.s
    if s["running"]:
        s["running"] = False
        return
    if s["i"] >= len(s["y"]):
        st.session_state.s = s = fresh()
    s["running"] = True


def detect(det, value):
    det.update(value)
    return bool(getattr(det, "drift_detected", getattr(det, "change_detected", False)))


def retrain(s):
    model = new_model(s["task"])
    for bx, by in s["buf"]:
        model.learn_one(bx, by)
    s["model"], s["det"], s["since"] = model, new_detector(), 0
    s["status"], s["pending"] = "Retraining", False


def step(s):
    x = dict(zip(s["cols"], s["X"][s["i"]]))
    y = s["y"][s["i"]]
    reg = s["task"] == "Regression"
    p = s["model"].predict_one(x)
    if p is None:
        p = "?"
    s["metric"].update(y, p)
    if reg:
        s["var"].update(y)
        sd = s["var"].get() ** 0.5
        score = min(abs(y - p) / sd, 1.0) if sd > 0 else 1.0
    else:
        score = float(p != y)
    s["i"] += 1
    s["since"] += 1
    s["win"].append(score)
    s["xs"].append(s["i"])
    s["ys"].append(sum(s["win"]) / len(s["win"]))
    s["buf"].append((x, y))
    if s["since"] > WARMUP and detect(s["det"], score):
        s["drifts"].append((s["i"], s["ys"][-1]))
        s["status"], s["pending"] = "Drift Detected", True
        return
    s["model"].learn_one(x, y)
    if s["status"] == "Retraining" and s["since"] >= WARMUP:
        s["status"] = "Normal"


def figure(s):
    k = max(1, len(s["xs"]) // 2000)
    dx = [v for d, _ in s["drifts"] for v in (d, d, None)]
    dy = [v for _ in s["drifts"] for v in (0, 1, None)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=s["xs"][::k], y=s["ys"][::k], mode="lines",
        line=dict(color="#1f77b4", width=2), name="Error"))
    fig.add_trace(go.Scatter(
        x=dx, y=dy, mode="lines", line=dict(color="rgba(214,39,40,0.6)", width=1.5, dash="dash"),
        hoverinfo="skip", name="Drift"))
    fig.add_trace(go.Scatter(
        x=[d for d, _ in s["drifts"]], y=[v for _, v in s["drifts"]], mode="markers",
        marker=dict(color="#d62728", size=10, symbol="diamond"), name="Drift point"))
    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
        template="plotly_white", uirevision="keep", hovermode="x",
        xaxis=dict(range=[0, len(s["y"])], title="Samples"),
        yaxis=dict(range=[0, 1.02], title=f"Rolling error (last {PLOT_WIN})"),
    )
    return fig


def render(s, ph):
    reg = s["task"] == "Regression"
    val = s["metric"].get()
    ph["status"].metric("Status", STATUS[s["status"]])
    ph["n"].metric("Samples processed", f'{s["i"]:,} / {len(s["y"]):,}')
    ph["d"].metric("Drift count", len(s["drifts"]))
    ph["m"].metric(
        f'{"MAE" if reg else "Accuracy"} (last {WINDOW})',
        f"{val:.3f}" if reg else f"{val:.1%}")
    cfg = {"displayModeBar": False}
    try:
        ph["chart"].plotly_chart(figure(s), width="stretch", config=cfg)
    except TypeError:
        ph["chart"].plotly_chart(figure(s), use_container_width=True, config=cfg)


if "s" not in st.session_state:
    st.session_state.s = fresh()
s = st.session_state.s

st.title("Model Drift Detection Dashboard")

c1, c2, c3, c4 = st.columns([1.2, 2.2, 1.6, 1])
c1.selectbox("Model type", TASKS, key="task", on_change=restart)
c2.file_uploader("Upload CSV (optional, last column = target)", type="csv",
                 key="file", on_change=restart)
c3.slider("Delay (ms)", 0, 500, 50, 10, key="delay")
c4.button("⏹ Stop" if s["running"] else "▶ Start", on_click=toggle, type="primary", key="go")

if s["msg"]:
    st.error(s["msg"])

m1, m2, m3, m4 = st.columns(4)
ph = {"status": m1.empty(), "n": m2.empty(), "d": m3.empty(), "m": m4.empty(), "chart": st.empty()}
render(s, ph)

delay = st.session_state.get("delay", 50) / 1000
while s["running"] and s["i"] < len(s["y"]):
    if s["pending"]:
        retrain(s)
        render(s, ph)
    step(s)
    render(s, ph)
    time.sleep(delay)

if s["running"]:
    s["running"] = False
    st.rerun()
