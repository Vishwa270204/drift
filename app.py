import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import copy

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="Got `batch_size`")
st.set_page_config(page_title="Model Drift Dashboard", page_icon="📈", layout="wide")

FAMILIES = {"SGD (linear)": "sgd", "MLP (neural net)": "mlp", "Random forest": "rf"}
EPOCHS = {"sgd": 20, "mlp": 30, "rf": 1}
SENS = {"Low (fewer alerts)": 4.0, "Medium": 3.0, "High (more alerts)": 2.0}
NO_RETRAIN = "No retrain (keep old model)"


# ------------------------------------------------------------------ model helpers
def make_est(family, task, seed):
    reg = task == "regression"
    if family == "sgd":
        kw = dict(alpha=1e-4, learning_rate="constant", eta0=0.01, random_state=seed)
        return SGDRegressor(**kw) if reg else SGDClassifier(loss="log_loss", **kw)
    if family == "mlp":
        kw = dict(hidden_layer_sizes=(64, 32), learning_rate_init=0.003, random_state=seed)
        return MLPRegressor(**kw) if reg else MLPClassifier(**kw)
    kw = dict(n_estimators=100, min_samples_leaf=2, random_state=seed, n_jobs=1)
    return RandomForestRegressor(**kw) if reg else RandomForestClassifier(**kw)


class Bundle:
    """Preprocessing + estimator. Fitted from scratch or fine-tuned with partial_fit."""

    def __init__(self, family, task, num, cat, classes, seed):
        self.family, self.task, self.num, self.cat = family, task, num, cat
        self.classes, self.seed = classes, seed
        self.mu, self.sd = 0.0, 1.0

    def _target(self, y):
        return (np.asarray(y, float) - self.mu) / self.sd if self.task == "regression" else np.asarray(y)

    def _partial(self, Xt, y, epochs):
        yy, rng = self._target(y), np.random.default_rng(self.seed)
        for _ in range(epochs):
            p = rng.permutation(len(yy))
            if self.task == "classification" and not hasattr(self.est, "classes_"):
                self.est.partial_fit(Xt[p], yy[p], classes=self.classes)
            else:
                self.est.partial_fit(Xt[p], yy[p])

    def fit(self, X, y):
        steps = []
        if self.num:
            steps.append(("n", Pipeline([("i", SimpleImputer(strategy="median", keep_empty_features=True)),
                                         ("s", StandardScaler())]), self.num))
        if self.cat:
            steps.append(("c", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.cat))
        self.pre = ColumnTransformer(steps).fit(X)
        y = np.asarray(y)
        if self.task == "regression":
            self.mu, self.sd = float(y.mean()), float(y.std()) or 1.0
        elif len(np.unique(y)) < 2 and self.family != "rf":
            raise ValueError("only one class in this data")
        self.est = make_est(self.family, self.task, self.seed)
        Xt = np.asarray(self.pre.transform(X), float)
        if hasattr(self.est, "partial_fit"):
            self._partial(Xt, y, EPOCHS[self.family])
        else:
            self.est.fit(Xt, self._target(y))
        return self

    def fine_tune(self, X, y, epochs=5):
        y = np.asarray(y)
        if self.task == "classification":
            keep = np.isin(y, self.classes)
            X, y = X[keep], y[keep]
        self._partial(np.asarray(self.pre.transform(X), float), y, epochs)
        return self

    def predict(self, X):
        p = self.est.predict(np.asarray(self.pre.transform(X), float))
        return p * self.sd + self.mu if self.task == "regression" else np.asarray(p).astype(str)


def row_err(task, y, p):
    if task == "regression":
        return np.abs(np.asarray(y, float) - np.asarray(p, float))
    return (np.asarray(y) != np.asarray(p)).astype(float)


def fit_new(family, task, num, cat, classes, seed, X, y, cap=30000):
    if len(y) > cap:
        pick = np.sort(np.random.default_rng(seed).choice(len(y), cap, replace=False))
        X, y = X.iloc[pick], y[pick]
    return Bundle(family, task, num, cat, classes, seed).fit(X, y)


# ------------------------------------------------------------------ data prep
@st.cache_data(show_spinner="Reading CSV...")
def read(data):
    return pd.read_csv(data)


def build_frame(df, num, cat):
    parts = {c: pd.to_numeric(df[c], errors="coerce").astype(float).to_numpy() for c in num}
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
            ids = s.nunique() / n > 0.99 and bool(np.all(np.mod(s.dropna(), 1) == 0))
            if not ids:
                num.append(c)
        elif s.nunique() <= 50:
            cat.append(c)
    return num, cat


# ------------------------------------------------------------------ page
st.title("📈 Model Drift Dashboard")
up = st.file_uploader("Upload a CSV (rows in time order, or pick a time column)", type=["csv"])
if up is None:
    st.stop()
df = read(up.getvalue())

c1, c2, c3, c4 = st.columns(4)
target = c1.selectbox("Target column", df.columns, index=len(df.columns) - 1)
tcol = c2.selectbox("Time column (optional)", ["(row order)"] + [c for c in df.columns if c != target])
family_label = c3.selectbox("Model", list(FAMILIES))
family = FAMILIES[family_label]
tgt = df[target]
guess = "regression" if (pd.api.types.is_numeric_dtype(tgt) and tgt.nunique() > 10) else "classification"
task = c4.radio("Task", ["classification", "regression"], index=["classification", "regression"].index(guess), horizontal=True)

with st.expander("Settings"):
    s1, s2, s3, s4, s5, s6 = st.columns(6)
    train_pct = s1.slider("Initial training %", 20, 70, 40, 5)
    B = int(s2.number_input("Batch size (rows)", 20, 5000, 100))
    sens = s3.selectbox("Drift sensitivity", list(SENS), index=1)
    delay = int(s4.number_input("Label delay (batches)", 0, 20, 0))
    wait = int(s5.number_input("Batches to wait after alert before retraining", 0, 20, 2,
                               help="More waiting = more post-drift labelled data, but a longer time on the stale model."))
    manual = int(s6.number_input("Manual drift start batch (0 = auto-detect)", 0, 10000, 0))
    seed = 42

if st.button("Run analysis", type="primary"):
    d = df.copy()
    if tcol != "(row order)":
        tv = pd.to_numeric(d[tcol], errors="coerce") if pd.api.types.is_numeric_dtype(d[tcol]) \
            else pd.to_datetime(d[tcol], errors="coerce", utc=True)
        d = d.loc[tv.notna()].assign(_t=tv[tv.notna()]).sort_values("_t", kind="stable").drop(columns="_t")
    d = d[d[target].notna()].reset_index(drop=True)
    if len(d) < 300:
        st.error("Need at least 300 labelled rows."); st.stop()
    num, cat = pick_features(d, {target, tcol})
    if not num and not cat:
        st.error("No usable feature columns found."); st.stop()
    X = build_frame(d, num, cat)
    y = d[target].astype(float).to_numpy() if task == "regression" else d[target].astype(str).to_numpy()
    classes = np.unique(y) if task == "classification" else None

    n_tr = int(len(d) * train_pct / 100)
    n_fit = int(n_tr * 0.8)                       # baseline fits on first 80% of training; last 20% = reference errors
    base = Bundle(family, task, num, cat, classes, seed).fit(X.iloc[:n_fit], y[:n_fit])
    ref_err = row_err(task, y[n_fit:n_tr], base.predict(X.iloc[n_fit:n_tr]))
    ref_m, ref_s = ref_err.mean(), ref_err.std() + 1e-12

    Xs, ys = X.iloc[n_tr:].reset_index(drop=True), y[n_tr:]
    N = len(ys); nb = int(np.ceil(N / B)); bid = np.arange(N) // B
    base_err = row_err(task, ys, base.predict(Xs))
    bm = np.array([base_err[bid == k].mean() for k in range(nb)])
    sizes = np.array([(bid == k).sum() for k in range(nb)])
    thr = ref_m + SENS[sens] * ref_s / np.sqrt(sizes)
    flag = bm > thr

    start = None
    if manual > 0:
        start = manual - 1
    else:
        for k in range(nb - 1):
            if flag[k] and flag[k + 1]:          # two consecutive bad batches = alert
                start = k; break
    out = dict(bm=bm, thr=thr, ref_m=ref_m, nb=nb, start=start, delay=delay, task=task, results=None, msg=None)

    if start is not None:
        retrain_at = start + 1 + delay + wait    # batch after which the retrain happens
        last_lab = retrain_at - delay            # newest batch whose labels have arrived
        out["retrain_at"] = retrain_at
        lab = np.isin(bid, np.arange(0, last_lab + 1))
        new_m = lab & (bid >= start)
        old_m = np.arange(N) < 0
        Xold = pd.concat([X.iloc[:n_tr], Xs[lab & (bid < start)]]).reset_index(drop=True)
        yold = np.concatenate([y[:n_tr], ys[lab & (bid < start)]])
        Xnew, ynew = Xs[new_m].reset_index(drop=True), ys[new_m]
        ev = bid > retrain_at
        if ev.sum() < 30 or len(ynew) < 30:
            out["msg"] = ("Drift was found too late in the stream to evaluate retraining "
                          "(not enough later batches or too little post-drift labelled data). "
                          "Lower the batch size, wait fewer batches, or use a longer file.")
        else:
            Xev, yev = Xs[ev], ys[ev]
            def fit_on(Xa, ya):
                return fit_new(family, task, num, cat, classes, seed, Xa, ya)
            Xall, yall = pd.concat([Xold, Xnew]).reset_index(drop=True), np.concatenate([yold, ynew])
            win = min(len(yall), max(2 * len(ynew), 200))
            strategies = {
                NO_RETRAIN: (lambda: base, len(y[:n_fit])),
                "New data only (from scratch)": (lambda: fit_on(Xnew, ynew), len(ynew)),
                "Old + new data (from scratch)": (lambda: fit_on(Xall, yall), len(yall)),
                "Old + new, new weighted 3x": (lambda: fit_on(pd.concat([Xold] + [Xnew] * 3).reset_index(drop=True),
                                                              np.concatenate([yold] + [ynew] * 3)), len(yold) + 3 * len(ynew)),
                "Recent window only (from scratch)": (lambda: fit_on(Xall.iloc[-win:], yall[-win:]), win),
            }
            if hasattr(base.est if hasattr(base, "est") else None, "partial_fit"):
                strategies["Fine-tune old model on new data"] = (
                    lambda: copy.deepcopy(base).fine_tune(Xnew, ynew), len(ynew))
            res, errs, notes = [], {}, []
            for name, (mk, n_used) in strategies.items():
                try:
                    m = mk()
                    e = row_err(task, yev, m.predict(Xev))
                    errs[name] = np.array([e[bid[ev] == k].mean() for k in range(retrain_at + 1, nb)])
                    res.append({"Strategy": name, "Training rows used": int(n_used), "Mean error": float(e.mean())})
                except Exception as exc:
                    notes.append(f"{name}: skipped ({exc})")
            r = pd.DataFrame(res)
            b0 = r.loc[r["Strategy"] == NO_RETRAIN, "Mean error"].iloc[0]
            r["Change vs no retrain (%)"] = (b0 - r["Mean error"]) / max(b0, 1e-12) * 100   # positive = better
            out.update(results=r, errs=errs, notes=notes, n_old=len(yold), n_new=len(ynew))
    st.session_state["out"] = out

out = st.session_state.get("out")
if out is None:
    st.info("Pick your columns and click **Run analysis**.")
    st.stop()

err_name = "MAE" if out["task"] == "regression" else "Error rate"
xs = np.arange(1, out["nb"] + 1)
start, retrain_at = out["start"], out.get("retrain_at")

# ---- headline numbers
k1, k2, k3, k4 = st.columns(4)
k1.metric("Drift", "Detected" if start is not None else "None found")
k2.metric("Drift starts at batch", start + 1 if start is not None else "n/a")
res = out["results"]
if res is not None:
    alt = res[res["Strategy"] != NO_RETRAIN].sort_values("Mean error")
    best = alt.iloc[0] if len(alt) else None
    k3.metric("Retrained at batch", retrain_at + 1 + 0)
    if best is not None:
        k4.metric("Best strategy", best["Strategy"], f"{best['Change vs no retrain (%)']:+.1f}% vs no retrain")

# ---- main graph: model drift
fig = go.Figure()
fig.add_trace(go.Scatter(x=xs, y=out["bm"], mode="lines+markers", name=f"{err_name} of current model"))
fig.add_trace(go.Scatter(x=xs, y=out["thr"], mode="lines", name="Drift threshold", line=dict(dash="dash", color="#d1342f")))
fig.add_hline(y=out["ref_m"], line_dash="dot", line_color="gray", annotation_text="error at training time")
if start is not None:
    fig.add_vrect(x0=start + 1, x1=out["nb"], fillcolor="#d1342f", opacity=0.06, line_width=0)
    fig.add_vline(x=start + 1, line_color="#d1342f", annotation_text="drift starts")
if retrain_at is not None:
    fig.add_vline(x=retrain_at + 1, line_color="#2e9e5b", annotation_text="retrain")
fig.update_layout(title="Model drift: error of the frozen model on incoming batches", xaxis_title="Batch", yaxis_title=err_name, height=380)
st.plotly_chart(fig, use_container_width=True)

if start is None:
    st.success("No drift detected: error stayed within the expected range, so no retraining is needed.")
    st.stop()
if out["msg"]:
    st.warning(out["msg"]); st.stop()

# ---- strategy comparison after retraining
st.subheader("After retraining: how does each strategy perform?")
st.caption(f"Old data = {out['n_old']:,} rows from before the drift. New data = {out['n_new']:,} labelled rows since the drift began. "
           "All strategies are scored on the same later batches that none of them saw.")
l, r_ = st.columns(2)
fig = go.Figure()
for name, e in out["errs"].items():
    fig.add_trace(go.Scatter(x=np.arange(retrain_at + 2, out["nb"] + 1), y=e, mode="lines+markers", name=name,
                             line=dict(dash="dash" if name == NO_RETRAIN else "solid")))
fig.update_layout(title=f"{err_name} per batch after retraining", xaxis_title="Batch", yaxis_title=err_name, height=380)
l.plotly_chart(fig, use_container_width=True)
srt = res.sort_values("Mean error")
bar = go.Figure(go.Bar(x=srt["Mean error"], y=srt["Strategy"], orientation="h",
                       marker_color=["#888" if s == NO_RETRAIN else "#2e9e5b" for s in srt["Strategy"]],
                       text=srt["Mean error"].round(4), textposition="auto"))
bar.update_layout(title=f"Average {err_name} (lower is better)", yaxis=dict(autorange="reversed"), height=380)
r_.plotly_chart(bar, use_container_width=True)
st.dataframe(srt.round(4), hide_index=True)
for n in out.get("notes", []):
    st.caption(n)

if best is not None and best["Change vs no retrain (%)"] > 0:
    st.success(f"Recommended: **{best['Strategy']}** ({best['Change vs no retrain (%)']:+.1f}% vs keeping the old model). "
               "Keep monitoring; when drift is flagged again, repeat the same retrain with fresher data.")
else:
    st.info("No retraining strategy beat the old model on later batches. Wait for more post-drift labels, or check whether the alert was noise.")
