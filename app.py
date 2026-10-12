import copy
import io
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

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="Got `batch_size`")
st.set_page_config(page_title="Model Drift Dashboard", page_icon="📈", layout="wide")

FAMILIES = {"SGD (linear)": "sgd", "MLP (neural net)": "mlp", "Random forest": "rf"}
EPOCHS = {"sgd": 20, "mlp": 30, "rf": 1}
SENS = {"Low (fewer alerts)": 4.0, "Medium": 3.0, "High (more alerts)": 2.0}
KEEP = "Keep current model (no retrain)"
NEWO, BOTH = "New data only", "Old + new data"
WGT, WIN = "Old + new, new weighted 3x", "Recent window only"
INC = "Incremental (mini-batch partial_fit on new data)"
ONL = "Online (one row at a time)"
CONT = "Continual (new data + replay of old data)"
TRF = "Transfer (pretrain on source CSV, fine-tune on new)"
MB = 64   # mini-batch size for incremental / continual updates


# ------------------------------------------------------------------ models
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
    def __init__(self, family, task, num, cat, classes, seed):
        self.family, self.task, self.num, self.cat = family, task, num, cat
        self.classes, self.seed, self.mu, self.sd = classes, seed, 0.0, 1.0

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

    def update_batches(self, X, y, bs, epochs, replay=None, rng=None):
        """Sequential partial_fit in mini-batches; optionally mix each batch with replayed old rows."""
        y = np.asarray(y)
        if self.task == "classification":
            keep = np.isin(y, self.classes)
            X, y = X[keep], y[keep]
        Xt = np.asarray(self.pre.transform(X), float)
        if replay is not None:
            Rt, ry = np.asarray(self.pre.transform(replay[0]), float), np.asarray(replay[1])
        for s in range(0, len(y), bs):
            xb, yb = Xt[s:s + bs], y[s:s + bs]
            if replay is not None and len(ry):
                pick = rng.choice(len(ry), min(MB, len(ry)), replace=False)
                xb, yb = np.vstack([xb, Rt[pick]]), np.concatenate([yb, ry[pick]])
            self._partial(xb, yb, epochs)
        return self

    def predict(self, X):
        p = self.est.predict(np.asarray(self.pre.transform(X), float))
        return p * self.sd + self.mu if self.task == "regression" else np.asarray(p).astype(str)


def row_err(task, y, p):
    if task == "regression":
        return np.abs(np.asarray(y, float) - np.asarray(p, float))
    return (np.asarray(y) != np.asarray(p)).astype(float)


def ref_stats(task, e):
    m, s = float(e.mean()), float(e.std())
    s = max(s, 0.1) if task == "classification" else s
    return m, max(s, 0.1 * m, 1e-9)


def fit_new(S, X, y, cap=30000):
    if len(y) > cap:
        pick = np.sort(np.random.default_rng(S["seed"]).choice(len(y), cap, replace=False))
        X, y = X.iloc[pick], y[pick]
    return Bundle(S["family"], S["task"], S["num"], S["cat"], S["classes"], S["seed"]).fit(X, y)


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
            if not (s.nunique() / n > 0.99 and bool(np.all(np.mod(s.dropna(), 1) == 0))):  # skip id-like
                num.append(c)
        elif s.nunique() <= 50:
            cat.append(c)
    return num, cat


# ------------------------------------------------------------------ streaming engine
def log(S, event, detail):
    S["events"].append({"Batch": S["k"], "Event": event, "Detail": detail})


def new_rows(S):
    """Labelled rows available since the suspected drift start."""
    al = S["alert"]
    if al is None:
        return 0
    known = S["k"] - S["delay"]                       # batches whose labels have arrived
    return max(0, min(known * S["B"], S["N"]) - al["start"] * S["B"])


def step(S):
    """Receive one more batch: predict with the deployed model, then check drift on batches whose labels arrived."""
    k, B, n_tr = S["k"], S["B"], S["n_tr"]
    a, b = n_tr + k * B, min(n_tr + (k + 1) * B, len(S["y"]))
    e = row_err(S["task"], S["y"][a:b], S["model"].predict(S["X"].iloc[a:b]))
    S["errs"].append(float(e.mean()))
    S["thr"].append(S["ref_m"] + S["z"] * S["ref_s"] / np.sqrt(b - a))
    S["ver"].append(S["version"])
    S["k"] = k + 1
    j = k - S["delay"]
    if j >= 0 and S["ver"][j] == S["version"]:
        flagged = S["errs"][j] > S["thr"][j]
        if S["alert"] is None:
            S["bad"] = S["bad"] + [j] if flagged else []
            if len(S["bad"]) >= 2 and S["k"] > S["snooze_until"]:        # 2 bad batches in a row = alert
                S["alert"], S["cand"], S["good"] = dict(start=S["bad"][0], status="waiting"), None, 0
                S["bad"] = []
                log(S, "Drift alert", f"errors above threshold since batch {S['alert']['start'] + 1}")
        elif S["alert"]["status"] == "waiting":
            S["good"] = 0 if flagged else S["good"] + 1
            if S["good"] >= 3:                                           # it went away by itself
                S["alert"] = None
                log(S, "Alert cleared", "error returned to normal before retraining was needed")
    al = S["alert"]
    if al and al["status"] == "waiting" and new_rows(S) >= S["min_rows"]:
        al["status"] = "ready"


def get_data(S, with_holdout):
    n_tr, B = S["n_tr"], S["B"]
    s0 = n_tr + S["alert"]["start"] * B
    s1 = n_tr + min((S["k"] - S["delay"]) * B, S["N"])
    X, y = S["X"], S["y"]
    return X.iloc[:s0], y[:s0], X.iloc[s0:s1], y[s0:s1]


def build(S, name, Xo, yo, Xn, yn):
    if name == KEEP:
        return S["model"]
    if name == NEWO:
        return fit_new(S, Xn, yn)
    Xa, ya = pd.concat([Xo, Xn]).reset_index(drop=True), np.concatenate([yo, yn])
    if name == BOTH:
        return fit_new(S, Xa, ya)
    if name == WGT:
        return fit_new(S, pd.concat([Xo] + [Xn] * 3).reset_index(drop=True), np.concatenate([yo] + [yn] * 3))
    if name == WIN:
        w = min(len(ya), max(2 * len(yn), 200))
        return fit_new(S, Xa.iloc[-w:], ya[-w:])
    rng = np.random.default_rng(S["seed"])
    if name == TRF:
        return fit_new(S, *S["source"]).update_batches(Xn, yn, MB, 3)
    m = copy.deepcopy(S["model"])
    if name == INC:
        return m.update_batches(Xn, yn, MB, 2)
    if name == ONL:
        return m.update_batches(Xn.iloc[-1500:], yn[-1500:], 1, 1)
    sel = rng.choice(len(yo), min(500, len(yo)), replace=False)          # CONT: replay buffer of old rows
    return m.update_batches(Xn, yn, MB, 2, replay=(Xo.iloc[sel], yo[sel]), rng=rng)


def compute_cands(S):
    """Score every option on the newest labelled rows, which are held out of training."""
    Xo, yo, Xn, yn = get_data(S, True)
    hold = min(int(max(50, 0.25 * len(yn))), len(yn) - 30)
    Xh, yh, Xt, yt = Xn.iloc[-hold:], yn[-hold:], Xn.iloc[:-hold], yn[:-hold]
    can = hasattr(S["model"].est, "partial_fit")
    no_pf = "random forest has no partial_fit"
    names, skipped = [KEEP, NEWO, BOTH, WGT, WIN], []
    if can:
        names += [INC, ONL, CONT]
    else:
        skipped += [(n, no_pf) for n in (INC, ONL, CONT)]
    if can and S["source"] is not None:
        names.append(TRF)
    else:
        skipped.append((TRF, no_pf if not can else S["src_msg"]))
    rows, errs = [], {}
    for n in names:
        try:
            e = row_err(S["task"], yh, build(S, n, Xo, yo, Xt, yt).predict(Xh))
            errs[n] = e
            rows.append({"Strategy": n, "Holdout error": float(e.mean())})
        except Exception as exc:
            skipped.append((n, str(exc)))
    t = pd.DataFrame(rows)
    keep = t.loc[t["Strategy"] == KEEP, "Holdout error"].iloc[0]
    t["Better than keeping (%)"] = (keep - t["Holdout error"]) / max(keep, 1e-12) * 100
    return dict(table=t, errs=errs, hold=hold, n_old=len(yo), n_new=len(yn), skipped=skipped)


def apply_choice(S, name):
    al, c = S["alert"], S["cand"]
    S["comps"].append(c["table"].assign(Decision=len(S["comps"]) + 1))
    if name == KEEP:
        S["snooze_until"] = S["k"] + S["snooze"]
        log(S, "Kept current model", f"no alerts for {S['snooze']} batches")
    else:
        Xo, yo, Xn, yn = get_data(S, False)
        S["model"] = build(S, name, Xo, yo, Xn, yn)
        S["version"] += 1
        S["ref_m"], S["ref_s"] = ref_stats(S["task"], c["errs"][name])
        keep = float(c["errs"][KEEP].mean())
        S["retrains"].append(dict(version=S["version"], strategy=name, applied=S["k"], start=al["start"],
                                  est_new=float(c["errs"][name].mean()), est_old=keep))
        log(S, "Retrained", f"v{S['version']} = {name} (old {c['n_old']:,} rows, new {c['n_new']:,} rows)")
    S["alert"], S["cand"], S["bad"], S["good"] = None, None, [], 0


# ------------------------------------------------------------------ setup
st.title("📈 Model Drift Dashboard")
up = st.file_uploader("Upload a CSV (rows in time order, or pick a time column)", type=["csv"])
if up is None:
    st.stop()
df = read(up.getvalue())

c1, c2, c3, c4 = st.columns(4)
target = c1.selectbox("Target column", df.columns, index=len(df.columns) - 1)
tcol = c2.selectbox("Time column (optional)", ["(row order)"] + [c for c in df.columns if c != target])
family = FAMILIES[c3.selectbox("Model", list(FAMILIES))]
guess = "regression" if (pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique() > 10) else "classification"
task = c4.radio("Task", ["classification", "regression"], index=["classification", "regression"].index(guess), horizontal=True)

with st.expander("Settings"):
    s1, s2, s3, s4, s5, s6 = st.columns(6)
    train_pct = s1.slider("Initial training %", 20, 70, 40, 5)
    B = int(s2.number_input("Batch size (rows)", 20, 5000, 100))
    sens = s3.selectbox("Drift sensitivity", list(SENS), index=1)
    delay = int(s4.number_input("Label delay (batches)", 0, 20, 0))
    min_rows = int(s5.number_input("Min new labelled rows before asking", 100, 100000, 300,
                                   help="After a drift alert, the app waits (and keeps monitoring) until this many post-drift labelled rows exist."))
    snooze = int(s6.number_input("Snooze after 'keep' (batches)", 0, 200, 10))
    src_file = st.file_uploader("Optional: source-domain CSV for transfer learning (same feature columns and target)",
                                type=["csv"], key="src")

if st.button("Start monitoring", type="primary"):
    d = df.copy()
    if tcol != "(row order)":
        tv = pd.to_numeric(d[tcol], errors="coerce") if pd.api.types.is_numeric_dtype(d[tcol]) \
            else pd.to_datetime(d[tcol], errors="coerce", utc=True)
        d = d.loc[tv.notna()].assign(_t=tv[tv.notna()]).sort_values("_t", kind="stable").drop(columns="_t")
    d = d[d[target].notna()].reset_index(drop=True)
    num, cat = pick_features(d, {target, tcol})
    if len(d) < 400 or not (num or cat):
        st.error("Need at least 400 labelled rows and at least one usable feature column."); st.stop()
    X = build_frame(d, num, cat)
    y = d[target].astype(float).to_numpy() if task == "regression" else d[target].astype(str).to_numpy()
    n_tr = int(len(d) * train_pct / 100)
    n_fit = int(n_tr * 0.8)
    source, src_msg = None, "no source CSV was uploaded"
    if src_file is not None:
        try:
            sd = read(src_file.getvalue())
            miss = [c for c in num + cat + [target] if c not in sd.columns]
            if miss:
                src_msg = f"source CSV is missing columns {miss}"
            else:
                sx = build_frame(sd, num, cat)
                if task == "regression":
                    sy = pd.to_numeric(sd[target], errors="coerce").to_numpy(float)
                    ok = ~np.isnan(sy)
                else:
                    sy = sd[target].astype(str).to_numpy()
                    ok = np.isin(sy, np.unique(y))
                sx, sy = sx[ok].reset_index(drop=True), sy[ok]
                if len(sy) < 50:
                    src_msg = f"only {len(sy)} usable source rows (need 50)"
                else:
                    source, src_msg = (sx, sy), f"{len(sy):,} source rows"
        except Exception as exc:
            src_msg = f"could not use the source CSV: {exc}"
    S = dict(X=X, y=y, task=task, family=family, num=num, cat=cat, seed=42, n_tr=n_tr, B=B, N=len(y) - n_tr,
             nb=int(np.ceil((len(y) - n_tr) / B)), z=SENS[sens], delay=delay, min_rows=min_rows, snooze=snooze,
             classes=np.unique(y) if task == "classification" else None,
             version=0, k=0, errs=[], thr=[], ver=[], bad=[], good=0, alert=None, snooze_until=0,
             cand=None, events=[], retrains=[], comps=[], source=source, src_msg=src_msg)
    try:
        S["model"] = Bundle(family, task, num, cat, S["classes"], 42).fit(X.iloc[:n_fit], y[:n_fit])
    except Exception as exc:
        st.error(f"Could not train the first model: {exc}"); st.stop()
    S["ref_m"], S["ref_s"] = ref_stats(task, row_err(task, y[n_fit:n_tr], S["model"].predict(X.iloc[n_fit:n_tr])))
    st.session_state["S"] = S

S = st.session_state.get("S")
if S is None:
    st.info("Choose your columns and click **Start monitoring**.")
    st.stop()


# ------------------------------------------------------------------ controls (run before drawing)
def ready(S):
    return S["alert"] is not None and S["alert"]["status"] == "ready"


def run(S, n):
    i = 0
    while S["k"] < S["nb"] and not ready(S) and (n is None or i < n):
        step(S)
        i += 1


done = S["k"] >= S["nb"]
b1, b2, b3, b4 = st.columns(4)
off = ready(S) or done
if b1.button("Next batch", disabled=off):
    run(S, 1)
if b2.button("Next 10 batches", disabled=off):
    run(S, 10)
if b3.button("Run until a decision is needed", type="primary", disabled=off):
    run(S, None)
if b4.button("Reset"):
    del st.session_state["S"]
    st.rerun()

al = S["alert"]
status = ("Finished" if done and not ready(S) else "Decision needed" if ready(S)
          else "Drift suspected, collecting data" if al else "Monitoring")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Batches received", f"{S['k']} / {S['nb']}")
m2.metric("Deployed model", f"v{S['version']}")
m3.metric("Retrains so far", len(S["retrains"]))
m4.metric("Status", status)

# ------------------------------------------------------------------ main graph
# ------------------------------------------------------------------
# Main graph (safe against mismatched monitoring arrays)
# ------------------------------------------------------------------
if S["k"]:
    fig = go.Figure()

    # Convert monitoring history into arrays
    ver_all = np.asarray(S["ver"])
    errs_all = np.asarray(S["errs"], dtype=float)

    # Ensure every plotted array has the same length
    n = min(S["k"], len(ver_all), len(errs_all), len(S["thr"]))

    xs = np.arange(1, n + 1)
    ver = ver_all[:n]
    errs = errs_all[:n]
    thresholds = np.asarray(S["thr"], dtype=float)[:n]

    for v in sorted(np.unique(ver)):
        idx = np.flatnonzero(ver == v)

        # Safe because idx is generated from the aligned version array
        fig.add_trace(
            go.Scatter(
                x=xs[idx],
                y=errs[idx],
                mode="lines+markers",
                name=f"Model v{v}"
            )
        )

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=thresholds,
            mode="lines",
            name="Drift threshold",
            line=dict(dash="dash", color="#d1342f")
        )
    )

    for r in S["retrains"]:
        batch_x = r["applied"] + 1
        if 1 <= batch_x <= n:
            fig.add_vline(
                x=batch_x,
                line_color="#2e9e5b",
                annotation_text=f"Retrain v{r['version']}"
            )

    if S["alert"] is not None:
        drift_x = S["alert"]["start"] + 1
        if 1 <= drift_x <= n:
            fig.add_vline(
                x=drift_x,
                line_color="#d1342f",
                annotation_text="Drift starts"
            )

    fig.update_layout(
        title="Model drift: error per incoming batch",
        xaxis_title="Batch",
        yaxis_title="MAE" if S["task"] == "regression" else "Error rate",
        height=380
    )

    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------ waiting / decision
if al and al["status"] == "waiting":
    st.info(f"Drift suspected since batch {al['start'] + 1}, but only {new_rows(S):,} of {S['min_rows']:,} new labelled rows "
            "have arrived. Not enough to retrain safely, so monitoring continues. You will be asked once there is enough data "
            "(or the alert clears itself if the error returns to normal).")

if ready(S):
    if S["cand"] is None:
        with st.spinner("Testing every retraining option on the newest labelled rows..."):
            S["cand"] = compute_cands(S)
    c = S["cand"]
    t = c["table"].sort_values("Holdout error").reset_index(drop=True)
    st.warning(f"**Drift confirmed since batch {al['start'] + 1}. What should we retrain on?**  \n"
               f"Old data (before the drift): {c['n_old']:,} rows. New data (since the drift): {c['n_new']:,} rows. "
               f"Each option below was trained without the newest {c['hold']:,} labelled rows and scored on them.")
    st.plotly_chart(go.Figure(go.Bar(
        x=t["Holdout error"], y=t["Strategy"], orientation="h", text=t["Holdout error"].round(4),
        marker_color=["#888" if s == KEEP else "#2e9e5b" for s in t["Strategy"]])).update_layout(
        title="Error on the newest labelled rows (lower is better)", yaxis=dict(autorange="reversed"), height=300))
    if c["skipped"]:
        st.caption("Not available here: " + "; ".join(f"{n} ({why})" for n, why in c["skipped"]))
    choice = st.radio("Apply", list(t["Strategy"]), index=0, key=f"choice_{S['k']}",
                      help="Preselected = lowest error. Pick another option if you prefer.")
    a1, a2 = st.columns([1, 3])
    if a1.button("Apply", type="primary"):
        apply_choice(S, choice)
        st.rerun()
    if a2.button("Skip for now (keep current model)"):
        apply_choice(S, KEEP)
        st.rerun()

# ------------------------------------------------------------------ results after retraining
if S["retrains"]:
    st.subheader("How is each retrained model performing now?")
    rows = []
    for r in S["retrains"]:
        ver = np.array(S["ver"]); errs = np.array(S["errs"])
        before = errs[r["start"]:r["applied"]]
        after = errs[ver == r["version"]]
        rows.append({"Version": f"v{r['version']}", "Strategy": r["strategy"], "Applied at batch": r["applied"] + 1,
                     "Old model error since drift": before.mean() if len(before) else np.nan,
                     "This model's error so far": after.mean() if len(after) else np.nan,
                     "Batches measured": int(len(after))})
    out = pd.DataFrame(rows)
    out["Improvement (%)"] = (out["Old model error since drift"] - out["This model's error so far"]) / out["Old model error since drift"] * 100
    st.dataframe(out.round(4), hide_index=True)

if S["events"]:
    with st.expander("Event log"):
        st.dataframe(pd.DataFrame(S["events"]), hide_index=True)

if S["comps"]:
    st.subheader("Which strategy is best overall?")
    allc = pd.concat(S["comps"])
    lb = (allc.groupby("Strategy").agg(avg_gain=("Better than keeping (%)", "mean"), n=("Decision", "count"))
          .reset_index().sort_values("avg_gain", ascending=False))
    st.plotly_chart(go.Figure(go.Bar(
        x=lb["avg_gain"], y=lb["Strategy"], orientation="h", text=lb["avg_gain"].round(1),
        marker_color=["#888" if n == KEEP else "#2e9e5b" for n in lb["Strategy"]])).update_layout(
        title="Average error reduction vs keeping the current model (%, higher is better)",
        yaxis=dict(autorange="reversed"), height=320))
    top = lb[lb["Strategy"] != KEEP].iloc[0] if (lb["Strategy"] != KEEP).any() else None
    if top is not None and top["avg_gain"] > 0:
        st.success(f"Best so far: **{top['Strategy']}**, averaging {top['avg_gain']:+.1f}% vs keeping the model "
                   f"across {int(top['n'])} drift decision(s). More drift events make this ranking more reliable.")
    else:
        st.info("No retraining strategy has beaten keeping the current model yet.")
