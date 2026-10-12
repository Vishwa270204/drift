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
st.markdown("<style>.block-container{padding-top:1.5rem;padding-bottom:1rem}</style>", unsafe_allow_html=True)

FAMILIES = {"SGD (linear)": "sgd", "MLP (neural net)": "mlp", "Random forest": "rf"}
EPOCHS = {"sgd": 20, "mlp": 30, "rf": 1}
SENS = {"Low (fewer alerts)": 4.0, "Medium": 3.0, "High (more alerts)": 2.0}
MB = 64  # mini-batch size for incremental / continual / transfer updates

# Strategy names. Primary ones are always offered; advanced ones only when they can really run.
KEEP = "Keep current model"
RECENT, NEWO, BOTH, CONT = "Recent window only", "New data only", "Old + new data", "Continual learning"
WGT, INC, ONL, TRF = "Old + new, new weighted 3x", "Incremental update", "Online (one row at a time)", "Transfer learning"
PRIMARY, ADV = [RECENT, NEWO, BOTH, CONT], [WGT, INC, ONL, TRF]
HOW = {
    RECENT: "Last N labelled rows (old + new, in time order) → new model trained from scratch.",
    NEWO: "Only labelled rows since the drift began → new model trained from scratch.",
    BOTH: "All labelled history before the drift + new rows → new model trained from scratch.",
    CONT: "New rows + a random replay sample of old rows. SGD: current model updated with partial_fit on mixed mini-batches. "
          "MLP / random forest: new model trained from scratch on new + replay rows.",
    WGT: "Old + new rows, with each new row repeated 3 times → trained from scratch.",
    INC: "SGD only: current model updated with partial_fit on new rows in mini-batches (no old data).",
    ONL: "SGD only: current model updated one row at a time, single pass over the latest rows.",
    TRF: "Fresh model pretrained on the uploaded source CSV, then fine-tuned on new rows (SGD / MLP).",
}
REG_SEL, CLS_SEL = ["MAE", "RMSE"], ["Error rate", "Recall of class of interest"]


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
    """Preprocessing (fitted on its own training data only) + estimator + regression target scaling."""

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
        y = np.asarray(y)
        if len(y) < 30:
            raise ValueError(f"only {len(y)} usable labelled rows (need 30)")
        if self.task == "classification" and len(np.unique(y)) < 2:
            raise ValueError("training rows contain only one class")
        steps = []
        if self.num:
            steps.append(("n", Pipeline([("i", SimpleImputer(strategy="median", keep_empty_features=True)),
                                         ("s", StandardScaler())]), self.num))
        if self.cat:
            steps.append(("c", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.cat))
        self.pre = ColumnTransformer(steps).fit(X)
        if self.task == "regression":
            self.mu, self.sd = float(y.mean()), float(y.std()) or 1.0
        self.est = make_est(self.family, self.task, self.seed)
        Xt = np.asarray(self.pre.transform(X), float)
        if hasattr(self.est, "partial_fit"):
            self._partial(Xt, y, EPOCHS[self.family])
        else:
            self.est.fit(Xt, self._target(y))
        return self

    def update_batches(self, X, y, bs, epochs, replay=None, rng=None):
        """Sequential partial_fit using this model's existing preprocessing, class list and target scaling."""
        y = np.asarray(y)
        if self.task == "classification":
            keep = np.isin(y, self.classes)
            X, y = X[keep], y[keep]
        if len(y) == 0:
            raise ValueError("no usable labelled rows for the update")
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
        if len(X) == 0:
            return np.array([])
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


# ------------------------------------------------------------------ evaluation metrics
def metrics(S, y, p):
    y, p = np.asarray(y), np.asarray(p)
    if S["task"] == "regression":
        d = y.astype(float) - p.astype(float)
        return {"MAE": float(np.abs(d).mean()), "RMSE": float(np.sqrt((d ** 2).mean()))}
    pos = S["pos"]
    tp, fp, fn = int(((p == pos) & (y == pos)).sum()), int(((p == pos) & (y != pos)).sum()), int(((p != pos) & (y == pos)).sum())
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"Error rate": float((y != p).mean()), "Precision": prec, "Recall": rec, "F1": 2 * prec * rec / max(prec + rec, 1e-12)}


def sel_loss(m, crit):
    return 1 - m["Recall"] if crit == "Recall of class of interest" else m[crit]


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


def prepare_source(sd, num, cat, target, task, classes):
    """Validate an optional source-domain CSV. Returns ((X, y) or None, message)."""
    miss = [c for c in num + cat + [target] if c not in sd.columns]
    if miss:
        return None, f"source CSV is missing columns {miss}"
    sx = build_frame(sd, num, cat)
    if task == "regression":
        sy = pd.to_numeric(sd[target], errors="coerce").to_numpy(float)
        ok = ~np.isnan(sy)
    else:
        sy = sd[target].astype(str).to_numpy()
        ok = np.isin(sy, classes)
    sx, sy = sx[ok].reset_index(drop=True), sy[ok]
    if len(sy) < 50:
        return None, f"only {len(sy)} usable source rows (need 50) with compatible target labels"
    if task == "classification" and len(np.unique(sy)) < 2:
        return None, "source target has fewer than two known classes"
    return (sx, sy), f"{len(sy):,} source rows"


# ------------------------------------------------------------------ streaming engine
def log(S, event, detail):
    S["events"].append({"Batch": S["k"], "Event": event, "Detail": detail})


def new_rows(S):
    """Labelled rows available since the suspected drift start (respects label delay)."""
    al = S["alert"]
    if al is None:
        return 0
    return max(0, min((S["k"] - S["delay"]) * S["B"], S["N"]) - al["start"] * S["B"])


def step(S):
    """Receive one batch: predict with the deployed model; labels of batch k-delay arrive now and feed the detector."""
    k, B, n_tr = S["k"], S["B"], S["n_tr"]
    a, b = n_tr + k * B, min(n_tr + (k + 1) * B, len(S["y"]))
    e = row_err(S["task"], S["y"][a:b], S["model"].predict(S["X"].iloc[a:b]))
    S["errs"].append(float(e.mean()))
    S["thr"].append(S["ref_m"] + S["z"] * S["ref_s"] / np.sqrt(b - a))
    S["ver"].append(S["version"])
    S["k"] = k + 1
    j = k - S["delay"]
    if j >= 0 and S["ver"][j] == S["version"]:          # ignore batches scored by an earlier model version
        flagged = S["errs"][j] > S["thr"][j]
        if S["alert"] is None:
            if S["k"] <= S["snooze_until"]:
                S["bad"] = []                           # snoozed: discard evidence so it cannot fire the moment snooze ends
            else:
                S["bad"] = S["bad"] + [j] if flagged else []
                if len(S["bad"]) >= 2:                  # 2 bad batches in a row = alert
                    S["alert"], S["cand"], S["good"] = dict(start=S["bad"][0], status="waiting"), None, 0
                    S["bad"] = []
                    log(S, "Drift alert", f"errors above threshold since batch {S['alert']['start'] + 1}")
        elif S["alert"]["status"] == "waiting":
            S["good"] = 0 if flagged else S["good"] + 1
            if S["good"] >= 3:
                S["alert"] = None
                log(S, "Alert cleared", "error returned to normal before retraining was needed")
    al = S["alert"]
    if al and al["status"] == "waiting" and new_rows(S) >= S["min_rows"]:
        al["status"] = "ready"


def get_data(S):
    """Chronological split at the drift start: old = labelled rows before it, new = labelled rows since."""
    n_tr, B = S["n_tr"], S["B"]
    s0 = n_tr + S["alert"]["start"] * B
    s1 = n_tr + min((S["k"] - S["delay"]) * B, S["N"])
    X, y = S["X"], S["y"]
    return X.iloc[:s0], y[:s0], X.iloc[s0:s1], y[s0:s1]


# ------------------------------------------------------------------ strategy -> data selection + training procedure
def availability(S):
    """name -> None if it can run, else the reason it cannot."""
    fam, sgd = S["family"], S["family"] == "sgd"
    no_inc = "needs an SGD model (MLP partial_fit is not equivalent to retraining; random forest has none)"
    a = {n: None for n in PRIMARY + [WGT]}
    a[INC] = a[ONL] = None if sgd else no_inc
    a[TRF] = (None if S["source"] is not None else S["src_msg"]) if fam in ("sgd", "mlp") else "needs an SGD or MLP model"
    return a


def build(S, name, Xo, yo, Xn, yn):
    """Return a NEW fitted model (or the deployed one for KEEP). Never mutates the deployed model."""
    rng = np.random.default_rng(S["seed"])
    if name == KEEP:
        return S["model"]
    if name == NEWO:
        return fit_new(S, Xn, yn)
    Xa, ya = pd.concat([Xo, Xn]).reset_index(drop=True), np.concatenate([yo, yn])
    if name == BOTH:
        return fit_new(S, Xa, ya)
    if name == RECENT:
        w = min(S["win"], len(ya))
        return fit_new(S, Xa.iloc[-w:], ya[-w:])
    if name == WGT:
        return fit_new(S, pd.concat([Xo] + [Xn] * 3).reset_index(drop=True), np.concatenate([yo] + [yn] * 3))
    if name == TRF:
        return fit_new(S, *S["source"]).update_batches(Xn, yn, MB, 3)
    if name == CONT:
        sel = np.sort(rng.choice(len(yo), min(S["replay"], len(yo)), replace=False))   # replay sample of history
        if S["family"] == "sgd":                       # supported incremental update on a copy
            return copy.deepcopy(S["model"]).update_batches(Xn, yn, MB, 2, replay=(Xo.iloc[sel], yo[sel]), rng=rng)
        return fit_new(S, pd.concat([Xo.iloc[sel], Xn]).reset_index(drop=True), np.concatenate([yo[sel], yn]))
    m = copy.deepcopy(S["model"])
    if name == INC:
        return m.update_batches(Xn, yn, MB, 2)
    return m.update_batches(Xn.iloc[-1500:], yn[-1500:], 1, 1)   # ONL


def compute_cands(S):
    """Train every runnable strategy on rows BEFORE the newest `hold` rows; score all on those same held-out rows."""
    Xo, yo, Xn, yn = get_data(S)
    hold = min(500, int(max(50, 0.25 * len(yn))), len(yn) - 30)
    if hold < 20 or len(yo) < 30:
        raise ValueError("not enough labelled rows to hold out an evaluation set")
    Xh, yh, Xt, yt = Xn.iloc[-hold:], yn[-hold:], Xn.iloc[:-hold], yn[:-hold]
    crit, notes = S["sel"], []
    if S["task"] == "classification" and crit.startswith("Recall") and (yh == S["pos"]).sum() < 10:
        crit = "Error rate"
        notes.append(f"Fewer than 10 '{S['pos']}' rows in the evaluation set, so recall is unreliable. Selection uses error rate instead.")
    if S["task"] == "classification":
        top = pd.Series(yh).value_counts(normalize=True).iloc[0]
        if top > 0.9:
            notes.append(f"Evaluation set is highly imbalanced ({top:.0%} one class): error rate can look good while the minority class is missed.")
    if hold < 100:
        notes.append(f"Only {hold} evaluation rows: differences between strategies may be sampling noise.")
    avail = availability(S)
    rows, errs, preds, skipped = [], {}, {}, [(n, r) for n, r in avail.items() if r]
    for n in [KEEP] + [x for x in PRIMARY + ADV if avail[x] is None]:
        try:
            p = build(S, n, Xo, yo, Xt, yt).predict(Xh)
            m = metrics(S, yh, p)
            errs[n], preds[n] = row_err(S["task"], yh, p), p
            rows.append({"Strategy": n, "Selection loss": sel_loss(m, crit), **m})
        except Exception as exc:
            skipped.append((n, f"failed: {exc}"))
    t = pd.DataFrame(rows)
    keep = t.loc[t["Strategy"] == KEEP, "Selection loss"].iloc[0]
    t["Better than keeping (%)"] = (keep - t["Selection loss"]) / max(abs(keep), 1e-12) * 100
    return dict(table=t, errs=errs, preds=preds, y_hold=yh, hold=hold, n_old=len(yo), n_new=len(yn),
                skipped=skipped, crit=crit, notes=notes)


def apply_choice(S, name):
    """Returns an error message on failure; the deployed model and version change only on success."""
    c = S["cand"]
    if name != KEEP:
        try:
            Xo, yo, Xn, yn = get_data(S)
            new = build(S, name, Xo, yo, Xn, yn)          # refit on ALL labelled rows (evaluation rows included)
        except Exception as exc:
            return f"Retraining failed ({exc}). The deployed model was not changed."
        al = S["alert"]
        S["model"], S["version"] = new, S["version"] + 1
        S["ref_m"], S["ref_s"] = ref_stats(S["task"], c["errs"][name])   # threshold from the held-out estimate
        S["retrains"].append(dict(version=S["version"], strategy=name, applied=S["k"], start=al["start"]))
        log(S, "Retrained", f"v{S['version']} = {name} (old {c['n_old']:,} rows, new {c['n_new']:,} rows)")
    else:
        S["snooze_until"] = S["k"] + S["snooze"]
        log(S, "Skipped retraining", f"model unchanged; no alerts for {S['snooze']} batches")
    S["comps"].append(c["table"].assign(Decision=len(S["comps"]) + 1))
    S["alert"], S["cand"], S["bad"], S["good"] = None, None, [], 0
    return None


# ------------------------------------------------------------------ configuration
st.title("📈 Model Drift Dashboard")
with st.container(border=True):
    st.subheader("1. Configuration")
    up = st.file_uploader("CSV file (rows in time order, or pick a time column)", type=["csv"])
    if up is None:
        st.stop()
    df = read(up.getvalue())
    c1, c2, c3, c4 = st.columns(4)
    target = c1.selectbox("Target column", df.columns, index=len(df.columns) - 1)
    tcol = c2.selectbox("Time column (optional)", ["(row order)"] + [c for c in df.columns if c != target])
    family = FAMILIES[c3.selectbox("Model", list(FAMILIES))]
    guess = "regression" if (pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique() > 10) else "classification"
    task = c4.radio("Task", ["classification", "regression"], index=["classification", "regression"].index(guess), horizontal=True)
    s1, s2, s3, s4, s5, s6 = st.columns(6)
    train_pct = s1.slider("Initial training %", 20, 70, 40, 5)
    B = int(s2.number_input("Batch size (rows)", 20, 5000, 100))
    sens = s3.selectbox("Drift sensitivity", list(SENS), index=1)
    delay = int(s4.number_input("Label delay (batches)", 0, 20, 0))
    min_rows = int(s5.number_input("Min new labelled rows", 100, 100000, 300,
                                   help="After a drift alert, monitoring continues until this many post-drift labelled rows exist."))
    snooze = int(s6.number_input("Snooze after skipping (batches)", 0, 200, 10))
    with st.expander("Advanced settings"):
        a1, a2, a3 = st.columns(3)
        win = int(a1.number_input("Recent window (rows)", 50, 100000, 500))
        replay = int(a2.number_input("Continual replay sample (old rows)", 50, 20000, 500))
        seed = int(a3.number_input("Random seed", 0, 9999, 42))
        a4, a5 = st.columns(2)
        if task == "classification":
            vc = df[target].dropna().astype(str).value_counts()
            opts = sorted(vc.index)[:50]
            dflt = vc.index[-1] if len(vc) == 2 else vc.index[0]
            pos = a5.selectbox("Class of interest (precision / recall / F1)", opts, index=opts.index(dflt) if dflt in opts else 0)
            sel = a4.selectbox("Strategy selection criterion", CLS_SEL, help="Lower loss wins. Recall criterion uses 1 - recall.")
        else:
            pos, sel = None, a4.selectbox("Strategy selection criterion", REG_SEL)
        src_file = st.file_uploader("Optional source-domain CSV for transfer learning (same feature columns and target)",
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
        classes = np.unique(y) if task == "classification" else None   # assumption: label space is known up front
        n_tr = int(len(d) * train_pct / 100)
        n_fit = int(n_tr * 0.8)
        source, src_msg = None, "no source CSV was uploaded"
        if src_file is not None:
            try:
                source, src_msg = prepare_source(read(src_file.getvalue()), num, cat, target, task, classes)
            except Exception as exc:
                src_msg = f"could not use the source CSV: {exc}"
        S = dict(X=X, y=y, task=task, family=family, num=num, cat=cat, seed=seed, n_tr=n_tr, B=B, N=len(y) - n_tr,
                 nb=int(np.ceil((len(y) - n_tr) / B)), z=SENS[sens], delay=delay, min_rows=min_rows, snooze=snooze,
                 win=win, replay=replay, sel=sel, pos=pos, classes=classes, source=source, src_msg=src_msg,
                 version=0, k=0, errs=[], thr=[], ver=[], bad=[], good=0, alert=None, snooze_until=0,
                 cand=None, events=[], retrains=[], comps=[])
        try:
            S["model"] = Bundle(family, task, num, cat, classes, seed).fit(X.iloc[:n_fit], y[:n_fit])
            S["ref_m"], S["ref_s"] = ref_stats(task, row_err(task, y[n_fit:n_tr], S["model"].predict(X.iloc[n_fit:n_tr])))
        except Exception as exc:
            st.error(f"Could not train the first model: {exc}"); st.stop()
        st.session_state["S"] = S
        st.session_state.pop("confirm_reset", None)

S = st.session_state.get("S")
if S is None:
    st.info("Choose your columns and click **Start monitoring**.")
    st.stop()
ERR = "MAE" if S["task"] == "regression" else "Error rate"


def ready(S):
    return S["alert"] is not None and S["alert"]["status"] == "ready"


def run(S, n):
    i = 0
    while S["k"] < S["nb"] and not ready(S) and (n is None or i < n):
        step(S)
        i += 1


# ------------------------------------------------------------------ monitoring controls + metrics
with st.container(border=True):
    st.subheader("2. Monitoring")
    done = S["k"] >= S["nb"]
    off = ready(S) or done
    b1, b2, b3, b4 = st.columns([3, 2, 2, 1.5])
    if b1.button("▶ Run until a decision is needed", type="primary", disabled=off, use_container_width=True):
        run(S, None)
    if b2.button("Next batch", disabled=off, use_container_width=True):
        run(S, 1)
    if b3.button("Next 10 batches", disabled=off, use_container_width=True):
        run(S, 10)
    if b4.button("Reset", use_container_width=True):
        if S["k"] > 0:
            st.session_state["confirm_reset"] = True
        else:
            del st.session_state["S"]; st.rerun()
    if st.session_state.get("confirm_reset"):
        st.warning("Reset the running session? Progress, model versions and history will be lost.")
        r1, r2, _ = st.columns([1, 1, 4])
        if r1.button("Yes, reset"):
            del st.session_state["S"]; st.session_state.pop("confirm_reset"); st.rerun()
        if r2.button("Cancel"):
            st.session_state.pop("confirm_reset"); st.rerun()

    al = S["alert"]
    status = ("Finished" if done and not ready(S) else "Decision needed" if ready(S)
              else "Drift suspected, collecting data" if al else "Monitoring")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Batches received", f"{S['k']} / {S['nb']}")
    m2.metric("Deployed model", f"v{S['version']}")
    m3.metric("Retraining events", len(S["retrains"]))
    m4.metric("Status", status)

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
    if al and al["status"] == "waiting":
        st.info(f"Drift suspected since batch {al['start'] + 1}, but only {new_rows(S):,} of {S['min_rows']:,} new labelled rows have "
                "arrived. Monitoring continues; you will be asked once there is enough data, or the alert clears if error returns to normal.")

# ------------------------------------------------------------------ decision panel
if ready(S):
    if S["cand"] is None:
        try:
            with st.spinner("Evaluating every runnable strategy on the same held-out rows..."):
                S["cand"] = compute_cands(S)
        except Exception as exc:
            st.error(f"Could not evaluate strategies: {exc}")
            if st.button("Skip retraining (keep current model)"):
                S["cand"] = dict(table=pd.DataFrame(), errs={}, n_old=0, n_new=0)
                apply_choice(S, KEEP); st.rerun()
            st.stop()
    c, t = S["cand"], S["cand"]["table"].sort_values("Selection loss").reset_index(drop=True)
    rec, k_loss = t.iloc[0]["Strategy"], float(t.loc[t["Strategy"] == KEEP, "Selection loss"].iloc[0])
    with st.container(border=True):
        st.subheader("3. Drift confirmed: decide how to adapt")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Drift start batch", al["start"] + 1)
        d2.metric("Historical labelled rows", f"{c['n_old']:,}")
        d3.metric("New labelled rows", f"{c['n_new']:,}")
        d4.metric("Held-out evaluation rows", f"{c['hold']:,}")
        lab = f"{c['crit']} (lower is better)" if c["crit"] in ("Error rate", "MAE", "RMSE") else "1 − recall of class of interest (lower is better)"
        fig = go.Figure(go.Bar(x=t["Selection loss"], y=t["Strategy"], orientation="h", text=t["Selection loss"].round(4),
                               marker_color=["#888" if s == KEEP else "#5b8def" if s in ADV else "#2e9e5b" for s in t["Strategy"]]))
        fig.update_layout(title="Candidate comparison on the same held-out rows", xaxis_title=lab,
                          yaxis=dict(autorange="reversed"), height=60 + 38 * len(t), margin=dict(t=45, b=30))
        st.plotly_chart(fig)
        if rec == KEEP:
            st.info("**Recommendation: skip retraining.** No candidate beat the current model on the held-out rows.")
        else:
            gain = t.iloc[0]["Better than keeping (%)"]
            st.success(f"**Recommended: {rec}.** It had the lowest {c['crit']} on the held-out rows ({t.iloc[0]['Selection loss']:.4g} vs "
                       f"{k_loss:.4g} for the current model, {gain:.1f}% lower). This is measured on recent data only and does not "
                       "guarantee better performance on future data.")
        for n in c["notes"]:
            st.warning(n)
        if c["skipped"]:
            st.caption("Not offered: " + "; ".join(f"{n} ({why})" for n, why in c["skipped"]))

        avail_prim = [n for n in PRIMARY if n in set(t["Strategy"])]
        avail_adv = [n for n in ADV if n in set(t["Strategy"])]
        pk = S["k"]
        prim = st.radio("Retraining strategy", avail_prim, index=avail_prim.index(rec) if rec in avail_prim else 0,
                        key=f"prim_{pk}", horizontal=True)
        choice = prim
        if avail_adv:
            with st.expander("Advanced strategies", expanded=rec in avail_adv):
                dflt = "(use the strategy selected above)"
                adv = st.radio("Override with an advanced strategy", [dflt] + avail_adv, key=f"adv_{pk}",
                               index=([dflt] + avail_adv).index(rec) if rec in avail_adv else 0)
                choice = prim if adv == dflt else adv
        st.caption(f"**{choice}:** {HOW[choice]}")
        row = t[t["Strategy"] == choice].iloc[0]
        show = [x for x in ("Error rate", "Precision", "Recall", "F1", "MAE", "RMSE") if x in t.columns]
        st.caption("Held-out metrics for this option: " + ", ".join(f"{x} {row[x]:.4g}" for x in show))
        if S["task"] == "classification":
            with st.expander("Confusion matrix on held-out rows for the selected option"):
                labs = sorted(set(c["y_hold"]) | set(c["preds"][choice]))
                cm = pd.crosstab(pd.Categorical(c["y_hold"], labs), pd.Categorical(c["preds"][choice], labs), dropna=False)
                st.plotly_chart(go.Figure(go.Heatmap(z=cm.values, x=labs, y=labs, text=cm.values, texttemplate="%{text}",
                                                     colorscale="Blues")).update_layout(xaxis_title="Predicted", yaxis_title="Actual",
                                                                                      yaxis=dict(autorange="reversed"), height=300))
        with st.expander("How each strategy trains"):
            for n in PRIMARY + ADV:
                st.markdown(f"- **{n}:** {HOW[n]}")
        e1, e2, _ = st.columns([1, 2, 4])
        if e1.button("Apply", type="primary"):
            msg = apply_choice(S, choice)
            if msg:
                st.error(msg)
            else:
                st.rerun()
        if e2.button("Skip retraining (keep current model)"):
            apply_choice(S, KEEP)
            st.rerun()

# ------------------------------------------------------------------ results
if S["retrains"]:
    with st.container(border=True):
        st.subheader("4. Post-retraining performance")
        ver, errs, rows = np.array(S["ver"]), np.array(S["errs"]), []
        for r in S["retrains"]:
            before, after = errs[r["start"]:r["applied"]], errs[ver == r["version"]]
            rows.append({"Version": f"v{r['version']}", "Strategy": r["strategy"], "Applied at batch": r["applied"] + 1,
                         f"Previous model {ERR} since drift": before.mean() if len(before) else np.nan,
                         f"This model {ERR} so far": after.mean() if len(after) else np.nan, "Batches measured": int(len(after))})
        out = pd.DataFrame(rows)
        a_, b_ = out.columns[3], out.columns[4]
        out["Improvement (%)"] = (out[a_] - out[b_]) / out[a_] * 100
        st.dataframe(out.round(4), hide_index=True)
        st.caption(f"{ERR} is the per-batch prediction error of the deployed model on live batches (labels counted once they arrive).")

if S["comps"]:
    with st.expander("Historical strategy comparison (all decisions so far)"):
        allc = pd.concat(S["comps"])
        lb = (allc.groupby("Strategy").agg(avg_gain=("Better than keeping (%)", "mean"), n=("Decision", "count"))
              .reset_index().sort_values("avg_gain", ascending=False))
        st.plotly_chart(go.Figure(go.Bar(
            x=lb["avg_gain"], y=lb["Strategy"], orientation="h", text=lb["avg_gain"].round(1),
            marker_color=["#888" if n == KEEP else "#2e9e5b" for n in lb["Strategy"]])).update_layout(
            title="Average improvement over keeping the model, using each decision's own criterion (%, higher is better)",
            yaxis=dict(autorange="reversed"), height=60 + 38 * len(lb)))
        top = lb[lb["Strategy"] != KEEP].iloc[0] if (lb["Strategy"] != KEEP).any() else None
        if top is not None and top["avg_gain"] > 0:
            st.success(f"Best so far: **{top['Strategy']}** ({top['avg_gain']:+.1f}% on average over {int(top['n'])} decision(s)). "
                       "Few decisions means a weak ranking.")
        else:
            st.info("No retraining strategy has beaten keeping the current model yet.")

if S["events"]:
    with st.expander("Event log", expanded=False):
        st.dataframe(pd.DataFrame(S["events"]), hide_index=True)
