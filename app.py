
import copy
import hashlib
import io
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, precision_recall_fscore_support, r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="Got `batch_size`")

st.set_page_config(page_title="CSV Drift & Learning Strategy Lab", page_icon="📈", layout="wide")

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
NONE_LABEL = "(none)"
MAX_ONEHOT = 50          # categorical features with more levels are not one-hot encoded for models
PSI_EPS = 1e-4           # proportion floor so empty bins do not produce log(0)
FAMILIES = {
    "SGD (linear model, supports partial_fit)": "sgd",
    "MLP neural network (supports partial_fit)": "mlp",
    "Random forest (no partial_fit)": "rf",
}
FAMILY_NAME = {v: k for k, v in FAMILIES.items()}
EPOCHS = {"sgd": 20, "mlp": 30, "rf": 1}
IMPROVE_COL = "Change vs baseline (%, positive = better)"
HIGHER_BETTER = {
    "MAE": False, "RMSE": False, "R²": True, "Accuracy": True,
    "Precision (weighted)": True, "Recall (weighted)": True,
    "F1 (weighted)": True, "F1 (macro)": True,
}
REG_METRICS = ["MAE", "RMSE", "R²"]
CLS_METRICS = ["F1 (weighted)", "Accuracy", "F1 (macro)"]
STATUS_COLORS = {"Stable": "#2e9e5b", "Moderate": "#e0a100", "Significant": "#d1342f", "N/A": "#888888"}
POLICY_MONITOR = "Monitor and alert only (no model changes)"
POLICY_PARTIAL = "After an alert, update a shadow candidate with arriving labels (partial_fit)"
POLICY_RETRAIN = "On an alert, retrain a shadow candidate from scratch on labelled data so far"


# --------------------------------------------------------------------------------------
# Loading, profiling and ordering
# --------------------------------------------------------------------------------------
def parse_datetime(s):
    """Parse to naive UTC datetimes; unparseable values become NaT."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            out = pd.to_datetime(s, errors="coerce", format="mixed", utc=True)
        except (TypeError, ValueError):
            out = pd.to_datetime(s, errors="coerce", utc=True)
    return out.dt.tz_convert(None)


def _integer_like(values):
    return values.size > 0 and bool(np.all(np.mod(values, 1) == 0))


def prepare_dataframe(raw):
    """Return (clean copy, per-column info). Nothing is dropped; problems are flagged in notes."""
    df = raw.copy()
    df.columns = [str(c) for c in df.columns]
    records = []
    for col in df.columns:
        s = df[col]
        notes = []
        orig_dtype = str(s.dtype)
        if int(s.notna().sum()) == 0:
            kind = "empty"
        elif pd.api.types.is_bool_dtype(s):
            kind = "boolean"
        elif pd.api.types.is_numeric_dtype(s):
            kind = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(s):
            kind = "datetime"
        else:
            n_nn = int(s.notna().sum())
            num = pd.to_numeric(s, errors="coerce")
            ok = int(num.notna().sum())
            if ok / n_nn >= 0.9:
                kind = "numeric"
                bad = n_nn - ok
                notes.append(f"{bad} non-numeric entries set to missing" if bad else "numeric values stored as text")
                df[col] = num
            else:
                sample = s.dropna().astype(str).head(200)
                is_dt = len(sample) >= 5 and parse_datetime(sample).notna().mean() >= 0.9
                kind = "datetime-text" if is_dt else "categorical"
        if kind == "numeric" and pd.api.types.is_float_dtype(df[col]):
            n_inf = int(np.isinf(df[col].to_numpy(dtype=float)).sum())
            if n_inf:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                notes.append(f"{n_inf} infinite values set to missing")
        s = df[col]
        n_nn = int(s.notna().sum())
        n_unique = int(s.nunique(dropna=True))
        constant = n_nn > 0 and n_unique <= 1
        id_like = False
        if n_nn >= 20 and n_unique / n_nn > 0.9:
            if kind == "categorical":
                id_like = True
            elif kind == "numeric":
                lower = col.strip().lower()
                hint = lower in ("id", "index", "unnamed: 0") or lower.endswith("_id") or lower.startswith("id_")
                id_like = _integer_like(s.dropna().to_numpy(dtype=float)) and (n_unique / n_nn > 0.99 or hint)
        if constant:
            notes.append("constant column")
        if kind == "empty":
            notes.append("entirely missing")
        elif len(df) and (1 - n_nn / len(df)) > 0.5:
            notes.append("more than 50% missing")
        if id_like:
            notes.append("identifier-like / very high cardinality")
        records.append({
            "Column": col, "Pandas dtype": orig_dtype,
            "Detected type": kind, "Missing": len(df) - n_nn,
            "Missing (%)": round(100 * (len(df) - n_nn) / max(len(df), 1), 2),
            "Unique": n_unique, "Constant": constant, "Identifier-like": id_like,
            "Notes": "; ".join(notes),
        })
    return df, pd.DataFrame(records)


@st.cache_data(show_spinner="Reading and profiling the CSV...")
def load_and_prepare(data):
    last_err = None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            raw = pd.read_csv(io.BytesIO(data), encoding=enc)
            if raw.shape[1] <= 1:  # maybe a different delimiter
                try:
                    alt = pd.read_csv(io.BytesIO(data), encoding=enc, sep=None, engine="python")
                    if alt.shape[1] > raw.shape[1]:
                        raw = alt
                except Exception:
                    pass
            df, info = prepare_dataframe(raw)
            return df, info, enc, int(raw.duplicated().sum())
        except Exception as exc:  # parsing problems are reported to the user
            last_err = exc
    raise ValueError(f"Could not parse the file as CSV: {last_err}")


def order_by_time(df, time_col, kind):
    """Sort by the chosen time column (stable). Returns (frame, time values or None, n dropped, was sorted)."""
    if time_col is None:
        return df.reset_index(drop=True), None, 0, True
    tv = pd.to_numeric(df[time_col], errors="coerce") if kind == "numeric" else parse_datetime(df[time_col])
    valid = tv.notna().to_numpy()
    n_bad = int((~valid).sum())
    out = df.loc[valid].reset_index(drop=True)
    tv = tv[valid].reset_index(drop=True)
    was_sorted = bool(tv.is_monotonic_increasing)
    order = tv.argsort(kind="stable").to_numpy()
    return out.iloc[order].reset_index(drop=True), tv.iloc[order].reset_index(drop=True), n_bad, was_sorted


def family_of(kind):
    if kind == "numeric":
        return "num"
    if kind in ("categorical", "boolean"):
        return "cat"
    return None


def infer_task(s, kind):
    if kind in ("categorical", "boolean"):
        return "classification"
    vals = s.dropna().to_numpy(dtype=float)
    nun = len(np.unique(vals))
    if nun <= 2:
        return "classification"
    if _integer_like(vals) and (nun <= 10 or (nun <= 50 and nun / len(vals) < 0.02)):
        return "classification"
    return "regression"


def leakage_suspects(df, features, target):
    """Features that are (near-)copies of the target."""
    out = []
    tgt_num = pd.to_numeric(df[target], errors="coerce")
    for f in features:
        try:
            if df[f].astype(str).equals(df[target].astype(str)):
                out.append(f)
                continue
            f_num = pd.to_numeric(df[f], errors="coerce")
            both = f_num.notna() & tgt_num.notna()
            if both.sum() > 10 and f_num[both].std() > 0 and tgt_num[both].std() > 0:
                if abs(np.corrcoef(f_num[both], tgt_num[both])[0, 1]) > 0.999:
                    out.append(f)
        except Exception:
            continue
    return out


def csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def vline(fig, x, text, color="#d1342f"):
    fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=1, xref="x", yref="paper",
                  line=dict(color=color, dash="dash", width=1))
    fig.add_annotation(x=x, y=1, yref="paper", text=text, showarrow=False,
                       yanchor="bottom", font=dict(size=10, color=color))


# --------------------------------------------------------------------------------------
# PSI drift
# --------------------------------------------------------------------------------------
def status_of(psi, lo, hi):
    if psi is None or np.isnan(psi):
        return "N/A"
    return "Stable" if psi < lo else ("Moderate" if psi < hi else "Significant")


def numeric_edges(ref, n_bins):
    vals = ref.dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return None
    edges = np.unique(np.quantile(vals, np.linspace(0, 1, n_bins + 1)[1:-1]))
    return edges if edges.size else np.array([vals[0]])


def numeric_labels(edges):
    labs = [f"≤ {edges[0]:.4g}"]
    labs += [f"({edges[i]:.4g}, {edges[i + 1]:.4g}]" for i in range(len(edges) - 1)]
    labs += [f"> {edges[-1]:.4g}", "Missing"]
    return labs


def numeric_counts(s, edges):
    x = s.to_numpy(dtype=float)
    miss = np.isnan(x)
    idx = np.digitize(x[~miss], edges, right=True)
    return np.append(np.bincount(idx, minlength=len(edges) + 1), miss.sum())


def cat_counts(s, levels):
    miss = s.isna()
    vals = s[~miss].astype(str)
    vc = vals.value_counts()
    c = [int(vc.get(level, 0)) for level in levels]
    return np.array(c + [int(len(vals) - sum(c)), int(miss.sum())])


def build_reference(ref_df, columns, kinds, n_bins):
    """Per-feature reference bins/levels and counts from the reference window only."""
    spec = {}
    for col in columns:
        fam = family_of(kinds[col])
        if fam == "num":
            edges = numeric_edges(ref_df[col], n_bins)
            if edges is not None:
                spec[col] = ("num", edges, numeric_counts(ref_df[col], edges), numeric_labels(edges))
        elif fam == "cat" and ref_df[col].notna().any():
            levels = list(ref_df[col].dropna().astype(str).value_counts().head(20).index)
            spec[col] = ("cat", levels, cat_counts(ref_df[col], levels), levels + ["Other / unseen", "Missing"])
    return spec


def counts_for(entry, s):
    return numeric_counts(s, entry[1]) if entry[0] == "num" else cat_counts(s, entry[1])


def psi_value(ref_counts, cmp_counts):
    r, c = ref_counts.astype(float), cmp_counts.astype(float)
    if r.sum() == 0 or c.sum() == 0:
        return np.nan, np.zeros_like(r)
    pr, pc = np.clip(r / r.sum(), PSI_EPS, None), np.clip(c / c.sum(), PSI_EPS, None)
    contrib = (pc - pr) * np.log(pc / pr)
    return float(contrib.sum()), contrib


def compute_drift(ref_df, cmp_df, columns, kinds, n_bins, lo, hi):
    spec = build_reference(ref_df, columns, kinds, n_bins)
    rows, details = [], {}
    for col in columns:
        if col not in spec:
            rows.append({"Feature": col, "Type": family_of(kinds[col]) or kinds[col], "PSI": np.nan,
                         "Status": "N/A", "Note": "no valid reference values or unsupported type"})
            continue
        entry = spec[col]
        cc = counts_for(entry, cmp_df[col])
        psi, contrib = psi_value(entry[2], cc)
        row = {"Feature": col, "Type": "numeric" if entry[0] == "num" else "categorical", "PSI": psi,
               "Status": status_of(psi, lo, hi),
               "Missing % (reference)": round(100 * ref_df[col].isna().mean(), 2),
               "Missing % (comparison)": round(100 * cmp_df[col].isna().mean(), 2)}
        if entry[0] == "num":
            r = ref_df[col].dropna().to_numpy(dtype=float)
            c = cmp_df[col].dropna().to_numpy(dtype=float)
            sd = r.std() if r.size else 0
            row["Mean shift (reference std units)"] = round((c.mean() - r.mean()) / sd, 3) if (sd > 0 and c.size) else np.nan
        else:
            nn = max(int(cc[:-1].sum()), 1)
            row["Unseen / rare categories in comparison (%)"] = round(100 * cc[-2] / nn, 2)
        rows.append(row)
        details[col] = pd.DataFrame({
            "Bin": entry[3],
            "Reference (%)": 100 * entry[2] / max(entry[2].sum(), 1),
            "Comparison (%)": 100 * cc / max(cc.sum(), 1),
            "PSI contribution": contrib,
        })
    res = pd.DataFrame(rows)
    total = res["PSI"].sum(skipna=True)
    res["Share of total PSI (%)"] = (100 * res["PSI"] / total).round(1) if total > 0 else 0.0
    res = res.sort_values("PSI", ascending=False, na_position="last").reset_index(drop=True)
    return res, details


# --------------------------------------------------------------------------------------
# Models (preprocessing is fitted on training data only)
# --------------------------------------------------------------------------------------
def model_frame(lf, num_cols, cat_cols):
    parts = {c: lf[c].astype(float).to_numpy() for c in num_cols}
    for c in cat_cols:
        s = lf[c]
        parts[c] = np.where(s.notna(), s.astype(str), "(missing)")
    return pd.DataFrame(parts, index=lf.index)


def make_estimator(family, task, seed):
    reg = task == "regression"
    if family == "sgd":
        if reg:
            return SGDRegressor(alpha=1e-4, learning_rate="constant", eta0=0.01, random_state=seed)
        return SGDClassifier(loss="log_loss", alpha=1e-4, learning_rate="constant", eta0=0.01, random_state=seed)
    if family == "mlp":
        kw = dict(hidden_layer_sizes=(64, 32), learning_rate_init=0.003, batch_size="auto", random_state=seed)
        return MLPRegressor(**kw) if reg else MLPClassifier(**kw)
    if reg:
        return RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=seed, n_jobs=1)
    return RandomForestClassifier(n_estimators=100, min_samples_leaf=2, random_state=seed, n_jobs=1)


class ModelBundle:
    """Preprocessing + estimator (+ target scaling for regression)."""

    def __init__(self, family, task, num_cols, cat_cols, classes, seed):
        self.family, self.task, self.seed = family, task, seed
        self.classes = None if classes is None else np.asarray(classes)
        steps = []
        if num_cols:
            steps.append(("num", Pipeline([
                ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("sc", StandardScaler())]), list(num_cols)))
        if cat_cols:
            steps.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(cat_cols)))
        self.pre = ColumnTransformer(steps, remainder="drop")
        self.est = make_estimator(family, task, seed)
        self.mu, self.sd = 0.0, 1.0

    @property
    def supports_partial(self):
        return hasattr(self.est, "partial_fit")

    def fit_preprocessing(self, X, y):
        self.pre.fit(X)
        if self.task == "regression":
            y = np.asarray(y, dtype=float)
            self.mu = float(y.mean())
            sd = float(y.std())
            self.sd = sd if sd > 1e-12 else 1.0

    def transform(self, X):
        return np.asarray(self.pre.transform(X), dtype=float)

    def _target(self, y):
        if self.task == "regression":
            return (np.asarray(y, dtype=float) - self.mu) / self.sd
        return np.asarray(y)

    def update_arrays(self, Xt, y, epochs=1):
        """partial_fit on already-transformed rows (only for estimators that support it)."""
        if not self.supports_partial:
            raise RuntimeError(f"{type(self.est).__name__} does not support partial_fit")
        y = np.asarray(y)
        if self.task == "classification" and self.classes is not None:
            keep = np.isin(y, self.classes)
            if not keep.all():
                Xt, y = Xt[keep], y[keep]
        if len(y) == 0:
            return
        yy = self._target(y)
        for _ in range(epochs):
            if self.task == "classification" and not hasattr(self.est, "classes_"):
                self.est.partial_fit(Xt, yy, classes=self.classes)
            else:
                self.est.partial_fit(Xt, yy)

    def train(self, X, y, epochs=None):
        """Initial training: epochs of partial_fit if supported, otherwise a normal fit."""
        Xt, y = self.transform(X), np.asarray(y)
        if self.supports_partial:
            self.update_arrays(Xt, y, epochs or EPOCHS[self.family])
        else:
            if self.task == "classification" and self.classes is not None:
                keep = np.isin(y, self.classes)
                Xt, y = Xt[keep], y[keep]
            self.est.fit(Xt, self._target(y))

    def predict(self, X):
        p = self.est.predict(self.transform(X))
        if self.task == "regression":
            return p * self.sd + self.mu
        return np.asarray(p).astype(str)


def compute_metrics(task, y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if task == "regression":
        try:
            r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan
        except Exception:
            r2 = np.nan
        return {"MAE": float(mean_absolute_error(y_true, y_pred)),
                "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))), "R²": r2}
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {"Accuracy": float(accuracy_score(y_true, y_pred)), "Precision (weighted)": float(p),
            "Recall (weighted)": float(r), "F1 (weighted)": float(f),
            "F1 (macro)": float(f1_score(y_true, y_pred, average="macro", zero_division=0))}


def row_errors(task, y_true, y_pred):
    if task == "regression":
        return np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))
    return (np.asarray(y_true) != np.asarray(y_pred)).astype(float)


def paired_bootstrap_gain(task, metric, y, p_base, p_other, n_boot=100, seed=0):
    """95% CI of (other - baseline) on `metric`, flipped so positive = better."""
    rng = np.random.default_rng(seed)
    sign = 1.0 if HIGHER_BETTER[metric] else -1.0
    n = len(y)
    gains = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        mb = compute_metrics(task, y[idx], p_base[idx])[metric]
        mo = compute_metrics(task, y[idx], p_other[idx])[metric]
        gains.append(sign * (mo - mb))
    gains = np.array(gains)
    gains = gains[~np.isnan(gains)]
    if gains.size == 0:
        return np.nan, np.nan
    return float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))


def mean_ci(e, n_boot=300, seed=0):
    rng = np.random.default_rng(seed)
    e = np.asarray(e, dtype=float)
    boots = e[rng.integers(0, len(e), (n_boot, len(e)))].mean(axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _split(idx, test_size, strat, seed):
    try:
        return train_test_split(idx, test_size=test_size, random_state=seed,
                                stratify=None if strat is None else strat[idx])
    except ValueError:
        return train_test_split(idx, test_size=test_size, random_state=seed)


def make_partitions(n, chrono, train_pct, adapt_pct, seed, y, task):
    """Train / adaptation / held-out test. Chronological when order is meaningful, else seeded random."""
    n_tr, n_ad = int(n * train_pct / 100), int(n * adapt_pct / 100)
    idx = np.arange(n)
    if chrono:
        return idx[:n_tr], idx[n_tr:n_tr + n_ad], idx[n_tr + n_ad:]
    strat = y if task == "classification" else None
    rest, test = _split(idx, n - n_tr - n_ad, strat, seed)
    train, adapt = _split(rest, n_ad, strat, seed)
    return np.sort(train), np.sort(adapt), np.sort(test)


@st.cache_resource(show_spinner="Training baseline model...", max_entries=3)
def train_baseline(_X, _y, _x, key, family, task, num_cols, cat_cols, chrono,
                   train_pct, adapt_pct, seed, max_train_rows):
    n = len(_y)
    tr, ad, te = make_partitions(n, chrono, train_pct, adapt_pct, seed, _y, task)
    if len(np.unique(_y[tr])) < 2 and task == "classification":
        raise ValueError("The training partition contains only one class; a classifier cannot be trained.")
    rng = np.random.default_rng(seed)
    tr_fit = np.sort(rng.choice(tr, size=max_train_rows, replace=False)) if len(tr) > max_train_rows else tr
    classes = np.unique(np.concatenate([_y[tr], _y[ad]])) if task == "classification" else None
    bundle = ModelBundle(family, task, num_cols, cat_cols, classes, seed)
    bundle.fit_preprocessing(_X.iloc[tr_fit], _y[tr_fit])
    bundle.train(_X.iloc[tr_fit], _y[tr_fit])
    pred_te, pred_tr = bundle.predict(_X.iloc[te]), bundle.predict(_X.iloc[tr_fit])
    if task == "regression":
        naive_pred, naive_desc = np.full(len(te), _y[tr_fit].mean()), "always predict the training mean"
    else:
        vals, cnt = np.unique(_y[tr_fit], return_counts=True)
        naive_pred = np.full(len(te), vals[np.argmax(cnt)])
        naive_desc = f"always predict the majority training class ('{vals[np.argmax(cnt)]}')"
    out = {
        "bundle": bundle, "parts": (tr, ad, te), "tr_fit": tr_fit, "classes": classes,
        "pred_test": pred_te, "pred_train": pred_tr,
        "metrics_test": compute_metrics(task, _y[te], pred_te),
        "metrics_train": compute_metrics(task, _y[tr_fit], pred_tr),
        "naive_metrics": compute_metrics(task, _y[te], naive_pred), "naive_desc": naive_desc,
        "stream": None,
    }
    if chrono:
        sidx = np.concatenate([ad, te])
        sp = bundle.predict(_X.iloc[sidx])
        out["stream"] = pd.DataFrame({
            "pos": sidx, "x": _x.iloc[sidx].to_numpy(), "y_true": _y[sidx], "y_pred": sp,
            "error": row_errors(task, _y[sidx], sp),
            "Partition": np.where(np.isin(sidx, te), "Held-out test", "Adaptation"),
        })
    return out


# --------------------------------------------------------------------------------------
# Concept drift detector (simplified ADWIN-style)
# --------------------------------------------------------------------------------------
class SimpleADWIN:
    """Adaptive-window change detector on a stream bounded in [0, 1] (variance-aware Hoeffding-type bound).

    Simplified re-implementation: keeps at most `max_window` recent values, checks every
    `check_every` updates, and when two sub-windows have significantly different means it
    drops the older part and signals a change.
    """

    def __init__(self, delta=0.002, max_window=1000, min_sub=30, check_every=8):
        self.delta, self.max_window, self.min_sub, self.check_every = delta, max_window, min_sub, check_every
        self.window, self.t = [], 0

    def update(self, x):
        self.window.append(float(x))
        if len(self.window) > self.max_window:
            del self.window[0]
        self.t += 1
        n = len(self.window)
        if self.t % self.check_every or n < 2 * self.min_sub:
            return False
        arr = np.asarray(self.window)
        csum = np.cumsum(arr)
        ks = np.arange(self.min_sub, n - self.min_sub + 1)
        n0 = ks.astype(float)
        n1 = n - n0
        gap_mean = np.abs(csum[ks - 1] / n0 - (csum[-1] - csum[ks - 1]) / n1)
        mr = 1.0 / n0 + 1.0 / n1
        dd = np.log(2.0 * np.log(max(n, 3)) / self.delta)
        eps = np.sqrt(2.0 * mr * arr.var() * dd) + (2.0 / 3.0) * dd * mr
        j = int(np.argmax(gap_mean - eps))
        if gap_mean[j] - eps[j] > 0:
            del self.window[:int(ks[j])]
            return True
        return False


def normalize_errors(task, err, ref_len):
    """Map errors into [0, 1]. Regression: divide by 4x the mean error of the first rows, then clip."""
    err = np.asarray(err, dtype=float)
    if task == "classification":
        return err
    scale = 4.0 * err[:max(ref_len, 1)].mean() + 1e-12
    return np.clip(err / scale, 0.0, 1.0)


def run_adwin(norm_err, delta):
    det = SimpleADWIN(delta)
    return [i for i, e in enumerate(norm_err) if det.update(e)]


# --------------------------------------------------------------------------------------
# Learning strategy comparison
# --------------------------------------------------------------------------------------
def prepare_source(src_bytes, num_cols, cat_cols, target, task, classes, max_rows, seed):
    """Validate an optional source-domain CSV for transfer learning. Returns (X, y, message)."""
    raw, _, _, _ = load_and_prepare(src_bytes)
    need = list(num_cols) + list(cat_cols) + [target]
    missing = [c for c in need if c not in raw.columns]
    if missing:
        return None, None, f"Source CSV is missing required columns: {missing}"
    sdf = raw.copy()
    for c in num_cols:
        sdf[c] = pd.to_numeric(sdf[c], errors="coerce")
        if sdf[c].isna().mean() > 0.5:
            return None, None, f"Source column '{c}' is mostly non-numeric, so features are incompatible."
    if task == "regression":
        sdf[target] = pd.to_numeric(sdf[target], errors="coerce")
        sdf = sdf[sdf[target].notna()]
        ys = sdf[target].to_numpy(dtype=float)
    else:
        sdf = sdf[sdf[target].notna()]
        ys = sdf[target].astype(str).to_numpy()
        keep = np.isin(ys, classes)
        sdf, ys = sdf[keep], ys[keep]
    if len(sdf) < 50:
        return None, None, f"Only {len(sdf)} usable labelled source rows (need at least 50 with compatible targets)."
    sdf = sdf.reset_index(drop=True)
    if len(sdf) > max_rows:
        pick = np.sort(np.random.default_rng(seed).choice(len(sdf), max_rows, replace=False))
        sdf, ys = sdf.iloc[pick].reset_index(drop=True), ys[pick]
    return model_frame(sdf, num_cols, cat_cols), ys, f"Source data OK: {len(sdf):,} labelled rows."


def run_strategies(X, y, perf, task, family, o, source):
    tr, ad, te = perf["parts"]
    base = perf["bundle"]
    tr_fit = perf["tr_fit"]
    Xtr, ytr, Xad, yad, Xte, yte = X.iloc[tr_fit], y[tr_fit], X.iloc[ad], y[ad], X.iloc[te], y[te]
    rng = np.random.default_rng(o["seed"])
    availability, preds = [], {}
    B = o["batch_size"]

    def batches(n):
        return [(s, min(s + B, n)) for s in range(0, n, B)]

    preds["Baseline (no update)"] = perf["pred_test"]
    availability.append(("Baseline (no update)", "Available", "Trained on the training partition only."))

    can_partial = base.supports_partial
    why_no = (f"{FAMILY_NAME[family]} has no partial_fit, so it cannot be updated incrementally.")
    Xad_t = base.transform(Xad) if can_partial else None

    def run_incremental(m, replay=None):
        for s, e in batches(len(yad)):
            Xb, yb = Xad_t[s:e], yad[s:e]
            if replay is not None and len(replay[1]) > 0:
                k = min(o["replay_per_batch"], len(replay[1]))
                pick = rng.choice(len(replay[1]), size=k, replace=False)
                Xb, yb = np.vstack([Xb, replay[0][pick]]), np.concatenate([yb, replay[1][pick]])
            m.update_arrays(Xb, yb, o["epochs"])
        return m

    if can_partial:
        preds["Incremental (mini-batch partial_fit)"] = run_incremental(copy.deepcopy(base)).predict(Xte)
        availability.append(("Incremental (mini-batch partial_fit)", "Available",
                             f"Baseline updated sequentially on {len(yad):,} adaptation rows in batches of {B}."))
        m = copy.deepcopy(base)
        rows = np.arange(len(yad))[-o["online_max_rows"]:]
        for i in rows:
            m.update_arrays(Xad_t[i:i + 1], yad[i:i + 1], 1)
        preds["Online (one row at a time)"] = m.predict(Xte)
        cap = "" if len(rows) == len(yad) else f" (only the most recent {len(rows):,} of {len(yad):,} rows, to limit run time)"
        availability.append(("Online (one row at a time)", "Available",
                             f"Single pass, one row per update{cap}."))
        rep_n = min(o["replay_size"], len(ytr))
        if rep_n > 0:
            sel = np.sort(rng.choice(len(ytr), size=rep_n, replace=False))
            buf = (base.transform(Xtr.iloc[sel]), ytr[sel])
            preds["Continual (replay)"] = run_incremental(copy.deepcopy(base), buf).predict(Xte)
            availability.append(("Continual (replay)", "Available",
                                 f"Replay buffer of {rep_n:,} randomly selected earlier rows; {min(o['replay_per_batch'], rep_n)} replayed per batch."))
        else:
            availability.append(("Continual (replay)", "Unavailable", "Replay buffer size is 0 (it would equal incremental learning)."))
    else:
        for name in ("Incremental (mini-batch partial_fit)", "Online (one row at a time)", "Continual (replay)"):
            availability.append((name, "Unavailable", why_no))

    if not can_partial:
        availability.append(("Transfer + fine-tuning", "Unavailable", why_no + " Fine-tuning needs a model whose weights can be updated."))
    elif source is None:
        availability.append(("Transfer + fine-tuning", "Unavailable", o["source_msg"]))
    else:
        m = copy.deepcopy(base)
        m.est = make_estimator(family, task, o["seed"] + 13)
        m.update_arrays(m.transform(source[0]), source[1], EPOCHS[family])        # pretrain on source domain
        m.update_arrays(m.transform(Xtr), ytr, max(3, EPOCHS[family] // 3))         # fine-tune on target training data
        for s, e in batches(len(yad)):                                              # then adapt on the stream
            m.update_arrays(Xad_t[s:e], yad[s:e], o["epochs"])
        preds["Transfer + fine-tuning"] = m.predict(Xte)
        availability.append(("Transfer + fine-tuning", "Available",
                             f"Pretrained on {len(source[1]):,} source-domain rows, fine-tuned on target training data, then adapted on the stream."))

    Xfull = pd.concat([Xtr, Xad]).reset_index(drop=True)
    yfull = np.concatenate([ytr, yad])
    if len(yfull) > o["max_train_rows"]:
        pick = np.sort(rng.choice(len(yfull), o["max_train_rows"], replace=False))
        Xfull, yfull = Xfull.iloc[pick], yfull[pick]
    num_cols = [c for c in Xfull.columns if c in o["num_cols"]]
    cat_cols = [c for c in Xfull.columns if c in o["cat_cols"]]
    full = ModelBundle(family, task, num_cols, cat_cols, perf["classes"], o["seed"])
    full.fit_preprocessing(Xfull, yfull)
    full.train(Xfull, yfull)
    preds["Full retrain on train + adaptation (reference)"] = full.predict(Xte)
    availability.append(("Full retrain on train + adaptation (reference)", "Available",
                         "New model fitted from scratch on all earlier labelled rows. Not incremental; shown as a reference."))

    metric = o["metric"]
    rows = []
    base_pred = preds["Baseline (no update)"]
    base_val = compute_metrics(task, yte, base_pred)[metric]
    for name, p in preds.items():
        mt = compute_metrics(task, yte, p)
        row = {"Strategy": name, **mt}
        if name == "Baseline (no update)":
            row[IMPROVE_COL], row["Gain vs baseline, 95% CI"], row["Distinguishable from baseline?"] = 0.0, "—", "—"
        else:
            sign = 1.0 if HIGHER_BETTER[metric] else -1.0
            row[IMPROVE_COL] = sign * (mt[metric] - base_val) / max(abs(base_val), 1e-9) * 100
            lo, hi = paired_bootstrap_gain(task, metric, yte, base_pred, p, seed=o["seed"])
            row["Gain vs baseline, 95% CI"] = f"[{lo:.4g}, {hi:.4g}]"
            row["Distinguishable from baseline?"] = "Yes" if (lo > 0 or hi < 0) else "No (within noise)"
        rows.append(row)
    return {"metrics": pd.DataFrame(rows), "availability": pd.DataFrame(availability, columns=["Strategy", "Status", "Notes"]),
            "preds": preds, "y_test": yte, "metric": metric, "n_adapt": len(yad)}


# --------------------------------------------------------------------------------------
# Live-stream simulation
# --------------------------------------------------------------------------------------
def run_simulation(stream, spec, lo, hi, B, model, policy, delay, adwin_delta, cooldown, epochs):
    n = len(stream)
    nb = int(np.ceil(n / B))
    adwin = SimpleADWIN(adwin_delta)
    rows, alerts = [], []
    versions = [{"Version": "v0 (production baseline)", "Batch": 0, "Trigger": "initial training", "Action": "trained on the training partition", "Labelled rows used": 0}]
    if model:
        base, task, Xs, ys = model["base"], model["task"], model["Xs"], model["ys"]
        cand = copy.deepcopy(base)
        prod_pred = base.predict(Xs)
        cand_pred = np.empty(n, dtype=prod_pred.dtype if task == "classification" else float)
        norm_err = normalize_errors(task, row_errors(task, ys, prod_pred), B)
        trigger_batch, last_retrain = None, -10**6
    for k in range(nb):
        a, b = k * B, min((k + 1) * B, n)
        batch = stream.iloc[a:b]
        psis = {}
        for col, entry in spec.items():
            psis[col] = psi_value(entry[2], counts_for(entry, batch[col]))[0]
        valid = {c: v for c, v in psis.items() if not np.isnan(v)}
        psi_max = max(valid.values()) if valid else np.nan
        top = max(valid, key=valid.get) if valid else ""
        status = status_of(psi_max, lo, hi)
        data_alert, perf_alert = status == "Significant", False
        if data_alert:
            alerts.append({"Batch": k + 1, "Type": "Input (feature) drift", "Detail": f"max PSI {psi_max:.3f} on '{top}'"})
        row = {"Batch": k + 1, "Start row": a, "End row": b, "Rows": b - a, "Max PSI": psi_max,
               "Mean PSI": float(np.mean(list(valid.values()))) if valid else np.nan,
               "Features at/above high threshold": int(sum(v >= hi for v in valid.values())),
               "Top drifting feature": top, "Data drift status": status}
        if model:
            cand_pred[a:b] = cand.predict(Xs.iloc[a:b])        # predict first ...
            j = k - delay                                      # ... labels for batch j arrive now
            if j >= 0:
                ja, jb = j * B, min((j + 1) * B, n)
                hits = [e for e in norm_err[ja:jb] if adwin.update(e)]
                if hits:
                    perf_alert = True
                    alerts.append({"Batch": k + 1, "Type": "Prediction-error change (ADWIN)",
                                   "Detail": f"detected after labels of batch {j + 1} arrived"})
            if (data_alert or perf_alert) and trigger_batch is None:
                trigger_batch = k
            if policy == POLICY_PARTIAL and trigger_batch is not None and j >= 0:
                ja, jb = j * B, min((j + 1) * B, n)
                cand.update_arrays(cand.transform(Xs.iloc[ja:jb]), ys[ja:jb], epochs)
                versions.append({"Version": f"v{len(versions)} (shadow candidate)", "Batch": k + 1,
                                 "Trigger": f"alert at batch {trigger_batch + 1}", "Action": f"partial_fit on labels of batch {j + 1}",
                                 "Labelled rows used": jb - ja})
            elif policy == POLICY_RETRAIN and (data_alert or perf_alert) and k - last_retrain >= cooldown:
                n_lab = min(n, (j + 1) * B) if j >= 0 else 0
                if n_lab >= 30:
                    Xall = pd.concat([model["X_train"], Xs.iloc[:n_lab]]).reset_index(drop=True)
                    yall = np.concatenate([model["y_train"], ys[:n_lab]])
                    new = ModelBundle(model["family"], task, model["num_cols"], model["cat_cols"], model["classes"], model["seed"])
                    new.fit_preprocessing(Xall, yall)
                    new.train(Xall, yall)
                    cand, last_retrain = new, k
                    versions.append({"Version": f"v{len(versions)} (shadow candidate)", "Batch": k + 1, "Trigger": "alert",
                                     "Action": "retrained from scratch", "Labelled rows used": int(len(yall))})
                else:
                    alerts.append({"Batch": k + 1, "Type": "Retraining skipped", "Detail": "too few labels have arrived yet"})
        rows.append(row)
    if model:
        keep = ["MAE", "RMSE"] if task == "regression" else ["Accuracy", "F1 (weighted)"]
        for k, row in enumerate(rows):
            a, b = row["Start row"], row["End row"]
            for who, pr in (("Production", prod_pred), ("Candidate", cand_pred)):
                mt = compute_metrics(task, ys[a:b], pr[a:b])
                for m in keep:
                    row[f"{who} {m}"] = mt[m]
            row["Labels available at batch"] = k + 1 + delay
    return {"batches": pd.DataFrame(rows), "alerts": pd.DataFrame(alerts, columns=["Batch", "Type", "Detail"]),
            "versions": pd.DataFrame(versions), "task": model["task"] if model else None,
            "trigger_batch": (trigger_batch + 1) if (model and trigger_batch is not None) else None}


# ======================================================================================

# ======================================================================================
# SIMPLE SINGLE-PAGE DASHBOARD
# ======================================================================================
st.title("📈 Model Drift Monitor")
st.caption("One page • persistent drift chart • steady incoming-data flow • alerts without automatic page refresh")

st.info(
    "This dashboard replays rows from an uploaded CSV as a stream simulation. "
    "A drift alert does not stop incoming rows. PSI measures input-feature changes; "
    "confirming performance/concept drift requires real target labels."
)

uploaded = st.file_uploader("Upload CSV data", type=["csv"], key="dashboard_csv")
if uploaded is None:
    st.stop()

try:
    df_all, info, encoding, n_dup = load_and_prepare(uploaded.getvalue())
except Exception as exc:
    st.error(f"Could not read this CSV: {exc}")
    st.stop()

if df_all.empty or len(df_all.columns) == 0:
    st.error("The CSV must contain at least one column and one row.")
    st.stop()

kinds = dict(zip(info["Column"], info["Detected type"]))
all_cols = list(df_all.columns)

# Keep configuration compact and visible; no tabs or sidebar.
with st.expander("Data and monitoring settings", expanded=True):
    c1, c2, c3 = st.columns(3)
    time_options = [NONE_LABEL] + [c for c in all_cols if kinds[c] in ("datetime", "datetime-text", "numeric")]
    time_choice = c1.selectbox("Time/order column (optional)", time_options, index=0)
    time_col = None if time_choice == NONE_LABEL else time_choice
    target_options = [NONE_LABEL] + [c for c in all_cols if c != time_col and kinds[c] in ("numeric", "categorical", "boolean")]
    target_choice = c2.selectbox("Target / actual outcome (optional)", target_options, index=0)
    target = None if target_choice == NONE_LABEL else target_choice
    features_options = [c for c in all_cols if c not in (time_col, target) and family_of(kinds[c]) is not None]
    features = c3.multiselect("Features to monitor", features_options, default=features_options[:min(5, len(features_options))])
    d1, d2, d3, d4 = st.columns(4)
    batch_size = int(d1.number_input("Rows per incoming batch", min_value=10, max_value=max(10, min(10000, len(df_all))), value=max(10, min(100, len(df_all)))))
    moderate_thr = float(d2.number_input("Moderate PSI threshold", min_value=0.01, max_value=5.0, value=0.10, step=0.01))
    significant_thr = float(d3.number_input("Drift alert threshold", min_value=0.02, max_value=10.0, value=0.25, step=0.01))
    bins = int(d4.number_input("PSI bins", min_value=3, max_value=20, value=10))
    strategy = st.selectbox(
        "Action to recommend when drift is detected",
        ["Notify only", "Retune existing model", "Retrain using new data only", "Retrain using old + new data"],
        help="This setting selects the response plan. The dashboard does not silently replace a production model."
    )
    start_button = st.button("▶ Start / refresh stream simulation", type="primary")

if significant_thr <= moderate_thr:
    st.error("The drift alert threshold must be higher than the moderate threshold.")
    st.stop()
if not features:
    st.warning("Choose at least one feature column to monitor.")
    st.stop()
if len(df_all) < 60:
    st.warning("PSI comparisons are more meaningful with at least 60 rows.")
    st.stop()

df_ord, tvals, n_bad_time, was_sorted = order_by_time(df_all, time_col, kinds.get(time_col))
if df_ord.empty:
    st.error("No usable rows remain after ordering by the selected time column.")
    st.stop()
if n_bad_time:
    st.warning(f"{n_bad_time} rows with invalid time values were excluded from the ordered stream.")
xvals = tvals if tvals is not None else pd.Series(np.arange(len(df_ord)))
ref_n = max(30, int(len(df_ord) * 0.4))
ref_n = min(ref_n, len(df_ord) - 20)
reference = df_ord.iloc[:ref_n][features].reset_index(drop=True)
stream = df_ord.iloc[ref_n:].reset_index(drop=True)
if len(stream) < 1:
    st.warning("Not enough rows remain for the stream after reserving reference data.")
    st.stop()

spec = build_reference(reference, list(features), kinds, bins)
if not spec:
    st.error("No supported numeric or categorical feature columns have valid reference values.")
    st.stop()

# Process once per explicit click; charts stay stable between interactions and no timer/rerun loop is used.
sig = hashlib.md5((uploaded.name + str(len(uploaded.getvalue())) + str(features) +
                   str(batch_size) + str(moderate_thr) + str(significant_thr) + str(bins) +
                   str(strategy) + str(time_col)).encode()).hexdigest()
if start_button or st.session_state.get("simple_dashboard_sig") != sig:
    records = []
    alerts = []
    for batch_no, start in enumerate(range(0, len(stream), batch_size), start=1):
        chunk = stream.iloc[start:start + batch_size]
        vals = {}
        for col, entry in spec.items():
            vals[col] = psi_value(entry[2], counts_for(entry, chunk[col]))[0]
        valid = {col: value for col, value in vals.items() if not np.isnan(value)}
        max_psi = max(valid.values()) if valid else np.nan
        top_feature = max(valid, key=valid.get) if valid else "N/A"
        status = status_of(max_psi, moderate_thr, significant_thr)
        records.append({
            "Batch": batch_no, "Rows received": min(start + len(chunk), len(stream)),
            "Batch size": len(chunk), "Max PSI": max_psi,
            "Mean PSI": float(np.mean(list(valid.values()))) if valid else np.nan,
            "Top feature": top_feature, "Status": status
        })
        if status == "Significant":
            alerts.append({
                "Batch": batch_no, "Status": "DRIFT DETECTED",
                "Feature": top_feature, "Max PSI": max_psi,
                "Recommended action": strategy
            })
    st.session_state["simple_dashboard_sig"] = sig
    st.session_state["simple_dashboard_batches"] = pd.DataFrame(records)
    st.session_state["simple_dashboard_alerts"] = pd.DataFrame(
        alerts, columns=["Batch", "Status", "Feature", "Max PSI", "Recommended action"]
    )

batches = st.session_state["simple_dashboard_batches"]
alerts = st.session_state["simple_dashboard_alerts"]

st.divider()
st.subheader("Live data flow")
st.caption(f"Reference window: first {len(reference):,} rows • Simulated incoming stream: {len(stream):,} rows • {len(batches):,} batches")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Rows in CSV", f"{len(df_all):,}")
k2.metric("Incoming batches", f"{len(batches):,}")
k3.metric("Drift alerts", f"{len(alerts):,}")
k4.metric("Latest status", batches.iloc[-1]["Status"] if len(batches) else "Waiting")

# Flow chart is always shown; the stream is not stopped or reset by an alert.
flow_fig = go.Figure()
flow_fig.add_trace(go.Scatter(
    x=batches["Batch"], y=batches["Rows received"], mode="lines+markers",
    name="Rows received", line=dict(width=3)
))
flow_fig.update_layout(
    title="Incoming data continues normally",
    xaxis_title="Stream batch", yaxis_title="Cumulative rows received",
    height=300, margin=dict(l=20, r=20, t=55, b=20)
)
st.plotly_chart(flow_fig, use_container_width=True, key="incoming_flow_chart")

st.subheader("Model / data drift graph")
drift_fig = go.Figure()
drift_fig.add_trace(go.Scatter(
    x=batches["Batch"], y=batches["Max PSI"], mode="lines+markers",
    name="Maximum feature PSI", line=dict(width=3),
    marker=dict(color=[STATUS_COLORS.get(s, "#888888") for s in batches["Status"]], size=8)
))
drift_fig.add_trace(go.Scatter(
    x=batches["Batch"], y=batches["Mean PSI"], mode="lines",
    name="Mean feature PSI", line=dict(dash="dot")
))
drift_fig.add_hline(y=moderate_thr, line_dash="dash", annotation_text="Moderate threshold")
drift_fig.add_hline(y=significant_thr, line_dash="dash", annotation_text="Alert threshold")
drift_fig.update_layout(
    title="Feature drift by incoming batch",
    xaxis_title="Stream batch", yaxis_title="PSI",
    height=380, margin=dict(l=20, r=20, t=55, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(drift_fig, use_container_width=True, key="drift_chart")

st.subheader("Drift notification")
if len(alerts):
    latest = alerts.iloc[-1]
    st.error(
        f"⚠️ Drift detected in batch {int(latest['Batch'])}: feature **{latest['Feature']}** "
        f"(maximum PSI = {latest['Max PSI']:.3f}). Incoming data continues. "
        f"Suggested response: **{latest['Recommended action']}**."
    )
    st.dataframe(alerts.tail(10), hide_index=True, use_container_width=True)
else:
    st.success("No significant feature drift detected in the current replay. Monitoring continues.")

st.subheader("Response strategy")
strategy_notes = {
    "Notify only": "Keep the current model unchanged, investigate the changed inputs, and continue monitoring.",
    "Retune existing model": "Try a controlled hyperparameter search on recent labelled data. Compare the tuned candidate against the current model on a separate validation set.",
    "Retrain using new data only": "Train a candidate on recent labelled data. Use this when the new regime is representative and older patterns are no longer useful; check for forgetting.",
    "Retrain using old + new data": "Train a candidate on a balanced mixture of historical and recent labelled data. This is a safer starting point when both old and new patterns may matter."
}
st.markdown(f"**Selected plan: {strategy}.** {strategy_notes[strategy]}")
st.caption(
    "Important: PSI detects changes in input distributions, not proof of worse predictions. "
    + ("Select a target and use real arriving labels to evaluate accuracy/error before approving a model change."
       if target else "No target was selected, so this dashboard cannot calculate model accuracy or confirm concept drift.")
)
st.subheader("Batch history")
st.dataframe(batches, hide_index=True, use_container_width=True)
st.download_button(
    "Download drift batch history",
    csv_bytes(batches), "drift_batch_history.csv", "text/csv"
)
if len(alerts):
    st.download_button(
        "Download drift alerts",
        csv_bytes(alerts), "drift_alerts.csv", "text/csv"
    )
