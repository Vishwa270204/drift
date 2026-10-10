"""Generic CSV Model Drift, Performance Monitoring & Learning Strategy Lab.

Upload any tabular CSV. The app profiles the data, measures feature-level drift (PSI),
trains a baseline scikit-learn model when a real target column is chosen, monitors
performance / concept drift when labels exist, compares model-update strategies that are
technically valid for the chosen model, and can replay the file as a simulated stream.
"""
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
# APP
# ======================================================================================
st.title("📈 Model Drift & Learning Strategy Lab")
st.caption("Upload any tabular CSV. Analyses adapt to your columns, and anything that needs labels, "
           "a compatible model, or a source model is marked unavailable instead of being faked.")
st.info("Drift in the *inputs* is evidence that data changed; it does not by itself prove a model got worse. "
        "Only real target labels can show that. Replaying a static CSV is a **simulation**, not a live data connection.")

st.header("1. Upload your CSV")
uploaded = st.file_uploader("CSV file", type=["csv"], key="main_csv")
if uploaded is None:
    st.write("Waiting for a CSV file. Nothing is generated or assumed until you upload one.")
    st.stop()

data_bytes = uploaded.getvalue()
fid = hashlib.md5(data_bytes).hexdigest()[:10]
try:
    df_all, info, encoding, n_dup = load_and_prepare(data_bytes)
except ValueError as exc:
    st.error(str(exc))
    st.stop()
if df_all.shape[0] == 0 or df_all.shape[1] == 0:
    st.error("The file has no rows or no columns. Please upload a CSV with a header row and at least some data.")
    st.stop()
if df_all.shape[0] < 30:
    st.warning(f"Only {df_all.shape[0]} rows: most analyses below will be unreliable or unavailable.")
if df_all.shape[0] > 300_000:
    st.warning("This is a large file; model training and simulations may be slow.")
kinds = dict(zip(info["Column"], info["Detected type"]))
all_cols = list(df_all.columns)

# ---------------------------------------------------------------- column setup (main page)
st.header("2. Choose columns")
time_opts = [c for c in all_cols if kinds[c] in ("datetime", "datetime-text", "numeric")]
auto_time = next((c for c in all_cols if kinds[c] in ("datetime", "datetime-text")), None)
c1, c2, c3 = st.columns(3)
time_choice = c1.selectbox(
    "Time column (optional)", [NONE_LABEL] + time_opts,
    index=([NONE_LABEL] + time_opts).index(auto_time) if auto_time else 0, key=f"time_{fid}",
    help="Rows are sorted by this column. With none, the file's row order is used for drift windows.")
time_col = None if time_choice == NONE_LABEL else time_choice
target_choice = c2.selectbox("Target column (optional)", [NONE_LABEL] + [c for c in all_cols if c != time_col],
                             key=f"target_{fid}", help="Real outcome labels. Without one, supervised metrics cannot be measured.")
target = None if target_choice == NONE_LABEL else target_choice
task = None
if target:
    guess = infer_task(df_all[target], kinds[target]) if kinds[target] in ("numeric", "categorical", "boolean") else "classification"
    task = c3.radio("Task type", ["classification", "regression"], horizontal=True,
                    index=["classification", "regression"].index(guess), key=f"task_{fid}_{target}",
                    help=f"Auto-detected as {guess}. Change it if that is wrong.")
else:
    c3.caption("No target selected, so only profiling and drift analysis are available.")

# default feature suggestions
excluded = {}
suggested = []
usable = [c for c in all_cols if c not in (target, time_col) and family_of(kinds[c]) and kinds[c] != "empty"]
for c in all_cols:
    if c == time_col:
        excluded[c] = "time column"
    elif c == target:
        excluded[c] = "target column"
    elif kinds[c] == "empty":
        excluded[c] = "entirely missing"
    elif not family_of(kinds[c]):
        excluded[c] = f"unsupported type ({kinds[c]}); usable only as a time column"
for c in usable:
    r = info.set_index("Column").loc[c]
    if r["Constant"]:
        excluded[c] = "constant column"
    elif r["Identifier-like"]:
        excluded[c] = "identifier-like / very high cardinality"
    else:
        suggested.append(c)
if target and kinds[target] in ("numeric", "categorical", "boolean"):
    for c in leakage_suspects(df_all, suggested, target):
        suggested.remove(c)
        excluded[c] = "possible target leakage (copy of the target)"
features = st.multiselect("Feature columns (used for drift and modelling)", usable, default=suggested, key=f"feat_{fid}",
                          help="Suggested defaults exclude the target, time column, constants, identifiers and suspected target copies.")
if excluded:
    with st.expander("Columns excluded by default and why"):
        st.dataframe(pd.DataFrame({"Column": list(excluded), "Reason": list(excluded.values())}), hide_index=True)
row_chrono = False
if time_col is None:
    row_chrono = st.checkbox("My rows are already in chronological order (treat row order as time for model splits, "
                             "concept-drift checks and the simulation)", key=f"chrono_{fid}")
chrono = time_col is not None or row_chrono

with st.expander("Modelling options (used when a target is selected)", expanded=bool(target)):
    m1, m2, m3, m4, m5 = st.columns(5)
    family_label = m1.selectbox("Model family", list(FAMILIES), key=f"fam_{fid}",
                                help="Only models with partial_fit can be updated incrementally.")
    family = FAMILIES[family_label]
    train_pct = m2.slider("Training %", 30, 80, 50, 5, key=f"trp_{fid}")
    adapt_pct = m3.slider("Adaptation %", 10, 40, 25, 5, key=f"adp_{fid}",
                          help="Later labelled rows used only to update models. The remainder is the held-out test set.")
    seed = int(m4.number_input("Random seed", 0, 9999, 42, key=f"seed_{fid}"))
    max_train_rows = int(m5.number_input("Max training rows", 1000, 500_000, 30_000, 1000, key=f"maxrows_{fid}",
                                         help="Larger training partitions are randomly subsampled for speed."))
    st.caption(f"Test share = {100 - train_pct - adapt_pct}%. Splits are chronological if a time column is chosen "
               "(or rows are declared chronological); otherwise a seeded random split is used.")

# ---------------------------------------------------------------- ordered frame + supervised prep
df_ord, tvals, n_bad_time, was_sorted = order_by_time(df_all, time_col, kinds.get(time_col))
if df_ord.empty:
    st.error("No rows have a valid value in the selected time column. Choose a different time column.")
    st.stop()
if n_bad_time:
    st.warning(f"{n_bad_time} rows have an unparseable time value and are excluded from analyses (they remain in the CSV).")
if time_col and not was_sorted:
    st.info(f"Rows were re-sorted by '{time_col}' because the file was not in time order.")
xvals = tvals if tvals is not None else pd.Series(np.arange(len(df_ord)))
n_rows = len(df_ord)

has_target = target is not None
sup_problems, perf = [], None
num_cols = tuple(f for f in features if family_of(kinds[f]) == "num")
cat_cols_all = [f for f in features if family_of(kinds[f]) == "cat"]
cat_cols = tuple(f for f in cat_cols_all if df_ord[f].nunique() <= MAX_ONEHOT)
dropped_cat = [f for f in cat_cols_all if f not in cat_cols]
lf = y_arr = X_model = x_lab = None
if has_target:
    if kinds[target] in ("empty", "datetime", "datetime-text"):
        sup_problems.append(f"Target '{target}' has type '{kinds[target]}', which cannot be used as a target.")
    else:
        lab_mask = df_ord[target].notna().to_numpy()
        lf = df_ord.loc[lab_mask].reset_index(drop=True)
        x_lab = xvals[lab_mask].reset_index(drop=True)
        if task == "regression":
            if kinds[target] != "numeric":
                sup_problems.append("Regression needs a numeric target; choose 'classification' or another column.")
            else:
                y_arr = lf[target].astype(float).to_numpy()
        else:
            y_arr = lf[target].astype(str).to_numpy()
            ncls = len(np.unique(y_arr))
            if ncls < 2:
                sup_problems.append("The target has fewer than two distinct values, so classification is impossible.")
            elif ncls > 50:
                sup_problems.append(f"The target has {ncls} classes. That is more than this app supports; consider regression or binning.")
        if not sup_problems:
            if len(lf) < 60:
                sup_problems.append(f"Only {len(lf)} labelled rows (need at least 60 for a train/adaptation/test split).")
            if train_pct + adapt_pct > 90:
                sup_problems.append("Training % + adaptation % must not exceed 90 (the test set needs at least 10%).")
            if not num_cols and not cat_cols:
                sup_problems.append("No usable feature columns are selected for modelling.")
    if not sup_problems:
        X_model = model_frame(lf, list(num_cols), list(cat_cols))
        sig_base = "|".join(map(str, [fid, time_col, target, task, num_cols, cat_cols, family, train_pct, adapt_pct,
                                      seed, chrono, max_train_rows]))
        try:
            perf = train_baseline(X_model, y_arr, x_lab, sig_base, family, task, num_cols, cat_cols, chrono,
                                  train_pct, adapt_pct, seed, max_train_rows)
            if len(perf["parts"][2]) < 10 or len(perf["parts"][0]) < 20:
                perf = None
                sup_problems.append("The held-out test (or training) partition is too small. Adjust the split percentages.")
        except Exception as exc:
            perf = None
            sup_problems.append(f"Baseline training failed: {exc}")
exports = {}
drift_summary = None

# ---------------------------------------------------------------- tabs
tabs = st.tabs(["Overview", "Data Quality", "Drift Detection", "Model Performance",
                "Learning Strategy Comparison", "Live Stream Simulation", "Results and Downloads"])

# ======================================================= Overview
with tabs[0]:
    st.subheader("Dataset overview")
    n_num = sum(kinds[c] == "numeric" for c in all_cols)
    n_cat = sum(kinds[c] in ("categorical", "boolean") for c in all_cols)
    a, b, c, d = st.columns(4)
    a.metric("Rows", f"{len(df_all):,}")
    b.metric("Columns", f"{len(all_cols):,}")
    c.metric("Numeric / categorical columns", f"{n_num} / {n_cat}")
    d.metric("Time ordering", time_col if time_col else ("row order (declared chronological)" if row_chrono else "row order"))
    st.markdown(f"**Target:** {target or 'none'}" + (f" ({task})" if target else "") +
                f"  \n**Features selected:** {len(features)}" +
                (f"  \n**High-cardinality categorical features excluded from the model (still used for drift):** {', '.join(dropped_cat)}" if dropped_cat else ""))

    st.subheader("What can be analysed with this setup?")
    stream_len = len(perf["stream"]) if perf and perf["stream"] is not None else 0
    checklist = [
        ("Data profiling & quality", "Available", "Works on any CSV."),
        ("Feature drift (PSI)", "Available" if features and n_rows >= 60 else "Unavailable",
         "Select feature columns and have at least 60 rows." if not (features and n_rows >= 60) else "Compares two row windows."),
        ("Supervised baseline model", "Available" if perf else "Unavailable",
         "Real target labels are used." if perf else (" ".join(sup_problems) if sup_problems else "No target column selected, so there are no ground-truth labels.")),
        ("Concept-drift / performance over time", "Available" if perf and chrono and stream_len >= 100 else "Unavailable",
         "Needs labels, chronological order and enough post-training rows." if not (perf and chrono and stream_len >= 100) else "Frozen baseline monitored on later rows."),
        ("Update-strategy comparison", "Available" if perf else "Unavailable",
         "Incremental/online/continual need a model with partial_fit; transfer needs a source CSV." if perf else "Needs a valid target and baseline model."),
        ("Live-stream simulation", "Available" if chrono and n_rows >= 100 else "Unavailable",
         ("Feature drift only (no target)." if not perf else "Feature drift plus performance.") if chrono and n_rows >= 100 else "Needs chronological order and at least 100 rows."),
    ]
    st.dataframe(pd.DataFrame(checklist, columns=["Analysis", "Status", "Notes"]), hide_index=True)

    st.subheader("Data drift versus concept drift")
    st.markdown("""
- **Data (covariate) drift:** the distribution of *input features* changes. It can be measured without labels, e.g. with PSI.
- **Concept drift:** the *relationship between inputs and the target* changes. It can only be confirmed with real labels,
  for example by seeing prediction errors grow over time.
- PSI alone **cannot** detect concept drift, and feature drift does not always hurt performance. Labels are often delayed;
  until they arrive, actual performance is unknown.
""")

# ======================================================= Data Quality
with tabs[1]:
    st.subheader("Data quality")
    miss_cells = int(df_all.isna().sum().sum())
    a, b, c, d, e = st.columns(5)
    a.metric("Rows", f"{len(df_all):,}")
    b.metric("Columns", len(all_cols))
    c.metric("Duplicate rows", f"{n_dup:,}", f"{100 * n_dup / len(df_all):.1f}%", delta_color="off")
    d.metric("Missing cells", f"{100 * miss_cells / max(df_all.size, 1):.2f}%")
    e.metric("Constant columns", int(info["Constant"].sum()))
    st.caption(f"Encoding read as {encoding}. Duplicates are counted but not removed. Missing values are never filled in the "
               "displayed data; models impute using training data only.")
    warns = []
    if (info["Detected type"] == "empty").any():
        warns.append("Entirely empty columns: " + ", ".join(info.loc[info["Detected type"] == "empty", "Column"]))
    if info["Constant"].any():
        warns.append("Constant columns carry no information: " + ", ".join(info.loc[info["Constant"], "Column"]))
    if info["Identifier-like"].any():
        warns.append("Identifier-like columns (excluded from features by default): " + ", ".join(info.loc[info["Identifier-like"], "Column"]))
    noted = info[info["Notes"].str.contains("non-numeric|infinite", regex=True)]
    for _, r in noted.iterrows():
        warns.append(f"'{r['Column']}': {r['Notes']}")
    for w in warns:
        st.warning(w)
    st.markdown("**Column profile**")
    st.dataframe(info, hide_index=True)
    exports["column_profile.csv"] = info
    st.markdown("**Data preview (first 50 rows)**")
    st.dataframe(df_all.head(50))

    num_list = [c for c in all_cols if kinds[c] == "numeric"]
    cat_list = [c for c in all_cols if kinds[c] in ("categorical", "boolean")]
    if num_list:
        st.markdown("**Numerical summary**")
        st.dataframe(df_all[num_list].describe().T)
    if cat_list:
        st.markdown("**Categorical summary**")
        rows = []
        for c in cat_list:
            vc = df_all[c].astype(str).where(df_all[c].notna()).value_counts()
            rows.append({"Column": c, "Unique": int(df_all[c].nunique()), "Most frequent": vc.index[0] if len(vc) else "",
                         "Most frequent (%)": round(100 * vc.iloc[0] / max(vc.sum(), 1), 2) if len(vc) else 0})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    left, right = st.columns(2)
    with left:
        miss_df = info[info["Missing"] > 0].sort_values("Missing (%)", ascending=False)
        if len(miss_df):
            fig = px.bar(miss_df, x="Missing (%)", y="Column", orientation="h", title="Missing values by column",
                         hover_data=["Missing"])
            fig.update_layout(yaxis=dict(autorange="reversed"), height=max(300, 28 * len(miss_df) + 100))
            st.plotly_chart(fig)
        else:
            st.success("No missing values.")
    with right:
        if cat_list:
            fig = px.bar(info[info["Column"].isin(cat_list)], x="Column", y="Unique", title="Unique values per categorical column")
            st.plotly_chart(fig)
    if cat_list:
        pick = st.selectbox("Inspect value counts for", cat_list, key=f"vc_{fid}")
        vc = df_all[pick].astype(str).where(df_all[pick].notna()).value_counts().head(25).reset_index()
        vc.columns = [pick, "Count"]
        st.plotly_chart(px.bar(vc, x=pick, y="Count", title=f"Top values of '{pick}'"))

# ======================================================= Drift Detection
with tabs[2]:
    st.subheader("Feature drift (Population Stability Index)")
    st.markdown("Compares each feature's distribution in an **earlier reference window** with a **later comparison window**. "
                "This is evidence about *inputs* only. See the Model Performance tab for whether predictions got worse.")
    drift_cols = list(features)
    if has_target and kinds.get(target) in ("numeric", "categorical", "boolean"):
        if st.checkbox("Also check the target column itself (label / prior drift)", key=f"dtarget_{fid}"):
            drift_cols.append(target)
    if not drift_cols or n_rows < 60:
        st.warning("Select at least one feature column and provide at least 60 rows to run drift detection.")
    else:
        w1, w2, w3 = st.columns(3)
        ref_rng = w1.slider("Reference window (% of rows)", 0, 100, (0, 50), key=f"ref_{fid}",
                            help="Percentages of the time-ordered rows.")
        cmp_rng = w2.slider("Comparison window (% of rows)", 0, 100, (50, 100), key=f"cmp_{fid}")
        n_bins = int(w3.number_input("Numeric bins (quantiles of the reference window)", 3, 30, 10, key=f"bins_{fid}"))
        t1, t2 = st.columns(2)
        thr_lo = float(t1.number_input("Moderate-drift threshold (PSI ≥)", 0.01, 5.0, 0.10, 0.01, key=f"tlo_{fid}"))
        thr_hi = float(t2.number_input("Significant-drift threshold (PSI ≥)", 0.02, 10.0, 0.25, 0.01, key=f"thi_{fid}"))
        st.caption(f"PSI < {thr_lo}: stable. {thr_lo} to < {thr_hi}: moderate shift. ≥ {thr_hi}: significant shift. "
                   "These are common rules of thumb, not statistical tests, and PSI is inflated for small windows. "
                   "Missing values form their own bin; categories not seen in the reference window fall into 'Other / unseen'.")
        ra, rb = int(n_rows * ref_rng[0] / 100), int(n_rows * ref_rng[1] / 100)
        ca, cb = int(n_rows * cmp_rng[0] / 100), int(n_rows * cmp_rng[1] / 100)
        if thr_hi <= thr_lo:
            st.error("The significant threshold must be larger than the moderate threshold.")
        elif ra < cb and ca < rb:
            st.error("The reference and comparison windows overlap. Adjust them so they cover different rows.")
        elif rb - ra < 30 or cb - ca < 30:
            st.error("Each window needs at least 30 rows.")
        else:
            if min(rb - ra, cb - ca) < 100:
                st.warning("A window has fewer than 100 rows; PSI values will be noisy.")
            ref_df, cmp_df = df_ord.iloc[ra:rb], df_ord.iloc[ca:cb]
            drift_res, drift_det = compute_drift(ref_df, cmp_df, drift_cols, kinds, n_bins, thr_lo, thr_hi)
            exports["drift_results.csv"] = drift_res
            counts = drift_res["Status"].value_counts()
            evaluated = drift_res[drift_res["Status"] != "N/A"]
            n_sig, n_mod = int(counts.get("Significant", 0)), int(counts.get("Moderate", 0))
            max_psi = float(evaluated["PSI"].max()) if len(evaluated) else np.nan
            drift_summary = {"max_psi": max_psi, "n_sig": n_sig, "n_mod": n_mod,
                             "status": status_of(max_psi, thr_lo, thr_hi)}
            st.markdown(f"Reference: rows {ra:,}–{rb:,} ({rb - ra:,} rows). Comparison: rows {ca:,}–{cb:,} ({cb - ca:,} rows)"
                        + (f", {xvals.iloc[ra]} → {xvals.iloc[rb - 1]} vs {xvals.iloc[ca]} → {xvals.iloc[cb - 1]}." if time_col else "."))
            a, b, c, d = st.columns(4)
            a.metric("Features evaluated", len(evaluated))
            b.metric("Significant drift", n_sig)
            c.metric("Moderate drift", n_mod)
            d.metric("Max PSI", f"{max_psi:.3f}" if not np.isnan(max_psi) else "n/a")
            if not len(evaluated):
                st.warning("No feature could be evaluated (reference window has no valid values).")
            elif n_sig == 0 and n_mod == 0:
                st.success("No notable feature-level drift between the selected windows. The distributions look similar.")
            else:
                top = evaluated.head(3)
                total_psi = evaluated["PSI"].sum()
                st.warning(f"Feature drift detected in {n_sig + n_mod} of {len(evaluated)} features. Largest contributors: "
                           + ", ".join(f"**{r.Feature}** (PSI {r.PSI:.3f}, {100 * r.PSI / total_psi:.0f}% of total)"
                                       for r in top.itertuples()))
            st.caption("Feature drift is evidence that inputs changed. It does not prove that model performance degraded.")
            if len(evaluated):
                fig = px.bar(evaluated, x="PSI", y="Feature", orientation="h", color="Status",
                             color_discrete_map=STATUS_COLORS, title="PSI by feature",
                             hover_data=["Share of total PSI (%)"])
                for thr, nm in ((thr_lo, "moderate"), (thr_hi, "significant")):
                    fig.add_vline(x=thr, line_dash="dot", annotation_text=nm, annotation_position="top")
                fig.update_layout(yaxis=dict(autorange="reversed"), height=max(320, 26 * len(evaluated) + 120))
                st.plotly_chart(fig)
            st.dataframe(drift_res.round(4), hide_index=True)
            st.download_button("Download drift results (CSV)", csv_bytes(drift_res), "drift_results.csv", "text/csv")
            if len(evaluated):
                feat = st.selectbox("Compare distributions for", list(evaluated["Feature"]), key=f"dfeat_{fid}")
                is_num = family_of(kinds[feat]) == "num"
                if is_num:
                    pdf = pd.concat([ref_df[[feat]].assign(Window="Reference"), cmp_df[[feat]].assign(Window="Comparison")])
                    fig = px.histogram(pdf.dropna(), x=feat, color="Window", barmode="overlay", opacity=0.6,
                                       histnorm="probability density", nbins=40, title=f"Distribution of '{feat}'",
                                       color_discrete_map={"Reference": "#1f77b4", "Comparison": "#ff7f0e"})
                    fig.update_yaxes(title="Density")
                else:
                    det = drift_det[feat].melt(id_vars=["Bin", "PSI contribution"], value_vars=["Reference (%)", "Comparison (%)"],
                                               var_name="Window", value_name="Share (%)")
                    fig = px.bar(det, x="Bin", y="Share (%)", color="Window", barmode="group", title=f"Category shares for '{feat}'")
                st.plotly_chart(fig)
                with st.expander("Per-bin PSI contributions"):
                    st.dataframe(drift_det[feat].round(4), hide_index=True)

# ======================================================= Model Performance
with tabs[3]:
    st.subheader("Baseline model performance")
    if not has_target:
        st.info("No target column was selected, so supervised model performance **cannot be measured**. "
                "No labels or actual values are invented. Profiling and PSI drift analysis remain available.")
    elif perf is None:
        st.warning("Supervised analysis is not available:\n\n" + "\n".join(f"- {p}" for p in sup_problems))
    else:
        tr, ad, te = perf["parts"]
        yte = y_arr[te]
        st.markdown(f"**Model:** {FAMILY_NAME[family]} · **Task:** {task} · **Features:** {len(num_cols)} numeric + "
                    f"{len(cat_cols)} categorical (one-hot). Preprocessing (imputation, scaling, encoding) is fitted on the training partition only. "
                    f"The target and time column are never used as features.")
        if dropped_cat:
            st.caption("Excluded from the model because of very many categories: " + ", ".join(dropped_cat))
        if len(lf) < len(df_ord):
            st.caption(f"{len(df_ord) - len(lf):,} rows with a missing target are excluded from supervised analysis.")
        split_txt = "chronological" if chrono else "seeded random (stratified for classification where possible)"
        st.markdown(f"**Split ({split_txt}):** training {len(tr):,} (model fitted on {len(perf['tr_fit']):,}) · "
                    f"adaptation {len(ad):,} (unused by the baseline) · held-out test {len(te):,}")
        if not chrono:
            st.warning("No chronological order was declared, so the split is random. Performance-over-time and drift-adaptation "
                       "conclusions are not available because earlier/later rows are mixed.")
        mt, mtr, nv = perf["metrics_test"], perf["metrics_train"], perf["naive_metrics"]
        comp = pd.DataFrame({"Held-out test": mt, "Training (in-sample)": mtr, f"Naive reference ({perf['naive_desc']})": nv}).T
        st.dataframe(comp.round(4))
        exports["baseline_test_predictions.csv"] = pd.DataFrame({"row_in_labelled_data": te, "actual": yte, "predicted": perf["pred_test"]})
        if task == "regression":
            ok = mt["RMSE"] < nv["RMSE"]
            st.markdown(f"**Summary:** on {len(te):,} unseen test rows the baseline has MAE {mt['MAE']:.4g}, RMSE {mt['RMSE']:.4g}, "
                        f"R² {mt['R²']:.3f}. The naive reference has RMSE {nv['RMSE']:.4g}, so the model is "
                        f"{'better' if ok else 'not better'} than that reference on RMSE.")
            l, r = st.columns(2)
            fig = px.scatter(x=yte, y=perf["pred_test"], labels={"x": "Actual", "y": "Predicted"}, title="Actual vs predicted (test)",
                             opacity=0.6)
            lim = [float(min(yte.min(), perf["pred_test"].min())), float(max(yte.max(), perf["pred_test"].max()))]
            fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="Perfect prediction", line=dict(dash="dash", color="gray")))
            l.plotly_chart(fig)
            if chrono:
                fig = go.Figure()
                xt = x_lab.iloc[te]
                fig.add_trace(go.Scatter(x=xt, y=yte, mode="lines", name="Actual"))
                fig.add_trace(go.Scatter(x=xt, y=perf["pred_test"], mode="lines", name="Predicted"))
                fig.update_layout(title="Actual and predicted over the test period", xaxis_title="Time / row", yaxis_title=target)
            else:
                fig = px.histogram(x=yte - perf["pred_test"], nbins=40, labels={"x": "Residual (actual - predicted)"}, title="Residual distribution (test)")
            r.plotly_chart(fig)
        else:
            maj = nv["Accuracy"]
            st.markdown(f"**Summary:** on {len(te):,} unseen test rows accuracy is {mt['Accuracy']:.3f} and weighted F1 {mt['F1 (weighted)']:.3f}. "
                        f"The majority-class reference reaches accuracy {maj:.3f}; "
                        f"{'the model beats it' if mt['Accuracy'] > maj else 'the model does not beat it'}.")
            labs = sorted(set(yte) | set(perf["pred_test"]))
            l, r = st.columns(2)
            if len(labs) <= 30:
                cm = confusion_matrix(yte, perf["pred_test"], labels=labs)
                fig = px.imshow(cm, x=labs, y=labs, text_auto=True, color_continuous_scale="Blues",
                                labels=dict(x="Predicted", y="Actual", color="Count"), title="Confusion matrix (test)")
                l.plotly_chart(fig)
            p, rc, f1, sup = precision_recall_fscore_support(yte, perf["pred_test"], labels=labs, zero_division=0)
            rep = pd.DataFrame({"Class": labs, "Precision": p, "Recall": rc, "F1": f1, "Support": sup})
            r.dataframe(rep.round(3), hide_index=True)
            dist = pd.concat([pd.DataFrame({"Class": y_arr[tr]}).assign(Partition="Training"),
                              pd.DataFrame({"Class": yte}).assign(Partition="Test")])
            dist = dist.groupby(["Partition", "Class"]).size().reset_index(name="Count")
            r.plotly_chart(px.bar(dist, x="Class", y="Count", color="Partition", barmode="group", title="Class balance"))
        st.caption("With small test sets these metrics have real sampling uncertainty; compare with the naive reference rather than trusting a single number.")

        # ---------------- concept drift
        st.divider()
        st.subheader("Concept drift, delayed labels and performance over time")
        st.markdown("""
The baseline model is **frozen** and monitored on the rows that come after its training data.
Rising errors can signal concept drift, but also covariate shift, noise or label problems; the data alone cannot separate these
causes. Detection requires **real labels**, **chronological order**, and enough rows. Labels often arrive late, so measured performance only covers rows whose labels have arrived.
""")
        stream = perf["stream"]
        if not chrono or stream is None:
            st.info("Declare chronological order (choose a time column or tick the row-order option) to monitor performance over time.")
        elif len(stream) < 100:
            st.info("Fewer than 100 post-training labelled rows: not enough to monitor performance over time reliably.")
        else:
            delay = int(st.number_input("Label delay (most recent rows whose labels have NOT arrived yet)", 0, max(len(stream) - 100, 0),
                                        0, key=f"delay_{fid}"))
            avail = stream.iloc[:len(stream) - delay].reset_index(drop=True)
            st.caption(f"Measurable rows: {len(avail):,} of {len(stream):,}. "
                       + ("With delayed labels, performance on the latest rows is unknown until they arrive." if delay else "No label delay assumed."))
            k_max = max(2, min(10, len(avail) // 30))
            K = st.slider("Number of evaluation windows", 2, k_max, min(5, k_max), key=f"K_{fid}")
            edges = np.linspace(0, len(avail), K + 1).astype(int)
            wrows = []
            for i in range(K):
                w = avail.iloc[edges[i]:edges[i + 1]]
                lo_c, hi_c = mean_ci(w["error"].to_numpy(), seed=seed + i)
                mtw = compute_metrics(task, w["y_true"], w["y_pred"])
                wrows.append({"Window": f"W{i + 1}", "From": w["x"].iloc[0], "To": w["x"].iloc[-1], "Rows": len(w),
                              "Mean error" if task == "regression" else "Error rate": w["error"].mean(),
                              "CI low": lo_c, "CI high": hi_c,
                              **({"RMSE": mtw["RMSE"]} if task == "regression" else {"Accuracy": mtw["Accuracy"], "F1 (weighted)": mtw["F1 (weighted)"]})})
            wdf = pd.DataFrame(wrows)
            exports["performance_by_window.csv"] = wdf
            st.dataframe(wdf.round({c: 4 for c in wdf.columns if pd.api.types.is_float_dtype(wdf[c])}), hide_index=True)
            ycol = "Mean error" if task == "regression" else "Error rate"
            fig = go.Figure(go.Scatter(x=wdf["Window"], y=wdf[ycol], mode="lines+markers", name=ycol,
                                       error_y=dict(type="data", symmetric=False, array=wdf["CI high"] - wdf[ycol],
                                                    arrayminus=wdf[ycol] - wdf["CI low"])))
            fig.update_layout(title=f"{ycol} per evaluation window (95% bootstrap CI)", xaxis_title="Window (chronological)",
                              yaxis_title="MAE" if task == "regression" else "Error rate")
            st.plotly_chart(fig)
            e1, e2 = avail["error"].to_numpy()[edges[0]:edges[1]], avail["error"].to_numpy()[edges[-2]:edges[-1]]
            rng = np.random.default_rng(seed)
            diff = (e2[rng.integers(0, len(e2), (500, len(e2)))].mean(axis=1) - e1[rng.integers(0, len(e1), (500, len(e1)))].mean(axis=1))
            dlo, dhi = np.percentile(diff, [2.5, 97.5])
            if dlo > 0:
                verdict = "Error is **higher** in the last window than in the first (the 95% interval excludes 0)."
            elif dhi < 0:
                verdict = "Error is **lower** in the last window than in the first (the 95% interval excludes 0)."
            else:
                verdict = "There is **no clear difference** in error between the first and last windows (the 95% interval includes 0)."
            st.markdown(f"{verdict} Difference in mean error (last − first): {e2.mean() - e1.mean():.4g}, 95% CI [{dlo:.4g}, {dhi:.4g}], "
                        f"with {len(e1):,} and {len(e2):,} rows. This is evidence about performance, not proof of *why* it changed.")
            if drift_summary:
                worse = dlo > 0
                feat_drift = drift_summary["status"] in ("Moderate", "Significant")
                interp = {(True, True): "Errors rose **and** input features drifted: consistent with data drift, possibly with concept drift too. The two cannot be separated from this evidence alone.",
                          (True, False): "Errors rose **without** notable feature drift: this pattern is more suggestive of concept drift, label noise or an unmeasured factor.",
                          (False, True): "Features drifted but no clear error increase was measured: data drift without demonstrated performance impact (so far).",
                          (False, False): "Neither notable feature drift nor a clear error increase was found."}[(worse, feat_drift)]
                st.info(interp)
            st.markdown("**ADWIN-style change detector on the prediction-error stream**")
            delta = st.select_slider("Confidence parameter δ (smaller = fewer false alarms, slower detection)",
                                     [0.0005, 0.001, 0.002, 0.005, 0.01, 0.05], value=0.002, key=f"delta_{fid}")
            ne = normalize_errors(task, avail["error"].to_numpy(), max(30, len(avail) // 5))
            hits = run_adwin(ne, delta)
            roll_w = max(10, len(avail) // 40)
            roll = avail["error"].rolling(roll_w, min_periods=max(3, roll_w // 3)).mean()
            fig = go.Figure(go.Scatter(x=avail["x"], y=roll, mode="lines", name=f"Rolling error (window {roll_w})"))
            for h in hits[:15]:
                vline(fig, avail["x"].iloc[h], "change")
            fig.update_layout(title="Frozen-baseline error over time with detected changes", xaxis_title="Time / row",
                              yaxis_title="Rolling mean absolute error" if task == "regression" else "Rolling error rate")
            st.plotly_chart(fig)
            if hits:
                hdf = pd.DataFrame({"Detection #": range(1, len(hits) + 1), "Stream row": hits, "Time / row": [avail["x"].iloc[h] for h in hits],
                                    "Partition": [avail["Partition"].iloc[h] for h in hits],
                                    "Earliest label-confirmed at stream row": [h + delay for h in hits]})
                st.warning(f"The detector signalled {len(hits)} change point(s) in the error stream. This suggests the error level changed; "
                           "confirm with a fresh labelled sample before acting.")
                st.dataframe(hdf, hide_index=True)
                exports["adwin_detections.csv"] = hdf
            else:
                st.success("No change in the error stream was detected at this sensitivity.")
            with st.expander("Detector requirements and limitations"):
                st.markdown("""
- Needs **ground-truth labels** for each monitored row, in chronological order; with delayed labels it can only react after they arrive.
- Works on a **bounded** signal: classification uses 0/1 errors; regression uses absolute errors scaled by 4× the early mean error and clipped to [0, 1].
- This is a **simplified** ADWIN-style implementation (bounded window, checks every few rows, variance-aware bound), not the reference library.
- It needs hundreds of labelled rows to detect modest shifts, and gives false alarms at the rate implied by δ only if the assumptions hold.
- It detects a change in *error level*, which can be caused by concept drift, covariate shift or noise. It cannot tell you which.
""")

# ======================================================= Learning Strategy Comparison
with tabs[4]:
    st.subheader("Compare model-update strategies")
    if perf is None:
        st.info("Strategy comparison needs real target labels and a trained baseline. "
                + ("No target column was selected." if not has_target else "See the Model Performance tab for what is blocking it."))
    else:
        tr, ad, te = perf["parts"]
        if len(ad) < 20 or len(te) < 20:
            st.warning("The adaptation or test partition has fewer than 20 rows; comparisons would be meaningless. Adjust the split percentages.")
        else:
            st.markdown("""
All strategies start from the **same trained baseline** (where applicable), update only on the **adaptation** rows,
and are scored on the **same held-out test rows**, which no strategy sees.
- **Incremental:** `partial_fit` on adaptation mini-batches. **Online:** one row at a time.
- **Continual (replay):** new batches are mixed with a small buffer of earlier rows. This reduces forgetting somewhat but is **not** a complete solution to catastrophic forgetting.
- **Transfer + fine-tuning:** only possible with a separate source dataset that has the same feature and target columns and a model whose weights can be updated. It is not simply retraining.
- **Full retrain** is included only as a non-incremental reference.
""")
            metric_opts = REG_METRICS if task == "regression" else CLS_METRICS
            s1, s2, s3, s4, s5 = st.columns(5)
            metric = s1.selectbox("Ranking metric", metric_opts, key=f"rankm_{fid}")
            bs = int(s2.number_input("Update batch size", 8, 2048, 64, key=f"sbs_{fid}"))
            epochs = int(s3.number_input("Epochs per batch", 1, 20, 2, key=f"sep_{fid}"))
            replay_size = int(s4.number_input("Replay buffer size", 0, 20000, 500, 50, key=f"srs_{fid}"))
            replay_pb = int(s5.number_input("Replay rows per batch", 1, 2048, 64, key=f"srb_{fid}"))
            src_file = st.file_uploader("Optional: source-domain CSV for transfer learning (same feature columns and target)", type=["csv"], key=f"src_{fid}")
            source, source_msg = None, "No source/pretrained dataset was provided, so transfer learning is not possible. Upload a compatible source CSV to enable it."
            if src_file is not None and perf["bundle"].supports_partial:
                try:
                    sx, sy, source_msg = prepare_source(src_file.getvalue(), list(num_cols), list(cat_cols), target, task,
                                                        perf["classes"], max_train_rows, seed)
                    source = (sx, sy) if sx is not None else None
                    (st.success if source else st.warning)(source_msg)
                except Exception as exc:
                    source_msg = f"Could not use the source CSV: {exc}"
                    st.warning(source_msg)
            opts = dict(metric=metric, batch_size=bs, epochs=epochs, replay_size=replay_size, replay_per_batch=replay_pb,
                        online_max_rows=3000, seed=seed, max_train_rows=max_train_rows, source_msg=source_msg,
                        num_cols=num_cols, cat_cols=cat_cols)
            sig = "|".join(map(str, [sig_base, metric, bs, epochs, replay_size, replay_pb, src_file.name if src_file else None,
                                     len(src_file.getvalue()) if src_file else 0]))
            if st.button("Run strategy comparison", type="primary", key=f"runstrat_{fid}"):
                try:
                    with st.spinner("Updating and evaluating strategies..."):
                        res = run_strategies(X_model, y_arr, perf, task, family, opts, source)
                    res["sig"] = sig
                    res["x_test"] = x_lab.iloc[te].to_numpy()
                    st.session_state["strat"] = res
                except Exception as exc:
                    st.error(f"Strategy comparison failed: {exc}")
            res = st.session_state.get("strat")
            if res is not None and not str(res["sig"]).startswith(str(fid)):
                res = None
            if res is not None:
                if res["sig"] != sig:
                    st.warning("Settings changed since this result was produced. Click 'Run strategy comparison' to refresh.")
                st.markdown("**Strategy availability**")
                st.dataframe(res["availability"], hide_index=True)
                mdf, mname = res["metrics"], res["metric"]
                st.markdown(f"**Metrics on the same held-out test set** ({len(res['y_test']):,} rows; ranking metric: {mname}, "
                            f"{'higher' if HIGHER_BETTER[mname] else 'lower'} is better)")
                st.dataframe(mdf.round(4), hide_index=True)
                exports["strategy_metrics.csv"] = mdf
                exports["strategy_test_predictions.csv"] = pd.DataFrame({"actual": res["y_test"], **{k: v for k, v in res["preds"].items()}})
                if len(mdf) >= 2:
                    best = mdf.loc[mdf[mname].idxmax() if HIGHER_BETTER[mname] else mdf[mname].idxmin()]
                    if best["Strategy"] == "Baseline (no update)":
                        st.info(f"No update strategy beat the baseline on {mname} on this test set. Updating did not help here, "
                                "which is plausible if no relevant drift occurred.")
                    else:
                        dist = best["Distinguishable from baseline?"]
                        st.success(f"Best on {mname}: **{best['Strategy']}** ({best[mname]:.4g}); change vs baseline "
                                   f"{best[IMPROVE_COL]:+.1f}%, 95% CI of the gain {best['Gain vs baseline, 95% CI']}. "
                                   + ("This difference is distinguishable from sampling noise." if dist == "Yes"
                                      else "This difference is within sampling noise, so do not read it as a real improvement."))
                st.caption("One dataset, one split, one seed. Treat rankings as indicative; repeat with other seeds and splits before deciding.")
                chart_metric = st.radio("Metric to chart", [c for c in mdf.columns if c in HIGHER_BETTER], horizontal=True,
                                        index=[c for c in mdf.columns if c in HIGHER_BETTER].index(mname), key=f"chm_{fid}")
                fig = px.bar(mdf, x="Strategy", y=chart_metric, color="Strategy", text_auto=".4g", title=f"{chart_metric} by strategy (held-out test)")
                fig.update_layout(showlegend=False, xaxis_title="")
                st.plotly_chart(fig)
                if chrono:
                    fig = go.Figure()
                    w = max(10, len(res["y_test"]) // 25)
                    for nm, p in res["preds"].items():
                        e = pd.Series(row_errors(task, res["y_test"], p)).rolling(w, min_periods=3).mean()
                        fig.add_trace(go.Scatter(x=res["x_test"], y=e, mode="lines", name=nm))
                    fig.update_layout(title="Rolling error over the test period by strategy", xaxis_title="Time / row",
                                      yaxis_title="Rolling MAE" if task == "regression" else "Rolling error rate")
                    st.plotly_chart(fig)
                pick = st.selectbox("Inspect predictions of", list(res["preds"]), key=f"inspect_{fid}")
                pp = res["preds"][pick]
                if task == "regression":
                    fig = px.scatter(x=res["y_test"], y=pp, opacity=0.6, labels={"x": "Actual", "y": "Predicted"}, title=f"Actual vs predicted: {pick}")
                    lim = [float(min(res["y_test"].min(), pp.min())), float(max(res["y_test"].max(), pp.max()))]
                    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="Perfect", line=dict(dash="dash", color="gray")))
                else:
                    labs = sorted(set(res["y_test"]) | set(pp))
                    fig = px.imshow(confusion_matrix(res["y_test"], pp, labels=labs), x=labs, y=labs, text_auto=True, color_continuous_scale="Blues",
                                    labels=dict(x="Predicted", y="Actual", color="Count"), title=f"Confusion matrix: {pick}")
                st.plotly_chart(fig)

# ======================================================= Live Stream Simulation
with tabs[5]:
    st.subheader("Live-stream simulation (replay of your static CSV)")
    st.warning("This replays rows from the uploaded file in chronological batches. It is a simulation, **not** a live data connection. "
               "Drift never stops the replay; alerts only trigger the action you choose, and the production model is never replaced automatically.")
    if not chrono or n_rows < 100:
        st.info("The simulation needs chronological order (choose a time column or declare row order as chronological) and at least 100 rows.")
    elif not features:
        st.info("Select at least one feature column to monitor.")
    else:
        use_model = perf is not None
        if use_model:
            n_tr = len(perf["parts"][0])
            ref_frame, stream_frame = lf.iloc[:n_tr][list(features)], lf.iloc[n_tr:][list(features)].reset_index(drop=True)
            st.caption(f"Reference = the model's training partition ({n_tr:,} rows). Stream = the {len(stream_frame):,} later labelled rows.")
        else:
            ref_pct = st.slider("Reference window = first (% of rows)", 10, 70, 40, key=f"simref_{fid}")
            cut = int(n_rows * ref_pct / 100)
            ref_frame, stream_frame = df_ord.iloc[:cut][list(features)], df_ord.iloc[cut:][list(features)].reset_index(drop=True)
            st.caption("No target selected: only input-feature drift is monitored. Collect labels to monitor performance.")
        if len(stream_frame) < 50 or len(ref_frame) < 30:
            st.warning("Reference or stream is too small for a simulation. Adjust the split or provide more rows.")
        else:
            g1, g2, g3, g4 = st.columns(4)
            def_b = int(max(50, len(stream_frame) // 20))
            bsz = int(g1.number_input("Batch size (rows)", 10, max(len(stream_frame), 10), min(def_b, len(stream_frame)), key=f"simB_{fid}"))
            s_lo = float(g2.number_input("Moderate PSI ≥", 0.01, 5.0, 0.10, 0.01, key=f"slo_{fid}"))
            s_hi = float(g3.number_input("Significant PSI ≥ (raises alert)", 0.02, 10.0, 0.25, 0.01, key=f"shi_{fid}"))
            n_sbins = int(g4.number_input("Numeric bins", 3, 20, 5, key=f"sbins_{fid}"))
            if bsz < 100:
                st.caption("Small batches make PSI noisy; consider batches of 100+ rows.")
            policy, delay_b, adelta, cooldown, sep = POLICY_MONITOR, 0, 0.002, 3, 2
            if use_model:
                h1, h2, h3 = st.columns(3)
                pols = [POLICY_MONITOR, POLICY_RETRAIN] + ([POLICY_PARTIAL] if perf["bundle"].supports_partial else [])
                policy = h1.selectbox("Action when an alert fires", pols, key=f"pol_{fid}",
                                      help="Candidates run in shadow mode; the production baseline is never replaced.")
                delay_b = int(h2.number_input("Label delay (batches)", 0, 20, 1, key=f"sdel_{fid}"))
                adelta = float(h3.select_slider("ADWIN δ", [0.0005, 0.001, 0.002, 0.005, 0.01, 0.05], value=0.002, key=f"sad_{fid}"))
                if not perf["bundle"].supports_partial:
                    st.caption(f"{FAMILY_NAME[family]} cannot be updated incrementally, so that action is not offered.")
            ssig = "|".join(map(str, [sig_base if use_model else fid, features, bsz, s_lo, s_hi, n_sbins, policy, delay_b, adelta]))
            if st.button("Run simulation", type="primary", key=f"runsim_{fid}"):
                try:
                    with st.spinner("Replaying the data in batches..."):
                        spec = build_reference(ref_frame, list(features), kinds, n_sbins)
                        model = None
                        if use_model:
                            tr_fit = perf["tr_fit"]
                            model = dict(base=perf["bundle"], task=task, Xs=X_model.iloc[n_tr:].reset_index(drop=True), ys=y_arr[n_tr:],
                                         X_train=X_model.iloc[tr_fit], y_train=y_arr[tr_fit], family=family, num_cols=list(num_cols),
                                         cat_cols=list(cat_cols), classes=perf["classes"], seed=seed)
                        sim = run_simulation(stream_frame, spec, s_lo, s_hi, bsz, model, policy, delay_b, adelta, cooldown, sep)
                        sim["sig"], sim["delay"], sim["thr"] = ssig, delay_b, (s_lo, s_hi)
                        sim["x"] = (x_lab.iloc[n_tr:] if use_model else xvals.iloc[len(ref_frame):]).reset_index(drop=True)
                        st.session_state["sim"] = sim
                except Exception as exc:
                    st.error(f"Simulation failed: {exc}")
            sim = st.session_state.get("sim")
            if sim is not None and not str(sim["sig"]).startswith(str(sig_base if use_model else fid)):
                sim = None
            if sim is not None:
                if sim["sig"] != ssig:
                    st.warning("Settings changed since this simulation ran. Click 'Run simulation' to refresh.")
                bdf, adf, vdf, stask = sim["batches"], sim["alerts"], sim["versions"], sim["task"]
                exports["simulation_batches.csv"] = bdf
                exports["simulation_alerts.csv"] = adf
                exports["simulation_model_versions.csv"] = vdf
                nb = len(bdf)
                cur = st.slider("Replay position: current batch", 1, nb, nb, key=f"cur_{fid}")
                row = bdf.iloc[cur - 1]
                a, b, c, d = st.columns(4)
                a.metric("Current batch", f"{cur} / {nb}")
                b.metric("Rows in batch", f"{int(row['Start row']):,}–{int(row['End row']):,}")
                c.metric("Input drift status", row["Data drift status"])
                d.metric("Max PSI", f"{row['Max PSI']:.3f}" if not np.isnan(row["Max PSI"]) else "n/a")
                seen = bdf.iloc[:cur]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=seen["Batch"], y=seen["Max PSI"], mode="lines+markers", name="Max PSI",
                                         marker=dict(color=[STATUS_COLORS[s] for s in seen["Data drift status"]], size=9)))
                fig.add_trace(go.Scatter(x=seen["Batch"], y=seen["Mean PSI"], mode="lines", name="Mean PSI", line=dict(dash="dot")))
                fig.add_hline(y=sim["thr"][1], line_dash="dash", line_color=STATUS_COLORS["Significant"], annotation_text="alert threshold")
                fig.update_layout(title="Input drift per batch (PSI vs reference)", xaxis_title="Batch", yaxis_title="PSI")
                st.plotly_chart(fig)
                if stask:
                    pm = "MAE" if stask == "regression" else "Accuracy"
                    measured = bdf[bdf["Labels available at batch"] <= cur]
                    if len(measured):
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=measured["Batch"], y=measured[f"Production {pm}"], mode="lines+markers", name="Production (frozen baseline)"))
                        fig.add_trace(go.Scatter(x=measured["Batch"], y=measured[f"Candidate {pm}"], mode="lines+markers", name="Shadow candidate",
                                                 line=dict(dash="dash")))
                        for _, vr in vdf[(vdf["Batch"] > 0) & (vdf["Batch"] <= cur)].iterrows():
                            fig.add_vline(x=vr["Batch"], line_color="lightgray", line_width=1)
                        for _, ar in adf[(adf["Batch"] <= cur) & (adf["Type"].str.startswith("Prediction"))].iterrows():
                            vline(fig, ar["Batch"], "perf alert")
                        fig.update_layout(title=f"{pm} per batch (shown once labels have arrived; gray lines = candidate updates)",
                                          xaxis_title="Batch (when predicted)", yaxis_title=pm)
                        st.plotly_chart(fig)
                        st.dataframe(measured.iloc[[-1]][["Batch"] + [c for c in measured.columns if c.startswith(("Production", "Candidate"))]].round(4), hide_index=True)
                    pending = bdf[(bdf["Labels available at batch"] > cur)].shape[0]
                    if pending:
                        st.info(f"{pending} most recent batch(es) have predictions but their labels have not arrived yet, so their performance is unknown.")
                    upd = vdf[(vdf["Batch"] > 0) & (vdf["Batch"] <= cur)]
                    if len(upd):
                        first_u = int(upd["Batch"].min())
                        after = measured[measured["Batch"] > first_u]
                        if len(after):
                            pw, cw = after[f"Production {pm}"].mean(), after[f"Candidate {pm}"].mean()
                            better = (cw < pw) if stask == "regression" else (cw > pw)
                            st.markdown(f"**Before vs after adaptation** (batches measured after the first candidate update, n={len(after)}): production {pm} {pw:.4g}, "
                                        f"candidate {cw:.4g}. The candidate was {'better' if better else 'not better'} on average over these batches. "
                                        "It is a shadow model and has **not** replaced production.")
                st.markdown("**Alerts so far**")
                st.dataframe(adf[adf["Batch"] <= cur], hide_index=True)
                if stask:
                    st.markdown("**Model version history**")
                    st.dataframe(vdf[vdf["Batch"] <= cur], hide_index=True)
                recs = []
                if (adf[adf["Batch"] <= cur]["Type"] == "Input (feature) drift").any():
                    recs.append("Input drift was flagged: inspect the drifting features, check for upstream data changes, and keep monitoring. This alone does not show the model got worse.")
                if stask and (adf[adf["Batch"] <= cur]["Type"].str.startswith("Prediction")).any():
                    recs.append("Prediction errors changed: verify labels, then consider retraining or updating a compatible model, and validate any candidate on recent labelled data before replacing production.")
                if stask and sim["delay"] > 0:
                    recs.append(f"Labels arrive {sim['delay']} batch(es) late: performance for the newest batches is unknown. Prioritise collecting labels.")
                if not stask:
                    recs.append("No target is selected: collect labels to measure performance and detect concept drift.")
                st.markdown("**Recommended next steps at this point in the replay**")
                for r_ in recs or ["No alerts so far: continue monitoring."]:
                    st.markdown(f"- {r_}")

# ======================================================= Results and Downloads
with tabs[6]:
    st.subheader("Results and downloads")
    st.markdown("Everything below is generated from **your** uploaded file and the settings you chose. Items appear after the relevant analysis has run.")
    if not exports:
        st.info("No results yet.")
    for name, frame in exports.items():
        st.download_button(f"Download {name}", csv_bytes(frame), name, "text/csv", key=f"dl_{name}")
    st.subheader("Limitations to keep in mind")
    st.markdown("""
- Without real target labels, model performance and concept drift **cannot** be measured; only input drift (PSI) is available.
- Incremental, online and continual strategies need an estimator with `partial_fit`; random forests are limited to retraining.
- Transfer learning needs a separate source dataset with matching features and target; it is unavailable otherwise.
- PSI thresholds are rules of thumb; small windows inflate PSI. The ADWIN-style detector is a simplified version and needs many labelled rows.
- One split and seed give one noisy comparison; results are indicative, not a universal ranking of methods.
- Replaying a CSV is a simulation. Nothing is deployed or replaced automatically.
""")
