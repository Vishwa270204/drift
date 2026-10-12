"""
Model Drift Dashboard
---------------------
A single-page Streamlit app for CSV-based streaming simulation, drift monitoring,
and user-controlled model adaptation.

Install:
    pip install streamlit pandas numpy plotly scikit-learn river

Run:
    streamlit run app.py

Important:
- This simulates streaming from a static CSV; it is not a live broker/production stream.
- Error-based drift monitoring requires ground-truth labels to become available.
- River's Hoeffding Adaptive Tree uses ADWIN internally; PSI is monitored separately
  for feature-distribution changes.
"""

from __future__ import annotations

import copy
import io
import warnings
from typing import Any

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
    river_drift = river_tree = None
    RIVER_OK = False

warnings.filterwarnings("ignore", category=UserWarning, module="river")

st.set_page_config(page_title="Model Drift Dashboard", page_icon="📈", layout="wide")

MIN_ROWS = 30
PALETTE = ["#2878b5", "#e08a00", "#7a4fb5", "#00897b", "#c2185b"]
CLASS_METRICS = ["Error rate", "Macro F1", "Fault-class miss rate"]
REG_METRICS = ["MAE", "RMSE"]


# ---------------------------------------------------------------------------
# CSV, schema and model helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Reading CSV...")
def read_csv_bytes(data: bytes) -> pd.DataFrame:
    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            frame = pd.read_csv(io.BytesIO(data), encoding=encoding)
            if frame.shape[1] == 1:
                frame = pd.read_csv(io.BytesIO(data), encoding=encoding, sep=None, engine="python")
            if frame.empty:
                raise ValueError("The CSV has no data rows.")
            if frame.columns.duplicated().any():
                raise ValueError("The CSV contains duplicate column names.")
            return frame
        except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
            last_error = exc
    raise ValueError(f"Could not read this CSV: {last_error}")


def normalize_target(series: pd.Series, task: str) -> np.ndarray:
    if task == "regression":
        vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(vals).all():
            raise ValueError("The selected regression target has missing or non-numeric values.")
        return vals
    return series.astype(str).to_numpy()


def infer_task(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > 10:
        return "regression"
    return "classification"


def choose_features(frame: pd.DataFrame, target: str, time_col: str | None,
                    selected_features: list[str]) -> tuple[list[str], list[str]]:
    excluded = {target}
    if time_col and time_col != "(row order)":
        excluded.add(time_col)
    features = [c for c in selected_features if c not in excluded]
    if not features:
        raise ValueError("Select at least one feature column other than the target/time column.")

    numeric, categorical = [], []
    for col in features:
        if frame[col].notna().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(frame[col]) and not pd.api.types.is_bool_dtype(frame[col]):
            numeric.append(col)
        else:
            categorical.append(col)
    if not numeric and not categorical:
        raise ValueError("No usable feature columns remain.")
    return numeric, categorical


def prepare_features(frame: pd.DataFrame, numeric: list[str],
                     categorical: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for col in numeric:
        vals = pd.to_numeric(frame[col], errors="coerce").astype(float)
        vals = vals.replace([np.inf, -np.inf], np.nan)
        out[col] = vals
    for col in categorical:
        out[col] = frame[col].where(frame[col].notna(), "(missing)").astype(str)
    return out


def psi_numeric(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Population Stability Index for one numeric feature."""
    ref = pd.to_numeric(reference, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    cur = pd.to_numeric(current, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(ref) < 2 or len(cur) < 2 or ref.nunique() < 2:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return float("nan")
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts = np.histogram(ref, bins=edges)[0].astype(float)
    cur_counts = np.histogram(cur, bins=edges)[0].astype(float)
    ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-6, None)
    cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    ref = reference.fillna("(missing)").astype(str)
    cur = current.fillna("(missing)").astype(str)
    categories = sorted(set(ref.unique()) | set(cur.unique()))
    if not categories:
        return float("nan")
    ref_counts = ref.value_counts().reindex(categories, fill_value=0).to_numpy(float)
    cur_counts = cur.value_counts().reindex(categories, fill_value=0).to_numpy(float)
    ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-6, None)
    cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def feature_psi(reference: pd.DataFrame, current: pd.DataFrame,
                numeric: list[str], categorical: list[str]) -> dict[str, float]:
    scores = {}
    for col in numeric:
        scores[col] = psi_numeric(reference[col], current[col])
    for col in categorical:
        scores[col] = psi_categorical(reference[col], current[col])
    return scores


class StreamModel:
    """Hoeffding Adaptive Tree with median imputation for numeric features."""

    def __init__(self, task: str, numeric: list[str], categorical: list[str],
                 hp: dict[str, Any], seed: int):
        if not RIVER_OK:
            raise RuntimeError("River is missing. Install it with: pip install river")
        self.task = task
        self.numeric = list(numeric)
        self.categorical = list(categorical)
        self.features = self.numeric + self.categorical
        self.hp = dict(hp)
        self.seed = int(seed)
        self.medians: dict[str, float] = {}
        self.fallback_class: str | None = None
        self.model = None

    def _clean_rows(self, x: pd.DataFrame) -> list[dict[str, Any]]:
        z = x[self.features].copy()
        for col in self.numeric:
            z[col] = pd.to_numeric(z[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            z[col] = z[col].fillna(self.medians.get(col, 0.0))
        for col in self.categorical:
            z[col] = z[col].where(z[col].notna(), "(missing)").astype(str)
        return z.to_dict("records")

    def _new_model(self):
        common = dict(
            grace_period=int(self.hp["grace_period"]),
            delta=float(self.hp["delta"]),
            tau=float(self.hp["tau"]),
            nominal_attributes=list(self.categorical) or None,
            drift_detector=river_drift.ADWIN(delta=float(self.hp["adwin_delta"])),
            seed=self.seed,
        )
        if self.task == "regression":
            return river_tree.HoeffdingAdaptiveTreeRegressor(**common)
        return river_tree.HoeffdingAdaptiveTreeClassifier(**common)

    def fit(self, x: pd.DataFrame, y: np.ndarray):
        if len(x) != len(y) or len(y) < MIN_ROWS:
            raise ValueError(f"Training needs at least {MIN_ROWS} aligned labeled rows.")
        self.medians = {
            col: float(pd.to_numeric(x[col], errors="coerce").median())
            if pd.to_numeric(x[col], errors="coerce").notna().any() else 0.0
            for col in self.numeric
        }
        if self.task == "classification":
            values, counts = np.unique(y.astype(str), return_counts=True)
            self.fallback_class = str(values[np.argmax(counts)])
        self.model = self._new_model()
        self.learn(x, y)
        return self

    def learn(self, x: pd.DataFrame, y: np.ndarray):
        if self.model is None:
            raise RuntimeError("Call fit() before learn().")
        if len(x) != len(y):
            raise ValueError("Features and target lengths do not match.")
        rows = self._clean_rows(x)
        for row, target in zip(rows, y):
            label = float(target) if self.task == "regression" else str(target)
            self.model.learn_one(row, label)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        out = []
        for row in self._clean_rows(x):
            pred = (self.model.predict_one(row) if self.task == "regression"
                    else self.model.predict_one(row))
            if self.task == "regression":
                out.append(0.0 if pred is None or not np.isfinite(pred) else float(pred))
            else:
                out.append(self.fallback_class if pred is None else str(pred))
        return np.asarray(out, dtype=float if self.task == "regression" else str)


def row_losses(task: str, y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    if task == "regression":
        return np.abs(np.asarray(y, dtype=float) - np.asarray(pred, dtype=float))
    return (np.asarray(y).astype(str) != np.asarray(pred).astype(str)).astype(float)


def metric_values(task: str, y: np.ndarray, pred: np.ndarray,
                  positive_class: str | None = None) -> dict[str, float]:
    if task == "regression":
        err = np.asarray(y, dtype=float) - np.asarray(pred, dtype=float)
        return {"MAE": float(np.mean(np.abs(err))),
                "RMSE": float(np.sqrt(np.mean(err ** 2)))}
    y = np.asarray(y).astype(str)
    pred = np.asarray(pred).astype(str)
    labels = np.unique(np.concatenate([y, pred]))
    result = {
        "Error rate": float(np.mean(y != pred)),
        "Macro F1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
    }
    if positive_class is not None and np.any(y == positive_class):
        _, recall, _, _ = precision_recall_fscore_support(
            y, pred, labels=[positive_class], average=None, zero_division=0
        )
        result["Fault-class miss rate"] = float(1 - recall[0])
    else:
        result["Fault-class miss rate"] = float("nan")
    return result


def metric_score(metric: str, values: dict[str, float]) -> float:
    value = values.get(metric, float("nan"))
    if metric == "Macro F1":
        return 1.0 - value
    return value


# ---------------------------------------------------------------------------
# Streaming session and adaptation
# ---------------------------------------------------------------------------

def start_session(frame: pd.DataFrame, target: str, time_col: str,
                  task: str, selected_features: list[str], config: dict[str, Any]):
    if not RIVER_OK:
        raise ValueError("River is not installed. Run: pip install river")

    data = frame.copy()
    if target not in data.columns:
        raise ValueError("Target column is missing.")
    if time_col != "(row order)":
        if time_col not in data.columns or time_col == target:
            raise ValueError("Choose a valid time column.")
        parsed = pd.to_datetime(data[time_col], errors="coerce", utc=True)
        # If values are numeric, retain numeric ordering instead of parsing as dates.
        if pd.api.types.is_numeric_dtype(data[time_col]):
            parsed = pd.to_numeric(data[time_col], errors="coerce")
        data["_stream_sort_key"] = parsed
        data = data.loc[data["_stream_sort_key"].notna()].sort_values(
            "_stream_sort_key", kind="stable"
        ).drop(columns="_stream_sort_key")
    data = data.loc[data[target].notna()].reset_index(drop=True)
    if len(data) < 400:
        raise ValueError("Use at least 400 rows with non-missing target values for this demo.")

    y = normalize_target(data[target], task)
    numeric, categorical = choose_features(data, target, time_col, selected_features)
    x = prepare_features(data, numeric, categorical)

    batch_size = int(config["batch_size"])
    train_end = int(len(data) * config["train_pct"] / 100)
    fit_end = int(train_end * 0.8)
    calibration_end = train_end
    if fit_end < MIN_ROWS or calibration_end - fit_end < MIN_ROWS:
        raise ValueError("Initial training/calibration split is too small.")
    if len(data) - train_end < 2 * batch_size:
        raise ValueError("Streaming section must contain at least two batches; lower batch size or training percentage.")

    hp = {
        "grace_period": config["grace_period"],
        "delta": config["delta"],
        "tau": config["tau"],
        "adwin_delta": config["adwin_delta"],
    }
    model = StreamModel(task, numeric, categorical, hp, config["seed"]).fit(
        x.iloc[:fit_end], y[:fit_end]
    )
    calibration_pred = model.predict(x.iloc[fit_end:calibration_end])
    calibration_loss = row_losses(task, y[fit_end:calibration_end], calibration_pred)
    ref_mean = float(np.mean(calibration_loss))
    ref_std = float(np.std(calibration_loss, ddof=1)) if len(calibration_loss) > 1 else 0.0

    return {
        "data": data, "X": x, "y": y, "target": target, "time_col": time_col,
        "task": task, "numeric": numeric, "categorical": categorical,
        "config": config, "hp": hp, "n_train": train_end, "fit_end": fit_end,
        "batch_size": batch_size, "stream_n": len(data) - train_end,
        "n_batches": int(np.ceil((len(data) - train_end) / batch_size)),
        "k": 0, "model": model, "model_version": 1,
        "ref_mean": ref_mean, "ref_std": ref_std,
        "batch_errors": [], "thresholds": [], "batch_versions": [],
        "batch_metrics": [], "psi_records": [], "events": [],
        "drift_streak": 0, "drift_alert": False, "drift_batch": None,
        "updates": [], "watch": None, "stream_predictions": [],
        "paused_for_decision": False,
        # UI defaults are initialized here because stream buttons can be clicked
        # before the configuration controls are revisited on a later rerun.
        "response_mode": "Manual",
        "auto_learn_stream": False,
        "selection_metric": "MAE" if task == "regression" else "Error rate",
        "validation_fraction": 0.25,
        "acceptance_tolerance": 0.0,
        "window_rows": 500,
        "label_delay": 0,
        "rollback_batches": 3,
        "rollback_tolerance": 0.0,
    }


def log_event(s: dict, event: str, detail: str):
    s["events"].append({
        "Batch": s["k"], "Event": event, "Details": detail,
        "Model version": s["model_version"],
    })


def calculate_threshold(s: dict, batch_len: int) -> float:
    # Heuristic threshold, not a formal statistical guarantee.
    return s["ref_mean"] + s["config"]["sensitivity_z"] * max(
        s["ref_std"], 0.05
    ) / np.sqrt(max(batch_len, 1))


def process_one_batch(s: dict):
    if s["k"] >= s["n_batches"]:
        return

    start = s["n_train"] + s["k"] * s["batch_size"]
    end = min(start + s["batch_size"], len(s["y"]))
    x_batch, y_batch = s["X"].iloc[start:end], s["y"][start:end]

    # Score before learning from this batch to avoid evaluating on fitted labels.
    predictions = s["model"].predict(x_batch)
    losses = row_losses(s["task"], y_batch, predictions)
    metrics = metric_values(s["task"], y_batch, predictions, s["config"].get("positive_class"))
    error = float(np.mean(losses))
    threshold = calculate_threshold(s, len(y_batch))

    # Feature drift compares the batch to the initial training/reference window.
    ref_x = s["X"].iloc[:s["fit_end"]]
    psi = feature_psi(ref_x, x_batch, s["numeric"], s["categorical"])
    max_psi = max((v for v in psi.values() if np.isfinite(v)), default=float("nan"))

    s["batch_errors"].append(error)
    s["thresholds"].append(threshold)
    s["batch_versions"].append(s["model_version"])
    s["batch_metrics"].append(metrics)
    s["psi_records"].append({"Batch": s["k"] + 1, **psi, "Max PSI": max_psi})
    s["stream_predictions"].append({
        "Batch": s["k"] + 1, "Rows": len(y_batch), "Model version": s["model_version"],
        "Error": error, **metrics, "Max PSI": max_psi,
    })
    s["k"] += 1

    error_flag = error > threshold
    psi_flag = np.isfinite(max_psi) and max_psi >= s["config"]["psi_threshold"]
    if error_flag:
        s["drift_streak"] += 1
    else:
        s["drift_streak"] = 0

    if not s["drift_alert"] and s["drift_streak"] >= 2:
        s["drift_alert"] = True
        s["drift_batch"] = s["k"] - 1
        s["paused_for_decision"] = s["config"].get("response_mode", "Manual") == "Manual"
        log_event(s, "Performance drift detected",
                  "Prediction error exceeded its reference threshold for two consecutive batches.")

    if psi_flag:
        log_event(s, "Feature drift signal",
                  f"Maximum PSI {max_psi:.3f} reached the configured threshold {s['config']['psi_threshold']:.3f}.")

    # Watch candidate model against current model on the same post-deployment batches.
    if s["watch"] is not None:
        w = s["watch"]
        candidate = w["candidate"]
        old_pred = w["old_model"].predict(x_batch)
        candidate_pred = candidate.predict(x_batch)
        old_score = metric_score(w["metric"], metric_values(
            s["task"], y_batch, old_pred, s["config"].get("positive_class")
        ))
        new_score = metric_score(w["metric"], metric_values(
            s["task"], y_batch, candidate_pred, s["config"].get("positive_class")
        ))
        w["old_scores"].append(old_score)
        w["new_scores"].append(new_score)
        if len(w["new_scores"]) >= s["config"]["rollback_batches"]:
            old_avg, new_avg = np.nanmean(w["old_scores"]), np.nanmean(w["new_scores"])
            if np.isfinite(old_avg) and np.isfinite(new_avg) and new_avg > old_avg + s["config"]["rollback_tolerance"]:
                s["model"] = w["old_model"]
                s["model_version"] += 1
                w["record"]["Outcome"] = "Rolled back"
                log_event(s, "Rollback", f"Candidate score {new_avg:.4f} was worse than shadow old-model score {old_avg:.4f}.")
            else:
                w["record"]["Outcome"] = "Kept"
                log_event(s, "Update retained", "Candidate was not worse than the old model on matched post-update batches.")
            s["updates"].append(w["record"])
            s["watch"] = None

    if s["config"]["auto_learn_stream"]:
        # This is an explicitly selected prequential/online mode. Labels are assumed
        # to be available after scoring the batch in this offline simulation.
        s["model"].learn(x_batch, y_batch)


def selected_update_data(s: dict, strategy: str, data_policy: str):
    stream_start = s["n_train"]
    end = stream_start + min(max(s["k"] - s["config"]["label_delay"], 0) * s["batch_size"], s["stream_n"])
    start = stream_start
    if s["drift_batch"] is not None:
        start = stream_start + s["drift_batch"] * s["batch_size"]

    if data_policy == "New drift-period data only":
        x = s["X"].iloc[start:end]
        y = s["y"][start:end]
    elif data_policy == "Recent sliding window":
        n = min(s["config"]["window_rows"], max(end, 0))
        x = s["X"].iloc[max(0, end - n):end]
        y = s["y"][max(0, end - n):end]
    elif data_policy == "Old + new data":
        x = s["X"].iloc[:end]
        y = s["y"][:end]
    else:  # Old + new, with recent rows repeated as a simple recency weighting
        x0, y0 = s["X"].iloc[:end], s["y"][:end]
        recent_start = max(0, end - s["config"]["window_rows"])
        x_recent, y_recent = s["X"].iloc[recent_start:end], s["y"][recent_start:end]
        x = pd.concat([x0, x_recent], ignore_index=True)
        y = np.concatenate([y0, y_recent])
    return x, np.asarray(y)


def perform_update(s: dict, strategy: str, data_policy: str):
    x_new, y_new = selected_update_data(s, strategy, data_policy)
    if len(y_new) < 2 * MIN_ROWS:
        st.error(f"Not enough eligible labeled rows ({len(y_new)}). At least {2 * MIN_ROWS} are needed for a train/validation split.")
        return

    # Keep a chronological holdout at the end. Never use these rows for candidate fit.
    hold_n = max(MIN_ROWS, int(len(y_new) * s["config"]["validation_fraction"]))
    hold_n = min(hold_n, len(y_new) - MIN_ROWS)
    x_fit, y_fit = x_new.iloc[:-hold_n], y_new[:-hold_n]
    x_hold, y_hold = x_new.iloc[-hold_n:], y_new[-hold_n:]

    old_model = copy.deepcopy(s["model"])
    metric = s["config"]["selection_metric"]

    try:
        if strategy == "Incremental learning":
            candidate = copy.deepcopy(s["model"]).learn(x_fit, y_fit)
        elif strategy == "Continual learning":
            # For this tree model, continual learning is implemented as continued
            # incremental updates plus an optional replay policy selected above.
            candidate = copy.deepcopy(s["model"]).learn(x_fit, y_fit)
        elif strategy == "Full retraining":
            candidate = StreamModel(
                s["task"], s["numeric"], s["categorical"], s["hp"], s["config"]["seed"]
            ).fit(x_fit, y_fit)
        elif strategy == "Retuning":
            # Small transparent search over tree settings; evaluate on holdout.
            trials = []
            for grace in (50, 100, 200, 400):
                hp = dict(s["hp"], grace_period=grace)
                cand = StreamModel(
                    s["task"], s["numeric"], s["categorical"], hp, s["config"]["seed"]
                ).fit(x_fit, y_fit)
                pred = cand.predict(x_hold)
                score = metric_score(metric, metric_values(
                    s["task"], y_hold, pred, s["config"].get("positive_class")
                ))
                trials.append((score, cand, hp))
            trials = [t for t in trials if np.isfinite(t[0])]
            if not trials:
                st.error("Retuning could not produce a valid candidate.")
                return
            _, candidate, chosen_hp = min(trials, key=lambda t: t[0])
            s["hp"] = chosen_hp
        elif strategy == "Transfer learning":
            st.warning("Transfer learning is not implemented for this Hoeffding Adaptive Tree. Choose another strategy or add a compatible pretrained model.")
            return
        else:
            st.error("Unknown learning strategy.")
            return

        old_pred = s["model"].predict(x_hold)
        new_pred = candidate.predict(x_hold)
        old_values = metric_values(s["task"], y_hold, old_pred, s["config"].get("positive_class"))
        new_values = metric_values(s["task"], y_hold, new_pred, s["config"].get("positive_class"))
        old_score, new_score = metric_score(metric, old_values), metric_score(metric, new_values)
        if not (np.isfinite(old_score) and np.isfinite(new_score)):
            st.error("Could not compare candidates: selected metric is undefined on the validation rows.")
            return

        improvement = old_score - new_score
        accepted = new_score <= old_score + s["config"]["acceptance_tolerance"]
        st.subheader("Candidate evaluation")
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Current model ({metric})", f"{old_score:.4f}")
        c2.metric(f"Candidate ({metric})", f"{new_score:.4f}", delta=f"{improvement:+.4f} lower-is-better")
        c3.metric("Decision", "Accepted" if accepted else "Rejected")

        if not accepted:
            log_event(s, "Candidate rejected", f"{strategy} score {new_score:.4f} was worse than current score {old_score:.4f}.")
            st.warning("The current model is retained because the candidate did not meet the acceptance rule.")
            return

        # Candidate has not seen the holdout labels. Deploy the evaluated candidate.
        # For incremental candidates this means only the fit subset is added to the old model.
        record = {
            "Version": s["model_version"] + 1,
            "Strategy": strategy,
            "Data policy": data_policy,
            "Rows used for fitting": len(y_fit),
            "Validation rows": len(y_hold),
            "Metric": metric,
            "Old score": old_score,
            "Candidate score": new_score,
            "Outcome": "Watching",
        }
        s["model"] = candidate
        s["model_version"] += 1
        s["watch"] = {
            "old_model": old_model,
            "candidate": copy.deepcopy(candidate),
            "metric": metric,
            "old_scores": [],
            "new_scores": [],
            "record": record,
        }
        s["drift_alert"] = False
        s["paused_for_decision"] = False
        s["drift_streak"] = 0
        log_event(s, "Model updated", f"Deployed v{s['model_version']} using {strategy} and {data_policy}.")
        st.success(f"Candidate deployed as model v{s['model_version']}. It will be monitored for {s['config']['rollback_batches']} future batches.")
    except Exception as exc:
        st.error(f"Update failed safely; current deployed model was not intentionally replaced. Details: {exc}")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def make_error_chart(s: dict):
    fig = go.Figure()
    xs = np.arange(1, len(s["batch_errors"]) + 1)
    versions = np.asarray(s["batch_versions"])
    errors = np.asarray(s["batch_errors"], dtype=float)
    thresholds = np.asarray(s["thresholds"], dtype=float)
    for i, version in enumerate(sorted(set(versions.tolist()))):
        mask = versions == version
        fig.add_trace(go.Scatter(
            x=xs[mask], y=errors[mask], mode="lines+markers",
            name=f"Model v{version}", line=dict(color=PALETTE[i % len(PALETTE)])
        ))
    if len(thresholds):
        fig.add_trace(go.Scatter(
            x=xs, y=thresholds, mode="lines", name="Performance threshold",
            line=dict(color="#c62828", dash="dash")
        ))
    if s["drift_batch"] is not None:
        fig.add_vline(x=s["drift_batch"] + 1, line_dash="dot", line_color="#c62828",
                      annotation_text="Drift alert", annotation_position="top right")
    fig.update_layout(
        title="Prediction error by incoming batch",
        xaxis_title="Batch", yaxis_title="MAE" if s["task"] == "regression" else "Error rate",
        height=380, margin=dict(t=55, b=40, l=45, r=20),
        legend=dict(orientation="h", y=-0.25),
    )
    st.plotly_chart(fig, use_container_width=True)


def make_psi_chart(s: dict):
    if not s["psi_records"]:
        st.info("Feature PSI will appear after the first batch.")
        return
    psi_df = pd.DataFrame(s["psi_records"]).set_index("Batch")
    cols = [c for c in psi_df.columns if c != "Max PSI"]
    if not cols:
        return
    fig = go.Figure()
    for i, col in enumerate(cols):
        fig.add_trace(go.Scatter(
            x=psi_df.index, y=psi_df[col], mode="lines+markers", name=col,
            line=dict(color=PALETTE[i % len(PALETTE)])
        ))
    fig.add_hline(y=s["config"]["psi_threshold"], line_dash="dash", line_color="#c62828",
                  annotation_text="PSI threshold")
    fig.update_layout(
        title="Feature drift (PSI by feature)",
        xaxis_title="Batch", yaxis_title="PSI", height=320,
        margin=dict(t=50, b=40, l=45, r=20),
        legend=dict(orientation="h", y=-0.3),
    )
    st.plotly_chart(fig, use_container_width=True)


def main():
    st.title("📈 Model Drift Dashboard")
    st.caption("Upload a CSV, simulate incoming batches, monitor performance and feature drift, then choose how to adapt.")

    if not RIVER_OK:
        st.error("River is required. Install it with `pip install river` and restart the app.")
        st.stop()

    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded is None:
        st.info("Upload a labeled CSV to begin. The app uses earlier rows for initial training and later rows as a simulated stream.")
        st.stop()

    try:
        frame = read_csv_bytes(uploaded.getvalue())
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    with st.expander("1. Data and initial model settings", expanded=True):
        target = st.selectbox("Target column (the value the model learns to predict)", frame.columns,
                              index=len(frame.columns) - 1)
        time_options = ["(row order)"] + [c for c in frame.columns if c != target]
        time_col = st.selectbox("Time/order column", time_options)
        task_guess = infer_task(frame[target].dropna())
        task = st.radio("Prediction task", ["classification", "regression"],
                        index=0 if task_guess == "classification" else 1, horizontal=True)
        default_features = [c for c in frame.columns if c not in {target, time_col}]
        selected_features = st.multiselect(
            "Feature columns", [c for c in frame.columns if c != target and c != time_col],
            default=default_features,
            help="Choose the input columns. The target is never used as a feature."
        )
        c1, c2, c3 = st.columns(3)
        train_pct = c1.slider("Initial data used for training/calibration (%)", 20, 70, 40, 5)
        batch_size = c2.number_input("Rows per incoming batch", min_value=20, max_value=5000, value=100, step=20)
        seed = c3.number_input("Random seed", min_value=0, max_value=10000, value=42)

        with st.expander("Advanced model and drift settings"):
            a1, a2, a3 = st.columns(3)
            grace_period = a1.number_input("Tree grace period", 10, 5000, 200, step=10)
            delta = a2.select_slider("Tree split confidence (delta)", options=[1e-3, 1e-5, 1e-7, 1e-9], value=1e-7,
                                     format_func=lambda v: f"{v:.0e}")
            tau = a3.number_input("Tree tie threshold (tau)", 0.001, 0.5, 0.05, step=0.01)
            a1, a2, a3 = st.columns(3)
            adwin_delta = a1.select_slider("Tree internal ADWIN delta", options=[0.1, 0.01, 0.002, 0.0005], value=0.002)
            sensitivity_z = a2.select_slider("Performance alert sensitivity", options=[2.0, 3.0, 4.0], value=3.0)
            psi_threshold = a3.number_input("Feature PSI alert threshold", 0.01, 1.0, 0.2, step=0.05)
            classes = sorted(set(frame[target].dropna().astype(str)))[:50]
            positive_class = None
            if task == "classification" and classes:
                default_pos = "1" if "1" in classes else classes[-1]
                positive_class = st.selectbox("Fault/positive class (for miss rate)", classes,
                                              index=classes.index(default_pos))
        config = {
            "train_pct": train_pct, "batch_size": int(batch_size), "seed": int(seed),
            "grace_period": int(grace_period), "delta": float(delta), "tau": float(tau),
            "adwin_delta": float(adwin_delta), "sensitivity_z": float(sensitivity_z),
            "psi_threshold": float(psi_threshold), "positive_class": positive_class,
        }
        if st.button("Train initial model / restart simulation", type="primary"):
            try:
                st.session_state["drift_session"] = start_session(
                    frame, target, time_col, task, selected_features, config
                )
                st.success("Initial model trained. Start the stream below.")
            except Exception as exc:
                st.error(f"Could not start: {exc}")

    s = st.session_state.get("drift_session")
    if s is None:
        st.info("Configure the CSV and train the initial model to continue.")
        st.stop()

    st.divider()
    st.subheader("2. Streaming simulation")
    done = s["k"] >= s["n_batches"]
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("▶ Run all batches", type="primary", disabled=done):
        while s["k"] < s["n_batches"]:
            process_one_batch(s)
            if s["drift_alert"] and s["config"].get("response_mode", "Manual") == "Manual":
                break
        st.rerun()
    if c2.button("Next batch", disabled=done):
        process_one_batch(s)
        st.rerun()
    if c3.button("Next 10 batches", disabled=done):
        for _ in range(min(10, s["n_batches"] - s["k"])):
            process_one_batch(s)
            if s["drift_alert"] and s["config"].get("response_mode", "Manual") == "Manual":
                break
        st.rerun()
    if c4.button("Reset simulation"):
        st.session_state.pop("drift_session", None)
        st.rerun()

    current_status = "Finished" if done else "Drift alert" if s["drift_alert"] else "Monitoring"
    m = st.columns(4)
    m[0].metric("Batches processed", f"{s['k']} / {s['n_batches']}")
    m[1].metric("Model version", f"v{s['model_version']}")
    m[2].metric("Successful/attempted updates", len(s["updates"]))
    m[3].metric("Status", current_status)

    if s["drift_alert"]:
        st.warning("⚠️ Performance drift detected. Incoming data continues to be processed. Choose a response strategy below.")
    elif s["watch"]:
        st.info("Candidate deployed; monitoring it against the previous model on matched future batches.")
    elif done:
        st.success("The simulated stream has finished.")
    else:
        st.info("No confirmed performance drift alert. Feature PSI can still signal a change in input distributions.")

    if s["batch_errors"]:
        make_error_chart(s)
        make_psi_chart(s)

    st.subheader("3. Choose the response to drift")
    c1, c2 = st.columns(2)
    strategy = c1.selectbox("Learning strategy", [
        "Incremental learning", "Continual learning", "Full retraining", "Retuning", "Transfer learning"
    ])
    data_policy = c2.selectbox("Data used for updating", [
        "New drift-period data only",
        "Old + new data",
        "Recent sliding window",
        "Old + new, with recent rows repeated",
    ])
    c1, c2, c3 = st.columns(3)
    selection_metric_options = REG_METRICS if s["task"] == "regression" else CLASS_METRICS
    default_metric = "MAE" if s["task"] == "regression" else "Error rate"
    selection_metric = c1.selectbox("Validation metric (lower is better)", selection_metric_options,
                                    index=selection_metric_options.index(default_metric))
    validation_fraction = c2.slider("Validation share", 0.1, 0.4, 0.25, 0.05)
    acceptance_tolerance = c3.number_input("Allowed score degradation", 0.0, 1.0, 0.0, step=0.01)
    c1, c2, c3 = st.columns(3)
    window_rows = c1.number_input("Recent window rows", 60, 100000, 500, step=50)
    label_delay = c2.number_input("Label delay (batches)", 0, 50, 0)
    rollback_batches = c3.number_input("Future batches before rollback decision", 1, 20, 3)
    rollback_tolerance = st.number_input("Rollback tolerance (score units)", 0.0, 10.0, 0.0, step=0.01)
    response_mode = st.radio("Response mode", ["Manual", "Automatic"], horizontal=True)
    auto_learn_stream = st.checkbox(
        "Online learning: learn from each batch after scoring it",
        value=False,
        help="In this static-CSV simulation, labels are assumed to arrive after each batch is scored."
    )
    s["config"].update({
        "selection_metric": selection_metric,
        "validation_fraction": float(validation_fraction),
        "acceptance_tolerance": float(acceptance_tolerance),
        "window_rows": int(window_rows),
        "label_delay": int(label_delay),
        "rollback_batches": int(rollback_batches),
        "rollback_tolerance": float(rollback_tolerance),
        "response_mode": response_mode,
        "auto_learn_stream": auto_learn_stream,
    })

    if strategy == "Transfer learning":
        st.warning("Transfer learning is shown for comparison but is not supported by the current Hoeffding Adaptive Tree implementation.")
    if st.button("Evaluate and apply selected strategy", type="primary",
                 disabled=not s["drift_alert"] or done):
        perform_update(s, strategy, data_policy)

    st.subheader("4. Update history")
    if s["updates"]:
        st.dataframe(pd.DataFrame(s["updates"]), use_container_width=True, hide_index=True)
    else:
        st.caption("No completed rollback-watch decisions yet.")

    st.subheader("5. Incoming-batch metrics")
    if s["stream_predictions"]:
        st.dataframe(pd.DataFrame(s["stream_predictions"]).round(4),
                     use_container_width=True, hide_index=True)
    if s["events"]:
        with st.expander("Activity log", expanded=False):
            st.dataframe(pd.DataFrame(s["events"]), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
