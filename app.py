import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(
    page_title="Drift & Learning Strategy Lab",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Model Drift & Learning Strategy Comparison")
st.caption(
    "Compare incremental learning, continual learning, and transfer learning "
    "on a simulated live sensor-forecasting problem."
)

st.info(
    "This is a controlled simulation for learning and demonstration. "
    "Synthetic outcomes are available so the dashboard can measure prediction error. "
    "Real systems may have delayed or missing labels."
)

with st.sidebar:
    st.header("Simulation settings")
    n_rows = st.slider("Number of time steps", 1500, 10000, 4000, step=500)
    drift_point_pct = st.slider("Drift begins at (%)", 40, 80, 60, step=5)
    drift_type = st.selectbox(
        "Drift type",
        ["Concept drift", "Data drift", "Both data + concept drift"],
        index=2,
        help=(
            "Data drift changes the input distribution. Concept drift changes "
            "the relationship between inputs and the target."
        ),
    )
    noise = st.slider("Sensor/target noise", 0.1, 5.0, 1.0, 0.1)
    batch_size = st.slider("Update batch size", 16, 256, 64, step=16)
    replay_size = st.slider("Continual-learning replay examples", 0, 1000, 200, step=50)
    source_similarity = st.slider(
        "Transfer source similarity", 0.2, 1.0, 0.8, 0.1,
        help="Higher means the source machine/task is more similar to the target."
    )
    random_seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)
    run_button = st.button("Run comparison", type="primary", use_container_width=True)

if "results" not in st.session_state:
    st.session_state.results = None

def make_data(n, drift_point_pct, drift_type, noise, source_similarity, seed):
    """Generate a time-ordered sensor series with controllable drift."""
    rng = np.random.default_rng(int(seed))
    t = np.arange(n)
    drift_idx = int(n * drift_point_pct / 100.0)

    # Sensor features with seasonal and slowly changing behavior.
    temp = 25 + 4 * np.sin(2 * np.pi * t / 120) + 0.0015 * t + rng.normal(0, 0.7, n)
    pressure = 40 + 3 * np.sin(2 * np.pi * t / 75 + 0.6) + rng.normal(0, 0.8, n)
    vibration = 3 + 0.5 * np.sin(2 * np.pi * t / 35) + rng.normal(0, 0.15, n)

    is_data_drift = drift_type in ("Data drift", "Both data + concept drift")
    is_concept_drift = drift_type in ("Concept drift", "Both data + concept drift")

    if is_data_drift:
        temp[drift_idx:] += 5.0
        pressure[drift_idx:] += 4.0
        vibration[drift_idx:] += 0.8

    # Before drift, target follows a stable nonlinear-ish sensor relationship.
    y = (
        0.9 * temp
        + 0.55 * pressure
        + 4.5 * vibration
        + 2.5 * np.sin(temp / 5)
        + 0.012 * t
    )

    if is_concept_drift:
        # Relationship between inputs and target changes after the drift point.
        y[drift_idx:] = (
            0.45 * temp[drift_idx:]
            + 0.95 * pressure[drift_idx:]
            + 10.0 * vibration[drift_idx:]
            + 4.0 * np.cos(pressure[drift_idx:] / 6)
            + 0.004 * t[drift_idx:]
            + 8.0
        )

    y += rng.normal(0, noise, n)

    frame = pd.DataFrame({
        "time_step": t,
        "temperature": temp,
        "pressure": pressure,
        "vibration": vibration,
        "target": y,
        "period": np.where(t < drift_idx, "Before drift", "After drift"),
    })

    # Related source-domain dataset used only by the transfer strategy.
    n_source = max(500, drift_idx)
    src_rng = np.random.default_rng(int(seed) + 101)
    st = np.arange(n_source)
    src_temp = 24 + 3.5 * np.sin(2 * np.pi * st / 120) + src_rng.normal(0, 0.7, n_source)
    src_pressure = 39 + 2.5 * np.sin(2 * np.pi * st / 75 + 0.6) + src_rng.normal(0, 0.8, n_source)
    src_vibration = 2.8 + 0.4 * np.sin(2 * np.pi * st / 35) + src_rng.normal(0, 0.15, n_source)

    # Source task is similar, with controllable difference in its target mapping.
    similarity = float(source_similarity)
    src_y = (
        (0.9 * similarity + 0.25) * src_temp
        + (0.55 * similarity + 0.25) * src_pressure
        + (4.5 * similarity + 1.5) * src_vibration
        + 2.5 * np.sin(src_temp / 5)
        + 0.01 * st
        + src_rng.normal(0, noise, n_source)
    )
    source = pd.DataFrame({
        "temperature": src_temp,
        "pressure": src_pressure,
        "vibration": src_vibration,
        "target": src_y,
    })
    return frame, source, drift_idx

FEATURES = ["temperature", "pressure", "vibration"]

def new_model(seed):
    return MLPRegressor(
        hidden_layer_sizes=(48, 24),
        activation="relu",
        solver="adam",
        learning_rate_init=0.002,
        max_iter=1,
        warm_start=True,
        random_state=int(seed),
        shuffle=False,
    )

def fit_epochs(model, X, y, epochs=12):
    # Repeated fit with warm_start makes a compact educational training loop.
    for _ in range(epochs):
        model.fit(X, y)
    return model

def score(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }

def run_experiment(n_rows, drift_point_pct, drift_type, noise, batch_size, replay_size, source_similarity, seed):
    df, source, drift_idx = make_data(
        n_rows, drift_point_pct, drift_type, noise, source_similarity, seed
    )

    # Chronological partitions:
    # initial training ends before drift; adaptation data comes after drift;
    # final test is later than adaptation data and is never used for updates.
    train_end = max(200, int(drift_idx * 0.85))
    post_start = drift_idx
    post_len = n_rows - post_start
    adapt_end = post_start + max(50, int(post_len * 0.55))
    adapt_end = min(adapt_end, n_rows - 20)

    initial = df.iloc[:train_end].copy()
    post_adapt = df.iloc[post_start:adapt_end].copy()
    test = df.iloc[adapt_end:].copy()

    if len(post_adapt) < 20 or len(test) < 20:
        raise ValueError("Not enough post-drift rows. Increase time steps or move drift earlier.")

    # Standardization is fitted only on the original training data to avoid leakage.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(initial[FEATURES])
    X_adapt = scaler.transform(post_adapt[FEATURES])
    X_test = scaler.transform(test[FEATURES])
    y_train = initial["target"].to_numpy()
    y_adapt = post_adapt["target"].to_numpy()
    y_test = test["target"].to_numpy()

    # Shared original model, cloned by independently fitting the same architecture.
    baseline = fit_epochs(new_model(seed), X_train, y_train, epochs=18)
    baseline_pred = baseline.predict(X_test)
    baseline_metrics = score(y_test, baseline_pred)

    # Incremental: update same architecture in sequential mini-batches.
    incremental = fit_epochs(new_model(seed), X_train, y_train, epochs=18)
    for start in range(0, len(X_adapt), batch_size):
        end = min(start + batch_size, len(X_adapt))
        incremental = fit_epochs(incremental, X_adapt[start:end], y_adapt[start:end], epochs=3)
    inc_pred = incremental.predict(X_test)
    inc_metrics = score(y_test, inc_pred)

    # Continual: mini-batch updates plus replay samples from historical training data.
    continual = fit_epochs(new_model(seed), X_train, y_train, epochs=18)
    rng = np.random.default_rng(int(seed) + 7)
    X_old = X_train
    y_old = y_train
    for start in range(0, len(X_adapt), batch_size):
        end = min(start + batch_size, len(X_adapt))
        X_batch = X_adapt[start:end]
        y_batch = y_adapt[start:end]
        if replay_size > 0:
            count = min(replay_size, len(X_old))
            idx = rng.choice(len(X_old), size=count, replace=False)
            X_update = np.vstack([X_batch, X_old[idx]])
            y_update = np.concatenate([y_batch, y_old[idx]])
        else:
            X_update, y_update = X_batch, y_batch
        continual = fit_epochs(continual, X_update, y_update, epochs=3)
    cont_pred = continual.predict(X_test)
    cont_metrics = score(y_test, cont_pred)

    # Transfer: pretrain the same architecture on a related source domain, then adapt
    # using the target's pre-drift training data and fine-tune on post-drift labels.
    transfer_scaler = StandardScaler()
    X_source = transfer_scaler.fit_transform(source[FEATURES])
    X_initial_t = transfer_scaler.transform(initial[FEATURES])
    X_adapt_t = transfer_scaler.transform(post_adapt[FEATURES])
    X_test_t = transfer_scaler.transform(test[FEATURES])

    transfer = fit_epochs(new_model(int(seed) + 13), X_source, source["target"].to_numpy(), epochs=18)
    # Fine-tune on the target's pre-drift examples to establish the target domain.
    transfer = fit_epochs(transfer, X_initial_t, y_train, epochs=8)
    # Then adapt to the new regime.
    for start in range(0, len(X_adapt_t), batch_size):
        end = min(start + batch_size, len(X_adapt_t))
        transfer = fit_epochs(transfer, X_adapt_t[start:end], y_adapt[start:end], epochs=3)
    trans_pred = transfer.predict(X_test_t)
    trans_metrics = score(y_test, trans_pred)

    # Prediction series for charting.
    pred_df = pd.DataFrame({
        "time_step": test["time_step"].to_numpy(),
        "Actual": y_test,
        "No adaptation": baseline_pred,
        "Incremental": inc_pred,
        "Continual (replay)": cont_pred,
        "Transfer + fine-tuning": trans_pred,
    })
    metrics_df = pd.DataFrame([
        {"Method": "No adaptation (baseline)", **baseline_metrics},
        {"Method": "Incremental learning", **inc_metrics},
        {"Method": "Continual learning (replay)", **cont_metrics},
        {"Method": "Transfer learning + fine-tuning", **trans_metrics},
    ])
    metrics_df["MAE improvement vs baseline (%)"] = (
        (baseline_metrics["MAE"] - metrics_df["MAE"]) / max(baseline_metrics["MAE"], 1e-9) * 100
    )
    return df, metrics_df, pred_df, drift_idx, train_end, adapt_end

if run_button or st.session_state.results is None:
    try:
        with st.spinner("Generating data and training comparison models..."):
            st.session_state.results = run_experiment(
                n_rows, drift_point_pct, drift_type, noise, batch_size,
                replay_size, source_similarity, random_seed
            )
    except Exception as exc:
        st.error(f"Could not run the experiment: {exc}")
        st.stop()

df, metrics_df, pred_df, drift_idx, train_end, adapt_end = st.session_state.results

# Top-line metrics
baseline_row = metrics_df.iloc[0]
best_row = metrics_df.loc[metrics_df["MAE"].idxmin()]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows generated", f"{len(df):,}")
c2.metric("Drift starts at", f"t = {drift_idx:,}")
c3.metric("Best method (test MAE)", best_row["Method"])
c4.metric(
    "Best test MAE",
    f'{best_row["MAE"]:.3f}',
    f'{best_row["MAE improvement vs baseline (%)"]:.1f}% vs baseline'
)

tab1, tab2, tab3, tab4 = st.tabs([
    "Overview", "Model comparison", "Data & drift", "How to interpret"
])

with tab1:
    st.subheader("What happens around drift?")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["time_step"], y=df["target"], mode="lines", name="Actual target",
        line=dict(width=1.5)
    ))
    fig.add_vline(x=drift_idx, line_dash="dash", line_color="red",
                  annotation_text="Drift begins", annotation_position="top")
    fig.add_vline(x=adapt_end, line_dash="dot", line_color="green",
                  annotation_text="Test period begins", annotation_position="top")
    fig.update_layout(
        xaxis_title="Time step", yaxis_title="Target value",
        height=420, legend_title="Series", margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        f"**Timeline:** initial training ends at step `{train_end}`; drift begins at `{drift_idx}`; "
        f"adaptation data ends at `{adapt_end}`; the final test uses later unseen rows."
    )
    st.subheader("Key result")
    st.write(
        f"On the held-out post-drift test period, **{best_row['Method']}** achieved the lowest "
        f"MAE ({best_row['MAE']:.3f}) in this simulation. Results depend on the generated data "
        "and settings; they are not a universal ranking of the methods."
    )

with tab2:
    st.subheader("Performance on unseen post-drift data")
    display_metrics = metrics_df.copy()
    for col in ["MAE", "RMSE", "MAE improvement vs baseline (%)"]:
        display_metrics[col] = display_metrics[col].map(lambda x: f"{x:.3f}")
    st.dataframe(display_metrics, use_container_width=True, hide_index=True)
    metric_choice = st.radio("Metric to chart", ["MAE", "RMSE"], horizontal=True)
    fig_bar = px.bar(
        metrics_df, x="Method", y=metric_choice, color="Method",
        title=f"{metric_choice} on held-out post-drift data",
        text_auto=".3f"
    )
    fig_bar.update_layout(showlegend=False, xaxis_title="", yaxis_title=metric_choice, height=420)
    st.plotly_chart(fig_bar, use_container_width=True)
    st.caption("Lower MAE/RMSE is better. All methods are evaluated on the same later test period.")

    st.subheader("Actual vs predicted")
    chosen_methods = st.multiselect(
        "Choose prediction series",
        ["No adaptation", "Incremental", "Continual (replay)", "Transfer + fine-tuning"],
        default=["No adaptation", "Incremental", "Continual (replay)", "Transfer + fine-tuning"]
    )
    plot_df = pred_df.melt(
        id_vars=["time_step", "Actual"],
        value_vars=chosen_methods,
        var_name="Method", value_name="Prediction"
    )
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=pred_df["time_step"], y=pred_df["Actual"], mode="lines",
        name="Actual", line=dict(width=3, color="#222222")
    ))
    for method in chosen_methods:
        fig_line.add_trace(go.Scatter(
            x=pred_df["time_step"], y=pred_df[method], mode="lines", name=method
        ))
    fig_line.update_layout(
        xaxis_title="Time step", yaxis_title="Target",
        height=480, margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig_line, use_container_width=True)

with tab3:
    st.subheader("Generated sensor data")
    st.dataframe(df.head(30), use_container_width=True, hide_index=True)
    col_a, col_b = st.columns(2)
    with col_a:
        feature = st.selectbox("Feature distribution", FEATURES)
        fig_hist = px.histogram(
            df, x=feature, color="period", barmode="overlay", opacity=0.65,
            title=f"{feature} before vs after drift"
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    with col_b:
        corr = df[FEATURES + ["target"]].corr(numeric_only=True)
        fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto", title="Feature/target correlation")
        st.plotly_chart(fig_corr, use_container_width=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download simulated sensor data (CSV)",
        data=csv,
        file_name="simulated_sensor_drift.csv",
        mime="text/csv"
    )

with tab4:
    st.subheader("What each method means in this app")
    st.markdown("""
    - **No adaptation:** the original model is left unchanged after drift.
    - **Incremental learning:** the model is updated sequentially with post-drift labeled batches.
    - **Continual learning (replay):** the model is updated with new batches mixed with selected examples from its old training data. Replay is a simple continual-learning strategy; it is not the only one.
    - **Transfer learning + fine-tuning:** the model first learns from a related synthetic source machine/task, then adapts to target-machine data.

    **Fairness note:** the model architecture is kept the same for all strategies, but the training histories differ by design. This is an educational comparison, not a rigorous benchmark. Repeat experiments with several seeds and use a realistic dataset before drawing conclusions.

    **Label note:** supervised updates use the known synthetic target values. In a real live system, actual outcomes may arrive late. Do not use the model's own predictions as if they were ground-truth labels.

    **Operational note:** continue ingesting live data during adaptation, but validate the candidate model before replacing the deployed model. If degraded predictions are unsafe, use a validated fallback or operational safeguard.
    """)
    st.subheader("Download results")
    st.download_button(
        "Download comparison metrics (CSV)",
        data=metrics_df.to_csv(index=False).encode("utf-8"),
        file_name="learning_method_comparison.csv",
        mime="text/csv"
    )
    st.download_button(
        "Download held-out predictions (CSV)",
        data=pred_df.to_csv(index=False).encode("utf-8"),
        file_name="post_drift_predictions.csv",
        mime="text/csv"
    )

st.divider()
st.caption("Educational demo • Synthetic data • Streamlit + Plotly + scikit-learn")
