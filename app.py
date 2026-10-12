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
