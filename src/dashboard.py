"""AgroSense ML — Interactive Multi-Tab Streamlit Dashboard."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import (
    CROP_TYPES,
    SAFETY_MOISTURE_THRESHOLD,
    SAFETY_RAINFALL_THRESHOLD,
    load_artifacts,
    run_irrigation_prediction,
)

st.set_page_config(
    page_title="AgroSense ML — Smart Precision Irrigation",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .ag-badge {
        display: inline-block;
        padding: 0.45rem 0.9rem;
        border-radius: 0.6rem;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 0.03em;
        margin-bottom: 0.5rem;
    }
    .ag-badge-on {
        background-color: rgba(16, 185, 129, 0.16);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.45);
    }
    .ag-badge-off {
        background-color: rgba(59, 130, 246, 0.14);
        color: #3b82f6;
        border: 1px solid rgba(59, 130, 246, 0.4);
    }
    .ag-badge-override {
        background-color: rgba(245, 158, 11, 0.18);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.5);
    }
    .ag-badge-clear {
        background-color: rgba(34, 197, 94, 0.12);
        color: #22c55e;
        border: 1px solid rgba(34, 197, 94, 0.35);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model_bundle() -> tuple[Any, Any, dict[str, Any]]:
    """Load saved joblib pipelines and metrics JSON once per server process."""
    return load_artifacts()


SCENARIO_CONFIGS: dict[str, dict[str, float]] = {
    "Normal Field Cycle": {
        "moisture_target": 36.0,
        "temp_offset": 0.0,
        "rain_base": 1.2,
    },
    "Drought / Heatwave": {
        "moisture_target": 19.0,
        "temp_offset": 8.5,
        "rain_base": 0.0,
    },
    "Incoming Rainstorm (Safety Override)": {
        "moisture_target": 48.0,
        "temp_offset": -3.5,
        "rain_base": 21.5,
    },
    "Over-Saturated Soil (Safety Override)": {
        "moisture_target": 81.0,
        "temp_offset": -1.5,
        "rain_base": 2.0,
    },
}


def _simulate_sensor_tick(
    tick_idx: int,
    scenario: str,
    crop_type: str,
    prev_moisture: float | None = None,
) -> dict[str, Any]:
    """Generate one deterministic-yet-realistic ESP32 sensor reading for a simulation tick."""
    cfg = SCENARIO_CONFIGS[scenario]
    rng = np.random.default_rng(1000 + tick_idx * 37)

    hour = (6 + tick_idx) % 24
    diurnal_wave = math.cos((hour - 14.0) * math.pi / 12.0)

    temperature = float(
        np.clip(25.0 + cfg["temp_offset"] + 5.5 * diurnal_wave + rng.normal(0, 0.8), 8.0, 45.0)
    )
    humidity = float(
        np.clip(64.0 - 1.3 * (temperature - 22.0) + rng.normal(0, 2.5), 16.0, 96.0)
    )

    if prev_moisture is None:
        moisture = float(np.clip(cfg["moisture_target"] + rng.normal(0, 2.0), 8.0, 92.0))
    else:
        moisture = float(
            np.clip(
                0.65 * prev_moisture + 0.35 * cfg["moisture_target"] + rng.normal(0, 2.2),
                8.0,
                92.0,
            )
        )

    rainfall = float(np.clip(cfg["rain_base"] + rng.normal(0, 1.1), 0.0, 40.0))

    return {
        "tick": tick_idx,
        "device_id": "esp32-sim-plot-01",
        "soil_moisture": round(moisture, 2),
        "temperature": round(temperature, 2),
        "humidity": round(humidity, 2),
        "rainfall_forecast_mm": round(rainfall, 2),
        "soil_ph": round(float(np.clip(6.5 + rng.normal(0, 0.12), 5.0, 8.0)), 2),
        "nitrogen": round(float(np.clip(72.0 + rng.normal(0, 3.0), 15.0, 140.0)), 1),
        "phosphorus": round(float(np.clip(44.0 + rng.normal(0, 2.0), 8.0, 90.0)), 1),
        "potassium": round(float(np.clip(80.0 + rng.normal(0, 3.0), 15.0, 160.0)), 1),
        "crop_type": crop_type,
        "hour_of_day": int(hour),
    }


def _evaluate_tick(
    tick_idx: int,
    scenario: str,
    crop_type: str,
    clf: Any,
    reg: Any,
    metrics: dict[str, Any],
    prev_moisture: float | None = None,
) -> dict[str, Any]:
    """Simulate a sensor reading and score it with the live ML pipelines."""
    reading = _simulate_sensor_tick(
        tick_idx=tick_idx,
        scenario=scenario,
        crop_type=crop_type,
        prev_moisture=prev_moisture,
    )
    pred = run_irrigation_prediction(reading, clf, reg, metrics)
    return {
        **reading,
        "irrigation_action": pred["irrigation_action"],
        "irrigation_needed": pred["irrigation_needed"],
        "irrigation_probability": pred["irrigation_probability"],
        "recommended_water_liters": pred["recommended_water_liters"],
        "override_active": pred["override_active"],
        "override_reason": pred["override_reason"],
        "vpd_kpa": pred["engineered_features"]["vpd_kpa"],
        "top_driving_factors": pred["top_driving_factors"],
    }


def _init_or_step_history(
    scenario: str,
    crop_type: str,
    clf: Any,
    reg: Any,
    metrics: dict[str, Any],
    step: bool = False,
    reset: bool = False,
) -> list[dict[str, Any]]:
    """Manage simulated IoT sensor stream state."""
    if reset or "sensor_history" not in st.session_state:
        history: list[dict[str, Any]] = []
        prev_m: float | None = None
        for idx in range(18):
            row = _evaluate_tick(
                tick_idx=idx,
                scenario=scenario,
                crop_type=crop_type,
                clf=clf,
                reg=reg,
                metrics=metrics,
                prev_moisture=prev_m,
            )
            history.append(row)
            prev_m = row["soil_moisture"]
        st.session_state.sensor_history = history
        st.session_state.sim_tick_counter = 18
        return history

    history = st.session_state.sensor_history
    if step:
        next_tick = int(st.session_state.get("sim_tick_counter", len(history)))
        prev_m = history[-1]["soil_moisture"] if history else None
        row = _evaluate_tick(
            tick_idx=next_tick,
            scenario=scenario,
            crop_type=crop_type,
            clf=clf,
            reg=reg,
            metrics=metrics,
            prev_moisture=prev_m,
        )
        history.append(row)
        st.session_state.sensor_history = history[-40:]
        st.session_state.sim_tick_counter = next_tick + 1
    return st.session_state.sensor_history


def _render_live_stream_view(history: list[dict[str, Any]]) -> None:
    """Render live KPIs, status badges, and streaming Plotly charts."""
    latest = history[-1]
    hist_df = pd.DataFrame(history)

    badge_col1, badge_col2 = st.columns(2)
    with badge_col1:
        if latest["irrigation_action"] == "ON":
            st.markdown(
                '<div class="ag-badge ag-badge-on">PUMP STATUS: ON — IRRIGATION ACTIVE</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="ag-badge ag-badge-off">PUMP STATUS: OFF — STANDBY</div>',
                unsafe_allow_html=True,
            )

    with badge_col2:
        if latest["override_active"]:
            st.markdown(
                f'<div class="ag-badge ag-badge-override">SAFETY OVERRIDE ACTIVE — {latest["override_reason"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="ag-badge ag-badge-clear">SAFETY OVERRIDE: CLEAR (WITHIN SAFE LIMITS)</div>',
                unsafe_allow_html=True,
            )

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Soil Moisture", f"{latest['soil_moisture']:.1f} %")
    kpi2.metric("Water Volume", f"{latest['recommended_water_liters']:.2f} L/m²")
    kpi3.metric("ML Trigger Prob", f"{latest['irrigation_probability'] * 100:.1f} %")
    kpi4.metric("Rainfall Forecast", f"{latest['rainfall_forecast_mm']:.1f} mm")
    kpi5.metric("VPD / Temp", f"{latest['vpd_kpa']:.2f} kPa", delta=f"{latest['temperature']:.1f} °C")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig_moisture = make_subplots(specs=[[{"secondary_y": True}]])
        fig_moisture.add_trace(
            go.Scatter(
                x=hist_df["tick"],
                y=hist_df["soil_moisture"],
                mode="lines+markers",
                name="Soil Moisture (%)",
                line=dict(color="#10b981", width=3),
                marker=dict(size=6),
            ),
            secondary_y=False,
        )
        on_df = hist_df[hist_df["irrigation_action"] == "ON"]
        if not on_df.empty:
            fig_moisture.add_trace(
                go.Scatter(
                    x=on_df["tick"],
                    y=on_df["soil_moisture"],
                    mode="markers",
                    name="Pump ON Trigger",
                    marker=dict(color="#f59e0b", size=11, symbol="diamond"),
                ),
                secondary_y=False,
            )
        fig_moisture.add_trace(
            go.Bar(
                x=hist_df["tick"],
                y=hist_df["rainfall_forecast_mm"],
                name="Rainfall Forecast (mm)",
                marker_color="rgba(59, 130, 246, 0.35)",
            ),
            secondary_y=True,
        )
        fig_moisture.add_hline(
            y=SAFETY_MOISTURE_THRESHOLD,
            line_dash="dash",
            line_color="#ef4444",
            annotation_text="75% Moisture Safety Ceiling",
            annotation_position="top left",
        )
        fig_moisture.update_layout(
            title=dict(text="Live Soil Moisture & Rainfall Stream", y=0.96),
            height=380,
            margin=dict(l=20, r=20, t=78, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        fig_moisture.update_yaxes(title_text="Soil Moisture (%)", range=[0, 100], secondary_y=False)
        fig_moisture.update_yaxes(title_text="Rainfall (mm)", range=[0, 42], secondary_y=True)
        fig_moisture.update_xaxes(title_text="Sensor Tick")
        st.plotly_chart(fig_moisture, width="stretch")

    with chart_col2:
        fig_vol = make_subplots(specs=[[{"secondary_y": True}]])
        fig_vol.add_trace(
            go.Bar(
                x=hist_df["tick"],
                y=hist_df["recommended_water_liters"],
                name="Recommended Water (L/m²)",
                marker_color="#0ea5e9",
            ),
            secondary_y=False,
        )
        fig_vol.add_trace(
            go.Scatter(
                x=hist_df["tick"],
                y=hist_df["vpd_kpa"],
                mode="lines+markers",
                name="Vapor Pressure Deficit (kPa)",
                line=dict(color="#f97316", width=2.5),
            ),
            secondary_y=True,
        )
        fig_vol.update_layout(
            title=dict(text="Water Requirement & Atmospheric Demand (VPD)", y=0.96),
            height=380,
            margin=dict(l=20, r=20, t=78, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        fig_vol.update_yaxes(title_text="Water Volume (L/m²)", rangemode="tozero", secondary_y=False)
        fig_vol.update_yaxes(title_text="VPD (kPa)", rangemode="tozero", secondary_y=True)
        fig_vol.update_xaxes(title_text="Sensor Tick")
        st.plotly_chart(fig_vol, width="stretch")

    with st.expander("Latest ESP32 Sensor Telemetry & Recent Stream Log", expanded=False):
        display_cols = [
            "tick",
            "crop_type",
            "hour_of_day",
            "soil_moisture",
            "temperature",
            "humidity",
            "rainfall_forecast_mm",
            "irrigation_action",
            "irrigation_probability",
            "recommended_water_liters",
            "override_active",
        ]
        st.dataframe(hist_df[display_cols].tail(10).iloc[::-1], width="stretch", hide_index=True)


def render_tab_live_monitor(clf: Any, reg: Any, metrics: dict[str, Any]) -> None:
    """Tab 1: Live IoT sensor simulation and pump monitoring."""
    st.subheader("Live IoT Field Monitor & ESP32 Stream Simulator")
    st.caption(
        "Simulates incoming ESP32 telemetry scored live by the saved joblib ML pipelines "
        "and safety override controller."
    )

    c1, c2, c3, c4, c5 = st.columns([1.8, 1.1, 1.0, 0.9, 1.1])
    with c1:
        scenario = st.selectbox(
            "Field Scenario",
            options=list(SCENARIO_CONFIGS.keys()),
            key="sim_scenario",
        )
    with c2:
        crop_type = st.selectbox("Monitored Crop", options=list(CROP_TYPES), key="sim_crop")
    with c3:
        st.markdown("<div style='height: 1.7rem;'></div>", unsafe_allow_html=True)
        step_clicked = st.button("Step Sensor (+15m)", width="stretch", type="primary")
    with c4:
        st.markdown("<div style='height: 1.7rem;'></div>", unsafe_allow_html=True)
        reset_clicked = st.button("Reset Stream", width="stretch")
    with c5:
        st.markdown("<div style='height: 1.7rem;'></div>", unsafe_allow_html=True)
        auto_stream = st.toggle("Live Auto-Stream", value=False, key="sim_auto_stream")

    if auto_stream:

        @st.fragment(run_every=2)
        def _auto_stream_fragment() -> None:
            history = _init_or_step_history(
                scenario=scenario,
                crop_type=crop_type,
                clf=clf,
                reg=reg,
                metrics=metrics,
                step=True,
                reset=reset_clicked,
            )
            _render_live_stream_view(history)

        _auto_stream_fragment()
    else:
        history = _init_or_step_history(
            scenario=scenario,
            crop_type=crop_type,
            clf=clf,
            reg=reg,
            metrics=metrics,
            step=step_clicked,
            reset=reset_clicked,
        )
        _render_live_stream_view(history)


WHAT_IF_DEFAULTS: dict[str, Any] = {
    "wi_crop": "Rice",
    "wi_moisture": 24.0,
    "wi_temp": 34.0,
    "wi_humidity": 34.0,
    "wi_rain": 0.5,
    "wi_ph": 6.5,
    "wi_n": 70.0,
    "wi_p": 42.0,
    "wi_k": 80.0,
    "wi_hour": 14,
}


def _apply_what_if_preset(values: dict[str, Any]) -> None:
    """Update What-If slider keys safely inside a button callback."""
    for key, val in values.items():
        st.session_state[key] = val


def render_tab_what_if(clf: Any, reg: Any, metrics: dict[str, Any]) -> None:
    """Tab 2: Interactive What-If ML Predictor with local feature explainability."""
    st.subheader("What-If Agronomic ML Predictor & Explainability")
    st.caption(
        "Adjust soil, weather, NPK, crop, and time-of-day parameters to query the trained "
        "classification and water-volume regression pipelines in real time."
    )

    for key, default_val in WHAT_IF_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default_val

    p1, p2, p3, p4 = st.columns(4)
    p1.button(
        "Preset: Dry Heatwave",
        width="stretch",
        on_click=_apply_what_if_preset,
        args=(
            {
                "wi_crop": "Rice",
                "wi_moisture": 18.0,
                "wi_temp": 38.0,
                "wi_humidity": 24.0,
                "wi_rain": 0.0,
                "wi_hour": 14,
            },
        ),
    )
    p2.button(
        "Preset: Optimal Field",
        width="stretch",
        on_click=_apply_what_if_preset,
        args=(
            {
                "wi_crop": "Wheat",
                "wi_moisture": 56.0,
                "wi_temp": 22.0,
                "wi_humidity": 65.0,
                "wi_rain": 3.0,
                "wi_hour": 8,
            },
        ),
    )
    p3.button(
        "Preset: Heavy Rain Override",
        width="stretch",
        on_click=_apply_what_if_preset,
        args=(
            {
                "wi_crop": "Maize",
                "wi_moisture": 26.0,
                "wi_temp": 28.0,
                "wi_humidity": 78.0,
                "wi_rain": 21.0,
                "wi_hour": 13,
            },
        ),
    )
    p4.button(
        "Preset: Saturated Soil Override",
        width="stretch",
        on_click=_apply_what_if_preset,
        args=(
            {
                "wi_crop": "Cotton",
                "wi_moisture": 81.0,
                "wi_temp": 25.0,
                "wi_humidity": 70.0,
                "wi_rain": 1.0,
                "wi_hour": 10,
            },
        ),
    )

    col_inputs, col_output = st.columns([1.05, 1.35], gap="large")

    with col_inputs:
        st.markdown("#### Field & Sensor Controls")
        crop_type = st.selectbox("Crop Type", options=list(CROP_TYPES), key="wi_crop")
        soil_moisture = st.slider("Soil Moisture (%)", 0.0, 100.0, step=0.5, key="wi_moisture")
        temperature = st.slider("Temperature (°C)", 5.0, 48.0, step=0.5, key="wi_temp")
        humidity = st.slider("Relative Humidity (%)", 10.0, 100.0, step=1.0, key="wi_humidity")
        rainfall = st.slider(
            "24h Rainfall Forecast (mm)", 0.0, 40.0, step=0.5, key="wi_rain"
        )
        hour_of_day = st.slider("Hour of Day (0-23)", 0, 23, step=1, key="wi_hour")

        n_col1, n_col2 = st.columns(2)
        with n_col1:
            soil_ph = st.slider("Soil pH", 4.5, 8.5, step=0.1, key="wi_ph")
            nitrogen = st.slider("Nitrogen N (mg/kg)", 5.0, 150.0, step=1.0, key="wi_n")
        with n_col2:
            phosphorus = st.slider("Phosphorus P (mg/kg)", 5.0, 100.0, step=1.0, key="wi_p")
            potassium = st.slider("Potassium K (mg/kg)", 10.0, 175.0, step=1.0, key="wi_k")

    payload = {
        "soil_moisture": soil_moisture,
        "temperature": temperature,
        "humidity": humidity,
        "rainfall_forecast_mm": rainfall,
        "soil_ph": soil_ph,
        "nitrogen": nitrogen,
        "phosphorus": phosphorus,
        "potassium": potassium,
        "crop_type": crop_type,
        "hour_of_day": hour_of_day,
    }
    prediction = run_irrigation_prediction(payload, clf, reg, metrics)

    with col_output:
        st.markdown("#### Real-Time ML Decision & Rationale")
        if prediction["override_active"]:
            st.warning(
                f"**Safety Override Triggered (`override_active = true`)** — Pump forced **OFF**. "
                f"Reason: {prediction['override_reason']}."
            )
        elif prediction["irrigation_action"] == "ON":
            st.success(
                f"**Irrigation Recommended: TURN ON** — Supply **{prediction['recommended_water_liters']:.2f} L/m²**."
            )
        else:
            st.info("**Irrigation Not Needed: KEEP OFF** — Soil and atmospheric conditions are adequate.")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Pump Decision", prediction["irrigation_action"])
        m2.metric("ML Probability", f"{prediction['irrigation_probability'] * 100:.1f}%")
        m3.metric("Water (L/m²)", f"{prediction['recommended_water_liters']:.2f}")
        m4.metric(
            "Effective Moisture",
            f"{prediction['engineered_features']['effective_moisture']:.1f}%",
        )

        gauge_col, factors_col = st.columns([0.95, 1.25])
        with gauge_col:
            prob_pct = prediction["irrigation_probability"] * 100.0
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=prob_pct,
                    number={"suffix": "%", "valueformat": ".1f"},
                    title={"text": "Irrigation Need Probability"},
                    gauge={
                        "axis": {"range": [0, 100], "tickvals": [0, 25, 50, 75, 100]},
                        "bar": {"color": "#10b981" if prob_pct >= 50 else "#3b82f6"},
                        "steps": [
                            {"range": [0, 35], "color": "rgba(59, 130, 246, 0.15)"},
                            {"range": [35, 65], "color": "rgba(245, 158, 11, 0.15)"},
                            {"range": [65, 100], "color": "rgba(16, 185, 129, 0.18)"},
                        ],
                        "threshold": {
                            "line": {"color": "#ef4444", "width": 3},
                            "thickness": 0.75,
                            "value": 50,
                        },
                    },
                )
            )
            fig_gauge.update_layout(height=275, margin=dict(l=35, r=35, t=55, b=15))
            st.plotly_chart(fig_gauge, width="stretch")

        with factors_col:
            factors = prediction["top_driving_factors"]
            factor_df = pd.DataFrame(factors).iloc[::-1]
            colors = [
                "#10b981" if imp > 0 else "#ef4444" for imp in factor_df["impact"]
            ]
            fig_factors = go.Figure(
                go.Bar(
                    x=factor_df["impact"],
                    y=[f"{lbl} ({val})" for lbl, val in zip(factor_df["label"], factor_df["value"])],
                    orientation="h",
                    marker_color=colors,
                    text=[f"{imp:+.3f}" for imp in factor_df["impact"]],
                    textposition="auto",
                )
            )
            fig_factors.update_layout(
                title="Top 3 Model Driving Factors (Probability Shift)",
                height=270,
                margin=dict(l=10, r=20, t=45, b=20),
                xaxis_title="Impact on Irrigation Probability",
            )
            fig_factors.add_vline(x=0, line_color="#64748b", line_width=1)
            st.plotly_chart(fig_factors, width="stretch")

        st.markdown("**Explanation Summary**")
        for rank, factor in enumerate(prediction["top_driving_factors"], start=1):
            st.write(f"{rank}. {factor['description']}")

        eng = prediction["engineered_features"]
        st.caption(
            f"Derived Agronomic Indices — VPD: `{eng['vpd_kpa']:.3f} kPa` | "
            f"Dryness Index: `{eng['dryness_index']:.3f}` | "
            f"Diurnal ET Factor: `{eng['diurnal_et_factor']:.3f}` | "
            f"NPK Mean: `{eng['npk_balance']:.1f} mg/kg` | "
            f"Safety Cutoffs: Moisture > `{SAFETY_MOISTURE_THRESHOLD:.0f}%` or Rain > `{SAFETY_RAINFALL_THRESHOLD:.0f} mm`"
        )


def render_tab_model_performance(metrics: dict[str, Any]) -> None:
    """Tab 3: Model comparisons, confusion matrix heatmap, and feature importance."""
    st.subheader("Model Benchmarking & Feature Importance")
    if not metrics:
        st.warning("No metrics found in `models/metrics.json`. Run `python src/train.py` first.")
        return

    meta = metrics.get("metadata", {})
    clf_section = metrics["classification"]
    reg_section = metrics["regression"]

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total Dataset Rows", f"{meta.get('total_rows', 0):,}")
    s2.metric(
        "Train / Test Split",
        f"{meta.get('train_rows', 0):,} / {meta.get('test_rows', 0):,}",
    )
    s3.metric("Best Classifier", clf_section["best_model"])
    s4.metric("Best Regressor", reg_section["best_model"])

    clf_rows = []
    for model_name, vals in clf_section["models"].items():
        clf_rows.append(
            {
                "Model": model_name,
                "F1 Score": vals["f1"],
                "Accuracy": vals["accuracy"],
                "ROC-AUC": vals["roc_auc"],
                "Selected": "Best" if model_name == clf_section["best_model"] else "",
            }
        )
    clf_df = pd.DataFrame(clf_rows)

    reg_rows = []
    for model_name, vals in reg_section["models"].items():
        reg_rows.append(
            {
                "Model": model_name,
                "RMSE (L/m²)": vals["rmse"],
                "MAE (L/m²)": vals["mae"],
                "R²": vals["r2"],
                "Selected": "Best" if model_name == reg_section["best_model"] else "",
            }
        )
    reg_df = pd.DataFrame(reg_rows)

    tbl_col1, tbl_col2 = st.columns(2, gap="large")
    with tbl_col1:
        st.markdown("#### Classification Benchmark (Irrigation ON/OFF)")
        st.dataframe(clf_df, width="stretch", hide_index=True)

        fig_clf_cmp = go.Figure()
        for metric_col, color in [
            ("F1 Score", "#10b981"),
            ("Accuracy", "#3b82f6"),
            ("ROC-AUC", "#8b5cf6"),
        ]:
            fig_clf_cmp.add_trace(
                go.Bar(name=metric_col, x=clf_df["Model"], y=clf_df[metric_col], marker_color=color)
            )
        fig_clf_cmp.update_layout(
            barmode="group",
            yaxis=dict(range=[0.8, 1.02], title="Score"),
            height=300,
            margin=dict(l=20, r=20, t=25, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        st.plotly_chart(fig_clf_cmp, width="stretch")

    with tbl_col2:
        st.markdown("#### Regression Benchmark (Water Volume L/m²)")
        st.dataframe(reg_df, width="stretch", hide_index=True)

        fig_reg_cmp = go.Figure()
        for metric_col, color in [
            ("RMSE (L/m²)", "#f97316"),
            ("MAE (L/m²)", "#eab308"),
            ("R²", "#14b8a6"),
        ]:
            fig_reg_cmp.add_trace(
                go.Bar(name=metric_col, x=reg_df["Model"], y=reg_df[metric_col], marker_color=color)
            )
        fig_reg_cmp.update_layout(
            barmode="group",
            yaxis=dict(title="Metric Value"),
            height=300,
            margin=dict(l=20, r=20, t=25, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        st.plotly_chart(fig_reg_cmp, width="stretch")

    viz_col1, viz_col2 = st.columns(2, gap="large")
    with viz_col1:
        st.markdown("#### Confusion Matrix Heatmap")
        clf_names = list(clf_section["models"].keys())
        default_idx = clf_names.index(clf_section["best_model"])
        selected_cm_model = st.selectbox(
            "Classifier Model", options=clf_names, index=default_idx, key="cm_model_select"
        )
        cm = np.array(clf_section["models"][selected_cm_model]["confusion_matrix"])
        labels = clf_section["models"][selected_cm_model].get(
            "confusion_labels", ["OFF (0)", "ON (1)"]
        )

        fig_cm = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=[f"Pred {lbl}" for lbl in labels],
                y=[f"True {lbl}" for lbl in labels],
                text=cm.astype(str),
                texttemplate="%{text}",
                textfont={"size": 18},
                colorscale="Tealgrn",
                showscale=True,
            )
        )
        fig_cm.update_layout(
            title=f"Confusion Matrix — {selected_cm_model}",
            height=380,
            margin=dict(l=20, r=20, t=45, b=20),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_cm, width="stretch")

    with viz_col2:
        st.markdown("#### Feature Importance (Best Model)")
        imp_task = st.radio(
            "Task",
            options=["Classification", "Regression"],
            horizontal=True,
            key="imp_task_radio",
        )
        raw_imp = (
            clf_section["feature_importance"]
            if imp_task == "Classification"
            else reg_section["feature_importance"]
        )
        imp_df = pd.DataFrame(raw_imp).head(12).iloc[::-1]

        fig_imp = go.Figure(
            go.Bar(
                x=imp_df["importance"],
                y=imp_df["feature"],
                orientation="h",
                marker_color="#10b981" if imp_task == "Classification" else "#0ea5e9",
                text=[f"{v:.3f}" for v in imp_df["importance"]],
                textposition="auto",
            )
        )
        best_title = (
            clf_section["best_model"] if imp_task == "Classification" else reg_section["best_model"]
        )
        fig_imp.update_layout(
            title=f"Top 12 Features — {best_title} ({imp_task})",
            height=380,
            margin=dict(l=20, r=20, t=45, b=20),
            xaxis_title="Normalized Importance",
        )
        st.plotly_chart(fig_imp, width="stretch")


def main() -> None:
    st.title("AgroSense ML — Smart Precision Agriculture & Irrigation")
    st.caption(
        "Automated Irrigation Decisioning, Water Volume Forecasting, and Real-Time IoT Monitoring"
    )

    try:
        clf, reg, metrics = get_model_bundle()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
        return

    tab_live, tab_whatif, tab_perf = st.tabs(
        [
            "Live Monitor & Simulator",
            "What-If ML Predictor",
            "Model Performance",
        ]
    )

    with tab_live:
        render_tab_live_monitor(clf, reg, metrics)
    with tab_whatif:
        render_tab_what_if(clf, reg, metrics)
    with tab_perf:
        render_tab_model_performance(metrics)


if __name__ == "__main__":
    main()
