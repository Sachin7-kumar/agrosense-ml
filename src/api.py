"""FastAPI backend serving AgroSense ML irrigation predictions and model metrics."""

from __future__ import annotations

import json
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from collections import deque

from src.preprocess import (
    CLASSIFIER_PATH,
    METRICS_PATH,
    REGRESSOR_PATH,
    load_artifacts,
    run_batch_irrigation_prediction,
    run_irrigation_prediction,
)


class SensorReading(BaseModel):
    """ESP32-compatible agricultural IoT sensor payload."""

    soil_moisture: float = Field(..., ge=0.0, le=100.0, description="Volumetric soil moisture (%)")
    temperature: float = Field(..., ge=-10.0, le=60.0, description="Air temperature (°C)")
    humidity: float = Field(..., ge=0.0, le=100.0, description="Relative humidity (%)")
    rainfall_forecast_mm: float = Field(
        ..., ge=0.0, le=500.0, description="24-hour rainfall forecast (mm)"
    )
    soil_ph: float = Field(..., ge=0.0, le=14.0, description="Soil pH level")
    nitrogen: float = Field(..., ge=0.0, le=500.0, description="Soil Nitrogen (mg/kg)")
    phosphorus: float = Field(..., ge=0.0, le=500.0, description="Soil Phosphorus (mg/kg)")
    potassium: float = Field(..., ge=0.0, le=500.0, description="Soil Potassium (mg/kg)")
    crop_type: Literal["Wheat", "Rice", "Cotton", "Maize"] = Field(
        ..., description="Crop species cultivated in the monitored plot"
    )
    hour_of_day: int = Field(
        default=12, ge=0, le=23, description="Local hour of day (0-23) for ET calculation"
    )
    device_id: str | None = Field(
        default=None, max_length=64, description="Optional ESP32 sensor node identifier"
    )


class DrivingFactor(BaseModel):
    """Single model-derived local feature contribution."""

    feature: str
    label: str
    value: str
    impact: float
    direction: Literal["increases_irrigation_need", "decreases_irrigation_need", "neutral"]
    description: str


class PredictionResponse(BaseModel):
    """Irrigation prediction response including safety override and explainability."""

    device_id: str | None = None
    timestamp_utc: str
    irrigation_action: Literal["ON", "OFF"]
    irrigation_needed: bool
    irrigation_probability: float
    recommended_water_liters: float
    override_active: bool
    override_reason: str | None = None
    sensor_anomalies: list[str] = Field(default_factory=list)
    top_driving_factors: list[DrivingFactor]
    engineered_features: dict[str, float]
    model_used: dict[str, str]


class BatchPredictionRequest(BaseModel):
    """Batch of agricultural sensor readings for multi-plot farm deployments."""

    readings: list[SensorReading] = Field(
        ..., min_length=1, max_length=500, description="List of IoT node telemetry readings"
    )


class BatchPredictionResponse(BaseModel):
    """Aggregated response for a batch inference request."""

    total_samples: int
    irrigation_on_count: int
    irrigation_off_count: int
    total_water_liters: float
    timestamp_utc: str
    predictions: list[PredictionResponse]


class TelemetryStatsResponse(BaseModel):
    """Real-time aggregated telemetry statistics across all served requests."""

    total_predictions_served: int
    irrigation_on_events: int
    irrigation_off_events: int
    safety_overrides_triggered: int
    total_water_recommended_liters: float
    active_device_ids: list[str]
    buffer_capacity: int
    current_buffered_events: int


# In-memory circular telemetry buffer for IoT monitoring
TELEMETRY_LOG: deque[dict[str, Any]] = deque(maxlen=200)


def _log_telemetry(record: dict[str, Any]) -> None:
    """Store telemetry event in memory buffer."""
    TELEMETRY_LOG.append(record)


class HealthResponse(BaseModel):
    """Service health and loaded artifact metadata."""

    status: Literal["ok", "degraded"]
    service: str
    models_loaded: bool
    classifier_model: str | None = None
    regressor_model: str | None = None
    timestamp_utc: str


def _ensure_models_loaded(target_app: FastAPI) -> tuple[Any, Any, dict[str, Any]]:
    """Load and cache trained pipelines on app.state if not already loaded."""
    if getattr(target_app.state, "classifier", None) is None:
        try:
            clf, reg, metrics = load_artifacts(
                classifier_path=CLASSIFIER_PATH,
                regressor_path=REGRESSOR_PATH,
                metrics_path=METRICS_PATH,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        target_app.state.classifier = clf
        target_app.state.regressor = reg
        target_app.state.metrics = metrics
    return (
        target_app.state.classifier,
        target_app.state.regressor,
        target_app.state.metrics,
    )


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    """Preload joblib pipelines into memory on startup if available."""
    if CLASSIFIER_PATH.exists() and REGRESSOR_PATH.exists():
        _ensure_models_loaded(fastapi_app)
    else:
        fastapi_app.state.classifier = None
        fastapi_app.state.regressor = None
        fastapi_app.state.metrics = {}
    yield


app = FastAPI(
    title="AgroSense ML API",
    description="Smart Precision Agriculture & Automated Irrigation Prediction Service",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Return service readiness and loaded model names."""
    clf = getattr(app.state, "classifier", None)
    reg = getattr(app.state, "regressor", None)
    metrics = getattr(app.state, "metrics", {}) or {}

    if clf is None and CLASSIFIER_PATH.exists() and REGRESSOR_PATH.exists():
        clf, reg, metrics = _ensure_models_loaded(app)

    models_loaded = clf is not None and reg is not None
    clf_name = metrics.get("classification", {}).get("best_model") if models_loaded else None
    reg_name = metrics.get("regression", {}).get("best_model") if models_loaded else None

    return HealthResponse(
        status="ok" if models_loaded else "degraded",
        service="AgroSense ML API",
        models_loaded=models_loaded,
        classifier_model=clf_name,
        regressor_model=reg_name,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_irrigation(payload: SensorReading) -> PredictionResponse:
    """Predict irrigation action, water volume, safety override, and top driving factors."""
    clf, reg, metrics = _ensure_models_loaded(app)
    ts = datetime.now(timezone.utc).isoformat()
    result = run_irrigation_prediction(
        sensor_payload=payload.model_dump(),
        classifier_pipeline=clf,
        regressor_pipeline=reg,
        metrics_summary=metrics,
    )
    resp = PredictionResponse(
        device_id=payload.device_id,
        timestamp_utc=ts,
        **result,
    )
    _log_telemetry(
        {
            "timestamp_utc": ts,
            "device_id": payload.device_id or "unassigned",
            "crop_type": payload.crop_type,
            "action": resp.irrigation_action,
            "water_liters": resp.recommended_water_liters,
            "override": resp.override_active,
            "soil_moisture": payload.soil_moisture,
        }
    )
    return resp


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_irrigation_batch(batch: BatchPredictionRequest) -> BatchPredictionResponse:
    """Perform vectorized batch inference across multiple sensor nodes."""
    clf, reg, metrics = _ensure_models_loaded(app)
    ts = datetime.now(timezone.utc).isoformat()
    payload_dicts = [reading.model_dump() for reading in batch.readings]
    results = run_batch_irrigation_prediction(
        sensor_payloads=payload_dicts,
        classifier_pipeline=clf,
        regressor_pipeline=reg,
        metrics_summary=metrics,
    )

    preds: list[PredictionResponse] = []
    on_count = 0
    off_count = 0
    total_water = 0.0

    for idx, r in enumerate(results):
        single_resp = PredictionResponse(
            timestamp_utc=ts,
            **r,
        )
        preds.append(single_resp)
        if single_resp.irrigation_action == "ON":
            on_count += 1
            total_water += single_resp.recommended_water_liters
        else:
            off_count += 1

        payload_item = batch.readings[idx]
        _log_telemetry(
            {
                "timestamp_utc": ts,
                "device_id": payload_item.device_id or "unassigned",
                "crop_type": payload_item.crop_type,
                "action": single_resp.irrigation_action,
                "water_liters": single_resp.recommended_water_liters,
                "override": single_resp.override_active,
                "soil_moisture": payload_item.soil_moisture,
            }
        )

    return BatchPredictionResponse(
        total_samples=len(preds),
        irrigation_on_count=on_count,
        irrigation_off_count=off_count,
        total_water_liters=round(total_water, 2),
        timestamp_utc=ts,
        predictions=preds,
    )


@app.get("/telemetry/recent")
def get_recent_telemetry(limit: int = 20) -> list[dict[str, Any]]:
    """Return the most recent IoT telemetry events from the circular buffer."""
    clamped_limit = max(1, min(limit, len(TELEMETRY_LOG)))
    items = list(TELEMETRY_LOG)
    return items[-clamped_limit:][::-1]


@app.get("/telemetry/stats", response_model=TelemetryStatsResponse)
def get_telemetry_stats() -> TelemetryStatsResponse:
    """Return live summary statistics across the active session."""
    items = list(TELEMETRY_LOG)
    on_cnt = sum(1 for it in items if it.get("action") == "ON")
    off_cnt = sum(1 for it in items if it.get("action") == "OFF")
    overrides = sum(1 for it in items if it.get("override") is True)
    tot_water = sum(float(it.get("water_liters", 0.0)) for it in items)
    dev_ids = sorted(list({str(it.get("device_id")) for it in items if it.get("device_id")}))

    return TelemetryStatsResponse(
        total_predictions_served=len(items),
        irrigation_on_events=on_cnt,
        irrigation_off_events=off_cnt,
        safety_overrides_triggered=overrides,
        total_water_recommended_liters=round(tot_water, 2),
        active_device_ids=dev_ids,
        buffer_capacity=TELEMETRY_LOG.maxlen or 200,
        current_buffered_events=len(items),
    )


@app.get("/metrics")
def get_metrics() -> dict[str, Any]:
    """Return saved model evaluation metrics and feature importances."""
    if not METRICS_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Metrics file not found. Run `python src/train.py` first.",
        )
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return metrics
