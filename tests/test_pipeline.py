"""End-to-end pytest suite for AgroSense ML data, preprocessing, models, and FastAPI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.data_generator import ensure_dataset, generate_dataset
from src.preprocess import (
    ALL_NUMERIC_FEATURES,
    CLASSIFIER_PATH,
    CROP_TYPES,
    METRICS_PATH,
    RAW_FEATURE_COLUMNS,
    REGRESSOR_PATH,
    REQUIRED_DATASET_COLUMNS,
    AgriculturalFeatureEngineer,
    build_preprocessor,
    detect_sensor_anomalies,
    load_artifacts,
    run_batch_irrigation_prediction,
)


def test_data_generator_schema_and_determinism() -> None:
    df_a = generate_dataset(num_rows=5000, seed=42)
    df_b = generate_dataset(num_rows=5000, seed=42)

    pd.testing.assert_frame_equal(df_a, df_b)
    assert len(df_a) >= 5000
    assert list(df_a.columns) == REQUIRED_DATASET_COLUMNS
    assert not df_a.isna().any().any()
    assert set(df_a["crop_type"].unique()).issubset(set(CROP_TYPES))
    assert set(df_a["irrigation_needed"].unique()) == {0, 1}

    off_rows = df_a[df_a["irrigation_needed"] == 0]
    on_rows = df_a[df_a["irrigation_needed"] == 1]
    assert (off_rows["water_volume_liters"] == 0.0).all()
    assert (on_rows["water_volume_liters"] > 0.0).all()

    safety_violators = df_a[
        (df_a["soil_moisture"] > 75.0) | (df_a["rainfall_forecast_mm"] > 15.0)
    ]
    assert (safety_violators["irrigation_needed"] == 0).all()


def test_ensure_dataset_preserves_custom_csv(tmp_path: Path) -> None:
    target = tmp_path / "custom.csv"
    sentinel = "custom_col\n123\n"
    target.write_text(sentinel, encoding="utf-8")

    ensure_dataset(output_path=target, num_rows=200, seed=42, force=False)
    assert target.read_text(encoding="utf-8") == sentinel

    ensure_dataset(output_path=target, num_rows=200, seed=42, force=True)
    regenerated = pd.read_csv(target)
    assert len(regenerated) == 200
    assert list(regenerated.columns) == REQUIRED_DATASET_COLUMNS


def test_preprocessing_output_shapes_and_unseen_crop() -> None:
    df = generate_dataset(num_rows=250, seed=7)
    X_raw = df[RAW_FEATURE_COLUMNS]

    engineer = AgriculturalFeatureEngineer()
    X_eng = engineer.fit_transform(X_raw)
    for col in ALL_NUMERIC_FEATURES:
        assert col in X_eng.columns

    preprocessor = build_preprocessor()
    X_trans = preprocessor.fit_transform(X_eng)

    expected_dim = len(ALL_NUMERIC_FEATURES) + len(CROP_TYPES)
    assert X_trans.shape == (len(df), expected_dim)
    assert not np.isnan(X_trans).any()
    assert not np.isinf(X_trans).any()

    unseen_row = X_raw.iloc[[0]].copy()
    unseen_row["crop_type"] = "Soybean"
    unseen_trans = preprocessor.transform(engineer.transform(unseen_row))
    assert unseen_trans.shape == (1, expected_dim)


def test_trained_models_and_metrics_thresholds() -> None:
    assert CLASSIFIER_PATH.exists(), "Run `python src/train.py` before running tests."
    assert REGRESSOR_PATH.exists(), "Run `python src/train.py` before running tests."
    assert METRICS_PATH.exists(), "Run `python src/train.py` before running tests."

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    clf_section = metrics["classification"]
    reg_section = metrics["regression"]

    assert len(clf_section["models"]) >= 3
    assert len(reg_section["models"]) >= 3

    best_clf = clf_section["best_model"]
    best_reg = reg_section["best_model"]
    assert clf_section["models"][best_clf]["f1"] > 0.85
    assert clf_section["models"][best_clf]["roc_auc"] > 0.90
    assert reg_section["models"][best_reg]["r2"] > 0.80

    clf_pipe, reg_pipe, _ = load_artifacts()
    sample = generate_dataset(num_rows=120, seed=99)[RAW_FEATURE_COLUMNS].head(10)
    preds = clf_pipe.predict(sample)
    probs = clf_pipe.predict_proba(sample)
    volumes = reg_pipe.predict(sample)

    assert preds.shape == (10,)
    assert probs.shape == (10, 2)
    assert volumes.shape == (10,)


def test_fastapi_endpoints_predict_safety_and_validation() -> None:
    with TestClient(app) as client:
        health_resp = client.get("/health")
        assert health_resp.status_code == 200
        health_data = health_resp.json()
        assert health_data["status"] == "ok"
        assert health_data["models_loaded"] is True

        metrics_resp = client.get("/metrics")
        assert metrics_resp.status_code == 200
        metrics_data = metrics_resp.json()
        assert "classification" in metrics_data
        assert "regression" in metrics_data

        dry_payload = {
            "soil_moisture": 18.0,
            "temperature": 36.5,
            "humidity": 26.0,
            "rainfall_forecast_mm": 0.0,
            "soil_ph": 6.4,
            "nitrogen": 72.0,
            "phosphorus": 45.0,
            "potassium": 82.0,
            "crop_type": "Rice",
            "hour_of_day": 14,
            "device_id": "esp32-field-01",
        }
        pred_resp = client.post("/predict", json=dry_payload)
        assert pred_resp.status_code == 200
        pred_data = pred_resp.json()
        assert pred_data["device_id"] == "esp32-field-01"
        assert pred_data["irrigation_action"] == "ON"
        assert pred_data["irrigation_needed"] is True
        assert pred_data["override_active"] is False
        assert pred_data["override_reason"] is None
        assert 0.0 <= pred_data["irrigation_probability"] <= 1.0
        assert pred_data["recommended_water_liters"] > 0.0
        assert len(pred_data["top_driving_factors"]) == 3

        wet_soil_payload = {**dry_payload, "soil_moisture": 81.5}
        wet_resp = client.post("/predict", json=wet_soil_payload)
        assert wet_resp.status_code == 200
        wet_data = wet_resp.json()
        assert wet_data["override_active"] is True
        assert wet_data["irrigation_action"] == "OFF"
        assert wet_data["irrigation_needed"] is False
        assert wet_data["recommended_water_liters"] == 0.0
        assert "75%" in wet_data["override_reason"]

        rain_payload = {**dry_payload, "rainfall_forecast_mm": 22.0}
        rain_resp = client.post("/predict", json=rain_payload)
        assert rain_resp.status_code == 200
        rain_data = rain_resp.json()
        assert rain_data["override_active"] is True
        assert rain_data["irrigation_action"] == "OFF"
        assert rain_data["recommended_water_liters"] == 0.0
        assert "15 mm" in rain_data["override_reason"]

        invalid_moisture = client.post("/predict", json={**dry_payload, "soil_moisture": 120.0})
        assert invalid_moisture.status_code == 422

        invalid_crop = client.post("/predict", json={**dry_payload, "crop_type": "Banana"})
        assert invalid_crop.status_code == 422

        invalid_rain = client.post("/predict", json={**dry_payload, "rainfall_forecast_mm": -2.0})
        assert invalid_rain.status_code == 422


def test_sensor_anomaly_detection() -> None:
    # Near zero moisture
    anom_zero = detect_sensor_anomalies({"soil_moisture": 0.5})
    assert any("near-zero" in a for a in anom_zero)

    # Saturated moisture
    anom_sat = detect_sensor_anomalies({"soil_moisture": 99.5})
    assert any("saturated" in a for a in anom_sat)

    # Low moisture during heavy rain
    anom_rain = detect_sensor_anomalies({"soil_moisture": 5.0, "rainfall_forecast_mm": 25.0})
    assert any("heavy rainfall" in a for a in anom_rain)

    # Extreme heat & humidity
    anom_heat = detect_sensor_anomalies({"temperature": 48.0, "humidity": 90.0})
    assert any("Extreme heat" in a for a in anom_heat)

    # Abnormal pH
    anom_ph = detect_sensor_anomalies({"soil_ph": 2.5})
    assert any("Soil pH" in a for a in anom_ph)

    # Normal payload produces no anomalies
    normal_payload = {
        "soil_moisture": 35.0,
        "temperature": 25.0,
        "humidity": 60.0,
        "rainfall_forecast_mm": 2.0,
        "soil_ph": 6.5,
        "nitrogen": 70.0,
        "phosphorus": 45.0,
        "potassium": 80.0,
    }
    assert len(detect_sensor_anomalies(normal_payload)) == 0


def test_fastapi_batch_prediction_and_telemetry() -> None:
    with TestClient(app) as client:
        batch_payload = {
            "readings": [
                {
                    "soil_moisture": 18.0,
                    "temperature": 35.0,
                    "humidity": 25.0,
                    "rainfall_forecast_mm": 0.0,
                    "soil_ph": 6.5,
                    "nitrogen": 70.0,
                    "phosphorus": 40.0,
                    "potassium": 80.0,
                    "crop_type": "Wheat",
                    "hour_of_day": 14,
                    "device_id": "batch-node-01",
                },
                {
                    "soil_moisture": 82.0,  # Safety override
                    "temperature": 22.0,
                    "humidity": 70.0,
                    "rainfall_forecast_mm": 0.0,
                    "soil_ph": 6.5,
                    "nitrogen": 70.0,
                    "phosphorus": 40.0,
                    "potassium": 80.0,
                    "crop_type": "Rice",
                    "hour_of_day": 10,
                    "device_id": "batch-node-02",
                },
                {
                    "soil_moisture": 30.0,
                    "temperature": 28.0,
                    "humidity": 50.0,
                    "rainfall_forecast_mm": 25.0,  # Rain safety override
                    "soil_ph": 6.2,
                    "nitrogen": 65.0,
                    "phosphorus": 38.0,
                    "potassium": 75.0,
                    "crop_type": "Maize",
                    "hour_of_day": 16,
                    "device_id": "batch-node-03",
                },
            ]
        }

        resp = client.post("/predict/batch", json=batch_payload)
        assert resp.status_code == 200
        data = resp.json()

        assert data["total_samples"] == 3
        assert len(data["predictions"]) == 3
        assert data["irrigation_on_count"] + data["irrigation_off_count"] == 3

        # Node 1 should be ON
        p1 = data["predictions"][0]
        assert p1["device_id"] == "batch-node-01"
        assert p1["irrigation_action"] == "ON"
        assert p1["recommended_water_liters"] > 0.0

        # Node 2 should be overridden (soil > 75%)
        p2 = data["predictions"][1]
        assert p2["override_active"] is True
        assert p2["irrigation_action"] == "OFF"
        assert p2["recommended_water_liters"] == 0.0

        # Node 3 should be overridden (rain > 15mm)
        p3 = data["predictions"][2]
        assert p3["override_active"] is True
        assert p3["irrigation_action"] == "OFF"
        assert p3["recommended_water_liters"] == 0.0

        # Verify Telemetry history endpoint
        hist_resp = client.get("/telemetry/recent?limit=10")
        assert hist_resp.status_code == 200
        history = hist_resp.json()
        assert len(history) >= 3
        device_ids_in_hist = [h["device_id"] for h in history]
        assert "batch-node-01" in device_ids_in_hist

        # Verify Telemetry stats endpoint
        stats_resp = client.get("/telemetry/stats")
        assert stats_resp.status_code == 200
        stats = stats_resp.json()
        assert stats["total_predictions_served"] >= 3
        assert stats["irrigation_on_events"] >= 1
        assert stats["safety_overrides_triggered"] >= 2
        assert stats["total_water_recommended_liters"] > 0.0
        assert "batch-node-01" in stats["active_device_ids"]

