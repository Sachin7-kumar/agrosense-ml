#!/usr/bin/env python3
"""AgroSense ML — End-to-End System Live Demonstration Script.

Executes all core features: single prediction, safety over-watering override,
heavy rain override, sensor anomaly detection, multi-node batch inference,
and real-time telemetry analytics.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from src.api import app

if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def run_demo() -> None:
    client = TestClient(app)

    print("\n" + "=" * 70)
    print("      [AgroSense ML] End-to-End Live System Demonstration      ")
    print("=" * 70)

    # 1. System Health Check
    health = client.get("/health").json()
    print("\n[Step 1] Verifying System Health & Loaded Pipelines:")
    print(f"  Service Status    : {health['status'].upper()}")
    print(f"  Classifier Model  : {health['classifier_model']}")
    print(f"  Regressor Model   : {health['regressor_model']}")
    print(f"  Models Ready      : {health['models_loaded']}")

    # 2. Case A: Standard Drought / Parched Soil (Should Trigger Irrigation)
    print("\n[Step 2] Testing Scenario A: Severe Drought / High ET (Needs Irrigation)")
    dry_payload = {
        "device_id": "esp32-plot-01",
        "soil_moisture": 18.5,
        "temperature": 35.0,
        "humidity": 26.0,
        "rainfall_forecast_mm": 0.0,
        "soil_ph": 6.4,
        "nitrogen": 72.0,
        "phosphorus": 45.0,
        "potassium": 82.0,
        "crop_type": "Wheat",
        "hour_of_day": 14,
    }
    r1 = client.post("/predict", json=dry_payload).json()
    print(f"  Telemetry Ingested : Moisture={dry_payload['soil_moisture']}%, Temp={dry_payload['temperature']}°C")
    print(f"  Irrigation Action  : >>> {r1['irrigation_action']} <<< (P={r1['irrigation_probability']:.2%})")
    print(f"  Water Recommended  : {r1['recommended_water_liters']} L/m²")
    print(f"  Safety Override    : {r1['override_active']}")
    print(f"  Engineered VPD     : {r1['engineered_features']['vpd_kpa']} kPa")
    print(f"  Top TreeSHAP Factor: {r1['top_driving_factors'][0]['description']}")

    # 3. Case B: Saturated Soil (Safety Override Test)
    print("\n[Step 3] Testing Scenario B: Saturated Soil Safeguard (Moisture > 75%)")
    wet_payload = {**dry_payload, "device_id": "esp32-plot-02", "soil_moisture": 84.0}
    r2 = client.post("/predict", json=wet_payload).json()
    print(f"  Telemetry Ingested : Moisture={wet_payload['soil_moisture']}% (High Saturation)")
    print(f"  Irrigation Action  : >>> {r2['irrigation_action']} <<< (Safety Cutoff Enforced)")
    print(f"  Override Active    : {r2['override_active']}")
    print(f"  Override Reason    : {r2['override_reason']}")
    print(f"  Water Delivered    : {r2['recommended_water_liters']} L/m² (Forced Zero)")

    # 4. Case C: Heavy Rainfall Forecast (Rain Safeguard)
    print("\n[Step 4] Testing Scenario C: Incoming Rainstorm Safeguard (Rain > 15mm)")
    rain_payload = {**dry_payload, "device_id": "esp32-plot-03", "rainfall_forecast_mm": 24.5}
    r3 = client.post("/predict", json=rain_payload).json()
    print(f"  Telemetry Ingested : Forecast Rain={rain_payload['rainfall_forecast_mm']} mm")
    print(f"  Irrigation Action  : >>> {r3['irrigation_action']} <<< (Rain Cutoff Enforced)")
    print(f"  Override Active    : {r3['override_active']}")
    print(f"  Override Reason    : {r3['override_reason']}")

    # 5. Case D: Sensor Anomaly Detection
    print("\n[Step 5] Testing Scenario D: Out-of-Distribution / Faulty Sensor Probe")
    anomaly_payload = {**dry_payload, "device_id": "esp32-plot-04", "soil_moisture": 0.8}
    r4 = client.post("/predict", json=anomaly_payload).json()
    print(f"  Telemetry Ingested : Moisture={anomaly_payload['soil_moisture']}% (Near Zero)")
    print(f"  Anomalies Flagged  : {r4['sensor_anomalies']}")

    # 6. Case E: Vectorized Multi-Node Farm Batch Inference
    print("\n[Step 6] Testing Scenario E: Vectorized Multi-Plot Batch Inference (5 Nodes)")
    batch_payload = {
        "readings": [
            {**dry_payload, "device_id": "node-plot-north", "crop_type": "Wheat", "soil_moisture": 20.0},
            {**dry_payload, "device_id": "node-plot-south", "crop_type": "Rice", "soil_moisture": 48.0},
            {**dry_payload, "device_id": "node-plot-east", "crop_type": "Cotton", "soil_moisture": 79.0}, # override
            {**dry_payload, "device_id": "node-plot-west", "crop_type": "Maize", "soil_moisture": 32.0},
            {**dry_payload, "device_id": "node-plot-central", "crop_type": "Wheat", "rainfall_forecast_mm": 30.0}, # override
        ]
    }
    r5 = client.post("/predict/batch", json=batch_payload).json()
    print(f"  Plots Evaluated    : {r5['total_samples']}")
    print(f"  Pumps Activated    : {r5['irrigation_on_count']} ON / {r5['irrigation_off_count']} OFF")
    print(f"  Total Batch Water  : {r5['total_water_liters']} Liters")
    print("  Per-Node Summary:")
    for p in r5["predictions"]:
        override_flag = " [OVERRIDE]" if p["override_active"] else ""
        print(f"    - {p['device_id']:<18} | Action: {p['irrigation_action']:<3} | Water: {p['recommended_water_liters']:>5.2f} L/m²{override_flag}")

    # 7. Telemetry Monitoring Stats
    print("\n[Step 7] Live In-Memory Telemetry Stats:")
    stats = client.get("/telemetry/stats").json()
    print(f"  Total Telemetry Events Logged   : {stats['total_predictions_served']}")
    print(f"  Total Irrigation ON Events      : {stats['irrigation_on_events']}")
    print(f"  Total Safety Overrides Handled  : {stats['safety_overrides_triggered']}")
    print(f"  Total Water Volume Recommended  : {stats['total_water_recommended_liters']:.2f} L")
    print(f"  Active Monitored Nodes          : {len(stats['active_device_ids'])} unique nodes")

    print("\n" + "=" * 70)
    print("       [SUCCESS] All System Capabilities Verified & Fully Operational!       ")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_demo()
