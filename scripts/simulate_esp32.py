#!/usr/bin/env python3
"""AgroSense ML — ESP32 Microcontroller Edge Simulator.

Simulates an IoT sensor node streaming field telemetry to the AgroSense ML API,
actuating a simulated 5V relay valve, and enforcing fail-safe offline fallback logic.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx
import numpy as np

SCENARIO_PROFILES: dict[str, dict[str, float]] = {
    "normal": {
        "moisture_target": 38.0,
        "temp_base": 24.0,
        "rain_base": 1.0,
    },
    "drought": {
        "moisture_target": 18.0,
        "temp_base": 36.0,
        "rain_base": 0.0,
    },
    "rainstorm": {
        "moisture_target": 65.0,
        "temp_base": 19.0,
        "rain_base": 28.0,
    },
    "sensor_drift": {
        "moisture_target": 12.0,
        "temp_base": 26.0,
        "rain_base": 0.5,
    },
}


class ESP32EdgeNode:
    """Simulates an ESP32 hardware device running agricultural firmware."""

    def __init__(
        self,
        device_id: str = "esp32-node-01",
        crop_type: str = "Wheat",
        scenario: str = "normal",
        api_url: str = "http://127.0.0.1:8000/predict",
    ) -> None:
        self.device_id = device_id
        self.crop_type = crop_type
        self.scenario = scenario
        self.api_url = api_url
        self.relay_state: bool = False  # False = OFF, True = ON
        self.current_moisture: float = SCENARIO_PROFILES[scenario]["moisture_target"]
        self.step_counter: int = 0
        self.total_water_dispensed: float = 0.0
        self.override_events: int = 0

    def read_sensors(self) -> dict[str, Any]:
        """Simulate reading analog capacitive soil moisture and digital DHT22/NPK sensors."""
        cfg = SCENARIO_PROFILES[self.scenario]
        hour = (6 + self.step_counter) % 24
        diurnal = math.cos((hour - 14.0) * math.pi / 12.0)

        # Ambient temperature and inverse humidity
        temp = float(np.clip(cfg["temp_base"] + 5.0 * diurnal + np.random.normal(0, 0.6), 8.0, 48.0))
        hum = float(np.clip(70.0 - 1.2 * (temp - 20.0) + np.random.normal(0, 2.0), 18.0, 96.0))

        # Dynamic moisture depletion or recharge from previous irrigation
        moisture_drain = 0.4 if self.relay_state is False else -4.5
        target = cfg["moisture_target"]
        self.current_moisture = float(
            np.clip(self.current_moisture - moisture_drain + np.random.normal(0, 0.4), 8.0, 88.0)
        )

        rain = float(np.clip(cfg["rain_base"] + np.random.normal(0, 0.8), 0.0, 50.0))

        # Sensor drift glitch simulation
        if self.scenario == "sensor_drift" and self.step_counter > 4:
            moisture_val = 0.5  # Simulate detached probe
        else:
            moisture_val = round(self.current_moisture, 2)

        return {
            "device_id": self.device_id,
            "soil_moisture": moisture_val,
            "temperature": round(temp, 2),
            "humidity": round(hum, 2),
            "rainfall_forecast_mm": round(rain, 2),
            "soil_ph": round(float(np.clip(6.4 + np.random.normal(0, 0.08), 5.5, 7.8)), 2),
            "nitrogen": round(float(np.clip(70.0 + np.random.normal(0, 2.0), 20.0, 130.0)), 1),
            "phosphorus": round(float(np.clip(42.0 + np.random.normal(0, 1.5), 10.0, 80.0)), 1),
            "potassium": round(float(np.clip(78.0 + np.random.normal(0, 2.0), 20.0, 150.0)), 1),
            "crop_type": self.crop_type,
            "hour_of_day": int(hour),
        }

    def execute_local_failsafe(self, payload: dict[str, Any]) -> tuple[str, float, str]:
        """Enforce local offline heuristics if cloud API is unreachable."""
        m = payload["soil_moisture"]
        r = payload["rainfall_forecast_mm"]
        if m > 75.0 or r > 15.0:
            return "OFF", 0.0, "LOCAL_FAILSAFE: High moisture/rain safety cutoff"
        if m < 25.0:
            return "ON", 4.5, "LOCAL_FAILSAFE: Critical dry soil threshold triggered"
        return "OFF", 0.0, "LOCAL_FAILSAFE: Moisture adequate"

    def step(self, dry_run: bool = False) -> dict[str, Any]:
        """Execute one simulation cycle: read -> transmit/infer -> actuate relay."""
        self.step_counter += 1
        payload = self.read_sensors()
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")

        action: str
        water: float
        override: bool = False
        source: str

        if dry_run:
            action, water, reason = self.execute_local_failsafe(payload)
            source = "DRY_RUN (Simulated Local)"
        else:
            try:
                resp = httpx.post(self.api_url, json=payload, timeout=3.0)
                if resp.status_code == 200:
                    data = resp.json()
                    action = data["irrigation_action"]
                    water = float(data["recommended_water_liters"])
                    override = bool(data["override_active"])
                    source = f"API ({data['model_used']['classifier']})"
                else:
                    action, water, reason = self.execute_local_failsafe(payload)
                    source = f"API_ERROR_{resp.status_code} (Fallback)"
            except Exception:
                action, water, reason = self.execute_local_failsafe(payload)
                source = "OFFLINE_FALLBACK (Connection Refused)"

        # Physical actuator response
        self.relay_state = (action == "ON")
        if self.relay_state:
            self.total_water_dispensed += water
        if override:
            self.override_events += 1

        relay_str = "[RELAY ON]" if self.relay_state else "[RELAY OFF]"
        override_str = " [SAFETY OVERRIDE]" if override else ""

        print(
            f"[{ts}] Node: {self.device_id:<14} | "
            f"Moisture: {payload['soil_moisture']:>5.1f}% | "
            f"Temp: {payload['temperature']:>4.1f}°C | "
            f"Rain: {payload['rainfall_forecast_mm']:>4.1f}mm | "
            f"{relay_str:<12} | "
            f"Water: {water:>4.1f} L/m² | "
            f"{source}{override_str}"
        )

        return {
            "step": self.step_counter,
            "payload": payload,
            "action": action,
            "water": water,
            "relay_state": self.relay_state,
            "override": override,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate an ESP32 IoT node with AgroSense ML.")
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict", help="FastAPI predict URL")
    parser.add_argument("--device-id", default="esp32-node-01", help="Device identifier")
    parser.add_argument(
        "--scenario",
        choices=["normal", "drought", "rainstorm", "sensor_drift"],
        default="normal",
        help="Field scenario profile",
    )
    parser.add_argument(
        "--crop",
        choices=["Wheat", "Rice", "Cotton", "Maize"],
        default="Wheat",
        help="Monitored crop",
    )
    parser.add_argument("--count", type=int, default=10, help="Number of telemetry cycles (0=infinite)")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between telemetry ticks")
    parser.add_argument("--dry-run", action="store_true", help="Run without calling the API")
    args = parser.parse_args()

    print("\n=======================================================")
    print("   AgroSense ML — ESP32 Smart Edge Node Simulator      ")
    print("=======================================================")
    print(f"Device ID : {args.device_id}")
    print(f"Crop Type : {args.crop}")
    print(f"Scenario  : {args.scenario}")
    print(f"API Target: {args.url} (dry_run={args.dry_run})")
    print("=======================================================\n")

    node = ESP32EdgeNode(
        device_id=args.device_id,
        crop_type=args.crop,
        scenario=args.scenario,
        api_url=args.url,
    )

    try:
        cycles = 0
        while True:
            node.step(dry_run=args.dry_run)
            cycles += 1
            if args.count > 0 and cycles >= args.count:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[Simulator stopped by user]")

    print("\n---------------- Simulation Summary ----------------")
    print(f"Total Cycles Completed  : {node.step_counter}")
    print(f"Total Water Dispensed   : {node.total_water_dispensed:.2f} L/m²")
    print(f"Safety Overrides Fired  : {node.override_events}")
    print("----------------------------------------------------\n")


if __name__ == "__main__":
    main()
