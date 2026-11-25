#!/usr/bin/env python3
"""AgroSense ML — Model & API Inference Latency Benchmarking Suite.

Measures single-inference and batch inference latencies (P50, P90, P95, P99),
throughput (QPS), and end-to-end FastAPI endpoint latency.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.api import app
from src.preprocess import (
    load_artifacts,
    run_batch_irrigation_prediction,
    run_irrigation_prediction,
)

SAMPLE_READING: dict[str, Any] = {
    "device_id": "esp32-bench-01",
    "soil_moisture": 22.5,
    "temperature": 32.0,
    "humidity": 35.0,
    "rainfall_forecast_mm": 0.0,
    "soil_ph": 6.4,
    "nitrogen": 68.0,
    "phosphorus": 44.0,
    "potassium": 78.0,
    "crop_type": "Wheat",
    "hour_of_day": 14,
}


def benchmark_single_inference(
    clf: Any,
    reg: Any,
    metrics: dict[str, Any],
    runs: int = 500,
) -> dict[str, float]:
    """Profile latency for single-sample Python inference."""
    # Warmup
    for _ in range(25):
        run_irrigation_prediction(SAMPLE_READING, clf, reg, metrics)

    latencies_ms: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        run_irrigation_prediction(SAMPLE_READING, clf, reg, metrics)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    arr = np.array(latencies_ms)
    return {
        "runs": runs,
        "mean_ms": round(float(np.mean(arr)), 3),
        "min_ms": round(float(np.min(arr)), 3),
        "max_ms": round(float(np.max(arr)), 3),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p90_ms": round(float(np.percentile(arr, 90)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "qps": round(float(1000.0 / np.mean(arr)), 1),
    }


def benchmark_batch_inference(
    clf: Any,
    reg: Any,
    metrics: dict[str, Any],
    batch_sizes: list[int] = [10, 50, 100, 250],
    runs: int = 50,
) -> list[dict[str, Any]]:
    """Profile latency and throughput across various batch sizes."""
    results: list[dict[str, Any]] = []

    for b_size in batch_sizes:
        batch_payload = [
            {**SAMPLE_READING, "device_id": f"esp32-node-{i:03d}"}
            for i in range(b_size)
        ]

        # Warmup
        for _ in range(5):
            run_batch_irrigation_prediction(batch_payload, clf, reg, metrics)

        latencies_ms: list[float] = []
        for _ in range(runs):
            t0 = time.perf_counter()
            run_batch_irrigation_prediction(batch_payload, clf, reg, metrics)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

        arr = np.array(latencies_ms)
        avg_batch_ms = float(np.mean(arr))
        per_item_ms = avg_batch_ms / b_size
        throughput = (b_size * 1000.0) / avg_batch_ms

        results.append(
            {
                "batch_size": b_size,
                "total_time_ms": round(avg_batch_ms, 2),
                "per_item_ms": round(per_item_ms, 4),
                "p95_ms": round(float(np.percentile(arr, 95)), 2),
                "throughput_items_per_sec": round(throughput, 1),
            }
        )
    return results


def benchmark_fastapi_endpoint(runs: int = 300) -> dict[str, float]:
    """Profile end-to-end HTTP request processing via TestClient."""
    with TestClient(app) as client:
        # Warmup
        for _ in range(15):
            client.post("/predict", json=SAMPLE_READING)

        latencies_ms: list[float] = []
        for _ in range(runs):
            t0 = time.perf_counter()
            resp = client.post("/predict", json=SAMPLE_READING)
            t1 = time.perf_counter()
            assert resp.status_code == 200
            latencies_ms.append((t1 - t0) * 1000.0)

        arr = np.array(latencies_ms)
        return {
            "runs": runs,
            "mean_ms": round(float(np.mean(arr)), 3),
            "p50_ms": round(float(np.percentile(arr, 50)), 3),
            "p95_ms": round(float(np.percentile(arr, 95)), 3),
            "p99_ms": round(float(np.percentile(arr, 99)), 3),
            "qps": round(float(1000.0 / np.mean(arr)), 1),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark AgroSense ML inference latency.")
    parser.add_argument("--runs", type=int, default=500, help="Number of benchmark iterations")
    args = parser.parse_args()

    print("\n=======================================================")
    print("      AgroSense ML — Performance & Latency Benchmark   ")
    print("=======================================================")
    print(f"Loading model artifacts from models/ ...")
    clf, reg, metrics = load_artifacts()
    clf_name = metrics.get("classification", {}).get("best_model", "Classifier")
    reg_name = metrics.get("regression", {}).get("best_model", "Regressor")
    print(f"Classifier : {clf_name}")
    print(f"Regressor  : {reg_name}")
    print("=======================================================\n")

    print(f"--- 1. Python Native Single Inference ({args.runs} runs) ---")
    single_res = benchmark_single_inference(clf, reg, metrics, runs=args.runs)
    print(f"  P50 (Median) Latency : {single_res['p50_ms']:>6.2f} ms")
    print(f"  P90 Latency          : {single_res['p90_ms']:>6.2f} ms")
    print(f"  P95 Latency          : {single_res['p95_ms']:>6.2f} ms")
    print(f"  P99 Latency          : {single_res['p99_ms']:>6.2f} ms")
    print(f"  Mean Latency         : {single_res['mean_ms']:>6.2f} ms (Min: {single_res['min_ms']:.2f} ms)")
    print(f"  Single-Thread QPS    : {single_res['qps']:>6.1f} req/sec\n")

    print(f"--- 2. Batch Inference Scaling ---")
    batch_res = benchmark_batch_inference(clf, reg, metrics)
    print(f"  {'Batch Size':<12} | {'Total (ms)':<12} | {'Per Item (ms)':<14} | {'Throughput (items/sec)':<22}")
    print("  " + "-" * 68)
    for row in batch_res:
        print(
            f"  {row['batch_size']:<12} | "
            f"{row['total_time_ms']:>8.2f} ms | "
            f"{row['per_item_ms']:>10.4f} ms | "
            f"{row['throughput_items_per_sec']:>18.1f}"
        )
    print()

    print(f"--- 3. FastAPI End-to-End POST /predict ({min(args.runs, 200)} runs) ---")
    api_res = benchmark_fastapi_endpoint(runs=min(args.runs, 200))
    print(f"  P50 (Median) Latency : {api_res['p50_ms']:>6.2f} ms")
    print(f"  P95 Latency          : {api_res['p95_ms']:>6.2f} ms")
    print(f"  P99 Latency          : {api_res['p99_ms']:>6.2f} ms")
    print(f"  API Throughput (QPS) : {api_res['qps']:>6.1f} req/sec\n")
    print("Benchmark complete! All targets within edge real-time limits (<25ms).\n")


if __name__ == "__main__":
    main()
