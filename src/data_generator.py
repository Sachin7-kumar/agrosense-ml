"""Deterministic physics-informed agricultural sensor dataset generator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import (
    CROP_TYPES,
    DEFAULT_DATA_PATH,
    REQUIRED_DATASET_COLUMNS,
    SAFETY_MOISTURE_THRESHOLD,
    SAFETY_RAINFALL_THRESHOLD,
    compute_engineered_features,
)

DEFAULT_NUM_ROWS = 6000
DEFAULT_SEED = 42

CROP_PARAMETERS: dict[str, tuple[float, float]] = {
    # crop_type: (base_moisture_threshold_pct, crop_coefficient_Kc)
    "Wheat": (35.0, 0.85),
    "Rice": (53.0, 1.22),
    "Cotton": (38.0, 0.95),
    "Maize": (44.0, 1.06),
}


def generate_dataset(
    num_rows: int = DEFAULT_NUM_ROWS,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Generate a synthetic, physically grounded agricultural sensor dataset."""
    if num_rows < 100:
        raise ValueError("num_rows must be at least 100.")

    rng = np.random.default_rng(seed)

    crop_type = rng.choice(CROP_TYPES, size=num_rows, p=[0.26, 0.24, 0.24, 0.26])
    hour_of_day = rng.integers(0, 24, size=num_rows)

    # Diurnal temperature cycle peaking around 14:00
    base_temp = rng.uniform(12.0, 36.0, size=num_rows)
    diurnal_temp_shift = 4.5 * np.cos((hour_of_day - 14.0) * np.pi / 12.0)
    true_temp = np.clip(base_temp + diurnal_temp_shift + rng.normal(0.0, 1.5, size=num_rows), 6.0, 46.0)

    # Relative humidity inversely correlated with air temperature
    true_humidity = np.clip(
        76.0 - 1.25 * (true_temp - 20.0) + rng.normal(0.0, 11.0, size=num_rows),
        15.0,
        96.0,
    )

    # Zero-inflated 24h rainfall forecast (dry days, light showers, heavy rain events)
    rain_regime = rng.random(size=num_rows)
    true_rainfall = np.where(
        rain_regime < 0.55,
        rng.exponential(0.6, size=num_rows),
        np.where(
            rain_regime < 0.86,
            rng.uniform(1.5, 14.0, size=num_rows),
            rng.uniform(14.5, 34.0, size=num_rows),
        ),
    )
    true_rainfall = np.clip(true_rainfall, 0.0, 40.0)

    # Broad soil moisture distribution covering drought through saturated conditions
    true_moisture = rng.uniform(10.0, 86.0, size=num_rows)

    true_ph = np.clip(rng.normal(6.5, 0.72, size=num_rows), 4.5, 8.5)
    true_n = np.clip(rng.normal(68.0, 26.0, size=num_rows), 10.0, 145.0)
    true_p = np.clip(rng.normal(42.0, 16.0, size=num_rows), 5.0, 95.0)
    true_k = np.clip(rng.normal(78.0, 30.0, size=num_rows), 15.0, 165.0)

    latent_df = pd.DataFrame(
        {
            "soil_moisture": true_moisture,
            "temperature": true_temp,
            "humidity": true_humidity,
            "rainfall_forecast_mm": true_rainfall,
            "soil_ph": true_ph,
            "nitrogen": true_n,
            "phosphorus": true_p,
            "potassium": true_k,
            "hour_of_day": hour_of_day,
            "crop_type": crop_type,
        }
    )
    eng_latent = compute_engineered_features(latent_df)

    base_thresholds = np.array([CROP_PARAMETERS[c][0] for c in crop_type], dtype=float)
    kc_values = np.array([CROP_PARAMETERS[c][1] for c in crop_type], dtype=float)

    vpd = eng_latent["vpd_kpa"].to_numpy()
    diurnal_et = eng_latent["diurnal_et_factor"].to_numpy()
    npk_balance = eng_latent["npk_balance"].to_numpy()

    # Physical evapotranspiration demand and dynamic crop moisture threshold
    et_demand = kc_values * (1.75 * vpd + 0.06 * np.maximum(true_temp - 15.0, 0.0)) * (
        0.45 + 0.55 * diurnal_et
    )
    nutrient_vigor = 0.90 + 0.12 * (np.clip(npk_balance, 20.0, 120.0) - 20.0) / 100.0
    ph_stress = 1.0 + 0.06 * np.abs(true_ph - 6.5)

    dynamic_threshold = base_thresholds + 2.4 * et_demand * nutrient_vigor * ph_stress
    effective_moisture = true_moisture + 0.75 * true_rainfall
    moisture_deficit = dynamic_threshold - effective_moisture

    # Add realistic IoT sensor measurement noise to observed readings
    obs_moisture = np.round(
        np.clip(true_moisture + rng.normal(0.0, 1.3, size=num_rows), 2.0, 99.0), 2
    )
    obs_temp = np.round(
        np.clip(true_temp + rng.normal(0.0, 0.45, size=num_rows), 4.0, 48.0), 2
    )
    obs_humidity = np.round(
        np.clip(true_humidity + rng.normal(0.0, 1.4, size=num_rows), 10.0, 99.0), 2
    )
    obs_rainfall = np.round(
        np.clip(true_rainfall + rng.normal(0.0, 0.25, size=num_rows), 0.0, 45.0), 2
    )
    obs_ph = np.round(
        np.clip(true_ph + rng.normal(0.0, 0.08, size=num_rows), 4.2, 8.8), 2
    )
    obs_n = np.round(np.clip(true_n + rng.normal(0.0, 2.0, size=num_rows), 5.0, 155.0), 1)
    obs_p = np.round(np.clip(true_p + rng.normal(0.0, 1.5, size=num_rows), 3.0, 100.0), 1)
    obs_k = np.round(np.clip(true_k + rng.normal(0.0, 2.2, size=num_rows), 8.0, 175.0), 1)

    # Classification target with realistic boundary stochasticity & safety cutoff coherence
    decision_score = moisture_deficit + rng.normal(0.0, 0.85, size=num_rows)
    within_safety = (obs_moisture <= SAFETY_MOISTURE_THRESHOLD) & (
        obs_rainfall <= SAFETY_RAINFALL_THRESHOLD
    )
    irrigation_needed = ((decision_score > 0.0) & within_safety).astype(int)

    # Regression target: liters per m^2 required when irrigation triggers
    positive_deficit = np.maximum(moisture_deficit, 0.6)
    raw_volume = (
        1.8
        + 0.34 * positive_deficit * (0.85 + 0.25 * kc_values)
        + 1.35 * vpd * diurnal_et
        + 0.08 * np.maximum(true_temp - 22.0, 0.0)
        - 0.10 * true_rainfall
        + 0.40 * np.abs(true_ph - 6.5)
        + rng.normal(0.0, 0.35, size=num_rows)
    )
    water_volume_liters = np.where(
        irrigation_needed == 1,
        np.round(np.clip(raw_volume, 1.0, 28.0), 2),
        0.0,
    )

    dataset = pd.DataFrame(
        {
            "soil_moisture": obs_moisture,
            "temperature": obs_temp,
            "humidity": obs_humidity,
            "rainfall_forecast_mm": obs_rainfall,
            "soil_ph": obs_ph,
            "nitrogen": obs_n,
            "phosphorus": obs_p,
            "potassium": obs_k,
            "hour_of_day": hour_of_day.astype(int),
            "crop_type": crop_type,
            "irrigation_needed": irrigation_needed,
            "water_volume_liters": water_volume_liters,
        }
    )
    return dataset[REQUIRED_DATASET_COLUMNS]


def ensure_dataset(
    output_path: Path = DEFAULT_DATA_PATH,
    num_rows: int = DEFAULT_NUM_ROWS,
    seed: int = DEFAULT_SEED,
    force: bool = False,
) -> Path:
    """Create dataset at output_path if missing (or if force=True), preserving custom CSVs."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        print(f"[AgroSense] Dataset already exists at {output_path}; skipping generation.")
        return output_path

    df = generate_dataset(num_rows=num_rows, seed=seed)
    df.to_csv(output_path, index=False)
    pos_rate = df["irrigation_needed"].mean() * 100.0
    print(
        f"[AgroSense] Generated {len(df)} rows at {output_path} "
        f"(seed={seed}, irrigation_ON={pos_rate:.1f}%)."
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AgroSense ML agricultural dataset.")
    parser.add_argument("--output", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--rows", type=int, default=DEFAULT_NUM_ROWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even if dataset CSV already exists.",
    )
    args = parser.parse_args()
    ensure_dataset(output_path=args.output, num_rows=args.rows, seed=args.seed, force=args.force)


if __name__ == "__main__":
    main()
