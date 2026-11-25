"""Preprocessing, feature engineering, safety rules, and inference utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "sensor_irrigation_data.csv"
MODELS_DIR = PROJECT_ROOT / "models"
CLASSIFIER_PATH = MODELS_DIR / "irrigation_classifier.joblib"
REGRESSOR_PATH = MODELS_DIR / "water_regressor.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

CROP_TYPES: tuple[str, ...] = ("Wheat", "Rice", "Cotton", "Maize")

RAW_NUMERIC_FEATURES: list[str] = [
    "soil_moisture",
    "temperature",
    "humidity",
    "rainfall_forecast_mm",
    "soil_ph",
    "nitrogen",
    "phosphorus",
    "potassium",
    "hour_of_day",
]

CATEGORICAL_FEATURES: list[str] = ["crop_type"]
RAW_FEATURE_COLUMNS: list[str] = RAW_NUMERIC_FEATURES + CATEGORICAL_FEATURES

ENGINEERED_FEATURES: list[str] = [
    "vpd_kpa",
    "dryness_index",
    "effective_moisture",
    "diurnal_et_factor",
    "hour_sin",
    "hour_cos",
    "npk_balance",
]

ALL_NUMERIC_FEATURES: list[str] = RAW_NUMERIC_FEATURES + ENGINEERED_FEATURES

TARGET_CLASSIFICATION = "irrigation_needed"
TARGET_REGRESSION = "water_volume_liters"
REQUIRED_DATASET_COLUMNS: list[str] = RAW_FEATURE_COLUMNS + [
    TARGET_CLASSIFICATION,
    TARGET_REGRESSION,
]

SAFETY_MOISTURE_THRESHOLD = 75.0
SAFETY_RAINFALL_THRESHOLD = 15.0

FEATURE_DISPLAY_METADATA: dict[str, tuple[str, str]] = {
    "soil_moisture": ("Soil Moisture", "%"),
    "temperature": ("Air Temperature", "°C"),
    "humidity": ("Relative Humidity", "%"),
    "rainfall_forecast_mm": ("Rainfall Forecast (24h)", "mm"),
    "soil_ph": ("Soil pH", "pH"),
    "nitrogen": ("Nitrogen (N)", "mg/kg"),
    "phosphorus": ("Phosphorus (P)", "mg/kg"),
    "potassium": ("Potassium (K)", "mg/kg"),
    "hour_of_day": ("Hour of Day", "h"),
    "vpd_kpa": ("Vapor Pressure Deficit", "kPa"),
    "dryness_index": ("Dryness Index", ""),
    "effective_moisture": ("Effective Moisture", "%"),
    "diurnal_et_factor": ("Diurnal ET Factor", ""),
    "hour_sin": ("Time Cycle (sin)", ""),
    "hour_cos": ("Time Cycle (cos)", ""),
    "npk_balance": ("NPK Balance Index", "mg/kg"),
    "crop_type": ("Crop Type", ""),
}


def compute_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute physical agronomic features from raw sensor readings."""
    out = df.copy()
    soil_moisture = pd.to_numeric(out["soil_moisture"], errors="coerce").astype(float)
    temperature = pd.to_numeric(out["temperature"], errors="coerce").astype(float)
    humidity = pd.to_numeric(out["humidity"], errors="coerce").astype(float)
    rainfall = pd.to_numeric(out["rainfall_forecast_mm"], errors="coerce").astype(float)
    nitrogen = pd.to_numeric(out["nitrogen"], errors="coerce").astype(float)
    phosphorus = pd.to_numeric(out["phosphorus"], errors="coerce").astype(float)
    potassium = pd.to_numeric(out["potassium"], errors="coerce").astype(float)
    hour = pd.to_numeric(out["hour_of_day"], errors="coerce").astype(float)

    # Tetens equation for saturation vapor pressure (kPa) and Vapor Pressure Deficit
    sat_vp = 0.6108 * np.exp((17.27 * temperature) / (temperature + 237.3))
    vpd_kpa = sat_vp * np.clip(1.0 - humidity / 100.0, 0.0, 1.0)

    out["vpd_kpa"] = np.round(vpd_kpa, 4)
    out["dryness_index"] = np.round(
        (temperature / (np.maximum(soil_moisture, 0.0) + 5.0))
        * (1.0 - np.clip(humidity, 0.0, 100.0) / 150.0),
        4,
    )
    out["effective_moisture"] = np.round(soil_moisture + 0.75 * rainfall, 4)
    out["diurnal_et_factor"] = np.round(
        0.20 + 0.80 * np.exp(-0.5 * ((hour - 13.5) / 3.5) ** 2),
        4,
    )
    out["hour_sin"] = np.round(np.sin(2.0 * np.pi * hour / 24.0), 4)
    out["hour_cos"] = np.round(np.cos(2.0 * np.pi * hour / 24.0), 4)
    out["npk_balance"] = np.round((nitrogen + phosphorus + potassium) / 3.0, 4)
    return out


class AgriculturalFeatureEngineer(BaseEstimator, TransformerMixin):
    """Sklearn transformer adding domain-specific agronomic and temporal features."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> AgriculturalFeatureEngineer:
        missing = [col for col in RAW_FEATURE_COLUMNS if col not in X.columns]
        if missing:
            raise ValueError(f"Missing required input columns: {missing}")
        self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=RAW_FEATURE_COLUMNS)
        missing = [col for col in RAW_FEATURE_COLUMNS if col not in X.columns]
        if missing:
            raise ValueError(f"Missing required input columns: {missing}")
        engineered = compute_engineered_features(X[RAW_FEATURE_COLUMNS])
        return engineered[ALL_NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(ALL_NUMERIC_FEATURES + CATEGORICAL_FEATURES, dtype=object)


def build_preprocessor() -> ColumnTransformer:
    """Create ColumnTransformer for numeric scaling and crop type encoding."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    categories=[list(CROP_TYPES)],
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, ALL_NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def build_model_pipeline(estimator: BaseEstimator) -> Pipeline:
    """Wrap feature engineering, preprocessing, and estimator into a single Pipeline."""
    return Pipeline(
        steps=[
            ("feature_engineer", AgriculturalFeatureEngineer()),
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ]
    )


def validate_dataset_schema(df: pd.DataFrame) -> None:
    """Validate that a loaded dataset CSV contains all expected columns and non-empty rows."""
    missing = [col for col in REQUIRED_DATASET_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}. "
            f"Expected columns: {REQUIRED_DATASET_COLUMNS}"
        )
    if df.empty:
        raise ValueError("Dataset CSV is empty.")


def evaluate_safety_override(
    soil_moisture: float,
    rainfall_forecast_mm: float,
) -> tuple[bool, str | None]:
    """Enforce hard agronomic safety cutoffs preventing over-watering."""
    reasons: list[str] = []
    if soil_moisture > SAFETY_MOISTURE_THRESHOLD:
        reasons.append(
            f"Soil moisture ({soil_moisture:.1f}%) exceeds {SAFETY_MOISTURE_THRESHOLD:.0f}% safety threshold"
        )
    if rainfall_forecast_mm > SAFETY_RAINFALL_THRESHOLD:
        reasons.append(
            f"Forecasted rainfall ({rainfall_forecast_mm:.1f} mm) exceeds {SAFETY_RAINFALL_THRESHOLD:.0f} mm safety threshold"
        )
    if reasons:
        return True, "; ".join(reasons)
    return False, None


def _format_feature_value(feature: str, row: pd.Series) -> str:
    """Format raw or engineered feature value with physical units."""
    val = row[feature]
    if feature == "crop_type":
        return str(val)
    _, unit = FEATURE_DISPLAY_METADATA.get(feature, (feature, ""))
    num = float(val)
    if feature == "hour_of_day":
        return f"{int(round(num)):02d}:00"
    if unit:
        return f"{num:.2f} {unit}" if unit in ("kPa", "pH") else f"{num:.1f} {unit}"
    return f"{num:.2f}"


def compute_driving_factors(
    classifier_pipeline: Pipeline,
    raw_df: pd.DataFrame,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Compute top-K local driving factors for a single classification prediction."""
    feat_eng = classifier_pipeline.named_steps["feature_engineer"]
    preproc = classifier_pipeline.named_steps["preprocessor"]
    estimator = classifier_pipeline.named_steps["model"]

    engineered_df = feat_eng.transform(raw_df)
    X_trans = preproc.transform(engineered_df)
    feature_names = list(preproc.get_feature_names_out())

    # Prefer exact TreeSHAP contributions when estimator is XGBoost
    raw_contribs: dict[str, float] = {}
    if hasattr(estimator, "get_booster"):
        try:
            import xgboost as xgb

            booster = estimator.get_booster()
            shap_matrix = booster.predict(xgb.DMatrix(X_trans), pred_contribs=True)
            # Last column is expected bias; map log-odds delta to approximate probability scale
            p_actual = float(estimator.predict_proba(X_trans)[0, 1])
            prob_scale = max(p_actual * (1.0 - p_actual), 0.10)
            crop_sum = 0.0
            for idx, name in enumerate(feature_names):
                val = float(shap_matrix[0, idx]) * prob_scale
                if name.startswith("crop_type_"):
                    crop_sum += val
                else:
                    raw_contribs[name] = val
            raw_contribs["crop_type"] = crop_sum
        except Exception:
            raw_contribs = {}

    if not raw_contribs:
        p_actual = float(estimator.predict_proba(X_trans)[0, 1])
        groups: list[tuple[str, list[int], list[float]]] = []
        crop_indices: list[int] = []
        for idx, name in enumerate(feature_names):
            if name.startswith("crop_type_"):
                crop_indices.append(idx)
            else:
                groups.append((name, [idx], [0.0]))
        if crop_indices:
            uniform_crop = [1.0 / len(crop_indices)] * len(crop_indices)
            groups.append(("crop_type", crop_indices, uniform_crop))

        X_batch = np.repeat(X_trans, repeats=len(groups), axis=0)
        for row_idx, (_, col_indices, baseline_vals) in enumerate(groups):
            for col_idx, b_val in zip(col_indices, baseline_vals):
                X_batch[row_idx, col_idx] = b_val

        p_baselines = estimator.predict_proba(X_batch)[:, 1]
        for row_idx, (group_name, _, _) in enumerate(groups):
            raw_contribs[group_name] = p_actual - float(p_baselines[row_idx])

    row = engineered_df.iloc[0]
    ranked = sorted(raw_contribs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]

    factors: list[dict[str, Any]] = []
    for feature, impact in ranked:
        label, _ = FEATURE_DISPLAY_METADATA.get(feature, (feature, ""))
        display_val = _format_feature_value(feature, row)
        if impact > 0.002:
            direction = "increases_irrigation_need"
            phrase = "increases irrigation likelihood"
        elif impact < -0.002:
            direction = "decreases_irrigation_need"
            phrase = "decreases irrigation likelihood"
        else:
            direction = "neutral"
            phrase = "has minimal net shift on irrigation likelihood"

        factors.append(
            {
                "feature": feature,
                "label": label,
                "value": display_val,
                "impact": round(float(impact), 4),
                "direction": direction,
                "description": f"{label} ({display_val}) {phrase} ({impact:+.3f})",
            }
        )
    return factors


def load_artifacts(
    classifier_path: Path = CLASSIFIER_PATH,
    regressor_path: Path = REGRESSOR_PATH,
    metrics_path: Path = METRICS_PATH,
) -> tuple[Pipeline, Pipeline, dict[str, Any]]:
    """Load saved joblib pipelines and evaluation metrics JSON."""
    if not classifier_path.exists() or not regressor_path.exists():
        raise FileNotFoundError(
            "Trained model artifacts not found. Run `python src/train.py` first."
        )
    classifier = joblib.load(classifier_path)
    regressor = joblib.load(regressor_path)
    metrics: dict[str, Any] = {}
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return classifier, regressor, metrics


def detect_sensor_anomalies(sensor_payload: dict[str, Any]) -> list[str]:
    """Detect physically implausible readings or probable IoT probe hardware faults."""
    anomalies: list[str] = []
    moisture = float(sensor_payload.get("soil_moisture", 0.0))
    temp = float(sensor_payload.get("temperature", 0.0))
    hum = float(sensor_payload.get("humidity", 0.0))
    rain = float(sensor_payload.get("rainfall_forecast_mm", 0.0))
    ph = float(sensor_payload.get("soil_ph", 7.0))
    n = float(sensor_payload.get("nitrogen", 0.0))
    p = float(sensor_payload.get("phosphorus", 0.0))
    k = float(sensor_payload.get("potassium", 0.0))

    if moisture <= 1.5:
        anomalies.append("Soil moisture sensor reports near-zero (<1.5%); verify probe ground insertion.")
    elif moisture >= 98.5:
        anomalies.append("Soil moisture saturated (>98.5%); potential sensor pin short circuit.")

    if moisture < 10.0 and rain >= 20.0:
        anomalies.append("Desiccated soil reading during heavy rainfall (>20mm); probable probe displacement.")

    if temp > 46.0 and hum > 85.0:
        anomalies.append("Extreme heat (>46°C) combined with high humidity (>85%) represents an atypical microclimate.")

    if ph < 3.8 or ph > 9.5:
        anomalies.append(f"Soil pH ({ph:.1f}) outside standard agricultural soil range (3.8 - 9.5).")

    if n <= 1.0 or p <= 1.0 or k <= 1.0:
        anomalies.append("Depleted/zero NPK reading suggests optical or RS485 communication timeout.")

    return anomalies


def run_irrigation_prediction(
    sensor_payload: dict[str, Any],
    classifier_pipeline: Pipeline,
    regressor_pipeline: Pipeline,
    metrics_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute end-to-end prediction, safety override evaluation, and factor attribution."""
    raw_df = pd.DataFrame([{col: sensor_payload[col] for col in RAW_FEATURE_COLUMNS}])
    engineered_df = classifier_pipeline.named_steps["feature_engineer"].transform(raw_df)
    eng_row = engineered_df.iloc[0]

    proba_on = float(classifier_pipeline.predict_proba(raw_df)[0, 1])
    pred_class = int(classifier_pipeline.predict(raw_df)[0])

    override_active, override_reason = evaluate_safety_override(
        soil_moisture=float(sensor_payload["soil_moisture"]),
        rainfall_forecast_mm=float(sensor_payload["rainfall_forecast_mm"]),
    )

    if override_active:
        irrigation_needed = False
        irrigation_action = "OFF"
        recommended_water_liters = 0.0
    else:
        irrigation_needed = bool(pred_class == 1)
        irrigation_action = "ON" if irrigation_needed else "OFF"
        if irrigation_needed:
            raw_liters = float(regressor_pipeline.predict(raw_df)[0])
            recommended_water_liters = round(max(0.0, raw_liters), 2)
        else:
            recommended_water_liters = 0.0

    driving_factors = compute_driving_factors(classifier_pipeline, raw_df, top_k=3)
    sensor_anomalies = detect_sensor_anomalies(sensor_payload)

    clf_name = type(classifier_pipeline.named_steps["model"]).__name__
    reg_name = type(regressor_pipeline.named_steps["model"]).__name__
    if metrics_summary:
        clf_name = metrics_summary.get("classification", {}).get("best_model", clf_name)
        reg_name = metrics_summary.get("regression", {}).get("best_model", reg_name)

    return {
        "irrigation_action": irrigation_action,
        "irrigation_needed": irrigation_needed,
        "irrigation_probability": round(proba_on, 4),
        "recommended_water_liters": recommended_water_liters,
        "override_active": override_active,
        "override_reason": override_reason,
        "sensor_anomalies": sensor_anomalies,
        "top_driving_factors": driving_factors,
        "engineered_features": {
            "vpd_kpa": round(float(eng_row["vpd_kpa"]), 4),
            "dryness_index": round(float(eng_row["dryness_index"]), 4),
            "effective_moisture": round(float(eng_row["effective_moisture"]), 2),
            "diurnal_et_factor": round(float(eng_row["diurnal_et_factor"]), 4),
            "npk_balance": round(float(eng_row["npk_balance"]), 2),
        },
        "model_used": {
            "classifier": clf_name,
            "regressor": reg_name,
        },
    }


def compute_batch_driving_factors(
    classifier_pipeline: Pipeline,
    engineered_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    top_k: int = 3,
) -> list[list[dict[str, Any]]]:
    """Vectorized calculation of driving factors across a batch of samples."""
    preproc = classifier_pipeline.named_steps["preprocessor"]
    estimator = classifier_pipeline.named_steps["model"]
    X_trans = preproc.transform(engineered_df)
    feature_names = list(preproc.get_feature_names_out())
    n_rows = len(engineered_df)

    if hasattr(estimator, "get_booster"):
        try:
            import xgboost as xgb

            booster = estimator.get_booster()
            shap_matrix = booster.predict(xgb.DMatrix(X_trans), pred_contribs=True)
            p_actual_arr = estimator.predict_proba(X_trans)[:, 1]

            batch_factors: list[list[dict[str, Any]]] = []
            for i in range(n_rows):
                p_actual = float(p_actual_arr[i])
                prob_scale = max(p_actual * (1.0 - p_actual), 0.10)
                raw_contribs: dict[str, float] = {}
                crop_sum = 0.0
                for col_idx, name in enumerate(feature_names):
                    val = float(shap_matrix[i, col_idx]) * prob_scale
                    if name.startswith("crop_type_"):
                        crop_sum += val
                    else:
                        raw_contribs[name] = val
                raw_contribs["crop_type"] = crop_sum

                row = engineered_df.iloc[i]
                ranked = sorted(raw_contribs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
                factors: list[dict[str, Any]] = []
                for feature, impact in ranked:
                    label, _ = FEATURE_DISPLAY_METADATA.get(feature, (feature, ""))
                    display_val = _format_feature_value(feature, row)
                    if impact > 0.002:
                        direction = "increases_irrigation_need"
                        phrase = "increases irrigation likelihood"
                    elif impact < -0.002:
                        direction = "decreases_irrigation_need"
                        phrase = "decreases irrigation likelihood"
                    else:
                        direction = "neutral"
                        phrase = "has minimal net shift on irrigation likelihood"
                    factors.append(
                        {
                            "feature": feature,
                            "label": label,
                            "value": display_val,
                            "impact": round(float(impact), 4),
                            "direction": direction,
                            "description": f"{label} ({display_val}) {phrase} ({impact:+.3f})",
                        }
                    )
                batch_factors.append(factors)
            return batch_factors
        except Exception:
            pass

    # Generic fallback
    return [
        compute_driving_factors(classifier_pipeline, raw_df.iloc[[i]], top_k=top_k)
        for i in range(n_rows)
    ]


def run_batch_irrigation_prediction(
    sensor_payloads: list[dict[str, Any]],
    classifier_pipeline: Pipeline,
    regressor_pipeline: Pipeline,
    metrics_summary: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Execute vectorized batch prediction across multiple sensor readings."""
    if not sensor_payloads:
        return []

    rows = [{col: p[col] for col in RAW_FEATURE_COLUMNS} for p in sensor_payloads]
    raw_df = pd.DataFrame(rows)
    engineered_df = classifier_pipeline.named_steps["feature_engineer"].transform(raw_df)

    proba_on_arr = classifier_pipeline.predict_proba(raw_df)[:, 1]
    pred_class_arr = classifier_pipeline.predict(raw_df)
    reg_liters_arr = np.maximum(regressor_pipeline.predict(raw_df), 0.0)

    clf_name = type(classifier_pipeline.named_steps["model"]).__name__
    reg_name = type(regressor_pipeline.named_steps["model"]).__name__
    if metrics_summary:
        clf_name = metrics_summary.get("classification", {}).get("best_model", clf_name)
        reg_name = metrics_summary.get("regression", {}).get("best_model", reg_name)

    all_factors = compute_batch_driving_factors(classifier_pipeline, engineered_df, raw_df, top_k=3)

    results: list[dict[str, Any]] = []
    for idx, payload in enumerate(sensor_payloads):
        proba_on = float(proba_on_arr[idx])
        pred_class = int(pred_class_arr[idx])

        override_active, override_reason = evaluate_safety_override(
            soil_moisture=float(payload["soil_moisture"]),
            rainfall_forecast_mm=float(payload["rainfall_forecast_mm"]),
        )

        if override_active:
            irrigation_needed = False
            irrigation_action = "OFF"
            rec_liters = 0.0
        else:
            irrigation_needed = bool(pred_class == 1)
            irrigation_action = "ON" if irrigation_needed else "OFF"
            rec_liters = round(float(reg_liters_arr[idx]), 2) if irrigation_needed else 0.0

        factors = all_factors[idx]
        anomalies = detect_sensor_anomalies(payload)
        eng_row = engineered_df.iloc[idx]

        results.append(
            {
                "device_id": payload.get("device_id"),
                "irrigation_action": irrigation_action,
                "irrigation_needed": irrigation_needed,
                "irrigation_probability": round(proba_on, 4),
                "recommended_water_liters": rec_liters,
                "override_active": override_active,
                "override_reason": override_reason,
                "sensor_anomalies": anomalies,
                "top_driving_factors": factors,
                "engineered_features": {
                    "vpd_kpa": round(float(eng_row["vpd_kpa"]), 4),
                    "dryness_index": round(float(eng_row["dryness_index"]), 4),
                    "effective_moisture": round(float(eng_row["effective_moisture"]), 2),
                    "diurnal_et_factor": round(float(eng_row["diurnal_et_factor"]), 4),
                    "npk_balance": round(float(eng_row["npk_balance"]), 2),
                },
                "model_used": {
                    "classifier": clf_name,
                    "regressor": reg_name,
                },
            }
        )
    return results
