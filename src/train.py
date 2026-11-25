"""Model training, evaluation, comparison, and artifact persistence for AgroSense ML."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_generator import ensure_dataset
from src.preprocess import (
    CLASSIFIER_PATH,
    DEFAULT_DATA_PATH,
    METRICS_PATH,
    MODELS_DIR,
    RAW_FEATURE_COLUMNS,
    REGRESSOR_PATH,
    TARGET_CLASSIFICATION,
    TARGET_REGRESSION,
    build_model_pipeline,
    validate_dataset_schema,
)


def _build_candidate_classifiers(seed: int = 42) -> dict[str, Any]:
    """Instantiate classification candidates with clean XGBoost fallback."""
    candidates: dict[str, Any] = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=seed),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=16,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        ),
    }
    try:
        from xgboost import XGBClassifier

        candidates["XGBoost"] = XGBClassifier(
            n_estimators=220,
            max_depth=5,
            learning_rate=0.07,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            eval_metric="logloss",
            n_jobs=-1,
        )
    except ImportError:
        candidates["HistGradientBoosting"] = HistGradientBoostingClassifier(
            max_iter=220,
            max_depth=6,
            learning_rate=0.07,
            random_state=seed,
        )
    return candidates


def _build_candidate_regressors(seed: int = 42) -> dict[str, Any]:
    """Instantiate regression candidates with clean XGBoost fallback."""
    candidates: dict[str, Any] = {
        "Ridge Regression": Ridge(alpha=1.0, random_state=seed),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=16,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        ),
    }
    try:
        from xgboost import XGBRegressor

        candidates["XGBoost"] = XGBRegressor(
            n_estimators=220,
            max_depth=5,
            learning_rate=0.07,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            n_jobs=-1,
        )
    except ImportError:
        candidates["HistGradientBoosting"] = HistGradientBoostingRegressor(
            max_iter=220,
            max_depth=6,
            learning_rate=0.07,
            random_state=seed,
        )
    return candidates


def extract_feature_importance(
    pipeline: Pipeline,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    scoring: str,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Extract normalized global feature importances from a fitted pipeline."""
    preproc = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["model"]
    feature_names = list(preproc.get_feature_names_out())

    if hasattr(estimator, "feature_importances_"):
        raw_weights = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        raw_weights = np.abs(np.asarray(estimator.coef_, dtype=float).ravel())
    else:
        perm = permutation_importance(
            pipeline,
            X_eval,
            y_eval,
            n_repeats=5,
            random_state=seed,
            scoring=scoring,
            n_jobs=-1,
        )
        # Permutation on raw features has length len(RAW_FEATURE_COLUMNS)
        raw_names = list(X_eval.columns)
        raw_imp = np.maximum(perm.importances_mean, 0.0)
        total = float(raw_imp.sum()) or 1.0
        return sorted(
            [
                {"feature": name, "importance": round(float(val / total), 5)}
                for name, val in zip(raw_names, raw_imp)
            ],
            key=lambda item: item["importance"],
            reverse=True,
        )

    raw_weights = np.maximum(raw_weights, 0.0)
    total_weight = float(raw_weights.sum()) or 1.0
    normalized = raw_weights / total_weight
    return sorted(
        [
            {"feature": str(name), "importance": round(float(weight), 5)}
            for name, weight in zip(feature_names, normalized)
        ],
        key=lambda item: item["importance"],
        reverse=True,
    )


def train_models(
    data_path: Path = DEFAULT_DATA_PATH,
    models_dir: Path = MODELS_DIR,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    """Train, evaluate, compare, and persist classification and regression models."""
    if not data_path.exists():
        if data_path == DEFAULT_DATA_PATH:
            ensure_dataset(output_path=data_path, seed=seed)
        else:
            raise FileNotFoundError(f"Dataset file not found: {data_path}")

    df = pd.read_csv(data_path)
    validate_dataset_schema(df)

    X = df[RAW_FEATURE_COLUMNS].copy()
    y_clf = df[TARGET_CLASSIFICATION].astype(int)
    y_reg = df[TARGET_REGRESSION].astype(float)

    X_train, X_test, y_clf_train, y_clf_test, y_reg_train_all, y_reg_test_all = train_test_split(
        X,
        y_clf,
        y_reg,
        test_size=test_size,
        random_state=seed,
        stratify=y_clf,
    )

    # --- Task 1: Irrigation Action Classification ---
    clf_results: dict[str, dict[str, Any]] = {}
    fitted_classifiers: dict[str, Pipeline] = {}

    for name, estimator in _build_candidate_classifiers(seed=seed).items():
        pipeline = build_model_pipeline(estimator)
        pipeline.fit(X_train, y_clf_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        clf_results[name] = {
            "accuracy": round(float(accuracy_score(y_clf_test, y_pred)), 4),
            "f1": round(float(f1_score(y_clf_test, y_pred)), 4),
            "roc_auc": round(float(roc_auc_score(y_clf_test, y_proba)), 4),
            "confusion_matrix": confusion_matrix(y_clf_test, y_pred).tolist(),
            "confusion_labels": ["OFF (0)", "ON (1)"],
        }
        fitted_classifiers[name] = pipeline

    best_clf_name = max(
        clf_results,
        key=lambda m: (clf_results[m]["f1"], clf_results[m]["roc_auc"]),
    )
    best_clf_pipeline = fitted_classifiers[best_clf_name]
    clf_importance = extract_feature_importance(
        best_clf_pipeline, X_test, y_clf_test, scoring="f1", seed=seed
    )

    # --- Task 2: Water Volume Regression (evaluated on triggered irrigation events) ---
    train_pos_mask = y_clf_train == 1
    test_pos_mask = y_clf_test == 1
    X_reg_train = X_train.loc[train_pos_mask]
    y_reg_train = y_reg_train_all.loc[train_pos_mask]
    X_reg_test = X_test.loc[test_pos_mask]
    y_reg_test = y_reg_test_all.loc[test_pos_mask]

    if X_reg_train.empty or X_reg_test.empty:
        raise ValueError("Insufficient positive irrigation rows to train regression models.")

    reg_results: dict[str, dict[str, Any]] = {}
    fitted_regressors: dict[str, Pipeline] = {}

    for name, estimator in _build_candidate_regressors(seed=seed).items():
        pipeline = build_model_pipeline(estimator)
        pipeline.fit(X_reg_train, y_reg_train)
        y_pred = np.maximum(pipeline.predict(X_reg_test), 0.0)

        reg_results[name] = {
            "rmse": round(float(root_mean_squared_error(y_reg_test, y_pred)), 4),
            "mae": round(float(mean_absolute_error(y_reg_test, y_pred)), 4),
            "r2": round(float(r2_score(y_reg_test, y_pred)), 4),
        }
        fitted_regressors[name] = pipeline

    best_reg_name = min(
        reg_results,
        key=lambda m: (reg_results[m]["rmse"], -reg_results[m]["r2"]),
    )
    best_reg_pipeline = fitted_regressors[best_reg_name]
    reg_importance = extract_feature_importance(
        best_reg_pipeline, X_reg_test, y_reg_test, scoring="r2", seed=seed
    )

    models_dir.mkdir(parents=True, exist_ok=True)
    classifier_out = models_dir / CLASSIFIER_PATH.name
    regressor_out = models_dir / REGRESSOR_PATH.name
    metrics_out = models_dir / METRICS_PATH.name

    joblib.dump(best_clf_pipeline, classifier_out)
    joblib.dump(best_reg_pipeline, regressor_out)

    feature_names_out = list(
        best_clf_pipeline.named_steps["preprocessor"].get_feature_names_out()
    )

    metrics_payload: dict[str, Any] = {
        "metadata": {
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_path": str(data_path),
            "total_rows": int(len(df)),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "positive_irrigation_rate": round(float(y_clf.mean()), 4),
            "regression_subset": "irrigation_needed == 1",
            "regression_train_rows": int(len(X_reg_train)),
            "regression_test_rows": int(len(X_reg_test)),
            "feature_names": feature_names_out,
        },
        "classification": {
            "best_model": best_clf_name,
            "selection_metric": "f1",
            "models": clf_results,
            "feature_importance": clf_importance,
        },
        "regression": {
            "best_model": best_reg_name,
            "selection_metric": "rmse",
            "models": reg_results,
            "feature_importance": reg_importance,
        },
    }

    metrics_out.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print("\n=== AgroSense ML Training Summary ===")
    print(f"Dataset: {data_path} ({len(df)} rows)")
    print("\n[Classification]")
    for model_name, scores in clf_results.items():
        tag = " <-- BEST" if model_name == best_clf_name else ""
        print(
            f"  {model_name:<22} | F1={scores['f1']:.4f} | "
            f"Acc={scores['accuracy']:.4f} | ROC-AUC={scores['roc_auc']:.4f}{tag}"
        )
    print("\n[Regression]")
    for model_name, scores in reg_results.items():
        tag = " <-- BEST" if model_name == best_reg_name else ""
        print(
            f"  {model_name:<22} | RMSE={scores['rmse']:.4f} | "
            f"MAE={scores['mae']:.4f} | R2={scores['r2']:.4f}{tag}"
        )
    print(f"\nSaved artifacts to {models_dir}")
    return metrics_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AgroSense ML models.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_models(data_path=args.data, models_dir=args.models_dir, seed=args.seed)


if __name__ == "__main__":
    main()
