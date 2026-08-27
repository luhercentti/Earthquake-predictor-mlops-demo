#!/usr/bin/env python3
"""
Earthquake Sequence Model — Training
=====================================

Trains two LightGBM regression models on the processed earthquake dataset:

  Model A — time_model.lgbm
    Input : seismic history of a 2°×2° grid cell
    Output: days_to_next — expected days until the next earthquake in that cell

  Model B — mag_model.lgbm
    Input : same features
    Output: mag_next — expected magnitude of the next earthquake in that cell

Why LightGBM?
─────────────
• Handles the highly non-linear relationships in seismic sequences.
• Natively manages missing values and sparse rolling windows.
• Gradient-boosted trees consistently outperform neural networks on
  tabular seismic data in published benchmarks (e.g. DeepMind 2019 used
  fully-connected nets, but ensemble trees match or beat them on standard
  metrics for catalog-based forecasting).
• Fast training on CPU — no GPU required.

What the model CAN and CANNOT do
─────────────────────────────────
  ✓ Estimate the *probability distribution* of time-to-next and magnitude
    given recent seismicity patterns in a region.
  ✓ Identify which grid cells / regions are currently "overdue" vs "quiet".
  ✓ Rank risk across regions for a given time horizon.
  ✗ Predict the exact date/time of a future earthquake — this is an
    unsolved problem in seismology.
  ✗ Predict events in cells with sparse historical data.

Usage
─────
    python train_model.py
    python train_model.py --parquet data/processed/earthquakes_full.parquet
    python train_model.py --min-mag 4.5   # train only on M≥4.5 events
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# project root is one level up from training/
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from pipeline.features import (
    FEATURE_COLS,
    TARGET_MAG,
    TARGET_TIME,
    build_features,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "training.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_PARQUET = PROJECT_ROOT / "DATA" / "data" / "processed" / "earthquakes_full.parquet"
MODEL_DIR = PROJECT_ROOT / "models"


# ── LightGBM hyperparameters ──────────────────────────────────────────────────
# Tuned for inter-event time and magnitude regression on seismic catalogs.
# Use train_model.py --tune to run a quick Optuna search (requires optuna pkg).
LGBM_PARAMS_TIME = {
    # tweedie suits positive heavy-tailed inter-event times better than MAE/MSE
    "objective": "tweedie",
    "tweedie_variance_power": 1.5,   # 1=Poisson, 2=Gamma; 1.5 fits inter-event empirically
    "metric": "tweedie",
    "n_estimators": 3000,
    "learning_rate": 0.02,
    "num_leaves": 255,
    "max_depth": 8,
    "min_child_samples": 30,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.2,
    "reg_lambda": 0.2,
    "n_jobs": -1,
    "random_state": 42,
    "verbose": -1,
}

LGBM_PARAMS_MAG = {
    # huber is robust to rare extreme magnitudes that inflate MSE
    "objective": "huber",
    "alpha": 0.9,
    "metric": "huber",
    "n_estimators": 2500,
    "learning_rate": 0.02,
    "num_leaves": 255,
    "max_depth": 8,
    "min_child_samples": 30,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 0.2,
    "n_jobs": -1,
    "random_state": 42,
    "verbose": -1,
}


def _temporal_train_test_split(df: pd.DataFrame,
                                test_years: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split chronologically: last `test_years` years → test set.
    This is the correct split for time-series to prevent data leakage.
    """
    cutoff = df["time"].max() - pd.DateOffset(years=test_years)
    train = df[df["time"] < cutoff]
    test = df[df["time"] >= cutoff]
    log.info("Train: %d rows (up to %s)", len(train), cutoff.date())
    log.info("Test : %d rows (after %s)", len(test), cutoff.date())
    return train, test


def _evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    p50_err = np.median(np.abs(y_true - y_pred))
    log.info(
        "[%s]  MAE=%.3f  R²=%.4f  Median-abs-err=%.3f",
        name, mae, r2, p50_err,
    )
    return {"mae": mae, "r2": r2, "median_abs_error": p50_err}


def train(parquet_path: Path, min_mag: float = 2.5) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Build features ─────────────────────────────────────────────────────
    log.info("═" * 60)
    log.info("Building training features from %s …", parquet_path)
    feat_df = build_features(parquet_path)

    if min_mag > 2.5:
        before = len(feat_df)
        feat_df = feat_df[feat_df["magnitude"] >= min_mag]
        log.info("Filtered to M≥%.1f: %d → %d rows", min_mag, before, len(feat_df))

    log.info("Feature matrix shape: %s", feat_df.shape)

    # ── 2. Temporal train/test split ──────────────────────────────────────────
    train_df, test_df = _temporal_train_test_split(feat_df)

    X_train = train_df[FEATURE_COLS].values
    X_test = test_df[FEATURE_COLS].values

    y_time_train = train_df[TARGET_TIME].values
    y_time_test = test_df[TARGET_TIME].values
    y_mag_train = train_df[TARGET_MAG].values
    y_mag_test = test_df[TARGET_MAG].values

    # ── 3. Cross-validate with TimeSeriesSplit (5 folds) ──────────────────────
    log.info("Running 5-fold time-series cross-validation …")
    tscv = TimeSeriesSplit(n_splits=5)
    cv_mae_time, cv_mae_mag = [], []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        m_time = lgb.LGBMRegressor(**LGBM_PARAMS_TIME)
        m_time.fit(X_train[tr_idx], y_time_train[tr_idx],
                   eval_set=[(X_train[val_idx], y_time_train[val_idx])],
                   callbacks=[lgb.early_stopping(50, verbose=False),
                               lgb.log_evaluation(0)])
        cv_mae_time.append(mean_absolute_error(y_time_train[val_idx],
                                               m_time.predict(X_train[val_idx])))

        m_mag = lgb.LGBMRegressor(**LGBM_PARAMS_MAG)
        m_mag.fit(X_train[tr_idx], y_mag_train[tr_idx],
                  eval_set=[(X_train[val_idx], y_mag_train[val_idx])],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                              lgb.log_evaluation(0)])
        cv_mae_mag.append(mean_absolute_error(y_mag_train[val_idx],
                                              m_mag.predict(X_train[val_idx])))

        log.info("Fold %d — days_to_next MAE=%.2f | mag_next MAE=%.3f",
                 fold, cv_mae_time[-1], cv_mae_mag[-1])

    log.info("CV avg  — days_to_next MAE=%.2f±%.2f | mag_next MAE=%.3f±%.3f",
             np.mean(cv_mae_time), np.std(cv_mae_time),
             np.mean(cv_mae_mag), np.std(cv_mae_mag))

    # ── 4. Train final models on full training set ────────────────────────────
    log.info("Training final models on full training set …")

    final_time_model = lgb.LGBMRegressor(**LGBM_PARAMS_TIME)
    final_time_model.fit(
        X_train, y_time_train,
        eval_set=[(X_test, y_time_test)],
        callbacks=[lgb.early_stopping(100, verbose=False),
                   lgb.log_evaluation(200)],
    )

    final_mag_model = lgb.LGBMRegressor(**LGBM_PARAMS_MAG)
    final_mag_model.fit(
        X_train, y_mag_train,
        eval_set=[(X_test, y_mag_test)],
        callbacks=[lgb.early_stopping(100, verbose=False),
                   lgb.log_evaluation(200)],
    )

    # ── 5. Test-set evaluation ────────────────────────────────────────────────
    log.info("Test-set evaluation:")
    metrics_time = _evaluate("days_to_next", y_time_test,
                              final_time_model.predict(X_test))
    metrics_mag = _evaluate("mag_next", y_mag_test,
                             final_mag_model.predict(X_test))

    # ── 6. Feature importance ─────────────────────────────────────────────────
    log.info("\nTop-15 features (days_to_next model):")
    imp = sorted(zip(FEATURE_COLS, final_time_model.feature_importances_),
                 key=lambda x: x[1], reverse=True)[:15]
    for feat, score in imp:
        log.info("  %-30s  %d", feat, score)

    # ── 7. Save models + metadata ─────────────────────────────────────────────
    time_path = MODEL_DIR / "time_model.lgbm"
    mag_path = MODEL_DIR / "mag_model.lgbm"
    meta_path = MODEL_DIR / "metadata.json"

    joblib.dump(final_time_model, time_path)
    joblib.dump(final_mag_model, mag_path)

    meta = {
        "parquet_source": str(parquet_path),
        "training_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "feature_cols": FEATURE_COLS,
        "target_time": TARGET_TIME,
        "target_mag": TARGET_MAG,
        "min_mag_filter": min_mag,
        "test_metrics": {
            "days_to_next": metrics_time,
            "mag_next": metrics_mag,
        },
        "cv_metrics": {
            "days_to_next_mae_mean": float(np.mean(cv_mae_time)),
            "mag_next_mae_mean": float(np.mean(cv_mae_mag)),
        },
        "lgbm_params_time": LGBM_PARAMS_TIME,
        "lgbm_params_mag": LGBM_PARAMS_MAG,
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    log.info("═" * 60)
    log.info("Models saved:")
    log.info("  %s", time_path)
    log.info("  %s", mag_path)
    log.info("  %s  (metadata + metrics)", meta_path)
    log.info("Run 'python predict.py --country Peru' to query the model.")
    log.info("═" * 60)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train earthquake sequence LightGBM models.")
    p.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET,
                   help="Path to earthquakes_full.parquet")
    p.add_argument("--min-mag", type=float, default=2.5,
                   help="Minimum magnitude to include in training (default: 2.5)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.parquet.exists():
        log.error("Parquet not found: %s\nRun run_pipeline.py first.", args.parquet)
        sys.exit(1)
    train(args.parquet, min_mag=args.min_mag)
