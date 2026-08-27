"""
Feature engineering for the earthquake sequence prediction model.

Approach
────────
Each earthquake event becomes one training row.  Features describe the
seismic history of its 2°×2° grid cell in the preceding 7/30/90/365 days
(rolling window stats).  Labels are:
  • days_to_next  – days until the next event in the same cell (regression)
  • mag_next      – magnitude of that next event (regression)

⚠️  Important honesty note
   Earthquake science cannot predict the exact time of individual events.
   What this model learns is the *conditional distribution* of inter-event
   times and magnitudes given recent seismicity patterns in a cell.
   Outputs must be interpreted as probabilistic estimates, not exact forecasts.

Grid resolution: 2° × 2°  (~220 km at equator – appropriate for country/
regional forecasting; too coarse for city-level precision).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

log = logging.getLogger(__name__)

GRID_DEG = 2.0
# Training uses data from 1973+ (global seismic network reached reliable density)
TRAIN_START = pd.Timestamp("1973-01-01", tz="UTC")
# Cap labels at 365 days (events with no successor within a year are dropped)
MAX_DAYS_LABEL = 365
# Minimum events in a cell to be included in training
MIN_CELL_EVENTS = 5

FEATURE_COLS = [
    # rolling window stats (past 7/30/90/365 days in same cell)
    "count_7d",   "mean_mag_7d",   "max_mag_7d",   "min_mag_7d",   "std_mag_7d",   "mean_depth_7d",   "seismic_moment_7d",
    "count_30d",  "mean_mag_30d",  "max_mag_30d",  "min_mag_30d",  "std_mag_30d",  "mean_depth_30d",  "seismic_moment_30d",
    "count_90d",  "mean_mag_90d",  "max_mag_90d",  "min_mag_90d",  "std_mag_90d",  "mean_depth_90d",  "seismic_moment_90d",
    "count_365d", "mean_mag_365d", "max_mag_365d", "min_mag_365d", "std_mag_365d", "mean_depth_365d", "seismic_moment_365d",
    # Gutenberg-Richter b-value: slope of log(N)~M — low b = higher large-quake risk
    "b_value_365d",
    # inter-event time sequence features
    "days_since_last",
    "inter_event_mean_10",   # mean of last 10 inter-event times (cell rate)
    "inter_event_cv_10",     # coefficient of variation — high = aftershock clustering
    # current event
    "magnitude", "depth", "mag_last",
    # spatial & temporal context
    "lat_bin", "lon_bin",
    "month", "hour_of_day", "day_of_week", "year",
]

TARGET_TIME = "days_to_next"
TARGET_MAG = "mag_next"


def _assign_grid(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["lat_bin"] = (df["latitude"] // GRID_DEG * GRID_DEG).astype(float)
    df["lon_bin"] = (df["longitude"] // GRID_DEG * GRID_DEG).astype(float)
    df["cell_id"] = df["lat_bin"].astype(str) + "|" + df["lon_bin"].astype(str)
    return df


def _cell_features(g: pd.DataFrame) -> pd.DataFrame:
    """Vectorised rolling-window features for one grid cell."""
    g = g.sort_values("time").copy()
    g = g.set_index("time")

    # pre-compute seismic moment per event (used in rolling sum)
    g["_sm"] = 10.0 ** (1.5 * g["magnitude"])

    for window, label in [("7D", "7d"), ("30D", "30d"),
                          ("90D", "90d"), ("365D", "365d")]:
        # closed='left' → [t-w, t): past events only, excludes current
        kwargs = dict(window=window, min_periods=0, closed="left")
        mag_r = g["magnitude"].rolling(**kwargs)
        dep_r = g["depth"].rolling(**kwargs)
        sm_r  = g["_sm"].rolling(**kwargs)

        g[f"count_{label}"]          = mag_r.count().astype(float)
        g[f"mean_mag_{label}"]       = mag_r.mean().fillna(0)
        g[f"max_mag_{label}"]        = mag_r.max().fillna(0)
        g[f"min_mag_{label}"]        = mag_r.min().fillna(0)
        g[f"std_mag_{label}"]        = mag_r.std().fillna(0)
        g[f"mean_depth_{label}"]     = dep_r.mean().fillna(0)
        # seismic moment sum captures energy release better than raw count
        g[f"seismic_moment_{label}"] = sm_r.sum().fillna(0)

    # Gutenberg-Richter b-value (MLE): b = log10(e) / (mean_M - min_M + 0.05)
    # Valid when ≥5 events; default 1.0 otherwise (global average)
    b_denom = (g["mean_mag_365d"] - g["min_mag_365d"] + 0.05).clip(lower=0.01)
    g["b_value_365d"] = np.where(
        g["count_365d"] >= 5,
        np.log10(np.e) / b_denom,
        1.0,
    )

    # inter-event times (position-based rolling on last 10 events)
    idx_series = g.index.to_series()
    iet = idx_series.diff().dt.total_seconds().div(86400)
    g["days_since_last"] = iet.fillna(999.0)

    iet_roll = iet.rolling(10, min_periods=2)
    iet_mean = iet_roll.mean().fillna(999.0)
    iet_std  = iet_roll.std().fillna(0.0)
    g["inter_event_mean_10"] = iet_mean
    g["inter_event_cv_10"]   = (iet_std / iet_mean.clip(lower=0.01)).fillna(0.0)

    g["mag_last"]    = g["magnitude"].shift(1).fillna(0.0)
    g["days_to_next"] = -idx_series.diff(-1).dt.total_seconds().div(86400)
    g["mag_next"]    = g["magnitude"].shift(-1)

    return g.drop(columns=["_sm"]).reset_index()


def build_features(parquet_path: Path) -> pd.DataFrame:
    """
    Load the processed earthquake parquet, engineer features, and return a
    DataFrame with one row per event that has a valid successor in its cell.
    """
    log.info("Loading parquet: %s", parquet_path)
    df = pd.read_parquet(parquet_path)

    # Ensure tz-aware datetime
    if df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize("UTC")

    # Filter to reliable-coverage era and earthquakes only
    df = df[df["time"] >= TRAIN_START].copy()
    df = df[df["magnitude"].notna() & df["depth"].notna()].copy()
    log.info("Events after 1973 filter: %d", len(df))

    df = _assign_grid(df)

    # Process each cell independently
    cells = df["cell_id"].unique()
    log.info("Processing %d grid cells …", len(cells))

    frames = []
    for cell in tqdm(cells, desc="Feature engineering", unit="cell"):
        g = df[df["cell_id"] == cell][[
            "time", "magnitude", "depth", "lat_bin", "lon_bin",
            "month", "hour_of_day", "day_of_week", "year",
        ]].copy()

        if len(g) < MIN_CELL_EVENTS:
            continue

        feat = _cell_features(g)
        feat["cell_id"] = cell
        frames.append(feat)

    if not frames:
        raise ValueError("No cells survived the minimum-events filter.")

    out = pd.concat(frames, ignore_index=True)

    # Drop rows where the label is unavailable (last event in cell, or > max horizon)
    out = out.dropna(subset=[TARGET_TIME, TARGET_MAG])
    out = out[out[TARGET_TIME] > 0]
    out = out[out[TARGET_TIME] <= MAX_DAYS_LABEL]

    log.info("Training rows after label filtering: %d", len(out))
    return out


def get_latest_cell_features(parquet_path: Path,
                              as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    For inference: compute features for the *most recent event* in every
    cell as of `as_of` (defaults to now).  Returns one row per active cell.
    """
    if as_of is None:
        as_of = pd.Timestamp.now(tz="UTC")

    df = pd.read_parquet(parquet_path)
    if df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize("UTC")

    # Only look at the window of history needed for features
    lookback_start = as_of - pd.Timedelta(days=366)
    df = df[(df["time"] >= lookback_start) & (df["time"] <= as_of)].copy()
    df = df[df["magnitude"].notna() & df["depth"].notna()].copy()
    df = _assign_grid(df)

    frames = []
    for cell, g in df.groupby("cell_id"):
        g = g[["time", "magnitude", "depth", "lat_bin", "lon_bin",
               "month", "hour_of_day", "day_of_week", "year"]].copy()
        if len(g) < 1:
            continue
        feat = _cell_features(g)
        # Keep only the last event (most recent state of the cell)
        last = feat.sort_values("time").tail(1).copy()
        last["cell_id"] = cell
        frames.append(last)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)
