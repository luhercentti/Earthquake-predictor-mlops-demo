"""
Merge, deduplicate, clean, and feature-engineer the combined earthquake dataset.

Deduplication strategy
──────────────────────
Events that appear in both USGS layers, or in both USGS and IGP, are matched
by spatial + temporal proximity:
  • |Δt|  ≤ 60 seconds
  • |Δlat| ≤ 0.1°  (~11 km)
  • |Δlon| ≤ 0.1°
When a duplicate pair is found, we keep the record with the richer source
priority (global_m45 < south_america_m25 < IGP).

ML feature columns added
────────────────────────
  year, month, day_of_year, hour_of_day, day_of_week  (temporal)
  depth_km                                              (renamed depth)
  mag_class   (0=minor <3, 1=light 3-3.9, 2=moderate 4-4.9,
               3=strong 5-5.9, 4=major 6-6.9, 5=great >=7)
  is_south_america, is_peru                             (region flags)
  days_since_epoch                                      (float, for time-series)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Source priority for dedup: lower = gets dropped in favour of higher
SOURCE_PRIORITY = {"global_m45": 0, "south_america_m25": 1, "IGP": 2}

# Thresholds for considering two records the same event
DEDUP_DELTA_SEC = 60
DEDUP_DELTA_DEG = 0.10

# USGS CSV standard column names
USGS_RENAME = {
    "time": "time",
    "latitude": "latitude",
    "longitude": "longitude",
    "depth": "depth",
    "mag": "magnitude",
    "magType": "magType",
    "place": "place",
    "type": "event_type",
    "id": "usgs_id",
    "usgs_layer": "source",
}

FINAL_COLUMNS = [
    "time",
    "latitude",
    "longitude",
    "depth",
    "magnitude",
    "magType",
    "place",
    "event_type",
    "source",
    # derived
    "year",
    "month",
    "day_of_year",
    "hour_of_day",
    "day_of_week",
    "days_since_epoch",
    "mag_class",
    "is_south_america",
    "is_peru",
]

# South America bounding box (same as downloader)
SA_LAT = (-56.0, 13.0)
SA_LON = (-82.0, -34.0)
# Peru bounding box (approximate)
PE_LAT = (-18.4, -0.0)
PE_LON = (-81.4, -68.7)

# Reference epoch for days_since_epoch feature
EPOCH = pd.Timestamp("1900-01-01", tz="UTC")


def _normalise_usgs(df: pd.DataFrame) -> pd.DataFrame:
    """Rename USGS columns and parse the time column."""
    df = df.copy()
    df.rename(columns={k: v for k, v in USGS_RENAME.items() if k in df.columns},
              inplace=True)

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)

    # Keep only earthquake events (not quarry blasts, explosions, etc.)
    if "event_type" in df.columns:
        df = df[df["event_type"].str.lower().isin(["earthquake", "eq", ""])
                | df["event_type"].isna()].copy()

    return df


def _normalise_igp(df: pd.DataFrame) -> pd.DataFrame:
    """Align IGP columns to the shared schema."""
    df = df.copy()
    for col in ("place", "event_type", "usgs_id"):
        if col not in df.columns:
            df[col] = pd.NA
    if "magType" not in df.columns:
        df["magType"] = pd.NA
    return df


def _dedup_fast(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate events using a sort-and-scan approach.

    For each pair of adjacent rows (after sorting by time), if they fall
    within DEDUP_DELTA_SEC and DEDUP_DELTA_DEG, the one with lower source
    priority is dropped.  This is O(n) after the sort.
    """
    df = df.sort_values("time").reset_index(drop=True)
    df["_priority"] = df["source"].map(SOURCE_PRIORITY).fillna(0).astype(int)
    df["_t_sec"] = df["time"].astype("int64") // 1_000_000_000  # ns → s

    keep = np.ones(len(df), dtype=bool)

    t = df["_t_sec"].to_numpy()
    lat = df["latitude"].to_numpy(dtype=float)
    lon = df["longitude"].to_numpy(dtype=float)
    pri = df["_priority"].to_numpy(dtype=int)

    i = 0
    while i < len(df):
        if not keep[i]:
            i += 1
            continue
        j = i + 1
        while j < len(df) and (t[j] - t[i]) <= DEDUP_DELTA_SEC:
            if (keep[j]
                    and abs(lat[j] - lat[i]) <= DEDUP_DELTA_DEG
                    and abs(lon[j] - lon[i]) <= DEDUP_DELTA_DEG):
                # drop the lower-priority one
                if pri[i] >= pri[j]:
                    keep[j] = False
                else:
                    keep[i] = False
                    break  # i is dropped; outer loop will skip
            j += 1
        i += 1

    result = df[keep].drop(columns=["_priority", "_t_sec"])
    log.info("Dedup: %d → %d rows (removed %d duplicates)",
             len(df), len(result), len(df) - len(result))
    return result


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all ML-ready derived columns."""
    t = df["time"]

    df["year"] = t.dt.year
    df["month"] = t.dt.month
    df["day_of_year"] = t.dt.day_of_year
    df["hour_of_day"] = t.dt.hour
    df["day_of_week"] = t.dt.dayofweek  # 0=Monday

    df["days_since_epoch"] = (t - EPOCH).dt.total_seconds() / 86400.0

    mag = pd.to_numeric(df["magnitude"], errors="coerce")
    df["magnitude"] = mag
    df["mag_class"] = pd.cut(
        mag,
        bins=[-np.inf, 3.0, 4.0, 5.0, 6.0, 7.0, np.inf],
        labels=[0, 1, 2, 3, 4, 5],
    ).astype("Int8")

    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")

    df["is_south_america"] = (
        lat.between(*SA_LAT) & lon.between(*SA_LON)
    ).astype("Int8")

    df["is_peru"] = (
        lat.between(*PE_LAT) & lon.between(*PE_LON)
    ).astype("Int8")

    return df


def merge_and_clean(usgs_raw: pd.DataFrame,
                    igp_raw: pd.DataFrame | None) -> pd.DataFrame:
    """
    Merge USGS + IGP frames, deduplicate, clean, and add ML features.
    Returns a clean DataFrame ready for output.
    """
    usgs = _normalise_usgs(usgs_raw)
    frames = [usgs]

    if igp_raw is not None:
        igp = _normalise_igp(igp_raw)
        frames.append(igp)

    combined = pd.concat(frames, ignore_index=True)
    log.info("Combined before dedup: %d rows", len(combined))

    # Drop rows with missing essential fields
    before = len(combined)
    combined = combined.dropna(subset=["time", "latitude", "longitude", "magnitude"])
    log.info("Dropped %d rows with null time/lat/lon/magnitude", before - len(combined))

    combined = _dedup_fast(combined)
    combined = _add_features(combined)

    # Ensure consistent column order; add missing columns as NA
    for col in FINAL_COLUMNS:
        if col not in combined.columns:
            combined[col] = pd.NA

    combined = combined[FINAL_COLUMNS].sort_values("time").reset_index(drop=True)
    log.info("Final dataset: %d rows, %d columns", *combined.shape)
    return combined


def save_outputs(df: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    """Write Parquet (primary) and CSV (human-readable) outputs."""
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = out_dir / "earthquakes_full.parquet"
    csv_path = out_dir / "earthquakes_full.csv"

    df.to_parquet(parquet_path, index=False, compression="snappy")
    df.to_csv(csv_path, index=False)

    pq_mb = parquet_path.stat().st_size / 1_048_576
    csv_mb = csv_path.stat().st_size / 1_048_576

    log.info("Saved Parquet: %s  (%.1f MB)", parquet_path, pq_mb)
    log.info("Saved CSV:     %s  (%.1f MB)", csv_path, csv_mb)

    return {"parquet": parquet_path, "csv": csv_path}
