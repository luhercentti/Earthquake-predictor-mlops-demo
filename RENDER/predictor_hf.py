"""
HuggingFace-aware predictor — used only in the Render deployment.
Swapped in by the Dockerfile; the local serving/predictor.py is untouched.

Differences from serving/predictor.py:
  - ModelRegistry.load() downloads models from HF if not present locally.
  - run_forecast() uses models/cell_snapshot.parquet (uploaded to HF)
    instead of the 200 MB earthquake parquet that isn't in the container.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.features import FEATURE_COLS, get_latest_cell_features   # noqa: F401 (imported for re-use)

log = logging.getLogger(__name__)

HF_REPO_ID = os.getenv("HF_REPO_ID", "")
MODEL_DIR   = PROJECT_ROOT / "models"
# Cell snapshot is uploaded to HF alongside the model files
CELL_SNAPSHOT = MODEL_DIR / "cell_snapshot.parquet"

# ── Country / city reference data ────────────────────────────────────────────
COUNTRY_BBOX: dict[str, tuple[float, float, float, float]] = {
    "peru":          (-18.4,  -0.0,  -81.4, -68.7),
    "chile":         (-55.9, -17.5,  -75.7, -66.5),
    "ecuador":       ( -5.0,   1.5,  -81.0, -75.2),
    "colombia":      ( -4.2,  12.5,  -79.0, -66.9),
    "bolivia":       (-22.9,  -9.7,  -69.6, -57.5),
    "brazil":        (-33.7,   5.3,  -73.9, -34.8),
    "argentina":     (-55.1, -21.8,  -73.6, -53.6),
    "venezuela":     (  1.0,  12.2,  -73.3, -59.8),
    "mexico":        ( 14.5,  32.7, -118.4, -86.7),
    "usa":           ( 24.4,  49.4, -124.8, -66.9),
    "alaska":        ( 51.2,  71.5, -179.9, -129.9),
    "california":    ( 32.5,  42.0, -124.5, -114.1),
    "japan":         ( 24.0,  45.5,  122.9,  145.8),
    "indonesia":     (-11.0,   6.1,   95.0,  141.0),
    "turkey":        ( 36.0,  42.1,   26.0,   44.8),
    "iran":          ( 25.1,  39.8,   44.0,   63.3),
    "italy":         ( 36.6,  47.1,    6.6,   18.5),
    "greece":        ( 34.8,  41.7,   19.4,   29.6),
    "nepal":         ( 26.3,  30.5,   80.0,   88.2),
    "new zealand":   (-47.0, -34.4,  166.5,  178.5),
    "philippines":   (  4.6,  20.2,  116.9,  126.6),
    "india":         (  6.7,  35.5,   68.1,   97.4),
    "pakistan":      ( 23.6,  37.1,   60.9,   77.3),
    "china":         ( 18.2,  53.6,   73.5,  135.1),
}

CITY_COORDS: dict[str, tuple[float, float]] = {
    "lima":          (-12.05, -77.05),
    "cusco":         (-13.52, -71.97),
    "arequipa":      (-16.41, -71.54),
    "trujillo":      ( -8.11, -79.03),
    "santiago":      (-33.45, -70.67),
    "valparaiso":    (-33.04, -71.63),
    "quito":         ( -0.23, -78.52),
    "bogota":        (  4.71, -74.07),
    "buenos aires":  (-34.60, -58.38),
    "mexico city":   ( 19.43, -99.13),
    "los angeles":   ( 34.05,-118.24),
    "san francisco": ( 37.77,-122.42),
    "tokyo":         ( 35.68, 139.69),
    "jakarta":       ( -6.21, 106.84),
    "istanbul":      ( 41.01,  28.97),
    "kathmandu":     ( 27.72,  85.32),
    "manila":        ( 14.60, 120.98),
    "christchurch":  (-43.53, 172.64),
    "tehran":        ( 35.69,  51.39),
}


class ModelRegistry:
    _instance: ModelRegistry | None = None

    def __init__(self) -> None:
        self.time_model = None
        self.mag_model  = None
        self.metadata: dict = {}
        self._loaded = False

    @classmethod
    def get(cls) -> ModelRegistry:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _download_from_hf(self) -> None:
        if not HF_REPO_ID:
            raise FileNotFoundError(
                f"Models not found in {MODEL_DIR} and HF_REPO_ID env var is not set.\n"
                "Set HF_REPO_ID=your-username/your-repo in the Render environment."
            )
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ImportError("huggingface_hub is not installed in this environment.")

        log.info("Downloading models from HuggingFace: %s …", HF_REPO_ID)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="model",
            local_dir=str(MODEL_DIR),
            token=os.getenv("HF_TOKEN"),
        )
        log.info("Models downloaded successfully from HuggingFace.")

    def load(self) -> None:
        time_path = MODEL_DIR / "time_model.lgbm"
        mag_path  = MODEL_DIR / "mag_model.lgbm"
        meta_path = MODEL_DIR / "metadata.json"

        if not time_path.exists() or not mag_path.exists():
            self._download_from_hf()

        log.info("Loading models from %s …", MODEL_DIR)
        self.time_model = joblib.load(time_path)
        self.mag_model  = joblib.load(mag_path)
        self.metadata   = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        self._loaded = True
        log.info("Models loaded successfully.")

    @property
    def loaded(self) -> bool:
        return self._loaded


def _resolve_region(country, city, lat, lon, radius_km):
    if country:
        key = country.lower().strip()
        if key not in COUNTRY_BBOX:
            matches = [k for k in COUNTRY_BBOX if key in k or k in key]
            if not matches:
                raise ValueError(f"Unknown country '{country}'. Available: {', '.join(sorted(COUNTRY_BBOX))}")
            key = matches[0]
        bb = COUNTRY_BBOX[key]
        return key, (bb[0], bb[1]), (bb[2], bb[3])
    if city:
        key = city.lower().strip()
        if key not in CITY_COORDS:
            matches = [k for k in CITY_COORDS if key in k or k in key]
            if not matches:
                raise ValueError(f"Unknown city '{city}'. Available: {', '.join(sorted(CITY_COORDS))}")
            key = matches[0]
        clat, clon = CITY_COORDS[key]
        deg = radius_km / 111.0
        return f"{key} (±{radius_km:.0f} km)", (clat - deg, clat + deg), (clon - deg, clon + deg)
    if lat is not None and lon is not None:
        deg = radius_km / 111.0
        return (f"({lat:.2f}°, {lon:.2f}°) ±{radius_km:.0f} km",
                (lat - deg, lat + deg), (lon - deg, lon + deg))
    raise ValueError("Provide country, city, or lat+lon.")


def _nearest_place(lat: float, lon: float) -> str:
    # No catalog available in the container — return coordinate string
    lat_s = f"{abs(lat):.1f}°{'S' if lat < 0 else 'N'}"
    lon_s = f"{abs(lon):.1f}°{'W' if lon < 0 else 'E'}"
    return f"{lat_s}  {lon_s}"


def _build_summary(forecasts: list[dict], region: str, mae: float | None) -> str:
    if not forecasts:
        return f"No active seismic cells found in {region}."
    top = forecasts[0]
    days = top["estimated_days_to_next"]
    mag  = top["estimated_magnitude"]
    place = top["nearest_known_place"]
    accuracy = f" (model accuracy: ±{mae:.0f} days)" if mae else ""
    return (
        f"The most likely next earthquake in {region.title()} is expected in "
        f"approximately {days:.0f} day{'s' if days != 1 else ''}, "
        f"near {place}, with an estimated magnitude of M {mag:.1f}.{accuracy}"
    )


def run_forecast(
    country=None, city=None, lat=None, lon=None,
    radius_km=200.0, min_mag=None, top_n=10,
    parquet_path=None,   # ignored in this HF version
) -> dict:
    registry = ModelRegistry.get()
    if not registry.loaded:
        registry.load()

    region_name, lat_range, lon_range = _resolve_region(country, city, lat, lon, radius_km)

    if not CELL_SNAPSHOT.exists():
        raise FileNotFoundError(
            f"Cell snapshot not found at {CELL_SNAPSHOT}. "
            "Re-upload models with RENDER/upload_to_hf.py (it includes the snapshot)."
        )

    log.info("Loading cell snapshot for region '%s' …", region_name)
    cell_df = pd.read_parquet(CELL_SNAPSHOT)

    region_cells = cell_df[
        cell_df["lat_bin"].between(*lat_range) &
        cell_df["lon_bin"].between(*lon_range)
    ].copy()

    if min_mag is not None:
        region_cells = region_cells[region_cells["max_mag_365d"] >= min_mag]

    empty_result = {
        "region": region_name,
        "as_of_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "total_active_cells": 0,
        "summary": f"No active seismic cells found in {region_name}.",
        "forecasts": [],
        "model_test_mae_days": None,
    }

    if region_cells.empty:
        return empty_result

    X = region_cells[FEATURE_COLS].values
    region_cells = region_cells.copy()
    region_cells["est_days"] = np.clip(registry.time_model.predict(X), 0.1, None)
    region_cells["est_mag"]  = np.clip(registry.mag_model.predict(X), 1.0, 9.9)

    top = region_cells.sort_values("est_days").head(top_n)

    forecasts = []
    for rank, (_, row) in enumerate(top.iterrows(), 1):
        mw = round(float(row["est_mag"]), 2)
        forecasts.append({
            "rank": rank,
            "cell_lat": float(row["lat_bin"]),
            "cell_lon": float(row["lon_bin"]),
            "estimated_days_to_next": round(float(row["est_days"]), 2),
            "estimated_magnitude": mw,
            "events_last_365d": int(row.get("count_365d", 0)),
            "max_magnitude_last_365d": round(float(row.get("max_mag_365d", 0)), 2),
            "nearest_known_place": _nearest_place(row["lat_bin"] + 1.0, row["lon_bin"] + 1.0),
        })

    mae = registry.metadata.get("test_metrics", {}).get("days_to_next", {}).get("mae")

    return {
        "region": region_name,
        "as_of_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "total_active_cells": len(region_cells),
        "summary": _build_summary(forecasts, region_name, mae),
        "forecasts": forecasts,
        "model_test_mae_days": round(mae, 3) if mae else None,
    }


# health check helper
DEFAULT_PARQUET = CELL_SNAPSHOT   # used by app.py health endpoint check
