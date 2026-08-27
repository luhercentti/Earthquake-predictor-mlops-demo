"""
Core inference logic for the serving layer.
Loads trained LightGBM models once at startup and exposes a single
`run_forecast()` function that the FastAPI routes call.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.features import FEATURE_COLS, get_latest_cell_features

log = logging.getLogger(__name__)

MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_PARQUET = PROJECT_ROOT / "DATA" / "data" / "processed" / "earthquakes_full.parquet"

# ── Country / city reference data ────────────────────────────────────────────
# (min_lat, max_lat, min_lon, max_lon)
COUNTRY_BBOX: dict[str, tuple[float, float, float, float]] = {
    # Americas
    "peru":              (-18.4,  -0.0,  -81.4, -68.7),
    "chile":             (-55.9, -17.5,  -75.7, -66.5),
    "ecuador":           ( -5.0,   1.5,  -81.0, -75.2),
    "colombia":          ( -4.2,  12.5,  -79.0, -66.9),
    "bolivia":           (-22.9,  -9.7,  -69.6, -57.5),
    "brazil":            (-33.7,   5.3,  -73.9, -34.8),
    "argentina":         (-55.1, -21.8,  -73.6, -53.6),
    "venezuela":         (  1.0,  12.2,  -73.3, -59.8),
    "paraguay":          (-27.6, -19.3,  -62.6, -54.3),
    "uruguay":           (-34.9, -30.1,  -58.4, -53.2),
    "guyana":            (  1.2,   8.6,  -61.4, -56.5),
    "suriname":          (  1.8,   6.0,  -58.1, -53.9),
    "mexico":            ( 14.5,  32.7, -118.4, -86.7),
    "usa":               ( 24.4,  49.4, -124.8, -66.9),
    "alaska":            ( 51.2,  71.5, -179.9, -129.9),
    "california":        ( 32.5,  42.0, -124.5, -114.1),
    "utah":              ( 37.0,  42.0, -114.1, -109.0),
    "nevada":            ( 35.0,  42.0, -120.0, -114.0),
    "oregon":            ( 42.0,  46.2, -124.6, -116.5),
    "washington state":  ( 45.5,  49.0, -124.7, -116.9),
    "canada":            ( 41.7,  83.1, -141.0,  -52.6),
    "guatemala":         ( 13.7,  17.8,  -92.2,  -88.2),
    "el salvador":       ( 13.1,  14.4,  -90.1,  -87.7),
    "honduras":          ( 13.0,  16.5,  -89.4,  -83.2),
    "nicaragua":         ( 10.7,  15.0,  -87.7,  -83.1),
    "costa rica":        (  8.0,  11.2,  -85.9,  -82.6),
    "panama":            (  7.2,   9.6,  -83.0,  -77.2),
    "haiti":             ( 18.0,  20.1,  -74.5,  -71.6),
    "cuba":              ( 19.8,  23.3,  -84.9,  -74.1),
    "dominican republic":(17.5,  20.0,  -72.0,  -68.3),
    "puerto rico":       ( 17.9,  18.5,  -67.9,  -65.6),
    # Europe
    "italy":             ( 36.6,  47.1,    6.6,   18.5),
    "greece":            ( 34.8,  41.7,   19.4,   29.6),
    "turkey":            ( 36.0,  42.1,   26.0,   44.8),
    "portugal":          ( 36.9,  42.1,   -9.5,   -6.2),
    "spain":             ( 36.0,  43.8,   -9.3,    4.3),
    "romania":           ( 43.6,  48.3,   20.3,   29.7),
    "serbia":            ( 42.2,  46.2,   18.8,   23.0),
    "albania":           ( 39.6,  42.7,   19.3,   21.1),
    "bulgaria":          ( 41.2,  44.2,   22.4,   28.6),
    "austria":           ( 46.4,  49.0,    9.5,   17.2),
    "switzerland":       ( 45.8,  47.8,    5.9,   10.5),
    "france":            ( 41.3,  51.1,   -5.1,    9.6),
    "iceland":           ( 63.3,  66.6,  -24.5,  -13.5),
    "cyprus":            ( 34.6,  35.7,   32.3,   34.6),
    # Middle East & Central Asia
    "iran":              ( 25.1,  39.8,   44.0,   63.3),
    "iraq":              ( 29.1,  37.4,   38.8,   48.6),
    "saudi arabia":      ( 16.4,  32.2,   34.6,   55.7),
    "yemen":             ( 12.1,  19.0,   42.6,   54.0),
    "afghanistan":       ( 29.4,  38.5,   60.5,   74.9),
    "uzbekistan":        ( 37.2,  45.6,   56.0,   73.1),
    "tajikistan":        ( 36.7,  41.0,   67.4,   75.2),
    "kyrgyzstan":        ( 39.2,  43.3,   69.3,   80.3),
    "kazakhstan":        ( 40.6,  55.4,   50.3,   87.4),
    "turkmenistan":      ( 35.1,  42.8,   52.5,   66.7),
    "azerbaijan":        ( 38.4,  41.9,   44.8,   50.4),
    "armenia":           ( 38.8,  41.3,   43.4,   46.6),
    "georgia":           ( 41.0,  43.6,   40.0,   46.7),
    "israel":            ( 29.5,  33.3,   34.3,   35.9),
    "jordan":            ( 29.2,  33.4,   34.9,   39.3),
    "syria":             ( 32.3,  37.3,   35.7,   42.4),
    "lebanon":           ( 33.0,  34.7,   35.1,   36.6),
    # Asia-Pacific
    "japan":             ( 24.0,  45.5,  122.9,  145.8),
    "indonesia":         (-11.0,   6.1,   95.0,  141.0),
    "philippines":       (  4.6,  20.2,  116.9,  126.6),
    "china":             ( 18.2,  53.6,   73.5,  135.1),
    "india":             (  6.7,  35.5,   68.1,   97.4),
    "nepal":             ( 26.3,  30.5,   80.0,   88.2),
    "pakistan":          ( 23.6,  37.1,   60.9,   77.3),
    "myanmar":           ( 10.0,  28.5,   92.2,  101.2),
    "thailand":          (  5.6,  20.5,   97.4,  105.7),
    "vietnam":           (  8.4,  23.4,  102.1,  109.5),
    "taiwan":            ( 21.9,  25.3,  120.0,  122.0),
    "south korea":       ( 34.0,  38.6,  125.1,  129.6),
    "north korea":       ( 37.7,  42.7,  124.2,  130.7),
    "malaysia":          (  0.9,   7.4,   99.6,  119.3),
    "papua new guinea":  ( -11.7,  -1.3,  141.0,  155.9),
    "solomon islands":   ( -11.9,  -5.0,  155.5,  162.7),
    "vanuatu":           (-20.3, -13.1,  166.5,  170.2),
    "fiji":              (-20.7, -15.7,  177.3,  -179.9),
    "tonga":             (-23.5, -15.5, -177.2,  -173.7),
    "new zealand":       (-47.0, -34.4,  166.5,  178.5),
    "australia":         (-43.6,  -9.2,  113.2,  153.6),
    # Africa
    "morocco":           ( 27.7,  35.9,   -5.9,   -1.0),
    "algeria":           ( 19.0,  37.1,   -8.7,    9.0),
    "ethiopia":          (  3.4,  14.9,   33.0,   47.9),
    "kenya":             ( -4.7,   5.0,   33.9,   41.9),
    "tanzania":          (-11.7,  -1.0,   29.3,   40.4),
    "mozambique":        (-26.9,  -10.5,   32.3,   40.8),
    "malawi":            (-17.1,  -9.4,   32.7,   35.9),
    "congo":             (-13.5,   5.4,   11.8,   31.3),
    "cameroon":          (  1.7,  13.1,    8.5,   16.2),
    "djibouti":          ( 10.9,  12.7,   41.8,   43.5),
    "somalia":           ( -1.7,  12.0,   40.9,   51.4),
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
    """Singleton that loads both models once and exposes them for inference."""

    _instance: ModelRegistry | None = None

    def __init__(self) -> None:
        self.time_model = None
        self.mag_model = None
        self.metadata: dict = {}
        self._loaded = False

    @classmethod
    def get(cls) -> ModelRegistry:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self) -> None:
        time_path = MODEL_DIR / "time_model.lgbm"
        mag_path = MODEL_DIR / "mag_model.lgbm"
        meta_path = MODEL_DIR / "metadata.json"

        if not time_path.exists() or not mag_path.exists():
            raise FileNotFoundError(
                f"Models not found in {MODEL_DIR}. Run training/train_model.py first."
            )

        log.info("Loading models from %s …", MODEL_DIR)
        self.time_model = joblib.load(time_path)
        self.mag_model = joblib.load(mag_path)
        self.metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        self._loaded = True
        log.info("Models loaded successfully.")

    @property
    def loaded(self) -> bool:
        return self._loaded


def _lookup_place_in_catalog(name: str, parquet_path: Path) -> tuple[float, float] | None:
    """Search the USGS place column for a city/state/region name and return its centroid."""
    if not parquet_path.exists():
        return None
    try:
        ref = pd.read_parquet(parquet_path, columns=["latitude", "longitude", "place"])
        ref = ref.dropna(subset=["place"])
        matches = ref[ref["place"].str.contains(name, case=False, na=False)]
        if matches.empty:
            return None
        return float(matches["latitude"].mean()), float(matches["longitude"].mean())
    except Exception:
        return None


def _resolve_region(
    country: str | None,
    city: str | None,
    lat: float | None,
    lon: float | None,
    radius_km: float,
    parquet_path: Path | None = None,
) -> tuple[str, tuple[float, float], tuple[float, float]]:
    """Return (region_name, lat_range, lon_range)."""
    if country:
        key = country.lower().strip()
        if key not in COUNTRY_BBOX:
            matches = [k for k in COUNTRY_BBOX if key in k or k in key]
            if not matches:
                raise ValueError(
                    f"Unknown country '{country}'. "
                    f"Available: {', '.join(sorted(COUNTRY_BBOX))}"
                )
            key = matches[0]
        bb = COUNTRY_BBOX[key]
        return key, (bb[0], bb[1]), (bb[2], bb[3])

    if city:
        key = city.lower().strip()
        # 1. exact match in hardcoded list
        if key in CITY_COORDS:
            clat, clon = CITY_COORDS[key]
        else:
            # 2. partial match in hardcoded list
            partial = [k for k in CITY_COORDS if key in k or k in key]
            if partial:
                clat, clon = CITY_COORDS[partial[0]]
                key = partial[0]
            else:
                # 3. dynamic lookup in the earthquake catalog place column
                # use 400km radius for catalog-derived locations (less precise centroid)
                coords = _lookup_place_in_catalog(city, parquet_path) if parquet_path else None
                if coords:
                    clat, clon = coords
                    key = city
                    radius_km = max(radius_km, 400.0)
                else:
                    raise ValueError(
                        f"Unknown city '{city}'. "
                        f"Try a country name instead, or use --lat/--lon. "
                        f"Hardcoded cities: {', '.join(sorted(CITY_COORDS))}"
                    )
        deg = radius_km / 111.0
        return f"{key} (±{radius_km:.0f} km)", (clat - deg, clat + deg), (clon - deg, clon + deg)

    if lat is not None and lon is not None:
        deg = radius_km / 111.0
        return (
            f"({lat:.2f}°, {lon:.2f}°) ±{radius_km:.0f} km",
            (lat - deg, lat + deg),
            (lon - deg, lon + deg),
        )

    raise ValueError("Provide country, city, or lat+lon.")


def _nearest_place(lat: float, lon: float, place_df: pd.DataFrame | None) -> str:
    if place_df is None or place_df.empty:
        lat_s = f"{abs(lat):.1f}°{'S' if lat < 0 else 'N'}"
        lon_s = f"{abs(lon):.1f}°{'W' if lon < 0 else 'E'}"
        return f"{lat_s} {lon_s}"
    dists = (place_df["latitude"] - lat) ** 2 + (place_df["longitude"] - lon) ** 2
    place = str(place_df.loc[dists.idxmin(), "place"])
    return place if place and place != "nan" else f"{lat:.2f}, {lon:.2f}"


def _build_summary(forecasts: list[dict], region: str, mae: float | None) -> str:
    if not forecasts:
        return f"No active seismic cells found in {region}."
    top = forecasts[0]
    days = top["estimated_days_to_next"]
    mag = top["estimated_magnitude"]
    place = top["nearest_known_place"]
    accuracy = f" (model accuracy: ±{mae:.0f} days)" if mae else ""
    return (
        f"The most likely next earthquake in {region.title()} is expected in "
        f"approximately {days:.0f} day{'s' if days != 1 else ''}, "
        f"near {place}, with an estimated magnitude of M {mag:.1f}.{accuracy}"
    )


def run_forecast(
    country: str | None = None,
    city: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
    radius_km: float = 200.0,
    min_mag: float | None = None,
    top_n: int = 10,
    parquet_path: Path = DEFAULT_PARQUET,
) -> dict:
    """
    Run the earthquake forecast and return a structured dict matching
    the ForecastResponse schema.
    """
    registry = ModelRegistry.get()
    if not registry.loaded:
        registry.load()

    region_name, lat_range, lon_range = _resolve_region(
        country, city, lat, lon, radius_km, parquet_path
    )

    log.info("Computing cell features for region '%s' …", region_name)
    cell_df = get_latest_cell_features(parquet_path)

    if cell_df.empty:
        return {
            "region": region_name,
            "as_of_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "total_active_cells": 0,
            "summary": f"No seismic activity found near {region_name} in the past year.",
            "forecasts": [],
            "model_test_mae_days": None,
        }

    # Filter to region
    region_cells = cell_df[
        cell_df["lat_bin"].between(*lat_range) &
        cell_df["lon_bin"].between(*lon_range)
    ].copy()

    if min_mag is not None:
        region_cells = region_cells[region_cells["max_mag_365d"] >= min_mag]

    if region_cells.empty:
        return {
            "region": region_name,
            "as_of_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "total_active_cells": 0,
            "summary": f"No seismic activity found near {region_name} matching the filters. Try removing min_mag or increasing the radius.",
            "forecasts": [],
            "model_test_mae_days": None,
        }

    X = region_cells[FEATURE_COLS].values
    region_cells = region_cells.copy()
    region_cells["est_days"] = np.clip(registry.time_model.predict(X), 0.1, None)
    region_cells["est_mag"] = np.clip(registry.mag_model.predict(X), 1.0, 9.9)

    top = region_cells.sort_values("est_days").head(top_n)

    # Place name lookup from a sample of the catalog
    place_df = None
    if parquet_path.exists():
        try:
            ref = pd.read_parquet(parquet_path, columns=["latitude", "longitude", "place"])
            ref = ref.dropna(subset=["place"])
            place_df = ref.sample(min(50_000, len(ref)), random_state=42)
        except Exception:
            pass

    forecasts = []
    for rank, (_, row) in enumerate(top.iterrows(), 1):
        mw  = round(float(row["est_mag"]), 2)
        forecasts.append({
            "rank": rank,
            "cell_lat": float(row["lat_bin"]),
            "cell_lon": float(row["lon_bin"]),
            "estimated_days_to_next": round(float(row["est_days"]), 2),
            "estimated_magnitude": mw,
            "events_last_365d": int(row.get("count_365d", 0)),
            "max_magnitude_last_365d": round(float(row.get("max_mag_365d", 0)), 2),
            "nearest_known_place": _nearest_place(
                row["lat_bin"] + 1.0, row["lon_bin"] + 1.0, place_df
            ),
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
