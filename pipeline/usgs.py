"""
USGS ComCat (FDSN Event Service) downloader.

Strategy:
  - Pull TWO overlapping layers to maximise event count:
      Layer 1: Global M>=4.5, 1900-01-01 → today
      Layer 2: South America bounding box M>=2.5, 1960-01-01 → today
  - USGS caps each response at 20 000 events, so we iterate in
    90-day windows (safe even for the densest modern periods).
  - Completed windows are checkpointed as individual CSV files in
    data/raw/usgs/ so a crashed run can resume without re-downloading.
"""

import time
import logging
from datetime import date, timedelta
from pathlib import Path

import requests
import pandas as pd
from tqdm import tqdm

log = logging.getLogger(__name__)

USGS_API = "https://earthquake.usgs.gov/fdsnws/event/1/query"

# South America bounding box (generous – includes Ecuador, Colombia, Venezuela,
# Peru, Bolivia, Brazil, Chile, Argentina, Uruguay, Paraguay, Guyana, Suriname)
SA_BBOX = dict(minlatitude=-56.0, maxlatitude=13.0,
               minlongitude=-82.0, maxlongitude=-34.0)

LAYERS = [
    {
        "name": "global_m45",
        "params": {"minmagnitude": 4.5},
        "start": date(1900, 1, 1),
    },
    {
        "name": "south_america_m25",
        "params": {"minmagnitude": 2.5, **SA_BBOX},
        "start": date(1960, 1, 1),
    },
]

WINDOW_DAYS = 90          # days per USGS request
REQUEST_DELAY = 1.5       # seconds between requests (be a polite client)
MAX_RETRIES = 5
RETRY_BACKOFF = 10        # seconds for first retry (doubles each attempt)


def _date_windows(start: date, end: date, days: int):
    """Yield (window_start, window_end) pairs of at most `days` each."""
    cur = start
    while cur < end:
        yield cur, min(cur + timedelta(days=days - 1), end)
        cur += timedelta(days=days)


def _fetch_window(params: dict, retries: int = MAX_RETRIES) -> pd.DataFrame | None:
    """Fetch one time window from USGS, return DataFrame or None on failure."""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(USGS_API, params=params, timeout=60)
            if resp.status_code == 200:
                from io import StringIO
                df = pd.read_csv(StringIO(resp.text))
                return df
            if resp.status_code == 204:  # No content – empty window
                return pd.DataFrame()
            if resp.status_code == 429:
                wait = RETRY_BACKOFF * attempt
                log.warning("Rate-limited. Sleeping %ds …", wait)
                time.sleep(wait)
                continue
            log.warning("HTTP %s for %s (attempt %d)", resp.status_code, params, attempt)
        except requests.RequestException as exc:
            log.warning("Request error (attempt %d): %s", attempt, exc)
        time.sleep(RETRY_BACKOFF * attempt)
    return None


def download_usgs(raw_dir: Path, end_date: date | None = None) -> list[Path]:
    """
    Download all USGS layers into per-window CSV files under raw_dir.
    Returns list of paths to all raw CSVs (new + previously cached).
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    today = end_date or date.today()
    all_paths: list[Path] = []

    for layer in LAYERS:
        layer_dir = raw_dir / layer["name"]
        layer_dir.mkdir(exist_ok=True)

        windows = list(_date_windows(layer["start"], today, WINDOW_DAYS))
        log.info("Layer '%s': %d windows to fetch", layer["name"], len(windows))

        for w_start, w_end in tqdm(windows, desc=layer["name"], unit="window"):
            fname = layer_dir / f"{w_start}_{w_end}.csv"
            all_paths.append(fname)

            if fname.exists():
                continue  # already cached

            params = {
                "format": "csv",
                "starttime": w_start.isoformat(),
                "endtime": w_end.isoformat(),
                "orderby": "time-asc",
                **layer["params"],
            }
            df = _fetch_window(params)
            if df is None:
                log.error("Failed to fetch window %s – %s; skipping.", w_start, w_end)
                continue

            df.to_csv(fname, index=False)
            time.sleep(REQUEST_DELAY)

    return all_paths


def load_usgs(raw_dir: Path) -> pd.DataFrame:
    """Concatenate all cached USGS window CSVs into a single DataFrame."""
    frames = []
    for layer in LAYERS:
        layer_dir = raw_dir / layer["name"]
        if not layer_dir.exists():
            continue
        for p in sorted(layer_dir.glob("*.csv")):
            try:
                df = pd.read_csv(p, low_memory=False)
                df["usgs_layer"] = layer["name"]
                frames.append(df)
            except Exception as exc:
                log.warning("Could not read %s: %s", p, exc)

    if not frames:
        raise FileNotFoundError(f"No USGS raw files found under {raw_dir}")

    combined = pd.concat(frames, ignore_index=True)
    log.info("Loaded %d raw USGS rows before dedup", len(combined))
    return combined
