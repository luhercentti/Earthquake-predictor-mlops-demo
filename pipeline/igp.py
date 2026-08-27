"""
IGP (Instituto Geofísico del Perú) seismic catalog downloader.

The IGP catalog is published on Peru's open-data portal (CKAN).  We discover
the resource download URL via the CKAN JSON API so the script doesn't break
if the platform re-publishes the file at a new URL.

If the automatic download fails for any reason (network, site changes, etc.)
the module prints clear manual-download instructions and returns None, so the
rest of the pipeline can continue with USGS data only.

Known column layout for the 1960-2023 catalog CSV (semicolon-separated):
  FECHA_UTC  HORA_UTC  LATITUD  LONGITUD  PROFUNDIDAD  MAGNITUD  TIPO_MAGNITUD
  (date)     (time)    (float)  (float)   (km float)   (float)   (str)
"""

import logging
from pathlib import Path

import requests
import pandas as pd

log = logging.getLogger(__name__)

# CKAN open-data portal for Peru
CKAN_API = "https://www.datosabiertos.gob.pe/api/3/action/package_show"
# Dataset slug as it appears in the portal URL
IGP_DATASET_SLUG = "catalogo-sismico-1960-2023-instituto-geofisico-del-peru-igp"

# Fallback direct URL (known as of 2024 – may drift)
IGP_FALLBACK_URL = (
    "https://www.datosabiertos.gob.pe/sites/default/files/recurso/"
    "Catalogo1960_2023.csv"
)

MANUAL_INSTRUCTIONS = """
╔══════════════════════════════════════════════════════════════════╗
║  IGP MANUAL DOWNLOAD INSTRUCTIONS                               ║
║                                                                  ║
║  1. Open: https://www.datosabiertos.gob.pe/dataset/             ║
║           catalogo-sismico-1960-2023-instituto-geofisico-       ║
║           del-peru-igp                                          ║
║  2. Click the CSV resource link and download the file.          ║
║  3. Save it as:  data/raw/igp/igp_catalog.csv                   ║
║  4. Re-run the pipeline – it will pick up the file automatically.║
╚══════════════════════════════════════════════════════════════════╝
"""


def _discover_csv_url() -> str | None:
    """Try the CKAN API to find the CSV resource download URL."""
    try:
        resp = requests.get(CKAN_API, params={"id": IGP_DATASET_SLUG}, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        resources = data.get("result", {}).get("resources", [])
        for res in resources:
            fmt = (res.get("format") or "").upper()
            url = res.get("url", "")
            if fmt == "CSV" or url.lower().endswith(".csv"):
                return url
    except Exception as exc:
        log.debug("CKAN API discovery failed: %s", exc)
    return None


def download_igp(raw_dir: Path) -> Path | None:
    """
    Download the IGP Peru seismic catalog CSV to raw_dir/igp_catalog.csv.
    Returns the path on success, None if download fails (pipeline continues).
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / "igp_catalog.csv"

    if dest.exists():
        log.info("IGP catalog already cached at %s", dest)
        return dest

    # 1. Try CKAN API discovery
    url = _discover_csv_url()
    if url:
        log.info("IGP CSV URL discovered via CKAN: %s", url)
    else:
        log.warning("CKAN discovery failed; trying fallback URL.")
        url = IGP_FALLBACK_URL

    try:
        log.info("Downloading IGP catalog from %s …", url)
        with requests.get(url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            dest.write_bytes(resp.content)
        log.info("IGP catalog saved to %s (%s bytes)", dest, dest.stat().st_size)
        return dest
    except Exception as exc:
        log.warning("IGP auto-download failed: %s", exc)
        print(MANUAL_INSTRUCTIONS)
        return None


# ---------------------------------------------------------------------------
# Column normalisation
# ---------------------------------------------------------------------------

# Maps possible IGP column names → canonical names used in the merged dataset
_IGP_COL_MAP = {
    # date / time
    "FECHA_UTC": "date_str",
    "HORA_UTC": "time_str",
    "FECHA": "date_str",
    "HORA": "time_str",
    # coordinates
    "LATITUD": "latitude",
    "LONGITUD": "longitude",
    "LAT": "latitude",
    "LON": "longitude",
    # depth
    "PROFUNDIDAD": "depth",
    "PROF": "depth",
    # magnitude
    "MAGNITUD": "magnitude",
    "MAG": "magnitude",
    # magnitude type
    "TIPO_MAGNITUD": "magType",
    "TIPO_MAG": "magType",
}


def load_igp(raw_dir: Path) -> pd.DataFrame | None:
    """
    Load and normalise the IGP CSV.  Returns None if the file is absent.
    """
    path = raw_dir / "igp_catalog.csv"
    if not path.exists():
        log.warning("IGP file not found at %s; skipping IGP layer.", path)
        return None

    # Try semicolon first (common for Peruvian open-data CSVs), then comma
    for sep in (";", ","):
        try:
            df = pd.read_csv(path, sep=sep, low_memory=False)
            if df.shape[1] > 2:
                break
        except Exception:
            continue
    else:
        log.error("Could not parse IGP CSV with ; or , separators.")
        return None

    # Normalise column names (upper-case strip)
    df.columns = [c.strip().upper() for c in df.columns]
    df.rename(columns={k.upper(): v for k, v in _IGP_COL_MAP.items()}, inplace=True)

    # Build UTC datetime
    if "date_str" in df.columns and "time_str" in df.columns:
        df["time"] = pd.to_datetime(
            df["date_str"].astype(str) + " " + df["time_str"].astype(str),
            errors="coerce",
            utc=True,
        )
    elif "date_str" in df.columns:
        df["time"] = pd.to_datetime(df["date_str"], errors="coerce", utc=True)

    df["source"] = "IGP"
    log.info("Loaded %d IGP rows", len(df))
    return df
