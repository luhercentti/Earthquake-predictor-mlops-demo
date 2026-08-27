#!/usr/bin/env python3
"""
Earthquake Dataset Builder
==========================
Downloads the full USGS global earthquake catalog + IGP Peru catalog,
merges them, deduplicates, and writes a clean ML-ready dataset.

Usage
-----
    # Full run (downloads everything, ~20-40 min):
    python run_pipeline.py

    # Resume an interrupted run (skips already-downloaded windows):
    python run_pipeline.py

    # Skip the IGP Peru layer (USGS only):
    python run_pipeline.py --no-igp

    # Custom output directory:
    python run_pipeline.py --out-dir /path/to/output

    # Limit USGS to a specific end date (useful for reproducibility):
    python run_pipeline.py --end-date 2024-12-31

Expected output size
--------------------
    Layer                     Approx. rows   Notes
    ─────────────────────────────────────────────────────────────────
    USGS global M≥4.5         ~650 000       1900-present
    USGS South America M≥2.5  ~500 000       1960-present (overlaps)
    IGP Peru                  ~24 000        1960-2023, very dense
    ─────────────────────────────────────────────────────────────────
    After deduplication       ~900 000–1.1M  varies with cutoff dates

    earthquakes_full.parquet  ~40–60 MB  (snappy-compressed)
    earthquakes_full.csv      ~160–220 MB

Download time
─────────────
    USGS has no hard rate limit but asks for polite querying.
    With the 1.5 s inter-request delay this script uses:
      global_m45 layer   : ~500 windows × 1.5 s ≈ 13 min
      south_america_m25  : ~260 windows × 1.5 s ≈  7 min
    Total: roughly 20-30 min on the first run.
    Subsequent runs complete in seconds (all windows are cached).
"""

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

# project root is one level up from ingestion/
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.usgs import download_usgs, load_usgs
from pipeline.igp import download_igp, load_igp
from pipeline.merge import merge_and_clean, save_outputs

# ── logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "pipeline.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

# ── default paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / "DATA" / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "DATA" / "data" / "processed"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build the full earthquake ML dataset from USGS + IGP."
    )
    p.add_argument(
        "--no-igp", action="store_true",
        help="Skip the IGP Peru catalog layer (use USGS only).",
    )
    p.add_argument(
        "--no-download", action="store_true",
        help="Skip downloading; use whatever is already cached in data/raw/.",
    )
    p.add_argument(
        "--end-date", type=date.fromisoformat, default=None,
        metavar="YYYY-MM-DD",
        help="Override the latest date for USGS queries (default: today).",
    )
    p.add_argument(
        "--out-dir", type=Path, default=OUT_DIR,
        help=f"Directory for the processed outputs (default: {OUT_DIR}).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log.info("═" * 60)
    log.info("Earthquake Dataset Builder – starting")
    log.info("Raw cache : %s", RAW_DIR)
    log.info("Output    : %s", args.out_dir)
    log.info("═" * 60)

    # ── 1. Download ───────────────────────────────────────────────────────────
    if not args.no_download:
        log.info("Step 1/3 – Downloading USGS catalog …")
        download_usgs(RAW_DIR / "usgs", end_date=args.end_date)

        if not args.no_igp:
            log.info("Step 1b  – Downloading IGP Peru catalog …")
            download_igp(RAW_DIR / "igp")
    else:
        log.info("Step 1/3 – Skipping download (--no-download set)")

    # ── 2. Load raw data ──────────────────────────────────────────────────────
    log.info("Step 2/3 – Loading raw data into memory …")
    try:
        usgs_df = load_usgs(RAW_DIR / "usgs")
    except FileNotFoundError as exc:
        log.error("No USGS raw files found: %s", exc)
        sys.exit(1)

    igp_df = None
    if not args.no_igp:
        igp_df = load_igp(RAW_DIR / "igp")

    # ── 3. Merge, clean, save ─────────────────────────────────────────────────
    log.info("Step 3/3 – Merging, deduplicating, feature-engineering …")
    final_df = merge_and_clean(usgs_df, igp_df)

    paths = save_outputs(final_df, args.out_dir)

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("═" * 60)
    log.info("DONE — %d earthquake events in final dataset", len(final_df))
    log.info("")
    log.info("Output files:")
    for fmt, path in paths.items():
        mb = path.stat().st_size / 1_048_576
        log.info("  %-10s %s  (%.1f MB)", fmt.upper(), path, mb)
    log.info("")
    log.info("Column overview:")
    for col in final_df.columns:
        non_null = final_df[col].notna().sum()
        log.info("  %-22s  %d non-null (%.1f%%)",
                 col, non_null, 100 * non_null / len(final_df))
    log.info("")
    log.info("Magnitude distribution:")
    mag_labels = {0: "Minor  (<3.0)", 1: "Light  (3–3.9)",
                  2: "Moderate (4–4.9)", 3: "Strong (5–5.9)",
                  4: "Major  (6–6.9)", 5: "Great  (≥7.0)"}
    for cls, label in mag_labels.items():
        n = (final_df["mag_class"] == cls).sum()
        log.info("  Class %d  %-20s  %7d events", cls, label, n)
    log.info("")
    log.info("Regional coverage:")
    log.info("  South America: %d events", final_df["is_south_america"].sum())
    log.info("  Peru:          %d events", final_df["is_peru"].sum())
    log.info("═" * 60)


if __name__ == "__main__":
    main()
