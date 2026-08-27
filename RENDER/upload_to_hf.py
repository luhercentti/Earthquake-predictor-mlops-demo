#!/usr/bin/env python3
"""
Upload models + inference snapshot to Hugging Face Hub.

This pre-computes the cell features snapshot (tiny, ~1-3 MB) so the
deployed server never needs the 200 MB parquet at runtime.

Setup (one-time):
    pip install huggingface_hub
    huggingface-cli login          # saves token, OR set HF_TOKEN env var

Usage:
    python RENDER/upload_to_hf.py --repo your-username/earthquake-predictor-models

After uploading, set HF_REPO_ID in your Render service environment variables.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_DIR    = PROJECT_ROOT / "models"
PARQUET_PATH = PROJECT_ROOT / "DATA" / "data" / "processed" / "earthquakes_full.parquet"
SNAPSHOT_PATH = MODEL_DIR / "cell_snapshot.parquet"   # uploaded alongside models

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


def _build_snapshot() -> None:
    """Compute latest per-cell features and save into models/ for upload."""
    from pipeline.features import get_latest_cell_features

    if not PARQUET_PATH.exists():
        log.error("Parquet not found: %s — run ingestion/run_pipeline.py first.", PARQUET_PATH)
        sys.exit(1)

    log.info("Computing cell features snapshot …")
    df = get_latest_cell_features(PARQUET_PATH)
    df.to_parquet(SNAPSHOT_PATH, index=False, compression="snappy")
    log.info("Snapshot saved: %s  (%.0f KB, %d cells)",
             SNAPSHOT_PATH, SNAPSHOT_PATH.stat().st_size / 1024, len(df))


def upload(repo_id: str, private: bool, skip_snapshot: bool) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        log.error("huggingface_hub not installed.  Run: pip install huggingface_hub")
        sys.exit(1)

    if not MODEL_DIR.exists() or not (MODEL_DIR / "time_model.lgbm").exists():
        log.error("Models not found in %s — run training/train_model.py first.", MODEL_DIR)
        sys.exit(1)

    if not skip_snapshot:
        _build_snapshot()

    token = os.getenv("HF_TOKEN")
    api = HfApi()

    log.info("Creating/verifying HF repo: %s …", repo_id)
    api.create_repo(repo_id=repo_id, repo_type="model",
                    private=private, token=token, exist_ok=True)

    log.info("Uploading %s …", MODEL_DIR)
    api.upload_folder(
        folder_path=str(MODEL_DIR),
        repo_id=repo_id,
        repo_type="model",
        token=token,
        commit_message="Upload LightGBM earthquake forecast models + cell snapshot",
    )

    log.info("")
    log.info("✓  Upload complete: https://huggingface.co/%s", repo_id)
    log.info("")
    log.info("Next steps:")
    log.info("  1. Go to Render → your service → Environment")
    log.info("     Add variable:  HF_REPO_ID = %s", repo_id)
    log.info("  2. Deploy (or trigger a manual redeploy)")
    log.info("     The server will pull models automatically at startup.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, metavar="USER/REPO-NAME",
                   help="HuggingFace repo ID, e.g. johndoe/earthquake-predictor-models")
    p.add_argument("--private", action="store_true",
                   help="Make the repo private (requires HF_TOKEN with write access)")
    p.add_argument("--skip-snapshot", action="store_true",
                   help="Skip recomputing the cell snapshot (use existing models/cell_snapshot.parquet)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    upload(args.repo, args.private, args.skip_snapshot)
