#!/usr/bin/env python3
"""
Upload trained models + full processed dataset to Hugging Face Hub.

Everything the server needs at runtime is uploaded:
  - time_model.lgbm
  - mag_model.lgbm
  - metadata.json
  - earthquakes_full.parquet   (~50 MB, needed for place-name lookups)

Setup (one-time):
    pip install huggingface_hub
    huggingface-cli login

Usage:
    python RENDER/upload_to_hf.py --repo your-username/earthquake-predictor-models
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


def upload(repo_id: str, private: bool) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        log.error("huggingface_hub not installed.  Run: pip install huggingface_hub")
        sys.exit(1)

    if not MODEL_DIR.exists() or not (MODEL_DIR / "time_model.lgbm").exists():
        log.error("Models not found in %s — run training/train_model.py first.", MODEL_DIR)
        sys.exit(1)

    if not PARQUET_PATH.exists():
        log.error("Parquet not found: %s — run ingestion/run_pipeline.py first.", PARQUET_PATH)
        sys.exit(1)

    token = os.getenv("HF_TOKEN")
    api = HfApi()

    log.info("Creating/verifying HF repo: %s …", repo_id)
    api.create_repo(repo_id=repo_id, repo_type="model",
                    private=private, token=token, exist_ok=True)

    log.info("Uploading models from %s …", MODEL_DIR)
    api.upload_folder(
        folder_path=str(MODEL_DIR),
        repo_id=repo_id,
        repo_type="model",
        token=token,
        commit_message="Upload models",
    )

    log.info("Uploading parquet (%s) …", PARQUET_PATH)
    api.upload_file(
        path_or_fileobj=str(PARQUET_PATH),
        path_in_repo="earthquakes_full.parquet",
        repo_id=repo_id,
        repo_type="model",
        token=token,
        commit_message="Upload processed earthquake dataset",
    )

    log.info("")
    log.info("✓  Upload complete: https://huggingface.co/%s", repo_id)
    log.info("")
    log.info("Next steps:")
    log.info("  1. Set HF_REPO_ID = %s in your Render environment variables", repo_id)
    log.info("  2. Redeploy on Render")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, metavar="USER/REPO-NAME")
    p.add_argument("--private", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    upload(args.repo, args.private)
