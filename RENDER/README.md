# RENDER/ — Render Deployment Stage

Everything needed to deploy the earthquake forecast website to
[Render](https://render.com) lives here.

**Existing scripts are not modified.** The Dockerfile swaps in
`predictor_hf.py` at build time, so the local `serving/` folder stays
exactly as it is.

---

## Why render.yaml is NOT in this folder

Render scans only the **repo root** for `render.yaml`. It will not find it
inside a subdirectory. That file intentionally lives one level up at the
project root — it simply points Render to use `RENDER/Dockerfile` as the
build file.

---

## Files in this folder

| File | Purpose |
|---|---|
| `upload_to_hf.py` | **Run locally once after training.** Computes the cell snapshot, then uploads models + snapshot to HuggingFace Hub. |
| `predictor_hf.py` | Drop-in replacement for `serving/predictor.py` inside the container. Downloads models from HuggingFace at startup and uses the cell snapshot for inference instead of the full 200 MB parquet. |
| `Dockerfile` | Builds the container image. Copies `pipeline/` and `serving/` from the repo, then overlays `predictor_hf.py` as `serving/predictor.py`. |
| `requirements.txt` | Serving-only dependencies for the Docker image (includes `huggingface_hub`). |

---

## What is the "cell snapshot"?

The forecast model predicts based on the *current seismic state* of each
2°×2° grid cell: how many earthquakes happened in the last 7 / 30 / 90 / 365
days, what was the max magnitude, etc.

Locally, those features are computed on the fly from the full
`DATA/data/processed/earthquakes_full.parquet` (~200 MB). That file is
gitignored and is not in the Docker container.

`upload_to_hf.py` solves this by pre-computing those rolling-window
features once (result: one row per active cell, ~1–3 MB) and saving them as
`cell_snapshot.parquet`. It uploads this file alongside the `.lgbm` models
to HuggingFace. At runtime on Render, `predictor_hf.py` reads the snapshot
directly — no 200 MB file needed.

```
Full parquet (200 MB, local only)
    └── upload_to_hf.py computes rolling stats per cell
              └── cell_snapshot.parquet (~2 MB)  ← uploaded to HuggingFace
                        └── predictor_hf.py reads this at runtime on Render
```

---

## Full deployment workflow

### 1 — Upload models to HuggingFace (run once locally after training)

```bash
pip install huggingface_hub
huggingface-cli login              # log in with your HF account

python RENDER/upload_to_hf.py --repo YOUR_USERNAME/earthquake-predictor-models
```

This uploads:
```
HuggingFace repo/
├── time_model.lgbm          trained LightGBM time-to-next model
├── mag_model.lgbm           trained LightGBM magnitude model
├── metadata.json            feature list + test metrics
└── cell_snapshot.parquet    pre-computed cell features (~2 MB)
```

Re-run this every time you retrain the model.

---

### 2 — Connect repo to Render

1. Go to **render.com** → New → Web Service
2. Connect your GitHub repo
3. Render detects `render.yaml` at the repo root → click **Apply**
4. In **Environment Variables** add:
   ```
   HF_REPO_ID = YOUR_USERNAME/earthquake-predictor-models
   ```
   If your HuggingFace repo is private, also add:
   ```
   HF_TOKEN = your_hf_token
   ```
5. Click **Deploy**

---

### 3 — What happens at container startup

```
Container starts
  └── uvicorn app:app
        └── lifespan() → ModelRegistry.load()
              ├── models/ not found locally
              ├── downloads from HuggingFace (HF_REPO_ID):
              │     time_model.lgbm
              │     mag_model.lgbm
              │     metadata.json
              │     cell_snapshot.parquet
              └── server ready
                    GET  /          → web UI
                    GET  /health    → health check
                    GET  /docs      → Swagger
                    POST /forecast  → forecast API
```

First startup takes ~30–60 s while models download. Subsequent restarts
are instant (models are cached in the container's filesystem for the
duration of the deployment).

---

### Testing the Docker image locally before deploying

```bash
# from the project root:
docker build -f RENDER/Dockerfile -t earthquake-forecast .

docker run -p 8000:8000 \
  -e HF_REPO_ID=YOUR_USERNAME/earthquake-predictor-models \
  earthquake-forecast
```

Then open http://localhost:8000
