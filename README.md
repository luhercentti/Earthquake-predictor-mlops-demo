# Earthquake Predictor — MLOps Demo

End-to-end MLOps project that downloads the global earthquake catalog,
trains a LightGBM model, and serves probabilistic forecasts via a REST API.

---

## Project layout

```
├── pipeline/       Shared Python library — imported by all stages, not run directly
├── ingestion/      Stage 1 — downloads & processes the earthquake dataset
├── training/       Stage 2 — trains the LightGBM models
├── serving/        Stage 3 — FastAPI REST server for querying the model
├── models/         Output of training (auto-created): .lgbm files + metadata.json
├── DATA/           Output of ingestion (auto-created): raw CSV windows + processed parquet
└── requirements.txt
```

---

## Step-by-step execution

### Prerequisites

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

### Step 1 — Download & build the dataset

```bash
python ingestion/run_pipeline.py
```

**What it does:** queries the USGS earthquake API in 90-day windows (1900–today),
downloads the IGP Peru national catalog, merges and deduplicates everything,
and writes the final ML-ready dataset.

**Takes:** ~20–30 min on first run. Subsequent runs are instant (windows are cached).

**Produces:**
```
DATA/data/raw/usgs/          one CSV per 90-day window
DATA/data/raw/igp/           igp_catalog.csv
DATA/data/processed/         earthquakes_full.parquet  (~40–60 MB)
                             earthquakes_full.csv       (~160–220 MB)
```

➡ See `ingestion/README.md` for options and details.

---

### Step 2 — Train the models

```bash
python training/train_model.py
```

**What it does:** reads the parquet, engineers rolling-window features per 2°×2°
grid cell, trains two LightGBM regressors (time-to-next and magnitude), runs
5-fold time-series cross-validation, and saves the artifacts.

**Takes:** 5–15 min depending on dataset size.

**Produces:**
```
models/time_model.lgbm     predicts days until next earthquake in a cell
models/mag_model.lgbm      predicts expected magnitude of next earthquake
models/metadata.json       feature list, hyperparams, test-set metrics
training.log               full training log
```

➡ See `training/README.md` for options and details.

---

### Step 3 — Serve the model locally

```bash
cd serving
uvicorn app:app --reload --port 8000
```

**What it does:** starts a FastAPI server that loads the trained models and
exposes REST endpoints you can query in plain HTTP.

**Open in your browser:**
```
http://localhost:8000/docs      interactive Swagger UI — click "Try it out"
http://localhost:8000/redoc     alternative API docs
```

**Or query with curl:**
```bash
# Forecast for an entire country
curl -X POST http://localhost:8000/forecast \
     -H "Content-Type: application/json" \
     -d '{"country": "Peru"}'

# Forecast near a specific city
curl -X POST http://localhost:8000/forecast \
     -H "Content-Type: application/json" \
     -d '{"city": "Lima", "min_mag": 4.5}'

# Forecast by coordinates
curl -X POST http://localhost:8000/forecast \
     -H "Content-Type: application/json" \
     -d '{"lat": -12.0, "lon": -77.0, "radius_km": 300}'

# List all available countries
curl http://localhost:8000/regions/countries

# Health check
curl http://localhost:8000/health
```

➡ See `serving/README.md` for the full API reference.

---

## MLOps lifecycle diagram

```
[USGS API]──┐
            ├──► ingestion/run_pipeline.py ──► DATA/data/processed/
[IGP Peru]──┘                                         │
                                                      ▼
                                         training/train_model.py ──► models/
                                                      │
                                                      ▼
                                            serving/uvicorn app ──► REST API
                                                      │
                                                      ▼
                                    POST /forecast {"country": "Peru"}
```

---

## What the model can and cannot do

| | |
|---|---|
| ✅ Ranks regions by seismic activity and "overdue" state | |
| ✅ Estimates expected days to next event and likely magnitude per region | |
| ❌ Cannot give an exact date/time — no model in the world can (unsolved physics) | |

Outputs are **probabilistic estimates** based on historical seismicity patterns,
the same statistical basis USGS uses for Probabilistic Seismic Hazard Analysis (PSHA).

---

## Production path (future)

```
Local FastAPI  →  Docker image  →  KServe on Kubernetes
```

When ready to productionise, containerise `serving/` and deploy the image
to KServe. No code changes required — KServe wraps the existing FastAPI container.
