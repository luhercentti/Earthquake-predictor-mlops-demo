# serving/ — Stage 3: Model Serving

Runs a **FastAPI** REST server that loads the trained models and answers
earthquake forecast questions over HTTP.

**Run this after training has completed.**

---

## How to start the server

```bash
# from the serving/ directory:
cd serving
uvicorn app:app --reload --port 8000
```

The `--reload` flag restarts the server automatically when you edit any file —
useful during development. Remove it in production.

---

## Explore the API in your browser

Once the server is running, open:

```
http://localhost:8000/docs      ← Swagger UI — click "Try it out" on any endpoint
http://localhost:8000/redoc     ← ReDoc alternative view
```

---

## API endpoints

### `GET /health`
Checks whether the server is up, models are loaded, and the dataset is available.
```bash
curl http://localhost:8000/health
```

### `GET /regions/countries`
Lists all country names accepted by `/forecast`.
```bash
curl http://localhost:8000/regions/countries
```

### `GET /regions/cities`
Lists all city names accepted by `/forecast`.
```bash
curl http://localhost:8000/regions/cities
```

### `POST /forecast`
Main endpoint. Returns the top-N most active grid cells in the region,
ranked by expected time to next earthquake.

**By country:**
```bash
curl -X POST http://localhost:8000/forecast \
     -H "Content-Type: application/json" \
     -d '{"country": "Peru"}'
```

**By city (with a 300 km search radius):**
```bash
curl -X POST http://localhost:8000/forecast \
     -H "Content-Type: application/json" \
     -d '{"city": "Lima", "radius_km": 300}'
```

**By coordinates:**
```bash
curl -X POST http://localhost:8000/forecast \
     -H "Content-Type: application/json" \
     -d '{"lat": -12.0, "lon": -77.0, "radius_km": 200}'
```

**Filter by minimum recent activity (M≥4.5 in past year):**
```bash
curl -X POST http://localhost:8000/forecast \
     -H "Content-Type: application/json" \
     -d '{"country": "Chile", "min_mag": 4.5, "top_n": 5}'
```

---

## Request parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `country` | string | — | Country name (e.g. `"Peru"`) |
| `city` | string | — | City name (e.g. `"Lima"`) |
| `lat` + `lon` | float | — | Coordinates (use with `radius_km`) |
| `radius_km` | float | 200 | Search radius when using city or coordinates |
| `min_mag` | float | none | Only show cells active at M≥X in the past year |
| `top_n` | int | 10 | Number of cells to return (max 50) |

Provide exactly one of: `country`, `city`, or `lat`+`lon`.

---

## Response structure

```json
{
  "region": "peru",
  "as_of_utc": "2026-08-26T14:30:00+00:00",
  "total_active_cells": 38,
  "model_test_mae_days": 12.4,
  "forecasts": [
    {
      "rank": 1,
      "cell_lat": -16.0,
      "cell_lon": -74.0,
      "estimated_days_to_next": 1.8,
      "estimated_magnitude": 4.6,
      "events_last_365d": 142,
      "max_magnitude_last_365d": 6.1,
      "nearest_known_place": "45 km SW of Ica, Peru"
    },
    ...
  ],
  "disclaimer": "PROBABILISTIC ESTIMATE ONLY. ..."
}
```

---

## Files in this folder

| File | Purpose |
|---|---|
| `app.py` | FastAPI application: defines all routes, startup (model loading), CORS |
| `predictor.py` | Core inference logic: model loading singleton, region resolution, cell filtering, model calls |
| `schemas.py` | Pydantic models for request validation and response serialisation |

---

## Spatial resolution note

The model works on **2°×2° grid cells** (~220 km at the equator).
`nearest_known_place` gives the closest named location from the USGS catalog —
it is for orientation only, not a precise epicentre prediction.

---

## Production path

When ready to move beyond local:

```
1. Build a Docker image from this serving/ folder
2. Push to a container registry
3. Deploy to KServe on Kubernetes
   → KServe wraps the container and adds a standard inference protocol
   → No code changes required in app.py
```
