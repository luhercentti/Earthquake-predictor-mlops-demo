"""
Earthquake Forecast — FastAPI Application
==========================================

Start locally:
    uvicorn app:app --reload --port 8000

Then open:
    http://localhost:8000/docs      ← interactive Swagger UI (try it here!)
    http://localhost:8000/redoc     ← ReDoc alternative

Example curl requests:
    curl http://localhost:8000/health

    curl -X POST http://localhost:8000/forecast \\
         -H "Content-Type: application/json" \\
         -d '{"country": "Peru"}'

    curl -X POST http://localhost:8000/forecast \\
         -H "Content-Type: application/json" \\
         -d '{"city": "Lima", "min_mag": 4.5}'

    curl -X POST http://localhost:8000/forecast \\
         -H "Content-Type: application/json" \\
         -d '{"lat": -12.0, "lon": -77.0, "radius_km": 300}'

Production path (when ready):
    docker build -t earthquake-serving .
    # → deploy container to KServe on Kubernetes
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from schemas import ForecastRequest, ForecastResponse, HealthResponse
from predictor import DEFAULT_PARQUET, COUNTRY_BBOX, CITY_COORDS, ModelRegistry, run_forecast

log = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

_UI_HTML = (Path(__file__).parent / "index.html").read_text(encoding="utf-8")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup so the first request isn't slow."""
    registry = ModelRegistry.get()
    try:
        registry.load()
        log.info("Models loaded at startup.")
    except FileNotFoundError as exc:
        log.warning("Models not yet available: %s", exc)
    yield
    log.info("Shutting down.")


app = FastAPI(
    title="Earthquake Forecast API",
    description=(
        "Probabilistic earthquake forecasting API powered by LightGBM models "
        "trained on the USGS global catalog + IGP Peru catalog (1973–present).\n\n"
        "**⚠ Disclaimer:** Outputs are probabilistic estimates based on historical "
        "seismicity patterns — not deterministic predictions. "
        "Do not use for emergency response decisions."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_ui():
    """Serve the forecast web UI."""
    return _UI_HTML


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Liveness + readiness check."""
    registry = ModelRegistry.get()
    return {
        "status": "ok",
        "models_loaded": registry.loaded,
        "dataset_available": DEFAULT_PARQUET.exists(),
    }


@app.get("/regions/countries", tags=["Regions"])
async def list_countries() -> list[str]:
    """Return all supported country names for /forecast."""
    return sorted(COUNTRY_BBOX.keys())


@app.get("/regions/cities", tags=["Regions"])
async def list_cities() -> list[str]:
    """Return all supported city names for /forecast."""
    return sorted(CITY_COORDS.keys())


@app.post("/forecast", response_model=ForecastResponse, tags=["Forecast"])
async def forecast(req: ForecastRequest):
    """
    Earthquake sequence forecast for a country, city, or coordinates.

    Returns the top-N most seismically active grid cells (2°×2°) in the
    region, ranked by expected time to the next earthquake, with estimated
    magnitude and nearest known place name.
    """
    registry = ModelRegistry.get()
    if not registry.loaded:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Run training/train_model.py first.",
        )
    if not DEFAULT_PARQUET.exists():
        raise HTTPException(
            status_code=503,
            detail="Dataset not found. Run ingestion/run_pipeline.py first.",
        )

    try:
        result = run_forecast(
            country=req.country,
            city=req.city,
            lat=req.lat,
            lon=req.lon,
            radius_km=req.radius_km,
            min_mag=req.min_mag,
            top_n=req.top_n,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        log.exception("Forecast failed: %s", exc)
        raise HTTPException(status_code=500, detail="Internal forecast error.")

    return result
