"""Pydantic schemas for request validation and response serialization."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class ForecastRequest(BaseModel):
    """
    Provide exactly one of: country, city, or lat+lon.
    """
    country: str | None = Field(None, example="Peru")
    city: str | None = Field(None, example="Lima")
    lat: float | None = Field(None, ge=-90, le=90, example=-12.05)
    lon: float | None = Field(None, ge=-180, le=180, example=-77.05)
    radius_km: float = Field(200.0, gt=0, le=2000, description="Search radius when using lat/lon or city")
    min_mag: float | None = Field(None, ge=0, le=10, description="Only include cells active at this magnitude or above in the past year")
    top_n: int = Field(10, ge=1, le=50, description="Number of top cells to return")

    @model_validator(mode="after")
    def _check_at_least_one(self) -> ForecastRequest:
        has_country = self.country is not None
        has_city = self.city is not None
        has_coords = self.lat is not None and self.lon is not None
        if not (has_country or has_city or has_coords):
            raise ValueError("Provide at least one of: country, city, or lat+lon")
        if self.lat is not None and self.lon is None:
            raise ValueError("lon is required when lat is provided")
        if self.lon is not None and self.lat is None:
            raise ValueError("lat is required when lon is provided")
        return self


class CellForecast(BaseModel):
    rank: int
    cell_lat: float = Field(description="SW corner latitude of the 2°×2° grid cell")
    cell_lon: float = Field(description="SW corner longitude of the 2°×2° grid cell")
    estimated_days_to_next: float = Field(description="Expected days until next earthquake in this cell")
    estimated_magnitude: float = Field(description="Expected magnitude of next earthquake")
    events_last_365d: int = Field(description="Number of earthquakes recorded in this cell in the past year")
    max_magnitude_last_365d: float = Field(description="Largest earthquake in this cell in the past year")
    nearest_known_place: str


class ForecastResponse(BaseModel):
    region: str
    as_of_utc: str
    total_active_cells: int
    summary: str = Field(description="Plain-language answer to 'when and where is the next earthquake?'")
    forecasts: list[CellForecast]
    model_test_mae_days: float | None = Field(None, description="Hold-out MAE from training evaluation")
    disclaimer: str = (
        "PROBABILISTIC ESTIMATE ONLY. "
        "Earthquake science cannot predict exact occurrence times. "
        "These estimates reflect conditional inter-event time distributions "
        "derived from historical seismicity patterns. "
        "Do not use for emergency response or civil-protection decisions."
    )


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    dataset_available: bool
