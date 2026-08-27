# API Usage Examples

All examples assume the server is running at `http://localhost:8000`.

Start the server:
```bash
cd serving && uvicorn app:app --reload --port 8000
```

Open the interactive UI in a browser: **http://localhost:8000**  
Open the Swagger docs: **http://localhost:8000/docs**

---

## System

### Health check
```bash
curl http://localhost:8000/health
```
```json
{"status":"ok","models_loaded":true,"dataset_available":true}
```

---

## Regions

### List all available countries
```bash
curl http://localhost:8000/regions/countries
```

### List all available cities
```bash
curl http://localhost:8000/regions/cities
```

---

## Forecast

### By country
```bash
curl -X POST http://localhost:8000/forecast \
     -H "Content-Type: application/json" \
     -d '{"country": "Peru"}'
```

```bash
curl -X POST http://localhost:8000/forecast \
     -H "Content-Type: application/json" \
     -d '{"country": "Chile"}'
```

```bash
curl -X POST http://localhost:8000/forecast \
     -H "Content-Type: application/json" \
     -d '{"country": "Japan"}'
```

---

### By city (default radius: 200 km)
```bash
curl -X POST http://localhost:8000/forecast \
     -H "Content-Type: application/json" \
     -d '{"city": "Lima"}'
```

```bash
curl -X POST http://localhost:8000/forecast \
     -H "Content-Type: application/json" \
     -d '{"city": "Tokyo"}'
```

### By city with custom search radius
```bash
curl -X POST http://localhost:8000/forecast \
     -H "Content-Type: application/json" \
     -d '{"city": "Lima", "radius_km": 500}'
```

---

### By coordinates
```bash
# Lima, Peru
curl -X POST http://localhost:8000/forecast \
     -H "Content-Type: application/json" \
     -d '{"lat": -12.05, "lon": -77.05}'
```

```bash
# Northern Chile coast
curl -X POST http://localhost:8000/forecast \
     -H "Content-Type: application/json" \
     -d '{"lat": -20.0, "lon": -70.0, "radius_km": 400}'
```

---

### Filter by minimum magnitude (only show seismically significant zones)
```bash
# Cells where the largest event in the past year was M≥5.0
curl -X POST http://localhost:8000/forecast \
     -H "Content-Type: application/json" \
     -d '{"country": "Peru", "min_mag": 5.0}'
```

```bash
curl -X POST http://localhost:8000/forecast \
     -H "Content-Type: application/json" \
     -d '{"country": "Chile", "min_mag": 6.0}'
```

---

### Control how many results come back
```bash
# Top 5 cells only
curl -X POST http://localhost:8000/forecast \
     -H "Content-Type: application/json" \
     -d '{"country": "Peru", "top_n": 5}'
```

```bash
# Top 20 cells
curl -X POST http://localhost:8000/forecast \
     -H "Content-Type: application/json" \
     -d '{"country": "Indonesia", "top_n": 20}'
```

---

### Combine filters
```bash
# Top 5 cells in Peru with M≥4.5 activity, wide radius around Lima
curl -X POST http://localhost:8000/forecast \
     -H "Content-Type: application/json" \
     -d '{"city": "Lima", "radius_km": 600, "min_mag": 4.5, "top_n": 5}'
```

```bash
# Significant zones in Japan
curl -X POST http://localhost:8000/forecast \
     -H "Content-Type: application/json" \
     -d '{"country": "Japan", "min_mag": 5.5, "top_n": 10}'
```

---

## Response fields explained

| Field | Meaning |
|---|---|
| `summary` | Plain-language answer: when, where, and how strong |
| `total_active_cells` | How many 2°×2° grid cells had seismic activity in the past year |
| `forecasts[].estimated_days_to_next` | Expected days until the next earthquake in that cell |
| `forecasts[].estimated_magnitude` | Expected magnitude of that earthquake |
| `forecasts[].events_last_365d` | Number of earthquakes recorded in that cell in the past year |
| `forecasts[].max_magnitude_last_365d` | Largest earthquake in that cell in the past year |
| `forecasts[].nearest_known_place` | Closest named location from the USGS catalog |
| `model_test_mae_days` | Model accuracy: average error in days on the hold-out test set |

---

## Supported countries

`alaska`, `argentina`, `bolivia`, `brazil`, `california`, `chile`, `china`,
`colombia`, `ecuador`, `greece`, `india`, `indonesia`, `iran`, `italy`,
`japan`, `mexico`, `nepal`, `new zealand`, `pakistan`, `peru`, `philippines`,
`turkey`, `usa`, `venezuela`

## Supported cities

`arequipa`, `bogota`, `buenos aires`, `christchurch`, `cusco`, `istanbul`,
`jakarta`, `kathmandu`, `lima`, `los angeles`, `manila`, `mexico city`,
`quito`, `san francisco`, `santiago`, `tehran`, `tokyo`, `trujillo`, `valparaiso`
