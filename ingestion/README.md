# ingestion/ — Stage 1: Data Collection

Downloads the full earthquake catalog from two sources, merges them,
and writes the ML-ready dataset to `DATA/data/processed/`.

**Run this first, before training.**

---

## How to run

```bash
# from the project root:
python ingestion/run_pipeline.py
```

---

## What it does (step by step)

```
1. USGS global layer   (M≥4.5, 1900–today)
   → queries the USGS FDSN API in 90-day windows
   → saves each window to DATA/data/raw/usgs/global_m45/

2. USGS South America  (M≥2.5, 1960–today, bounding box −56°→13°N)
   → same approach, denser coverage for the region
   → saves to DATA/data/raw/usgs/south_america_m25/

3. IGP Peru catalog    (M≥1.0, 1960–2023)
   → downloads from Peru's open-data portal (datosabiertos.gob.pe)
   → saves to DATA/data/raw/igp/igp_catalog.csv

4. Merge + deduplicate
   → events that appear in multiple sources are matched by
     |Δt| ≤ 60 s  and  |Δlat/lon| ≤ 0.1°

5. Feature columns added
   → year, month, day_of_year, hour_of_day, day_of_week
   → days_since_epoch, mag_class, is_south_america, is_peru

6. Output
   → DATA/data/processed/earthquakes_full.parquet   (primary, ~40–60 MB)
   → DATA/data/processed/earthquakes_full.csv       (human-readable, ~160–220 MB)
```

---

## Runtime

| Run | Duration |
|---|---|
| First run | ~20–30 min (network-bound, ~760 USGS API requests) |
| Subsequent runs | ~10 s (all windows already cached in `DATA/data/raw/`) |

Each 90-day window is saved as soon as it downloads.
If the script is interrupted, re-run it — only missing windows are fetched.

---

## Options

```bash
# skip IGP Peru layer (USGS only)
python ingestion/run_pipeline.py --no-igp

# re-use cached raw files, skip all downloads
python ingestion/run_pipeline.py --no-download

# cap the USGS query at a specific date (useful for reproducibility)
python ingestion/run_pipeline.py --end-date 2024-12-31

# write the output to a custom directory
python ingestion/run_pipeline.py --out-dir /path/to/output
```

---

## Output columns

| Column | Description |
|---|---|
| `time` | UTC datetime |
| `latitude` / `longitude` | Epicentre coordinates |
| `depth` | Hypocentral depth (km) |
| `magnitude` | Event magnitude |
| `magType` | Magnitude type (mw, ml, mb …) |
| `place` | Human-readable location string |
| `source` | `global_m45`, `south_america_m25`, or `IGP` |
| `mag_class` | 0=Minor(<3) 1=Light 2=Moderate 3=Strong 4=Major 5=Great(≥7) |
| `is_south_america` | 1 if inside South America bounding box |
| `is_peru` | 1 if inside Peru bounding box |

---

## Next step

Once `DATA/data/processed/earthquakes_full.parquet` exists:

```
→  training/README.md    (Step 2)
```
