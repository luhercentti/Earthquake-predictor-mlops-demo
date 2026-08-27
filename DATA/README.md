# Earthquake Dataset Builder

Downloads the full **USGS global earthquake catalog** and the **IGP Peru national catalog**, merges them into a single deduplicated, ML-ready dataset, and writes it to `data/processed/`.

---

## What this pipeline does

### Data sources

| Source | Coverage | Min magnitude | Rows (approx.) |
|---|---|---|---|
| **USGS ComCat** – global layer | World, 1900 → today | M ≥ 4.5 | ~650 000 |
| **USGS ComCat** – South America layer | lat −56→13, lon −82→−34, 1960 → today | M ≥ 2.5 | ~500 000 |
| **IGP Peru** (Instituto Geofísico del Perú) | Peru, 1960 → 2023 | M ≥ 1.0 | ~24 000 |
| **After deduplication** | Combined | — | **~900 000 – 1.1 M** |

### Steps

```
1. Download  →  data/raw/usgs/{layer}/YYYY-MM-DD_YYYY-MM-DD.csv   (checkpointed)
               data/raw/igp/igp_catalog.csv
2. Load      →  concat all window CSVs per layer
3. Merge     →  align column names across USGS + IGP schemas
4. Dedup     →  remove duplicate events (|Δt| ≤ 60 s, |Δlat/lon| ≤ 0.1°)
5. Features  →  add ML columns (mag_class, temporal, region flags, …)
6. Save      →  data/processed/earthquakes_full.parquet  (~40–60 MB)
               data/processed/earthquakes_full.csv       (~160–220 MB)
```

### Output columns

| Column | Description |
|---|---|
| `time` | UTC datetime of the event |
| `latitude` / `longitude` | Epicentre coordinates |
| `depth` | Hypocentral depth (km) |
| `magnitude` | Moment or local magnitude |
| `magType` | Magnitude type (mw, ml, mb, …) |
| `place` | Human-readable location description |
| `event_type` | Always "earthquake" after filtering |
| `source` | `global_m45`, `south_america_m25`, or `IGP` |
| `year` / `month` / `day_of_year` / `hour_of_day` / `day_of_week` | Temporal features |
| `days_since_epoch` | Float days since 1900-01-01 (time-series feature) |
| `mag_class` | 0=Minor(<3) 1=Light(3–4) 2=Moderate(4–5) 3=Strong(5–6) 4=Major(6–7) 5=Great(≥7) |
| `is_south_america` | 1 if inside South America bounding box |
| `is_peru` | 1 if inside Peru bounding box |

---

## Setup

### 1 — Create and activate a virtual environment

```bash
# from the repo root (Earthquake-predictor-mlops-demo/)
python3.12 -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows
```

### 2 — Install dependencies

```bash
pip install -r DATA/requirements.txt
```

### 3 — Run the pipeline

```bash
cd DATA
python run_pipeline.py
```

The first run takes **20–30 minutes** (network-bound; USGS is queried in 90-day windows with a polite 1.5 s delay between requests).  
Every window is checkpointed as soon as it downloads, so if you interrupt and restart, only missing windows are re-fetched.

### Options

```
python run_pipeline.py --help

  --no-igp          Skip the IGP Peru layer (USGS data only)
  --no-download     Skip downloading; re-process already-cached raw files
  --end-date DATE   Limit USGS queries to this date (YYYY-MM-DD)
  --out-dir PATH    Override the output directory (default: data/processed/)
```

---

## Expected disk usage

```
data/raw/usgs/         ~250–350 MB   (all window CSVs, kept for resumability)
data/raw/igp/          ~2–5 MB
data/processed/        ~200–280 MB   (parquet + csv)
Total                  ~500–650 MB
```

---

## Project structure

```
DATA/
├── README.md               ← you are here
├── requirements.txt        ← Python dependencies
├── run_pipeline.py         ← entry point (run this)
└── pipeline/
    ├── usgs.py             ← USGS FDSN API downloader (checkpointed, resumable)
    ├── igp.py              ← IGP Peru catalog downloader + column normalisation
    └── merge.py            ← merge, dedup, feature engineering, save
```

After running the pipeline:

```
DATA/
├── data/
│   ├── raw/
│   │   ├── usgs/
│   │   │   ├── global_m45/          ← one CSV per 90-day window
│   │   │   └── south_america_m25/   ← one CSV per 90-day window
│   │   └── igp/
│   │       └── igp_catalog.csv
│   └── processed/
│       ├── earthquakes_full.parquet ← primary ML input
│       └── earthquakes_full.csv     ← human-readable copy
└── pipeline.log                     ← full run log
```
