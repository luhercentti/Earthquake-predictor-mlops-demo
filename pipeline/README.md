# pipeline/ — Shared Library

This folder is a **Python package, not a stage you run directly.**
It is imported by `ingestion/`, `training/`, and `serving/`.
You never execute files here manually.

---

## What each module does

| File | Purpose |
|---|---|
| `usgs.py` | Downloads the USGS earthquake catalog in 90-day batches via the FDSN API. Handles rate-limiting, retries, and checkpointing so interrupted downloads resume cleanly. |
| `igp.py` | Downloads the IGP Peru national seismic catalog from Peru's open-data portal (CKAN API). Falls back to a known direct URL if the API is unreachable, and prints manual-download instructions if both fail. |
| `merge.py` | Concatenates USGS and IGP frames, normalises column names, deduplicates events that appear in multiple sources (matching by ±60 s and ±0.1° lat/lon), and adds base ML columns (`mag_class`, `is_south_america`, `is_peru`, etc.). |
| `features.py` | Assigns each event to a 2°×2° grid cell and computes rolling-window statistics (event count, mean/max/std magnitude, mean depth) over the past 7, 30, 90, and 365 days. Also computes `days_since_last`, `days_to_next`, and `mag_next` labels used in training. |

---

## Data flow

```
usgs.py ──┐
          ├──► merge.py ──► features.py ──► training / serving
igp.py  ──┘
```

---

## Nothing to run here

If you are setting up the project for the first time, go to:

```
→  ingestion/README.md    (Step 1)
```
