# training/ — Stage 2: Model Training

Trains two LightGBM regression models on the earthquake dataset produced
by Stage 1 and saves the artifacts to `models/`.

**Run this after `ingestion/run_pipeline.py` has finished.**

---

## How to run

```bash
# from the project root:
python training/train_model.py
```

---

## What it does (step by step)

```
1. Load features
   → reads DATA/data/processed/earthquakes_full.parquet
   → assigns each event to a 2°×2° grid cell
   → computes rolling-window stats per cell (7/30/90/365 days):
       count, mean/max/std magnitude, mean depth
   → labels each event with:
       days_to_next  — days until the next earthquake in the same cell
       mag_next      — magnitude of that next earthquake

2. Temporal train/test split  (no data leakage)
   → test set = last 3 years of data
   → train set = everything before that

3. 5-fold time-series cross-validation
   → validates the model respects chronological order
   → reports MAE per fold

4. Train final models on full training set
   → Model A (time_model.lgbm): predicts days_to_next  [regression, MAE loss]
   → Model B (mag_model.lgbm):  predicts mag_next       [regression, MSE loss]

5. Evaluate on hold-out test set
   → logs MAE, R², and median absolute error

6. Save artifacts
   → models/time_model.lgbm
   → models/mag_model.lgbm
   → models/metadata.json   (features, params, test metrics)
   → training.log
```

---

## Why LightGBM?

- Handles non-linear patterns in seismic sequences better than linear models.
- Natively tolerates missing values in sparse rolling windows.
- Trains on CPU in minutes — no GPU required.
- Gradient-boosted trees match or beat deep learning on tabular seismic
  catalog data (benchmark: DeepMind aftershock paper, Nature 2019).

---

## Options

```bash
# train only on larger events (faster, less noise)
python training/train_model.py --min-mag 4.5

# use a different parquet file
python training/train_model.py --parquet /path/to/earthquakes_full.parquet
```

---

## Output

```
models/
├── time_model.lgbm      predicts days until next earthquake in a cell
├── mag_model.lgbm       predicts expected magnitude
└── metadata.json        feature list, hyperparameters, test-set MAE/R²
training.log             full training run log
```

---

## Understanding the metrics in metadata.json

| Metric | Meaning |
|---|---|
| `mae` (days_to_next) | On average, predictions are off by this many days |
| `r2` (days_to_next) | How much of the variance the model explains (0–1) |
| `mae` (mag_next) | On average, magnitude predictions are off by this much |

Earthquake inter-event times follow a heavy-tailed distribution (Omori-Utsu law),
so a MAE of ~10–30 days is typical and scientifically reasonable — not a sign of
a bad model. The model is learning *relative risk*, not exact timing.

---

## What the model predicts

Given the seismic history of a 2°×2° cell up to today, it estimates:
- **days_to_next**: expected days until the next earthquake in that cell
- **mag_next**: expected magnitude of that earthquake

These are conditional expectations, not certainties. Cells with low
`days_to_next` are statistically more active right now — not guaranteed to rupture.

---

## Next step

Once `models/time_model.lgbm` and `models/mag_model.lgbm` exist:

```
→  serving/README.md    (Step 3)
```
