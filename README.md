# Bike Count Estimation — SL Challenge (first shot)

First pass at the challenge. It runs end-to-end, but it is deliberately a
**draft to review** — not the final submission. Please read the open points
below before drawing conclusions.

## Task (short)

Predict the hourly `BikeCount`, compare several model families, and do it for
**two horizons**: next hour (+1 h) and next 24 hours (+24 h). Scored by **MSE**
on a hidden test set (same format, delivered June 3rd). Required: at least one
linear, one tree-based, and one neural-network model.

## Approach

- **Time series taken seriously:** the strongest signals are lag features
  (autocorr lag-1 ≈ 0.91, lag-24 ≈ 0.81, lag-168 ≈ 0.89).
- **Direct multi-step:** two separate models. The +1 h model may use `lag ≥ 1`,
  the +24 h model only `lag ≥ 24` (24 h ahead, the last hour is still unknown).
- **Features:** calendar (raw + cyclical sin/cos, `is_weekend`), weather mapped
  to 7 robust keyword buckets, weather measurements, lags + rolling means.
- **Cleaning:** exactly one corrupt row (`Weather Condition Null`, all-NaN, and
  also the only duplicate timestamp) is removed → 8759 rows.
- **Validation:** temporal holdout (last 61 days), **no** random k-fold (it
  would leak future information through the lag features).

## Results (validation MSE, last 61 days — NOT the real test MSE)

| Model | +1 h | +24 h |
|---|---|---|
| naive (lag1 / lag24) | 22,138 | 60,513 |
| naive (lag168) | 45,525 | 45,525 |
| Ridge | 11,836 | 30,416 |
| RandomForest | 3,688 | 23,035 |
| **XGBoost (best)** | **3,179** | **21,450** |
| MLP | 8,923 | 33,940 |

→ XGBoost wins both horizons; every trained model beats the baselines. Classic
finding: boosted trees > NN on small tabular data.

## Run the notebook

```bash
pip install -r requirements.txt
jupyter notebook bike_count_estimation.ipynb
```

`bike_count_estimation.ipynb` loads the data, trains/compares all models, picks
the best per horizon, and prints the test MSE at the end.

### On June 3rd
Change **one line** at the top of the notebook:
```python
TEST_PATH = "path/to/hidden_test_file.xlsx"
```
and re-run. The two printed MSE values are the numbers for the results slide.

## ⚠️ Don't misread this

While `TEST_PATH` still points to the public dataset, the printed "Test MSE"
(~730 / ~3080) is an **in-sample check** (model evaluated on its own training
rows) and **far too optimistic**. The realistic out-of-sample value is closer
to the holdout numbers above (~3200 / ~21500).

## Open / TODO (your input wanted here)

- [ ] **Slides** (2): approach + results. Numbers above are ready to use.
- [ ] **Optional tuning** (XGBoost with TimeSeriesSplit). Left out on purpose:
      gain ~5–15% MSE, changes nothing for pass/fail, and risks overfitting to
      the winter holdout. Discussion point.
- [ ] Check the assumption: the test set has the exact same column names and is
      a single contiguous hourly series sortable by (Month, Day, Hour).

## Files

- `bike_count_estimation.ipynb` — submission notebook (self-contained).
- `build_notebook.py` — generates the notebook (dev, optional).
- `prototype.py` — quick validation script (dev, optional).
- `challenge_public_dataset (1).xlsx` — public training dataset.
- `requirements.txt` — package versions (important: **xgboost** must be present).
