# Bike Count Estimation — SL Challenge

Submission notebook for the supervised learning bike count challenge. The
notebook is generated from `build_notebook.py` so the executable notebook and
source stay in sync.

## Task (short)

Predict the hourly `BikeCount`, compare several model families, and do it for
**two direct forecast horizons**:

- `+1h`: predict `BikeCount(t+1)` from information available at time `t`
- `+24h`: predict `BikeCount(t+24)` from information available at time `t`

The `+24h` task is **not** the sum over the next 24 hours. It is the bike count
of the hour 24 hours in the future. The metric is **MSE** on the hidden test set
delivered on June 3rd. Required: at least one linear, one tree-based, and one
neural-network model.

## Approach

- **Explicit target shift:** the supervised target is `BikeCount.shift(-horizon)`.
- **Time series taken seriously:** BikeCount lag features are restricted to
  values known at forecast origin `t` or earlier.
- **Features:** target-hour calendar (raw + cyclical sin/cos, `is_weekend`),
  target-hour weather mapped to 7 robust keyword buckets, weather measurements,
  BikeCount lags, and rolling means.
- **Cleaning:** exactly one corrupt row (`Weather Condition Null`, all-NaN, and
  also the only duplicate timestamp) is removed → 8759 rows.
- **Validation:** temporal holdout (last 61 days), **no** random k-fold (it
  would leak future information through the lag features).

## Results (validation MSE, last 61 days — NOT the real test MSE)

| Model | +1 h | +24 h |
|---|---|---|
| naive horizon lag | 22,138 | 60,513 |
| naive lag168 | 45,525 | 45,525 |
| Ridge | 11,582 | 28,223 |
| RandomForest | 4,035 | 25,438 |
| **XGBoost (best)** | **3,332** | **22,660** |
| MLP | 9,303 | 39,635 |

→ XGBoost wins both horizons; every trained model beats the simple horizon-lag
baseline. Classic finding: boosted trees > NN on small tabular data.

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
and re-run the final section. While `TEST_PATH = None`, the notebook refuses to
print fake test scores.

## Open / TODO (your input wanted here)

- [ ] **Slides** (2): approach + results. Numbers above are ready to use as
      validation numbers until the hidden test set is available.
- [ ] **Optional tuning** (XGBoost with TimeSeriesSplit). Left out on purpose:
      gain ~5–15% MSE, changes nothing for pass/fail, and risks overfitting to
      the winter holdout. Discussion point.
- [ ] Check the assumption: the test set has the exact same column names,
      includes `BikeCount`, and is a single contiguous hourly series sortable by
      `(Month, Day, Hour)`.

## Files

- `bike_count_estimation.ipynb` — submission notebook (self-contained).
- `build_notebook.py` — generates the notebook (dev, optional).
- `prototype.py` — quick validation script (dev, optional).
- `challenge_public_dataset.xlsx` — public training dataset.
- `requirements.txt` — package versions (important: **xgboost** must be present).
