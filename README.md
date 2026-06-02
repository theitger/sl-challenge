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
- **Validation / selection:** model selection runs on **leakage-free
  rolling-origin time-series cross-validation** (expanding window, 14-day test
  blocks). A single temporal holdout (last 61 days) is reported as an
  independent control. **No** random k-fold — it would leak future bike counts
  into the training folds through the lag/rolling features. Selection criterion:
  **mean MSE across all rolling-origin windows**.

## Results (rolling-origin CV MSE — NOT the real test MSE)

CV mean/median MSE per model, averaged over 14 rolling-origin windows per horizon:

| Model | +1 h (mean) | +1 h (median) | +24 h (mean) | +24 h (median) |
|---|---|---|---|---|
| **XGBoost tuned (selected)** | **4,370** | 4,111 | **13,890** | **10,768** |
| XGBoost | 4,459 | 4,169 | 14,583 | 12,480 |
| XGBoost log1p | 4,652 | 4,044 | 19,569 | 12,273 |
| XGBoost tuned log1p | 4,686 | 4,294 | 16,534 | 11,227 |
| RandomForest | 5,263 | 4,979 | 14,681 | 12,366 |
| MLP | 7,657 | 6,565 | 22,316 | 21,238 |
| Ridge | 10,803 | 10,052 | 19,096 | 16,028 |

→ **XGBoost wins both horizons under CV and the holdout** — Ridge (linear),
RandomForest, and MLP (neural net) lose under both. The two validation schemes
agree on the winning *family*; they differ only on one detail inside XGBoost
(below). Classic finding: boosted trees > NN on small tabular data.

**On `log1p` (CV vs holdout disagreement, handled transparently):** the
`log1p` target transform wins the single recent 61-day **holdout**, but across
the full year it is *noisier* — at +1h it is the **worst** of the four XGBoost
variants by mean CV MSE, and per-horizon CV selects plain **`XGBoost tuned`**
(no transform) at both horizons. Because the hidden test set spans a similar
multi-season range, we follow the lower-variance CV criterion and ship
`XGBoost tuned`. The holdout's `tuned log1p` pick is reported as a control, not
used for selection.

Reproduce the full selection backtest standalone: `python3 backtest_selection.py`.

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
- [x] **Per-horizon XGBoost tuning** — added and backtested (see Results note).
      Marginal for pass/fail, but a clean methodology point for the approach slide.
- [x] **Selection on rolling-origin time-series CV** — model selection moved from
      the single holdout to leakage-free rolling-origin CV; holdout kept as a
      control. CV selects `XGBoost tuned` at both horizons.
- [x] Test-set assumption: instructor confirmed the hidden set "looks similar to
      the public dataset" and **includes `BikeCount`** (so local MSE is
      computable). Still assumed: same column names, single contiguous hourly
      series sortable by `(Month, Day, Hour)`.

## Files

- `bike_count_estimation.ipynb` — submission notebook (self-contained).
- `backtest_selection.py` — standalone rolling-origin CV over all model families
  (reproduces the notebook's selection); `backtest_tuning.py` — the earlier
  tuning-only backtest.
- `build_notebook.py` — generates the notebook (dev, optional).
- `prototype.py` — quick validation script (dev, optional).
- `challenge_public_dataset.xlsx` — public training dataset.
- `requirements.txt` — package versions (important: **xgboost** must be present).
