"""Generate the submission notebook bike_count_estimation.ipynb."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

cells = []
md = lambda s: cells.append(new_markdown_cell(s))
co = lambda s: cells.append(new_code_cell(s))

md("""# Bike Count Estimation — Supervised Learning Challenge

**Task.** Predict the hourly `BikeCount` for the city of Münster from one year of
hourly data, and compare model families across two forecast horizons.

**Forecast framing (direct multi-step).** Each row carries its own calendar and
weather columns. The two horizons differ *only* in which lagged BikeCounts are
available at the forecast origin:

| Horizon | Available lags | Idea |
|---|---|---|
| **+1 h**  | lag ≥ 1   | the most recent count is known (autocorr ≈ 0.91) |
| **+24 h** | lag ≥ 24  | nothing from the previous 24 h is known yet |

**Models compared** (one per family, as required): **Ridge** (linear),
**Random Forest** & **XGBoost** (tree-based), **MLP** (neural network), against
**naive baselines** (persistence and weekly-seasonal).

**Metric.** Mean Squared Error (MSE).

**Deliverable behaviour.** This notebook (1) loads a dataset, (2) trains and
selects the best model per horizon on the public data, then (3) applies that
model to the held-out test set and reports its MSE. It is written to run *blind*
on the hidden test set delivered on June 3rd — it only assumes the same column
format and a contiguous hourly series.""")

co("""import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

RNG = 0

# Public training data (shipped with this notebook).
TRAIN_PATH = "challenge_public_dataset (1).xlsx"

# Hidden test set, delivered on June 3rd. Same format as the public dataset.
# >>> On June 3rd, set this to the path of the provided test file. <<<
# It defaults to the public dataset so the notebook runs end-to-end as a check.
TEST_PATH = "challenge_public_dataset (1).xlsx"
""")

md("""## 1. Load & clean

The raw file contains exactly one corrupt row (`Weather Condition Null`, all
sensor values `NaN`) which is also the only duplicated `(Month, Day, Hour)`
timestamp. Dropping it fixes both issues at once and leaves 8759 unique hourly
rows. We sort chronologically so lag features shift correctly.""")

co("""REQUIRED_COLS = ["Month", "Day", "Hour", "Weekday", "Weather",
                 "Temperature (°C)", "Humidity (%)", "Rain (mm)", "Wind (km/h)", "BikeCount"]

def load_clean(path):
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()                   # tolerate stray whitespace
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:                                           # fail loudly, not silently
        raise ValueError(
            f"Input file is missing expected columns {missing}. "
            f"Got {list(df.columns)}. The file must have the same format as the "
            f"public dataset.")
    df = df.dropna(subset=["BikeCount"]).copy()           # drop corrupt/duplicate row
    df = df.sort_values(["Month", "Day", "Hour"]).reset_index(drop=True)  # assumes one contiguous year
    return df

train_df = load_clean(TRAIN_PATH)
print(f"clean rows: {len(train_df)}  |  columns: {list(train_df.columns)}")
train_df.head()
""")

md("""## 2. Feature engineering

**Calendar.** Raw `Hour/Weekday/Month` (for trees) plus cyclical sin/cos
encodings (for linear/MLP) and an `is_weekend` flag (Weekday 5 & 6, where the
commute peak collapses).

**Weather.** The 35 messy free-text categories are reduced to 7 stable buckets
by keyword — robust to unseen strings in the hidden test set, unlike one-hot on
the raw text.

**Lags (the core of the time-series signal).** Autocorrelation is strong
(lag-1 ≈ 0.91, lag-24 ≈ 0.81, lag-168 ≈ 0.89), so lagged BikeCounts and rolling
means dominate. Which lags are allowed depends on the horizon (see top table).

**Boundary imputation.** The first rows of any series lack lag history. Missing
lags are filled with the per-`(Hour, is_weekend)` mean BikeCount learned on the
training data, so every test row gets a prediction (no silent row dropping).""")

co("""WEATHER_CATS = ["Thunder", "Snow", "Rain", "Fog", "Clear", "Cloudy", "Other"]

def weather_bucket(s):
    s = str(s).lower()
    if "thunder" in s:                                  return "Thunder"
    if "snow" in s or "ice" in s or "sleet" in s:       return "Snow"
    if "rain" in s or "drizzle" in s or "shower" in s:  return "Rain"
    if "fog" in s:                                      return "Fog"
    if "sunny" in s or "clear" in s:                    return "Clear"
    if "cloud" in s or "overcast" in s:                 return "Cloudy"
    return "Other"

def build_features(df, horizon):
    # Return (X, y, lagcols) for a horizon. df must be cleaned + sorted.
    d = df.copy()
    d["is_weekend"] = d["Weekday"].isin([5, 6]).astype(int)
    d["hour_sin"]  = np.sin(2 * np.pi * d["Hour"]  / 24)
    d["hour_cos"]  = np.cos(2 * np.pi * d["Hour"]  / 24)
    d["month_sin"] = np.sin(2 * np.pi * d["Month"] / 12)
    d["month_cos"] = np.cos(2 * np.pi * d["Month"] / 12)

    cat = pd.Categorical(d["Weather"].map(weather_bucket), categories=WEATHER_CATS)
    wdum = pd.get_dummies(cat, prefix="w").astype(int)    # stable 7 columns

    bc = d["BikeCount"]
    lags = [1, 2, 3, 24, 168] if horizon == 1 else [24, 25, 48, 168]
    for L in lags:
        d[f"lag{L}"] = bc.shift(L)
    if horizon == 1:
        d["roll3"]  = bc.shift(1).rolling(3).mean()
        d["roll24"] = bc.shift(1).rolling(24).mean()
        rollcols = ["roll3", "roll24"]
    else:
        d["roll24"] = bc.shift(24).rolling(24).mean()
        rollcols = ["roll24"]

    base = ["Hour", "Weekday", "Month", "is_weekend",
            "hour_sin", "hour_cos", "month_sin", "month_cos",
            "Temperature (°C)", "Humidity (%)", "Rain (mm)", "Wind (km/h)"]
    lagcols = [f"lag{L}" for L in lags] + rollcols
    X = pd.concat([d[base], wdum, d[lagcols]], axis=1)
    return X, bc, lagcols

def fit_impute(df_train):
    tmp = df_train.copy()
    tmp["is_weekend"] = tmp["Weekday"].isin([5, 6]).astype(int)
    grp  = tmp.groupby(["Hour", "is_weekend"])["BikeCount"].mean()
    glob = tmp["BikeCount"].mean()
    return grp, glob

def apply_impute(X, rows, lagcols, grp, glob):
    X = X.copy()
    key = list(zip(rows["Hour"], rows["Weekday"].isin([5, 6]).astype(int)))
    fill = pd.Series([grp.get(k, glob) for k in key], index=X.index)
    for c in lagcols:
        X[c] = X[c].fillna(fill)
    return X
""")

md("""## 3. Model definitions

The Ridge `alpha`, the MLP architecture, and the XGBoost settings were chosen on
the temporal holdout below. The MLP standardises both inputs and target. The
**best model** uses XGBoost; for the harder +24 h horizon a `log1p` target
transform gave a lower holdout MSE, so it is applied there.""")

co("""def make_models():
    mlp = lambda: TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(128, 64, 32), alpha=1e-3,
                         learning_rate_init=0.005, max_iter=800,
                         early_stopping=True, n_iter_no_change=20, random_state=RNG)),
        transformer=StandardScaler())
    return {
        "Ridge":        make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "RandomForest": RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=RNG),
        "XGBoost":      XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                                     subsample=0.8, colsample_bytree=0.8,
                                     n_jobs=-1, random_state=RNG),
        "MLP":          mlp(),
    }

def best_model(horizon):
    # Selected best model per horizon (XGBoost; log1p target for the +24 h case).
    xgb = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                       subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=RNG)
    if horizon == 24:
        return TransformedTargetRegressor(regressor=xgb, func=np.log1p, inverse_func=np.expm1)
    return xgb
""")

md("""## 4. Model comparison (temporal holdout)

We validate on the **last 61 days** of the public year and train on the rest —
a temporal split, never a random k-fold, which would leak future information
through the lag features. Numbers below are the validation MSE used for the
results slide.""")

co("""def evaluate_holdout(df, n_val_days=61):
    split = len(df) - 24 * n_val_days
    df_tr = df.iloc[:split]
    rows = {}
    for horizon in (1, 24):
        X, y, lagcols = build_features(df, horizon)
        grp, glob = fit_impute(df_tr)
        Xtr = apply_impute(X.iloc[:split], df.iloc[:split], lagcols, grp, glob)
        Xva = apply_impute(X.iloc[split:], df.iloc[split:], lagcols, grp, glob)
        ytr, yva = y.iloc[:split], y.iloc[split:]

        naive_lag = 1 if horizon == 1 else 24
        res = {
            f"naive (lag{naive_lag})": mean_squared_error(yva, y.shift(naive_lag).iloc[split:].fillna(glob)),
            "naive (lag168)":          mean_squared_error(yva, y.shift(168).iloc[split:].fillna(glob)),
        }
        for name, mdl in make_models().items():
            mdl.fit(Xtr, ytr)
            res[name] = mean_squared_error(yva, np.clip(mdl.predict(Xva), 0, None))
        rows[f"{horizon}h"] = res
    return pd.DataFrame(rows).round(1)

comparison = evaluate_holdout(train_df)
print("Validation MSE (last 61 days):")
comparison
""")

md("""## 5. Best model — train on all public data, evaluate on the test set

The selected model (XGBoost) is refit on the **full** public dataset for each
horizon, then applied to the test set loaded from `TEST_PATH`. Lag features for
the test set are built from its own BikeCount history; boundary rows use the
training imputation. Predictions are clipped at 0 (counts are non-negative).

**On June 3rd:** point `TEST_PATH` to the provided test file and re-run — the
two MSE values printed below are the hidden-test results for the presentation.""")

co("""def evaluate_on_test(train_df, test_df):
    grp, glob = fit_impute(train_df)             # imputation stats from public data
    results = {}
    for horizon in (1, 24):
        Xtr, ytr, lagcols = build_features(train_df, horizon)
        Xtr = apply_impute(Xtr, train_df, lagcols, grp, glob)

        Xte, yte, _ = build_features(test_df, horizon)
        Xte = apply_impute(Xte, test_df, lagcols, grp, glob)
        Xte = Xte.reindex(columns=Xtr.columns, fill_value=0)   # align columns defensively

        model = best_model(horizon)
        model.fit(Xtr, ytr)
        pred = np.clip(model.predict(Xte), 0, None)
        results[horizon] = mean_squared_error(yte, pred)
    return results

test_df = load_clean(TEST_PATH)
mse = evaluate_on_test(train_df, test_df)

print("=" * 46)
print(f"  Test MSE  (+1 h horizon) : {mse[1]:12.2f}")
print(f"  Test MSE  (+24 h horizon): {mse[24]:12.2f}")
print("=" * 46)
if TEST_PATH == TRAIN_PATH:
    print("\\n!!! WARNING: these are NOT the real test scores. TEST_PATH still points")
    print("    to the public dataset, so the model is evaluated in-sample (it was")
    print("    trained on the same rows) -> the MSE is optimistically low.")
    print("    On June 3rd set TEST_PATH to the hidden test file and re-run.")
    print("    Realistic out-of-sample expectation ~ the holdout table above")
    print("    (XGBoost: ~3200 for +1h, ~21500 for +24h).")
""")

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
})
with open("bike_count_estimation.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote bike_count_estimation.ipynb with", len(cells), "cells")
