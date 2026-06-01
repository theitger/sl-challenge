"""Generate the submission notebook bike_count_estimation.ipynb."""
import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


cells = []


def md(source):
    cells.append(new_markdown_cell(source))


def co(source):
    cells.append(new_code_cell(source))


md("""# Bike Count Estimation - Supervised Learning Challenge

This notebook predicts hourly bike counts in Muenster for two direct forecast
horizons:

| Horizon | Meaning |
|---|---|
| `+1h` | predict `BikeCount(t+1)` from information available at time `t` |
| `+24h` | predict `BikeCount(t+24)` from information available at time `t` |

So the 24-hour task is **not** the sum over the next 24 hours. It is the bike
count of the hour 24 hours in the future.

We compare one linear model, tree-based models, and one neural network. The
metric is mean squared error (MSE).""")


md("""## 1. Imports & setup""")

co("""import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

RNG = 0
HORIZONS = [1, 24]
VALIDATION_DAYS = 61
""")


md("""## 2. Input paths & configuration

On June 3rd, set `TEST_PATH` to the hidden test file and run the final section.
Until then it stays `None`, so the notebook cannot accidentally print
in-sample test scores from the public training data.""")

co("""TRAIN_PATH = "challenge_public_dataset.xlsx"
TEST_PATH = None  # Example on June 3rd: "challenge_hidden_test_dataset.xlsx"

REQUIRED_COLS = [
    "Month", "Day", "Hour", "Weekday", "Weather",
    "Temperature (°C)", "Humidity (%)", "Rain (mm)", "Wind (km/h)", "BikeCount",
]
WEATHER_CATS = ["Thunder", "Snow", "Rain", "Fog", "Clear", "Cloudy", "Other"]
""")


md("""## 3. Load and validate data""")

co("""def load_clean(path):
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()

    missing = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns {missing}. Got {list(df.columns)}")

    # The public data contains one corrupt row with NaN BikeCount and sensor values.
    df = df.dropna(subset=["BikeCount"]).copy()
    df = df.sort_values(["Month", "Day", "Hour"]).reset_index(drop=True)

    duplicate_count = df.duplicated(["Month", "Day", "Hour"]).sum()
    if duplicate_count:
        raise ValueError(f"Found {duplicate_count} duplicated timestamps after cleaning.")

    return df


train_df = load_clean(TRAIN_PATH)
print(f"Clean public rows: {len(train_df)}")
print(f"Date range: {train_df.iloc[0][['Month', 'Day', 'Hour']].to_dict()} -> "
      f"{train_df.iloc[-1][['Month', 'Day', 'Hour']].to_dict()}")
train_df.head()
""")


md("""## 4. Feature engineering

The supervised target is explicit: for horizon `h`, the target is
`BikeCount.shift(-h)`, i.e. `BikeCount(t+h)`.

Calendar and weather features describe the target hour `t+h`, because those
columns are available for each row in the challenge file. BikeCount lag
features are restricted to values known at forecast origin `t` or earlier.""")

co("""def weather_bucket(value):
    text = str(value).lower()
    if "thunder" in text:
        return "Thunder"
    if "snow" in text or "ice" in text or "sleet" in text:
        return "Snow"
    if "rain" in text or "drizzle" in text or "shower" in text:
        return "Rain"
    if "fog" in text:
        return "Fog"
    if "sunny" in text or "clear" in text:
        return "Clear"
    if "cloud" in text or "overcast" in text:
        return "Cloudy"
    return "Other"


def make_supervised_frame(df, horizon):
    d = df.copy()

    d["target"] = d["BikeCount"].shift(-horizon)

    d["target_hour"] = d["Hour"].shift(-horizon)
    d["target_weekday"] = d["Weekday"].shift(-horizon)
    d["target_month"] = d["Month"].shift(-horizon)
    d["target_temperature"] = d["Temperature (°C)"].shift(-horizon)
    d["target_humidity"] = d["Humidity (%)"].shift(-horizon)
    d["target_rain"] = d["Rain (mm)"].shift(-horizon)
    d["target_wind"] = d["Wind (km/h)"].shift(-horizon)
    d["target_weather"] = d["Weather"].shift(-horizon)

    d["target_is_weekend"] = d["target_weekday"].isin([5, 6]).astype(int)
    d["target_hour_sin"] = np.sin(2 * np.pi * d["target_hour"] / 24)
    d["target_hour_cos"] = np.cos(2 * np.pi * d["target_hour"] / 24)
    d["target_month_sin"] = np.sin(2 * np.pi * d["target_month"] / 12)
    d["target_month_cos"] = np.cos(2 * np.pi * d["target_month"] / 12)

    weather = pd.Categorical(d["target_weather"].map(weather_bucket), categories=WEATHER_CATS)
    weather_dummies = pd.get_dummies(weather, prefix="weather").astype(int)

    def target_lag(lag):
        shift = lag - horizon
        if shift < 0:
            raise ValueError(f"target lag {lag} is not known for horizon {horizon}")
        return d["BikeCount"].shift(shift)

    target_lags = [1, 2, 3, 24, 168] if horizon == 1 else [24, 25, 48, 168]
    lag_cols = []
    for lag in target_lags:
        col = f"bike_count_target_minus_{lag}"
        d[col] = target_lag(lag)
        lag_cols.append(col)

    d["rolling_24h_mean"] = target_lag(horizon).rolling(24).mean()
    rolling_cols = ["rolling_24h_mean"]
    if horizon == 1:
        d["rolling_3h_mean"] = target_lag(horizon).rolling(3).mean()
        rolling_cols = ["rolling_3h_mean", "rolling_24h_mean"]

    feature_cols = [
        "target_hour", "target_weekday", "target_month", "target_is_weekend",
        "target_hour_sin", "target_hour_cos", "target_month_sin", "target_month_cos",
        "target_temperature", "target_humidity", "target_rain", "target_wind",
        *lag_cols, *rolling_cols,
    ]

    supervised = pd.concat([d[feature_cols + ["target"]], weather_dummies], axis=1)
    supervised = supervised.dropna().reset_index(drop=True)

    X = supervised.drop(columns=["target"])
    y = supervised["target"]
    return X, y, supervised


for horizon in HORIZONS:
    X_check, y_check, _ = make_supervised_frame(train_df, horizon)
    print(f"+{horizon}h supervised rows: {len(X_check)} | features: {X_check.shape[1]}")
""")


md("""## 5. Models

We compare a few XGBoost configurations explicitly instead of hard-coding one:

- **XGBoost**: a single default configuration for both horizons.
- **XGBoost tuned**: hyper-parameters selected separately per forecast horizon
  on the temporal holdout. The +1h horizon benefits from more, slower boosting
  rounds; the +24h horizon from shallower trees.
- the **`log1p`** variants train on `log(1 + BikeCount)` and transform
  predictions back with `expm1`, which dampens the influence of very large
  count peaks.

The validation step below picks the best of all candidates per horizon, so the
choice between tuned / default / log1p is data-driven, not assumed.""")

co("""def make_xgboost(horizon):
    params_by_horizon = {
        1: {"n_estimators": 800, "learning_rate": 0.03, "max_depth": 5},
        24: {"n_estimators": 300, "learning_rate": 0.03, "max_depth": 4},
    }
    return XGBRegressor(
        **params_by_horizon[horizon],
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=RNG,
    )


def make_xgboost_baseline():
    return XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=RNG,
    )


def log1p_wrap(model):
    return TransformedTargetRegressor(
        regressor=clone(model),
        func=np.log1p,
        inverse_func=np.expm1,
    )


def make_models(horizon):
    mlp = TransformedTargetRegressor(
        regressor=make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                alpha=1e-3,
                learning_rate_init=0.005,
                max_iter=800,
                early_stopping=True,
                n_iter_no_change=20,
                random_state=RNG,
            ),
        ),
        transformer=StandardScaler(),
    )

    baseline = make_xgboost_baseline()
    tuned = make_xgboost(horizon)
    return {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            n_jobs=-1,
            random_state=RNG,
        ),
        "XGBoost": baseline,
        "XGBoost log1p": log1p_wrap(baseline),
        "XGBoost tuned": tuned,
        "XGBoost tuned log1p": log1p_wrap(tuned),
        "MLP": mlp,
    }


def mse_rmse(y_true, y_pred):
    pred = np.clip(y_pred, 0, None)
    mse = mean_squared_error(y_true, pred)
    return mse, np.sqrt(mse)
""")


md("""## 6. Validation results

We use a temporal holdout: the last 61 days are validation data. A random split
would mix future and past observations and is inappropriate for this time
series task.""")

co("""def evaluate_validation(df, validation_days=VALIDATION_DAYS):
    rows = []
    best_models = {}

    for horizon in HORIZONS:
        X, y, supervised = make_supervised_frame(df, horizon)
        split = len(X) - validation_days * 24
        if split <= 0:
            raise ValueError("Validation window is larger than the supervised dataset.")

        X_train, X_val = X.iloc[:split], X.iloc[split:]
        y_train, y_val = y.iloc[:split], y.iloc[split:]
        frame_val = supervised.iloc[split:]

        baselines = {
            f"naive target lag {horizon}": frame_val[f"bike_count_target_minus_{horizon}"],
            "naive target lag 168": frame_val["bike_count_target_minus_168"],
        }
        for name, pred in baselines.items():
            mse, rmse = mse_rmse(y_val, pred)
            rows.append({"Model": name, "Horizon": f"+{horizon}h", "MSE": mse, "RMSE": rmse})

        fitted = {}
        for name, model in make_models(horizon).items():
            candidate = clone(model)
            candidate.fit(X_train, y_train)
            mse, rmse = mse_rmse(y_val, candidate.predict(X_val))
            rows.append({"Model": name, "Horizon": f"+{horizon}h", "MSE": mse, "RMSE": rmse})
            fitted[name] = (candidate, mse)

        best_name = min(fitted, key=lambda model_name: fitted[model_name][1])
        best_models[horizon] = best_name

    results = pd.DataFrame(rows).sort_values(["Horizon", "MSE"]).reset_index(drop=True)
    return results, best_models


validation_results, best_model_by_horizon = evaluate_validation(train_df)
display(validation_results.round({"MSE": 1, "RMSE": 1}))
print("Best model by horizon:", best_model_by_horizon)
""")


md("""## 7. Hidden test evaluation

On June 3rd, set `TEST_PATH` above and run this section. The hidden file is
expected to have the same columns as the public dataset, including `BikeCount`,
so the final MSE/RMSE can be computed locally.""")

co("""def train_final_models(train_df, best_model_by_horizon):
    final_models = {}
    final_columns = {}

    for horizon, model_name in best_model_by_horizon.items():
        X_train, y_train, _ = make_supervised_frame(train_df, horizon)
        model = clone(make_models(horizon)[model_name])
        model.fit(X_train, y_train)
        final_models[horizon] = (model_name, model)
        final_columns[horizon] = X_train.columns

    return final_models, final_columns


def evaluate_hidden_test(train_df, test_path, best_model_by_horizon):
    test_df = load_clean(test_path)
    final_models, final_columns = train_final_models(train_df, best_model_by_horizon)

    rows = []
    predictions = {}
    for horizon in HORIZONS:
        X_test, y_test, _ = make_supervised_frame(test_df, horizon)
        X_test = X_test.reindex(columns=final_columns[horizon], fill_value=0)

        model_name, model = final_models[horizon]
        pred = np.clip(model.predict(X_test), 0, None)
        predictions[horizon] = pred

        mse, rmse = mse_rmse(y_test, pred)
        rows.append({"Model": model_name, "Horizon": f"+{horizon}h", "MSE": mse, "RMSE": rmse})

    return pd.DataFrame(rows), predictions


if TEST_PATH is None:
    print("TEST_PATH is not set. Set it on June 3rd to evaluate the hidden test dataset.")
else:
    hidden_results, hidden_predictions = evaluate_hidden_test(
        train_df,
        TEST_PATH,
        best_model_by_horizon,
    )
    display(hidden_results.round({"MSE": 2, "RMSE": 2}))
""")


nb = new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    },
)

with open("bike_count_estimation.ipynb", "w") as f:
    nbf.write(nb, f)

print("wrote bike_count_estimation.ipynb with", len(cells), "cells")
