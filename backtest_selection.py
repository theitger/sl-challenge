"""Rolling-origin backtest of the FULL model selection (not just XGBoost tuning).

The submission notebook selects the per-horizon model on a single temporal
holdout (last 61 days). This script re-runs the *same* model family comparison
(Ridge / RandomForest / XGBoost x4 / MLP) under an expanding-window rolling-origin
backtest across the whole year, so we can check:

    does TS-CV pick the SAME per-horizon winner as the single holdout?

Models and feature pipeline are kept verbatim-equivalent to build_notebook.py.
For each horizon we walk an expanding-window origin and score consecutive,
non-overlapping test blocks. Refit per origin, train on everything before it.

Usage: python3 backtest_selection.py [block_days] [step_days]
"""
import sys
import warnings
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
DATA = "challenge_public_dataset.xlsx"
WEATHER_CATS = ["Thunder", "Snow", "Rain", "Fog", "Clear", "Cloudy", "Other"]
HORIZONS = [1, 24]

# backtest geometry (override: python3 backtest_selection.py <block_days> <step_days>)
TEST_BLOCK_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 14
MIN_TRAIN_DAYS = 150          # warm-up before the first origin
STEP_DAYS = int(sys.argv[2]) if len(sys.argv) > 2 else TEST_BLOCK_DAYS


def load_clean(path):
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["BikeCount"]).copy()
    df = df.sort_values(["Month", "Day", "Hour"]).reset_index(drop=True)
    return df


def weather_bucket(value):
    text = str(value).lower()
    if "thunder" in text: return "Thunder"
    if "snow" in text or "ice" in text or "sleet" in text: return "Snow"
    if "rain" in text or "drizzle" in text or "shower" in text: return "Rain"
    if "fog" in text: return "Fog"
    if "sunny" in text or "clear" in text: return "Clear"
    if "cloud" in text or "overcast" in text: return "Cloudy"
    return "Other"


def make_supervised_frame(df, horizon):
    """Verbatim port of build_notebook.py's canonical pipeline."""
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
            raise ValueError(f"target lag {lag} not known for horizon {horizon}")
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
    return X, y


# ---- models: verbatim-equivalent to build_notebook.py make_models() ----
def make_xgboost(horizon):
    params_by_horizon = {
        1: {"n_estimators": 800, "learning_rate": 0.03, "max_depth": 5},
        24: {"n_estimators": 300, "learning_rate": 0.03, "max_depth": 4},
    }
    return XGBRegressor(**params_by_horizon[horizon], subsample=0.8,
                        colsample_bytree=0.8, n_jobs=-1, random_state=RNG)


def make_xgboost_baseline():
    return XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                        subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=RNG)


def log1p_wrap(model):
    return TransformedTargetRegressor(regressor=clone(model), func=np.log1p, inverse_func=np.expm1)


def make_models(horizon):
    mlp = TransformedTargetRegressor(
        regressor=make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(128, 64, 32), alpha=1e-3,
                         learning_rate_init=0.005, max_iter=800, early_stopping=True,
                         n_iter_no_change=20, random_state=RNG),
        ),
        transformer=StandardScaler(),
    )
    baseline = make_xgboost_baseline()
    tuned = make_xgboost(horizon)
    return {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "RandomForest": RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=RNG),
        "XGBoost": baseline,
        "XGBoost log1p": log1p_wrap(baseline),
        "XGBoost tuned": tuned,
        "XGBoost tuned log1p": log1p_wrap(tuned),
        "MLP": mlp,
    }


def mse(y_true, y_pred):
    return mean_squared_error(y_true, np.clip(y_pred, 0, None))


def backtest():
    df = load_clean(DATA)
    block = TEST_BLOCK_DAYS * 24
    min_train = MIN_TRAIN_DAYS * 24
    step = STEP_DAYS * 24

    model_names = list(make_models(1).keys())
    all_rows = []
    for horizon in HORIZONS:
        X, y = make_supervised_frame(df, horizon)
        n = len(X)
        origins = list(range(min_train, n - block + 1, step))
        print(f"\n===== horizon +{horizon}h | supervised rows {n} | "
              f"{len(origins)} test windows of {TEST_BLOCK_DAYS}d =====", flush=True)
        for origin in origins:
            Xtr, ytr = X.iloc[:origin], y.iloc[:origin]
            Xte, yte = X.iloc[origin:origin + block], y.iloc[origin:origin + block]
            row = {"horizon": horizon, "origin": origin}
            for name, model in make_models(horizon).items():
                m = clone(model)
                m.fit(Xtr, ytr)
                row[name] = mse(yte, m.predict(Xte))
            all_rows.append(row)
            print(f"  origin {origin:5d} done", flush=True)
    return pd.DataFrame(all_rows), model_names


def summarize(res, models):
    HOLDOUT_WINNER = {1: "XGBoost tuned log1p", 24: "XGBoost tuned log1p"}  # from notebook
    for horizon in HORIZONS:
        sub = res[res.horizon == horizon]
        print(f"\n########## horizon +{horizon}h  ({len(sub)} windows) ##########")
        mean_mse = {m: round(sub[m].mean(), 1) for m in models}
        median_mse = {m: round(sub[m].median(), 1) for m in models}
        print("mean   MSE :", mean_mse)
        print("median MSE :", median_mse)

        cv_winner_mean = min(mean_mse, key=mean_mse.get)
        cv_winner_median = min(median_mse, key=median_mse.get)
        wins = sub[models].idxmin(axis=1).value_counts().to_dict()
        print("best model per window:", wins)
        print(f"\n  HOLDOUT winner (notebook)      : {HOLDOUT_WINNER[horizon]}")
        print(f"  TS-CV winner (mean MSE)        : {cv_winner_mean}")
        print(f"  TS-CV winner (median MSE)      : {cv_winner_median}")
        agree = (cv_winner_mean == HOLDOUT_WINNER[horizon])
        print(f"  -> holdout and TS-CV agree?     : {'YES' if agree else 'NO'}")


if __name__ == "__main__":
    res, models = backtest()
    res.to_csv("backtest_selection_results.csv", index=False)
    summarize(res, models)
    print("\nwrote backtest_selection_results.csv")
