"""Regenerate submission/model_group18.joblib with the CV-selected +1h model.

The instructor's submission is next-hour (+1h) only: one saved model + an
evaluation notebook. Rolling-origin time-series CV selects `XGBoost tuned`
(no log1p) as the most robust +1h model; the previously saved bundle held
`XGBoost tuned log1p`, which is the weakest XGBoost variant across the year.

Feature pipeline is verbatim-equivalent to build_notebook.py / the submission
notebook for horizon 1, so the saved feature_columns line up on load.
"""
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor

DATA = "challenge_public_dataset.xlsx"
OUT = "submission/model_group18.joblib"
HORIZON = 1
RNG = 0
WEATHER_CATS = ["Thunder", "Snow", "Rain", "Fog", "Clear", "Cloudy", "Other"]


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


def make_supervised_frame(df, horizon=HORIZON):
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

    lag_cols = []
    for lag in [1, 2, 3, 24, 168]:
        col = f"bike_count_target_minus_{lag}"
        d[col] = target_lag(lag)
        lag_cols.append(col)
    d["rolling_24h_mean"] = target_lag(horizon).rolling(24).mean()
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
    return supervised.drop(columns=["target"]), supervised["target"]


def main():
    df = load_clean(DATA)
    X, y = make_supervised_frame(df, HORIZON)

    # CV-selected +1h model: XGBoost tuned (no log1p).
    model = XGBRegressor(n_estimators=800, learning_rate=0.03, max_depth=5,
                         subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=RNG)
    model.fit(X, y)

    bundle = {
        "model": model,
        "feature_columns": list(X.columns),
        "horizon": HORIZON,
        "model_name": "XGBoost tuned",
        "training_rows": int(len(X)),
    }
    joblib.dump(bundle, OUT)
    print(f"wrote {OUT}")
    print(f"  model_name    : {bundle['model_name']}")
    print(f"  horizon       : {bundle['horizon']}")
    print(f"  training_rows : {bundle['training_rows']}")
    print(f"  n_features    : {len(bundle['feature_columns'])}")
    print(f"  feature_columns: {bundle['feature_columns']}")


if __name__ == "__main__":
    main()
