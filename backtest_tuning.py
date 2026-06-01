"""Rolling-origin backtest: does PR#2's per-horizon XGBoost tuning actually help?

Compares, on MANY temporal test windows (not the single 61-day holdout the PR
reports), four XGBoost configurations using the notebook's canonical
`make_supervised_frame` feature pipeline:

  baseline  : main's plain XGBoost   (500, lr=0.05, depth=6)
  log1p     : main's declared best   (baseline wrapped in log1p target)
  tuned     : PR#2 per-horizon params (1h: 800/0.03/d5, 24h: 300/0.03/d4)
  tuned+log : PR#2 params + log1p     (to check complementarity)

For each horizon we walk an expanding-window origin and score consecutive,
non-overlapping test blocks. Refit per origin, train on everything before it.
"""
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None

RNG = 0
DATA = "challenge_public_dataset.xlsx"
WEATHER_CATS = ["Thunder", "Snow", "Rain", "Fog", "Clear", "Cloudy", "Other"]
HORIZONS = [1, 24]

# backtest geometry (override: python3 backtest_tuning.py <block_days> <step_days>)
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


def base_xgb(**over):
    p = dict(subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=RNG)
    p.update(over)
    return XGBRegressor(**p)


def make_variants(horizon):
    baseline = base_xgb(n_estimators=500, learning_rate=0.05, max_depth=6)
    tuned_params = {1: dict(n_estimators=800, learning_rate=0.03, max_depth=5),
                    24: dict(n_estimators=300, learning_rate=0.03, max_depth=4)}[horizon]
    tuned = base_xgb(**tuned_params)
    log1p = lambda m: TransformedTargetRegressor(regressor=clone(m), func=np.log1p, inverse_func=np.expm1)
    return {
        "baseline": baseline,
        "log1p": log1p(baseline),
        "tuned": tuned,
        "tuned+log": log1p(tuned),
    }


def mse(y_true, y_pred):
    return mean_squared_error(y_true, np.clip(y_pred, 0, None))


def backtest():
    df = load_clean(DATA)
    block = TEST_BLOCK_DAYS * 24
    min_train = MIN_TRAIN_DAYS * 24
    step = STEP_DAYS * 24

    all_rows = []
    for horizon in HORIZONS:
        X, y = make_supervised_frame(df, horizon)
        n = len(X)
        origins = list(range(min_train, n - block + 1, step))
        print(f"\n===== horizon +{horizon}h | supervised rows {n} | "
              f"{len(origins)} test windows of {TEST_BLOCK_DAYS}d =====")
        variant_names = list(make_variants(horizon).keys())
        for origin in origins:
            Xtr, ytr = X.iloc[:origin], y.iloc[:origin]
            Xte, yte = X.iloc[origin:origin + block], y.iloc[origin:origin + block]
            row = {"horizon": horizon, "origin": origin}
            for name, model in make_variants(horizon).items():
                m = clone(model)
                m.fit(Xtr, ytr)
                row[name] = mse(yte, m.predict(Xte))
            all_rows.append(row)
    return pd.DataFrame(all_rows), ["baseline", "log1p", "tuned", "tuned+log"]


def summarize(res, variants):
    for horizon in HORIZONS:
        sub = res[res.horizon == horizon]
        print(f"\n########## horizon +{horizon}h  ({len(sub)} windows) ##########")
        print("mean   MSE :", {v: round(sub[v].mean(), 1) for v in variants})
        print("median MSE :", {v: round(sub[v].median(), 1) for v in variants})
        ref = "baseline"
        print(f"\n  paired comparison vs '{ref}' (per window):")
        for v in variants:
            if v == ref:
                continue
            diff = sub[v] - sub[ref]
            wins = int((diff < 0).sum())
            print(f"    {v:10s}: beats baseline in {wins}/{len(sub)} windows | "
                  f"mean ΔMSE {diff.mean():+9.1f} | median ΔMSE {diff.median():+9.1f}")
        # tuned vs log1p head-to-head (the real PR decision)
        diff = sub["tuned"] - sub["log1p"]
        wins = int((diff < 0).sum())
        print(f"\n  tuned vs log1p (main's current best): tuned wins {wins}/{len(sub)} | "
              f"mean ΔMSE {diff.mean():+9.1f}")
        # DECISIVE comparison: tuned+log (proposed merge) vs log1p (current main)
        d = sub["tuned+log"] - sub["log1p"]
        wins = int((d < 0).sum())
        print(f"\n  >>> tuned+log1p (proposed) vs log1p (current main):")
        print(f"      wins {wins}/{len(sub)} windows | mean ΔMSE {d.mean():+9.1f} | "
              f"median ΔMSE {d.median():+9.1f}")
        if wilcoxon is not None and len(sub) >= 6:
            for label, dd in [("tuned vs baseline", sub["tuned"] - sub["baseline"]),
                              ("tuned+log vs log1p", d)]:
                try:
                    _, p = wilcoxon(dd.values)
                    print(f"      Wilcoxon {label}: p={p:.4f} "
                          f"({'SIGNIFICANT' if p < 0.05 else 'not sig.'} @0.05)")
                except ValueError as e:
                    print(f"      Wilcoxon {label}: n/a ({e})")
        # best variant per window distribution
        best = sub[variants].idxmin(axis=1).value_counts().to_dict()
        print("  best variant per window:", best)


if __name__ == "__main__":
    res, variants = backtest()
    res.to_csv("backtest_results.csv", index=False)
    summarize(res, variants)
    print("\nwrote backtest_results.csv")
