"""
Bike Count Estimation — validation prototype.
Goal: get real holdout-MSE numbers for all required model families and both
forecast horizons before polishing the submission notebook.

Forecast framing (confirmed): direct multi-step.
  - 1h model  : may use lags >= 1   (lag1..lag168)
  - 24h model : may use lags >= 24  (no lag < 24h available at forecast origin)
Each row carries its own calendar + weather columns; horizons differ ONLY in
which lag features are permitted.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

RNG = 0
DATA = "challenge_public_dataset.xlsx"

# ---------------------------------------------------------------- load + clean
def load_clean(path):
    df = pd.read_excel(path)
    # the single all-NaN row is also the only duplicate timestamp -> drop it
    df = df.dropna(subset=["BikeCount"]).copy()
    df = df.sort_values(["Month", "Day", "Hour"]).reset_index(drop=True)
    return df

# ---------------------------------------------------------------- weather bucket
def weather_bucket(s):
    s = str(s).lower()
    if "thunder" in s:                                   return "Thunder"
    if "snow" in s or "ice" in s or "sleet" in s:        return "Snow"
    if "rain" in s or "drizzle" in s or "shower" in s:   return "Rain"
    if "fog" in s:                                       return "Fog"
    if "sunny" in s or "clear" in s:                     return "Clear"
    if "cloud" in s or "overcast" in s:                  return "Cloudy"
    return "Other"

WEATHER_CATS = ["Thunder", "Snow", "Rain", "Fog", "Clear", "Cloudy", "Other"]

# ---------------------------------------------------------------- features
def build_features(df, horizon):
    """Return (X, y) for a given horizon. df must be cleaned + sorted."""
    d = df.copy()
    d["is_weekend"] = d["Weekday"].isin([5, 6]).astype(int)
    d["hour_sin"] = np.sin(2 * np.pi * d["Hour"] / 24)
    d["hour_cos"] = np.cos(2 * np.pi * d["Hour"] / 24)
    d["month_sin"] = np.sin(2 * np.pi * d["Month"] / 12)
    d["month_cos"] = np.cos(2 * np.pi * d["Month"] / 12)

    # weather one-hot on stable buckets (robust to unseen test strings)
    d["wbucket"] = d["Weather"].map(weather_bucket)
    d["wbucket"] = pd.Categorical(d["wbucket"], categories=WEATHER_CATS)
    wdum = pd.get_dummies(d["wbucket"], prefix="w").astype(int)

    bc = d["BikeCount"]
    if horizon == 1:
        lags = [1, 2, 3, 24, 168]
    else:  # 24h: nothing within the previous 24h is known
        lags = [24, 25, 48, 168]
    for L in lags:
        d[f"lag{L}"] = bc.shift(L)
    if horizon == 1:
        d["roll3"] = bc.shift(1).rolling(3).mean()
        d["roll24"] = bc.shift(1).rolling(24).mean()
    else:
        d["roll24"] = bc.shift(24).rolling(24).mean()

    base = ["Hour", "Weekday", "Month", "is_weekend",
            "hour_sin", "hour_cos", "month_sin", "month_cos",
            "Temperature (°C)", "Humidity (%)", "Rain (mm)", "Wind (km/h)"]
    lagcols = [f"lag{L}" for L in lags] + (["roll3", "roll24"] if horizon == 1 else ["roll24"])
    X = pd.concat([d[base], wdum, d[lagcols]], axis=1)
    y = bc
    return X, y, lagcols

# ---------------------------------------------------------------- impute boundary lags
def fit_impute(df_train, lagcols):
    """Per (Hour,is_weekend) mean BikeCount on TRAIN, for filling NaN lags."""
    tmp = df_train.copy()
    tmp["is_weekend"] = tmp["Weekday"].isin([5, 6]).astype(int)
    grp = tmp.groupby(["Hour", "is_weekend"])["BikeCount"].mean()
    glob = tmp["BikeCount"].mean()
    return grp, glob

def apply_impute(X, df_rows, lagcols, grp, glob):
    X = X.copy()
    key = list(zip(df_rows["Hour"], df_rows["Weekday"].isin([5, 6]).astype(int)))
    fill = pd.Series([grp.get(k, glob) for k in key], index=X.index)
    for c in lagcols:
        X[c] = X[c].fillna(fill)
    return X

# ---------------------------------------------------------------- models
def models():
    return {
        "Ridge":        make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "RandomForest": RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=RNG),
        "XGBoost":      XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                                     subsample=0.8, colsample_bytree=0.8,
                                     n_jobs=-1, random_state=RNG),
        "MLP":          make_pipeline(StandardScaler(),
                                      MLPRegressor(hidden_layer_sizes=(128, 64),
                                                   max_iter=400, early_stopping=True,
                                                   random_state=RNG)),
    }

# ---------------------------------------------------------------- run
def main():
    df = load_clean(DATA)
    n = len(df)
    # temporal holdout: last ~2 months (~1440h) as validation
    split = n - 24 * 61
    df_tr, df_va = df.iloc[:split], df.iloc[split:]
    print(f"rows: {n} | train: {len(df_tr)} | val: {len(df_va)} "
          f"(last {len(df_va)/24:.0f} days)\n")

    for horizon in (1, 24):
        X, y, lagcols = build_features(df, horizon)
        Xtr, ytr = X.iloc[:split], y.iloc[:split]
        Xva, yva = X.iloc[split:], y.iloc[split:]
        grp, glob = fit_impute(df_tr, lagcols)
        Xtr = apply_impute(Xtr, df_tr, lagcols, grp, glob)
        Xva = apply_impute(Xva, df_va, lagcols, grp, glob)

        # naive baselines
        naive_lag = 1 if horizon == 1 else 24
        naive_pred = df[f"_naive"] = y.shift(naive_lag)
        nb = mean_squared_error(yva, y.shift(naive_lag).iloc[split:].fillna(glob))
        seas = mean_squared_error(yva, y.shift(168).iloc[split:].fillna(glob))

        print(f"===== horizon = {horizon}h  (features: {list(X.columns)[-len(lagcols):]}) =====")
        print(f"  baseline persistence(lag{naive_lag}) MSE : {nb:10.1f}")
        print(f"  baseline seasonal(lag168)      MSE : {seas:10.1f}")
        for name, mdl in models().items():
            mdl.fit(Xtr, ytr)
            pred = mdl.predict(Xva)
            mse = mean_squared_error(yva, pred)
            print(f"  {name:13s} MSE : {mse:10.1f}   (RMSE {np.sqrt(mse):7.1f})")
        print()

if __name__ == "__main__":
    main()
