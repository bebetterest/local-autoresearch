import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def solve(problem, seed, budget):
    x = np.asarray(problem["train_x"])
    xp = np.asarray(problem["predict_x"])
    y = np.log1p(problem["train_y"])
    names = problem["features"]

    temp_idx = names.index("temp")
    atemp_idx = names.index("atemp")
    hum_idx = names.index("hum")
    wind_idx = names.index("windspeed")
    elapsed_idx = names.index("elapsed_days")

    def add_time_features(values):
        day = values[:, elapsed_idx]
        sinv = np.sin(2 * np.pi * day / 365.25)
        cosv = np.cos(2 * np.pi * day / 365.25)
        sq = day ** 2
        return np.column_stack((values, sinv, cosv, sq))

    trend_x = add_time_features(x)
    trend_xp = add_time_features(xp)
    sin_idx = x.shape[1]
    cos_idx = sin_idx + 1
    sq_idx = sin_idx + 2
    numeric = [temp_idx, atemp_idx, hum_idx, wind_idx, elapsed_idx, sin_idx, cos_idx, sq_idx]

    trend_categorical = [
        names.index(k) for k in (
            "season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit", "weekend",
        )
    ]
    residual_categorical = [
        names.index(k) for k in (
            "hr", "holiday", "weekday", "workingday", "weathersit", "weekend",
        )
    ]

    trend = make_pipeline(
        ColumnTransformer(
            [
                ("calendar", OneHotEncoder(handle_unknown="ignore", sparse_output=False), trend_categorical),
                ("numeric", StandardScaler(), numeric),
            ]
        ),
        Ridge(alpha=1.0),
    )
    trend.fit(trend_x, y)

    residuals = HistGradientBoostingRegressor(
        max_iter=100,
        categorical_features=residual_categorical,
        early_stopping=False,
        random_state=seed,
    )
    residuals.fit(x, y - trend.predict(trend_x))

    prediction = trend.predict(trend_xp) + residuals.predict(xp)
    return np.expm1(np.maximum(0, prediction)).tolist()
