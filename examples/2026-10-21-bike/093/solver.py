import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import TweedieRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def solve(problem, seed, budget):
    x = np.asarray(problem["train_x"], dtype=float)
    xp = np.asarray(problem["predict_x"], dtype=float)
    y = np.log1p(np.asarray(problem["train_y"], dtype=float))
    names = list(problem["features"])

    def time_features(arr):
        day = arr[:, names.index("elapsed_days")]
        return np.column_stack([
            np.sin(2.0 * np.pi * day / 365.25),
            np.cos(2.0 * np.pi * day / 365.25),
            np.log1p(day),
            day,
            day ** 2,
            np.exp(day / 365.25),
        ])

    trend_categorical = [names.index(k) for k in (
        "season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit", "weekend",
    )]
    numeric = [names.index(k) for k in ("temp", "atemp", "hum", "windspeed", "elapsed_days")]

    trend_x = np.column_stack([x, time_features(x)])
    trend_xp = np.column_stack([xp, time_features(xp)])
    n_orig = x.shape[1]
    numeric += [n_orig, n_orig + 1, n_orig + 2, n_orig + 3, n_orig + 4, n_orig + 5]

    trend = make_pipeline(
        ColumnTransformer([
            ("calendar", OneHotEncoder(handle_unknown="ignore", sparse_output=False), trend_categorical),
            ("numeric", StandardScaler(), numeric),
        ]),
        TweedieRegressor(power=1.5, alpha=1.0, max_iter=1000),
    )
    trend.fit(trend_x, y)

    res_cols = [names.index(k) for k in (
        "temp", "atemp", "hum", "windspeed", "hr", "holiday", "weekday", "workingday", "weathersit", "weekend",
    )]
    base = x[:, res_cols].copy()
    res_time = time_features(x)
    res_x = np.column_stack([base, res_time])
    res_xp = np.column_stack([xp[:, res_cols], time_features(xp)])

    cat_pos = [4, 5, 6, 7, 8, 9]
    residuals = HistGradientBoostingRegressor(
        max_iter=100,
        categorical_features=cat_pos,
        early_stopping=False,
        random_state=seed,
    )
    residuals.fit(res_x, y - trend.predict(trend_x))
    prediction = trend.predict(trend_xp) + residuals.predict(res_xp)
    return np.expm1(np.maximum(0.0, prediction)).tolist()
