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
    x = np.asarray(problem["train_x"])
    xp = np.asarray(problem["predict_x"])
    y = np.log1p(np.asarray(problem["train_y"]))
    names = list(problem["features"])
    elapsed_idx = names.index("elapsed_days")

    trend_categorical = [names.index(k) for k in (
        "season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit", "weekend",
    )]

    def trend_matrix(arr):
        day = arr[:, elapsed_idx]
        sinv = np.sin(2 * np.pi * day / 365.25)
        cosv = np.cos(2 * np.pi * day / 365.25)
        logv = np.log1p(day)
        sq = day ** 2
        cube = day ** 3
        return np.column_stack((arr, sinv, cosv, logv, sq, cube))

    trend_x = trend_matrix(x)
    trend_xp = trend_matrix(xp)
    orig_n = x.shape[1]
    numeric = [
        names.index(k) for k in ("temp", "atemp", "hum", "windspeed", "elapsed_days")
    ] + [orig_n, orig_n + 1, orig_n + 2, orig_n + 3, orig_n + 4]

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
    res_x = x[:, res_cols]
    res_xp = xp[:, res_cols]
    res_cat_pos = [4, 5, 6, 7, 8, 9]

    residual = HistGradientBoostingRegressor(
        max_iter=100,
        categorical_features=res_cat_pos,
        early_stopping=False,
        random_state=seed,
    )
    residual.fit(res_x, y - trend.predict(trend_x))

    prediction = trend.predict(trend_xp) + residual.predict(res_xp)
    return np.expm1(np.maximum(0, prediction)).tolist()
