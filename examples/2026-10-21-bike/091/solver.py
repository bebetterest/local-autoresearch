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
    numeric = [names.index(k) for k in ("temp", "atemp", "hum", "windspeed", "elapsed_days")]

    def trend_matrix(arr):
        day = arr[:, names.index("elapsed_days")]
        time_feats = np.column_stack([
            np.sin(2.0 * np.pi * day / 365.25),
            np.cos(2.0 * np.pi * day / 365.25),
            np.log1p(day),
            day,
            day ** 2,
            day ** 3,
        ])
        return np.column_stack([arr, time_feats])

    trend_x = trend_matrix(x)
    trend_xp = trend_matrix(xp)
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
    res_cat_pos = [res_cols.index(orig) for orig in residual_categorical]
    res_x = x[:, res_cols]
    res_xp = xp[:, res_cols]

    residuals = HistGradientBoostingRegressor(
        max_iter=100,
        categorical_features=res_cat_pos,
        early_stopping=False,
        random_state=seed,
    )
    residuals.fit(res_x, y - trend.predict(trend_x))

    prediction = trend.predict(trend_xp) + residuals.predict(res_xp)
    return np.expm1(np.maximum(0.0, prediction)).tolist()
