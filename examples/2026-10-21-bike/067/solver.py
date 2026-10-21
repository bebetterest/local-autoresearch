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
    y = np.log1p(problem["train_y"])
    names = problem["features"]

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

    def add_trend_features(arr):
        day = arr[:, names.index("elapsed_days")]
        return np.column_stack([
            arr,
            np.sin(2 * np.pi * day / 365.25),
            np.cos(2 * np.pi * day / 365.25),
            np.log1p(day),
            day ** 2,
            day / 365.25,
        ])

    trend_x = add_trend_features(x)
    trend_xp = add_trend_features(xp)
    numeric += [len(names), len(names) + 1, len(names) + 2, len(names) + 3, len(names) + 4]

    trend = make_pipeline(
        ColumnTransformer(
            [
                ("calendar", OneHotEncoder(handle_unknown="ignore", sparse_output=False), trend_categorical),
                ("numeric", StandardScaler(), numeric),
            ]
        ),
        TweedieRegressor(power=1.5, alpha=1.0, max_iter=1000),
    )
    trend.fit(trend_x, y)

    res_cols = [names.index(k) for k in (
        "temp", "atemp", "hum", "windspeed", "hr", "holiday", "weekday", "workingday", "weathersit", "weekend",
    )]
    res_cat_pos = [res_cols.index(orig) for orig in residual_categorical]
    res_x = x[:, res_cols]
    res_xp = xp[:, res_cols]

    # One substantive change: add elapsed_days to the residual model
    elapsed_idx = names.index("elapsed_days")
    elapsed_col = x[:, [elapsed_idx]]
    elapsed_colp = xp[:, [elapsed_idx]]
    res_x = np.column_stack([res_x, elapsed_col])
    res_xp = np.column_stack([res_xp, elapsed_colp])

    residuals = HistGradientBoostingRegressor(
        max_iter=100,
        categorical_features=res_cat_pos,
        early_stopping=False,
        random_state=seed,
    )
    residuals.fit(res_x, y - trend.predict(trend_x))

    log_pred = trend.predict(trend_xp) + residuals.predict(res_xp)
    pred = np.expm1(np.maximum(0, log_pred))
    return pred.tolist()
