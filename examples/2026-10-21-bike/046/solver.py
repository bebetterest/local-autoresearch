import os
os.environ["OMP_NUM_THREADS"] = "1"

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

    trend_inputs = []
    for values in (x, xp):
        day = values[:, names.index("elapsed_days")]
        trend_inputs.append(
            np.column_stack(
                (
                    values,
                    np.sin(2 * np.pi * day / 365.25),
                    np.cos(2 * np.pi * day / 365.25),
                    np.log1p(day),
                    day ** 2,
                )
            )
        )
    trend_x, trend_xp = trend_inputs
    numeric += [len(names), len(names) + 1, len(names) + 2, len(names) + 3]

    trend = make_pipeline(
        ColumnTransformer(
            [
                (
                    "calendar",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    trend_categorical,
                ),
                ("numeric", StandardScaler(), numeric),
            ]
        ),
        TweedieRegressor(power=1.5, alpha=1.0, max_iter=1000),
    )
    trend.fit(trend_x, y)

    res_cols = [names.index(k) for k in (
        "temp", "atemp", "hum", "windspeed", "hr", "holiday", "weekday", "workingday", "weathersit", "weekend", "elapsed_days",
    )]
    res_cat_pos = [res_cols.index(c) for c in residual_categorical]
    res_x = x[:, res_cols]
    res_xp = xp[:, res_cols]

    residuals = y - trend.predict(trend_x)
    residual_model = HistGradientBoostingRegressor(
        max_iter=100,
        categorical_features=res_cat_pos,
        early_stopping=False,
        random_state=seed,
    )
    residual_model.fit(res_x, residuals)

    prediction = trend.predict(trend_xp) + residual_model.predict(res_xp)
    return np.expm1(np.maximum(0, prediction)).tolist()
