"""Fit Tweedie time/calendar trend plus boosted residuals on weather/calendar only."""

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
    residual_numeric = [
        names.index(k) for k in ("temp", "atemp", "hum", "windspeed", "elapsed_days")
    ]

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
    numeric = [names.index(k) for k in ("temp", "atemp", "hum", "windspeed", "elapsed_days")]
    trend_numeric = numeric + [len(names), len(names) + 1, len(names) + 2, len(names) + 3]

    trend = make_pipeline(
        ColumnTransformer(
            [
                (
                    "calendar",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    trend_categorical,
                ),
                ("numeric", StandardScaler(), trend_numeric),
            ]
        ),
        TweedieRegressor(power=1.5, alpha=1.0, max_iter=1000),
    )
    trend.fit(trend_x, y)

    res_cols = residual_numeric + residual_categorical
    res_x = x[:, res_cols]
    res_xp = xp[:, res_cols]
    res_cat_pos = list(range(len(residual_numeric), len(res_cols)))

    residuals = HistGradientBoostingRegressor(
        max_iter=100,
        categorical_features=res_cat_pos,
        early_stopping=False,
        random_state=seed,
    )
    residuals.fit(res_x, y - trend.predict(trend_x))

    prediction = trend.predict(trend_xp) + residuals.predict(res_xp)
    return np.expm1(np.maximum(0, prediction)).tolist()
