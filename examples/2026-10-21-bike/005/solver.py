"""Fit quadratic time trend plus boosted residuals."""

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

    trend_categorical = [
        names.index(k)
        for k in (
            "season",
            "yr",
            "mnth",
            "hr",
            "holiday",
            "weekday",
            "workingday",
            "weathersit",
            "weekend",
        )
    ]
    residual_categorical = [
        names.index(k)
        for k in (
            "hr",
            "holiday",
            "weekday",
            "workingday",
            "weathersit",
            "weekend",
        )
    ]
    numeric = [names.index(k) for k in ("temp", "atemp", "hum", "windspeed", "elapsed_days")]

    n_orig = x.shape[1]
    elapsed_idx = names.index("elapsed_days")

    def trend_inputs(values):
        day = values[:, elapsed_idx].astype(float)
        return np.column_stack(
            (
                values,
                np.sin(2 * np.pi * day / 365.25),
                np.cos(2 * np.pi * day / 365.25),
                day ** 2,
            )
        )

    trend_x, trend_xp = trend_inputs(x), trend_inputs(xp)
    sin_idx = n_orig
    cos_idx = n_orig + 1
    quad_idx = n_orig + 2
    numeric += [sin_idx, cos_idx, quad_idx]

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
