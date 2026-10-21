import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def solve(problem, seed, budget):
    x = np.asarray(problem["train_x"], dtype=float)
    xp = np.asarray(problem["predict_x"], dtype=float)
    y = np.log1p(np.asarray(problem["train_y"], dtype=float))
    names = list(problem["features"])
    n_names = len(names)

    trend_categorical = [
        names.index(k) for k in (
            "season", "yr", "mnth", "hr", "holiday", "weekday", "workingday",
            "weathersit", "weekend",
        )
    ]
    residual_categorical = [
        names.index(k) for k in (
            "hr", "holiday", "weekday", "workingday", "weathersit", "weekend",
        )
    ]
    numeric = [names.index(k) for k in ("temp", "atemp", "hum", "windspeed", "elapsed_days")]

    elapsed_idx = names.index("elapsed_days")
    hr_idx = names.index("hr")

    def build_trend_inputs(values):
        day = values[:, elapsed_idx]
        hr = values[:, hr_idx]
        total_hours = day * 24 + hr
        lag24_idx = np.clip(np.floor(total_hours).astype(int) - 24, 0, len(y) - 1)
        lag168_idx = np.clip(np.floor(total_hours).astype(int) - 168, 0, len(y) - 1)
        lag24 = y[lag24_idx]
        lag168 = y[lag168_idx]

        arr = np.column_stack((
            values,
            np.sin(2 * np.pi * day / 365.25),
            np.cos(2 * np.pi * day / 365.25),
            day ** 2,
            lag24,
            lag168,
        ))
        numeric_cols = numeric + [n_names, n_names + 1, n_names + 2, n_names + 3, n_names + 4]
        return arr, numeric_cols

    trend_x, trend_numeric = build_trend_inputs(x)
    trend_xp, _ = build_trend_inputs(xp)

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
        PoissonRegressor(alpha=1.0, max_iter=1000),
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
