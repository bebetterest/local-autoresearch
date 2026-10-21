import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

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

    trend_categorical = [names.index(k) for k in (
        "season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit", "weekend",
    )]
    residual_categorical = [
        "hr", "holiday", "weekday", "workingday", "weathersit", "weekend",
    ]
    numeric = [names.index(k) for k in ("temp", "atemp", "hum", "windspeed", "elapsed_days")]

    trend_x = x
    trend_xp = xp
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

    res_names = ["temp", "atemp", "hum", "windspeed", "hr", "holiday", "weekday", "workingday", "weathersit", "weekend"]
    res_cols = [names.index(k) for k in res_names]
    res_base = x[:, res_cols]
    res_basep = xp[:, res_cols]

    elapsed_idx = names.index("elapsed_days")
    year_idx = names.index("yr")

    def add_time(arr, elapsed_values, year_values):
        day = np.asarray(elapsed_values, dtype=float)
        year = np.asarray(year_values, dtype=float)
        return np.column_stack([arr, day, np.sin(2 * np.pi * day / 365.25), np.cos(2 * np.pi * day / 365.25), year])

    res_x = add_time(res_base, x[:, elapsed_idx], x[:, year_idx])
    res_xp = add_time(res_basep, xp[:, elapsed_idx], xp[:, year_idx])

    cat_positions = [res_names.index(k) for k in residual_categorical]
    residuals = HistGradientBoostingRegressor(
        max_iter=100,
        categorical_features=cat_positions,
        early_stopping=False,
        random_state=seed,
    )
    residuals.fit(res_x, y - trend.predict(trend_x))

    prediction = trend.predict(trend_xp) + residuals.predict(res_xp)
    return np.expm1(np.maximum(0, prediction)).tolist()
