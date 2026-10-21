"""Fit Poisson time/calendar trend with spline time trend plus boosted residuals."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, SplineTransformer


def solve(problem, seed, budget):
    x = np.asarray(problem["train_x"], dtype=float)
    xp = np.asarray(problem["predict_x"], dtype=float)
    y = np.log1p(np.asarray(problem["train_y"], dtype=float))
    names = list(problem["features"])
    idx = {k: i for i, k in enumerate(names)}

    trend_categorical = [
        idx[k] for k in (
            "season", "yr", "mnth", "hr", "holiday", "weekday", "workingday",
            "weathersit", "weekend",
        )
    ]
    residual_categorical = [
        idx[k] for k in (
            "hr", "holiday", "weekday", "workingday", "weathersit", "weekend",
        )
    ]
    numeric = [idx[k] for k in ("temp", "atemp", "hum", "windspeed", "elapsed_days")]
    numeric_without_elapsed = [idx[k] for k in ("temp", "atemp", "hum", "windspeed")]
    elapsed_idx = idx["elapsed_days"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "calendar",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                trend_categorical,
            ),
            (
                "numeric",
                StandardScaler(),
                numeric_without_elapsed,
            ),
            (
                "spline",
                SplineTransformer(n_knots=5, degree=3, include_bias=False),
                [elapsed_idx],
            ),
        ],
        remainder="drop",
    )

    trend = make_pipeline(
        preprocessor,
        PoissonRegressor(alpha=1.0, max_iter=1000),
    )
    trend.fit(x, y)

    residuals = HistGradientBoostingRegressor(
        max_iter=100,
        categorical_features=residual_categorical,
        early_stopping=False,
        random_state=seed,
    )
    residuals.fit(x, y - trend.predict(x))

    prediction = trend.predict(xp) + residuals.predict(xp)
    return np.expm1(np.maximum(0, prediction)).tolist()
