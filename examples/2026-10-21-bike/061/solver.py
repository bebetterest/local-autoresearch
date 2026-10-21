import os
os.environ["OMP_NUM_THREADS"] = "1"
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
    idx = {k: i for i, k in enumerate(names)}

    elapsed = idx["elapsed_days"]
    trend_cat = [idx[k] for k in (
        "season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit", "weekend"
    )]
    trend_num = [idx[k] for k in ("temp", "atemp", "hum", "windspeed", "elapsed")]

    def add_time_features(arr):
        day = arr[:, elapsed].astype(float)
        return np.column_stack([
            arr,
            np.sin(2 * np.pi * day / 365.25),
            np.cos(2 * np.pi * day / 365.25),
            np.log1p(day),
            day ** 2,
        ])

    tx = add_time_features(x)
    txp = add_time_features(xp)
    n_base = tx.shape[1]
    trend_num_full = trend_num + [n_base - 4, n_base - 3, n_base - 2, n_base - 1]

    trend = make_pipeline([
        ColumnTransformer([
            ("calendar", OneHotEncoder(handle_unknown="ignore", sparse_output=False), trend_cat),
            ("numeric", StandardScaler(), trend_num_full),
        ]),
        TweedieRegressor(power=1.5, alpha=1.0, max_iter=1000),
    ])
    trend.fit(tx, y)
    trend_train_pred = trend.predict(tx)
    trend_pred_x = trend.predict(txp)

    res_cols = [idx[k] for k in (
        "temp", "atemp", "hum", "windspeed", "hr", "holiday", "weekday", "workingday", "weathersit", "weekend"
    )]
    res_cat = [res_cols.index(idx[k]) for k in (
        "hr", "holiday", "weekday", "workingday", "weathersit", "weekend"
    )]
    res_x = np.column_stack([x[:, res_cols], trend_train_pred[:, None], x[:, [elapsed]]])
    res_xp = np.column_stack([xp[:, res_cols], trend_pred_x[:, None], xp[:, [elapsed]]])

    residuals = HistGradientBoostingRegressor(
        max_iter=100,
        categorical_features=res_cat,
        early_stopping=False,
        random_state=seed,
    )
    residuals.fit(res_x, y - trend_train_pred)
    train_pred = trend_train_pred + residuals.predict(res_x)
    pred_log = trend_pred_x + residuals.predict(res_xp)
    log_ratio = np.mean(y - train_pred)
    pred_log = pred_log + log_ratio

    return np.expm1(np.maximum(0, pred_log)).tolist()
