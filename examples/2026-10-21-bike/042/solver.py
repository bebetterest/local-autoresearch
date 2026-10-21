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

    def idx(*keys):
        return [names.index(k) for k in keys]

    trend_categorical = idx(
        "season", "yr", "mnth", "hr", "holiday", "weekday",
        "workingday", "weathersit", "weekend",
    )
    trend_num = idx("temp", "atemp", "hum", "windspeed", "elapsed_days")

    def temporal_basis(arr):
        day = arr[:, names.index("elapsed_days")]
        return np.column_stack([
            np.sin(2 * np.pi * day / 365.25),
            np.cos(2 * np.pi * day / 365.25),
            np.log1p(day),
            day ** 2,
        ])

    trend_x = np.column_stack([x, temporal_basis(x)])
    trend_xp = np.column_stack([xp, temporal_basis(xp)])
    temp_cols = [x.shape[1] + i for i in range(temporal_basis(x).shape[1])]

    trend = make_pipeline(
        ColumnTransformer([
            (
                "calendar",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                trend_categorical,
            ),
            (
                "numeric",
                StandardScaler(),
                trend_num + temp_cols,
            ),
        ]),
        TweedieRegressor(power=1.5, alpha=1.0, max_iter=1000),
    )
    trend.fit(trend_x, y)

    res_cols = idx(
        "temp", "atemp", "hum", "windspeed", "hr", "holiday",
        "weekday", "workingday", "weathersit", "weekend",
    )
    residual_categorical = {
        "hr", "holiday", "weekday", "workingday", "weathersit", "weekend",
    }
    res_cat_pos = [
        i for i, orig_idx in enumerate(res_cols)
        if names[orig_idx] in residual_categorical
    ]

    res_x = x[:, res_cols]
    res_xp = xp[:, res_cols]

    residuals = HistGradientBoostingRegressor(
        max_iter=100,
        categorical_features=res_cat_pos,
        early_stopping=False,
        random_state=seed,
    )
    residuals.fit(res_x, y - trend.predict(trend_x))

    pred_log = trend.predict(trend_xp) + residuals.predict(res_xp)
    train_pred_log = trend.predict(trend_x) + residuals.predict(res_x)
    pred_log = pred_log + np.mean(y - train_pred_log)

    pred = np.expm1(np.maximum(0, pred_log))
    return pred.tolist()
