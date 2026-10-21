import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


def solve(problem, seed, budget):
    x = np.asarray(problem["train_x"], dtype=np.float64)
    xp = np.asarray(problem["predict_x"], dtype=np.float64)
    y = np.log1p(np.asarray(problem["train_y"], dtype=np.float64))
    names = problem["features"]
    col = {n: i for i, n in enumerate(names)}

    cat_names = [
        "season", "yr", "mnth", "hr", "holiday", "weekday",
        "workingday", "weathersit", "weekend",
    ]
    cat_idx = [col[n] for n in cat_names]

    model = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.05,
        categorical_features=cat_idx,
        early_stopping=False,
        random_state=seed,
    )
    model.fit(x, y)
    pred_log = model.predict(xp)
    pred = np.expm1(np.maximum(0.0, pred_log))
    return pred.tolist()
