import math
import numpy as np
from scipy.optimize import minimize


def _grid(n):
    side = math.ceil(math.sqrt(n))
    radius = 0.5 / side - 1e-12
    return [
        [(c + 0.5) / side, (row + 0.5) / side, radius]
        for row in range(side)
        for c in range(side)
    ][:n]


def _pack26(seed):
    r = 0.0936
    h = math.sqrt(3.0) * r
    circles = []

    y = r
    for j in range(5):
        circles.append([r + 2.0 * r * j, y, r])
    y += h
    for j in range(4):
        circles.append([2.0 * r + 2.0 * r * j, y, r])
    y += h
    for j in range(5):
        circles.append([r + 2.0 * r * j, y, r])
    y += h
    for j in range(4):
        circles.append([2.0 * r + 2.0 * r * j, y, r])
    y += h
    for j in range(5):
        circles.append([r + 2.0 * r * j, y, r])
    y += h
    for j in range(3):
        circles.append([2.0 * r + 2.0 * r * j, y, r])

    if seed % 2 == 1:
        circles = [[1.0 - x, y, r] for x, y, r in circles]

    return circles


def _max_violation(pts, r):
    pts = np.asarray(pts, dtype=float)
    n = len(pts)
    if n == 0:
        return 0.0

    d = np.minimum(
        np.minimum(pts[:, 0], 1.0 - pts[:, 0]),
        np.minimum(pts[:, 1], 1.0 - pts[:, 1]),
    )
    worst = float(np.max(np.maximum(0.0, r - d)))

    if n > 1:
        dx = pts[:, 0][:, None] - pts[:, 0][None, :]
        dy = pts[:, 1][:, None] - pts[:, 1][None, :]
        dist = np.sqrt(dx * dx + dy * dy)
        iu = np.triu_indices(n, k=1)
        ov = np.maximum(0.0, 2.0 * r - dist[iu])
        worst = max(worst, float(np.max(ov)))

    return worst


def _objective(v):
    pts = v[:-1].reshape(-1, 2)
    r = float(v[-1])

    d = np.minimum(
        np.minimum(pts[:, 0], 1.0 - pts[:, 0]),
        np.minimum(pts[:, 1], 1.0 - pts[:, 1]),
    )
    b = np.maximum(0.0, r - d)
    viol = float(np.sum(b * b))

    if len(pts) > 1:
        dx = pts[:, 0][:, None] - pts[:, 0][None, :]
        dy = pts[:, 1][:, None] - pts[:, 1][None, :]
        dist = np.sqrt(dx * dx + dy * dy)
        iu = np.triu_indices(len(pts), k=1)
        ov = np.maximum(0.0, 2.0 * r - dist[iu])
        viol += float(np.sum(ov * ov))

    return 1e6 * viol - r


def _optimize26(seed):
    base = _pack26(seed)
    pts = np.array([[c[0], c[1]] for c in base], dtype=float)
    r0 = float(base[0][2])
    x0 = np.concatenate([pts.ravel(), [r0]])

    bounds = [(0.0, 1.0)] * 52 + [(1e-6, 0.5)]
    res = minimize(
        _objective,
        x0,
        bounds=bounds,
        method="L-BFGS-B",
        options={"maxiter": 150, "ftol": 1e-12, "gtol": 1e-12},
    )

    v = res.x
    pts_opt = v[:-1].reshape(-1, 2)
    r_opt = float(v[-1])

    if _max_violation(pts_opt, r_opt) <= 1e-9:
        cand = [[float(x), float(y), r_opt] for x, y in pts_opt]
        if sum(c[2] for c in cand) > sum(c[2] for c in base) + 1e-12:
            return cand

    return None


def solve(n, seed, budget):
    if n == 26:
        base = _pack26(seed)
        best = [list(map(float, c)) for c in base]
        best_score = sum(c[2] for c in best)

        try:
            cand = _optimize26(seed)
            if cand is not None:
                score = sum(c[2] for c in cand)
                if score > best_score:
                    best = cand
                    best_score = score
        except Exception:
            pass

        return best

    return _grid(n)
