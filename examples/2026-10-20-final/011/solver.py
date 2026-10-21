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


def _violation(v, r):
    c = v.reshape(-1, 2)
    x = c[:, 0]
    y = c[:, 1]

    b = 0.0
    b += np.sum(np.maximum(0.0, r - x) ** 2)
    b += np.sum(np.maximum(0.0, r - (1.0 - x)) ** 2)
    b += np.sum(np.maximum(0.0, r - y) ** 2)
    b += np.sum(np.maximum(0.0, r - (1.0 - y)) ** 2)

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    iu = np.triu_indices(len(c), k=1)
    ov = np.maximum(0.0, 2.0 * r - dist[iu]) ** 2
    return float(b + np.sum(ov))


def _max_violation(centers, r):
    worst = 0.0
    pts = np.asarray(centers, dtype=float)
    for x, y in pts:
        d = min(x, 1.0 - x, y, 1.0 - y)
        worst = max(worst, r - d)

    n = len(pts)
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            xj, yj = pts[j]
            dist = math.hypot(xi - xj, yi - yj)
            worst = max(worst, 2.0 * r - dist)
    return worst


def _optimize_centers(r, centers, rng, iterations=100):
    v0 = np.array(centers, dtype=float).ravel()
    v0 = v0 + rng.uniform(-0.001, 0.001, size=v0.shape)
    res = minimize(
        lambda v: _violation(v, r),
        v0,
        method="Powell",
        bounds=[(0.0, 1.0)] * 52,
        options={"maxiter": iterations, "ftol": 1e-12, "xtol": 1e-12},
    )
    centers_opt = res.x.reshape(-1, 2)
    return centers_opt, float(res.fun)


def solve(n, seed, budget):
    if n == 26:
        base = _pack26(seed)
        best_circles = [list(map(float, c)) for c in base]
        best_score = sum(c[2] for c in best_circles)
        start_centers = np.array([[c[0], c[1]] for c in base], dtype=float)
        rng = np.random.default_rng(seed)

        for r in (0.0936, 0.09365, 0.0937, 0.09375, 0.0938, 0.0939, 0.094):
            for _ in range(2):
                centers_opt, _ = _optimize_centers(r, start_centers, rng, iterations=100)
                if _max_violation(centers_opt, r) < 1e-9:
                    circles = [[float(x), float(y), float(r)] for x, y in centers_opt]
                    score = sum(c[2] for c in circles)
                    if score > best_score + 1e-12:
                        best_score = score
                        best_circles = circles

        return best_circles

    return _grid(n)
