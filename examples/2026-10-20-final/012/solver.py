import math
import numpy as np
from scipy.optimize import differential_evolution


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
    centers = v.reshape(-1, 2)
    x = centers[:, 0]
    y = centers[:, 1]

    b = 0.0
    b += np.sum(np.maximum(0.0, r - x) ** 2)
    b += np.sum(np.maximum(0.0, r - (1.0 - x)) ** 2)
    b += np.sum(np.maximum(0.0, r - y) ** 2)
    b += np.sum(np.maximum(0.0, r - (1.0 - y)) ** 2)

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    iu = np.triu_indices(len(x), k=1)
    ov = np.maximum(0.0, 2.0 * r - dist[iu]) ** 2
    return float(b + np.sum(ov))


def _max_violation(centers, r):
    pts = np.asarray(centers, dtype=float)
    worst = 0.0
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


def _try_radius(r, seed, n=26):
    bounds = []
    for i in range(n):
        bounds.append((r, 1.0 - r))
        bounds.append((r, 1.0 - r))

    def obj(v):
        return _violation(v, r)

    try:
        res = differential_evolution(
            obj,
            bounds,
            seed=seed,
            maxiter=20,
            popsize=2,
            tol=1e-10,
            mutation=(0.3, 1.0),
            recombination=0.9,
            polish=False,
            updating='immediate',
            workers=1,
        )
        centers = res.x.reshape(-1, 2)
        if _max_violation(centers, r) < 1e-9:
            return [[float(x), float(y), float(r)] for x, y in centers]
    except Exception:
        pass
    return None


def solve(n, seed, budget):
    if n == 26:
        base = _pack26(seed)
        best_circles = [list(map(float, c)) for c in base]
        best_score = sum(c[2] for c in best_circles)

        cand = _try_radius(0.094, seed)
        if cand is not None:
            score = sum(c[2] for c in cand)
            if score > best_score:
                return cand

        return best_circles

    return _grid(n)
