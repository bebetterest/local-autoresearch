import math
import time


def _grid(n):
    side = math.ceil(math.sqrt(n))
    radius = 0.5 / side - 1e-12
    return [
        [(c + 0.5) / side, (row + 0.5) / side, radius]
        for row in range(side)
        for c in range(side)
    ][:n]


def _base26():
    r = 1.0 / (2.0 + 5.0 * math.sqrt(3.0)) - 1e-9
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
    for j in range(5):
        circles.append([r + 2.0 * r * j, y, r])
    y += h
    for j in range(4):
        circles.append([2.0 * r + 2.0 * r * j, y, r])
    y += h
    for j in range(3):
        circles.append([2.0 * r + 2.0 * r * j, y, r])
    return circles


def _valid(circles, tol=1e-9):
    n = len(circles)
    for x, y, r in circles:
        if r <= 0:
            return False
        if x - r < -tol or x + r > 1.0 + tol or y - r < -tol or y + r > 1.0 + tol:
            return False
    for i in range(n):
        xi, yi, ri = circles[i]
        for j in range(i + 1, n):
            xj, yj, rj = circles[j]
            dist = math.hypot(xi - xj, yi - yj)
            if dist < (ri + rj) - 1e-9:
                return False
    return True


def _local_improve(circles, time_limit=5.0):
    import time as _time
    import numpy as np
    from scipy.optimize import minimize

    start = _time.time()
    n = len(circles)
    v0 = []
    for c in circles:
        v0.extend([c[0], c[1], c[2]])
    v0 = np.array(v0, dtype=float)

    def unpack(v):
        x = v[0::3].copy()
        y = v[1::3].copy()
        r = v[2::3].copy()
        return x, y, r

    def obj(v):
        _, _, r = unpack(v)
        return -float(np.sum(r))

    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 0.5))

    cons = []
    for i in range(n):
        def c1(v, i=i):
            x, y, r = unpack(v)
            return float(x[i] - r[i])
        cons.append({'type': 'ineq', 'fun': c1})

        def c2(v, i=i):
            x, y, r = unpack(v)
            return float(1.0 - x[i] - r[i])
        cons.append({'type': 'ineq', 'fun': c2})

        def c3(v, i=i):
            x, y, r = unpack(v)
            return float(y[i] - r[i])
        cons.append({'type': 'ineq', 'fun': c3})

        def c4(v, i=i):
            x, y, r = unpack(v)
            return float(1.0 - y[i] - r[i])
        cons.append({'type': 'ineq', 'fun': c4})

    for i in range(n):
        for j in range(i + 1, n):
            def cp(v, i=i, j=j):
                x, y, r = unpack(v)
                return float((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 - (r[i] + r[j]) ** 2)
            cons.append({'type': 'ineq', 'fun': cp})

    res = minimize(
        obj, v0, method='SLSQP', bounds=bounds, constraints=cons,
        options={'maxiter': 10, 'ftol': 1e-10, 'disp': False}
    )

    if _time.time() - start < time_limit:
        v = res.x
        x, y, r = unpack(v)
        cand = []
        for i in range(n):
            cand.append([float(x[i]), float(y[i]), float(r[i])])
        if _valid(cand, tol=1e-9) and sum(c[2] for c in cand) > sum(c[2] for c in circles):
            return cand
    return circles


def solve(n, seed, budget):
    if n == 26:
        circles = _base26()
        if budget is not None and budget > 4.0:
            circles = _local_improve(circles, time_limit=min(budget, 5.0))
        return circles
    return _grid(n)
