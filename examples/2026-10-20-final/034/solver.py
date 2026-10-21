import math
import time
import random


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


def _perturb(circles, rng, pos_scale, rad_lo, rad_hi):
    out = []
    for x, y, r in circles:
        nx = min(1.0, max(0.0, x + rng.uniform(-pos_scale, pos_scale)))
        ny = min(1.0, max(0.0, y + rng.uniform(-pos_scale, pos_scale)))
        nr = max(1e-6, r * rng.uniform(rad_lo, rad_hi))
        out.append([nx, ny, nr])
    return out


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


def _local_improve(circles, time_limit=2.0, seed=0):
    start = time.time()
    n = len(circles)
    if n == 0:
        return circles

    import numpy as np
    from scipy.optimize import minimize

    v0 = np.array([v for c in circles for v in c], dtype=float)

    def unpack(v):
        x = v[0::3].copy()
        y = v[1::3].copy()
        r = v[2::3].copy()
        return x, y, r

    def obj(v):
        return -float(np.sum(unpack(v)[2]))

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

    def attempt(v_start, maxiter, ftol, current_circles):
        if time.time() - start > time_limit:
            return None
        res = minimize(
            obj,
            v_start,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': maxiter, 'ftol': ftol, 'disp': False},
        )
        if time.time() - start > time_limit:
            return None
        if res.x is None:
            return None
        v = res.x
        x, y, r = unpack(v)
        cand = [[float(x[i]), float(y[i]), float(r[i])] for i in range(n)]
        if _valid(cand, tol=1e-9) and sum(c[2] for c in cand) > sum(c[2] for c in current_circles) + 1e-12:
            return cand
        return None

    best = circles
    best_sum = sum(c[2] for c in circles)

    cand = attempt(v0, 30, 1e-12, circles)
    if cand is not None:
        best = cand
        best_sum = sum(c[2] for c in cand)
        cand2 = attempt(np.array([v for c in cand for v in c]), 30, 1e-12, cand)
        if cand2 is not None and sum(c[2] for c in cand2) > best_sum + 1e-12:
            best = cand2
            best_sum = sum(c[2] for c in cand2)

    return best


def _maximize_radii(circles, tol=1e-9):
    n = len(circles)
    new_r = []
    for i in range(n):
        x, y, r = circles[i]
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        for j in range(n):
            if i == j:
                continue
            xj, yj, rj = circles[j]
            dist = math.hypot(x - xj, y - yj)
            max_r = min(max_r, dist - rj)
        new_r.append(max(0.0, max_r))
    return [[c[0], c[1], new_r[i]] for i, c in enumerate(circles)]


def solve(n, seed, budget):
    if n == 26:
        base = _base26()
        best = base
        best_sum = sum(c[2] for c in base)

        starts = [base]
        rng = random.Random(seed)
        starts.append(_perturb(base, rng, 0.004, 0.97, 1.03))
        rng = random.Random(seed + 1)
        starts.append(_perturb(base, rng, 0.006, 0.95, 1.05))

        for idx, cand in enumerate(starts):
            remaining = (budget if budget is not None else 15.0) - (idx * 2.0)
            if remaining <= 0:
                break
            improved = _local_improve(cand, time_limit=min(2.0, remaining), seed=seed)
            if _valid(improved, tol=1e-9):
                s = sum(c[2] for c in improved)
                if s > best_sum + 1e-12:
                    best = improved
                    best_sum = s

        final_time = (budget if budget is not None else 15.0) - 2.0
        if final_time > 0.0:
            improved = _local_improve(best, time_limit=min(0.5, final_time), seed=seed)
            if _valid(improved, tol=1e-9):
                s = sum(c[2] for c in improved)
                if s > best_sum + 1e-12:
                    best = improved

        original_best = best
        best = _maximize_radii(best)
        if _valid(best, tol=1e-9):
            final_time = (budget if budget is not None else 15.0) - 2.0
            if final_time > 0.0:
                improved = _local_improve(best, time_limit=min(0.5, final_time), seed=seed)
                if _valid(improved, tol=1e-9):
                    s = sum(c[2] for c in improved)
                    if s > sum(c[2] for c in best) + 1e-12:
                        best = improved
        if _valid(best, tol=1e-9):
            return best
        return original_best
    return _grid(n)
