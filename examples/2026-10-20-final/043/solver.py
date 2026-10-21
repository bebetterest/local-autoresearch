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


def _rotate(circles, angle):
    c = math.cos(angle)
    s = math.sin(angle)
    cx = 0.5
    cy = 0.5
    out = []
    for x, y, r in circles:
        dx = x - cx
        dy = y - cy
        nx = cx + dx * c - dy * s
        ny = cy + dx * s + dy * c
        out.append([nx, ny, r])
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


def _max_radii_for_positions(circles):
    import numpy as np
    from scipy.optimize import linprog
    n = len(circles)
    if n == 0:
        return None
    c = -np.ones(n)
    bounds = [(0.0, 0.5) for _ in range(n)]
    A_ub = []
    b_ub = []
    for i, (x, y, r) in enumerate(circles):
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(x)
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(1.0 - x)
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(y)
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(1.0 - y)
    for i in range(n):
        xi, yi, ri = circles[i]
        for j in range(i + 1, n):
            xj, yj, rj = circles[j]
            d = math.hypot(xi - xj, yi - yj)
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(d)
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds, method='highs')
    if not res.success:
        return None
    r = res.x
    cand = [[float(circles[i][0]), float(circles[i][1]), float(r[i])] for i in range(n)]
    if _valid(cand, tol=1e-9):
        return cand
    return None


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

    def obj_jac(v):
        g = np.zeros_like(v)
        g[2::3] = -1.0
        return g

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
        def c1_jac(v, i=i):
            g = np.zeros_like(v)
            g[3 * i] = 1.0
            g[3 * i + 2] = -1.0
            return g
        cons.append({'type': 'ineq', 'fun': c1, 'jac': c1_jac})

        def c2(v, i=i):
            x, y, r = unpack(v)
            return float(1.0 - x[i] - r[i])
        def c2_jac(v, i=i):
            g = np.zeros_like(v)
            g[3 * i] = -1.0
            g[3 * i + 2] = -1.0
            return g
        cons.append({'type': 'ineq', 'fun': c2, 'jac': c2_jac})

        def c3(v, i=i):
            x, y, r = unpack(v)
            return float(y[i] - r[i])
        def c3_jac(v, i=i):
            g = np.zeros_like(v)
            g[3 * i + 1] = 1.0
            g[3 * i + 2] = -1.0
            return g
        cons.append({'type': 'ineq', 'fun': c3, 'jac': c3_jac})

        def c4(v, i=i):
            x, y, r = unpack(v)
            return float(1.0 - y[i] - r[i])
        def c4_jac(v, i=i):
            g = np.zeros_like(v)
            g[3 * i + 1] = -1.0
            g[3 * i + 2] = -1.0
            return g
        cons.append({'type': 'ineq', 'fun': c4, 'jac': c4_jac})

    for i in range(n):
        for j in range(i + 1, n):
            def cp(v, i=i, j=j):
                x, y, r = unpack(v)
                return float((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 - (r[i] + r[j]) ** 2)
            def cp_jac(v, i=i, j=j):
                g = np.zeros_like(v)
                x, y, r = unpack(v)
                dx = float(x[i] - x[j])
                dy = float(y[i] - y[j])
                rij = float(r[i] + r[j])
                g[3 * i] = 2.0 * dx
                g[3 * j] = -2.0 * dx
                g[3 * i + 1] = 2.0 * dy
                g[3 * j + 1] = -2.0 * dy
                g[3 * i + 2] = -2.0 * rij
                g[3 * j + 2] = -2.0 * rij
                return g
            cons.append({'type': 'ineq', 'fun': cp, 'jac': cp_jac})

    def attempt(v_start, maxiter, ftol, current_circles):
        if time.time() - start > time_limit:
            return None
        res = minimize(
            obj,
            v_start,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            jac=obj_jac,
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

    if time.time() - start < time_limit - 0.2:
        lp_cand = _max_radii_for_positions(best)
        if lp_cand is not None:
            s = sum(c[2] for c in lp_cand)
            if s > best_sum + 1e-12:
                best = lp_cand
                best_sum = s

    return best


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
        rng = random.Random(seed + 4)
        starts.append(_rotate(base, rng.uniform(-0.2, 0.2)))

        start_time = time.time()
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

        total_budget = budget if budget is not None else 15.0
        remaining = total_budget - (time.time() - start_time)
        if remaining > 0.5:
            rng = random.Random(seed + 2)
            cand = _perturb(best, rng, 0.003, 0.98, 1.02)
            improved = _local_improve(cand, time_limit=min(1.0, remaining), seed=seed)
            if _valid(improved, tol=1e-9):
                s = sum(c[2] for c in improved)
                if s > best_sum + 1e-12:
                    best = improved
                    best_sum = s

            remaining = total_budget - (time.time() - start_time)
            if remaining > 0.5:
                rng = random.Random(seed + 3)
                cand = _perturb(best, rng, 0.002, 0.99, 1.01)
                improved = _local_improve(cand, time_limit=min(1.0, remaining), seed=seed)
                if _valid(improved, tol=1e-9):
                    s = sum(c[2] for c in improved)
                    if s > best_sum + 1e-12:
                        best = improved
                        best_sum = s

        return best
    return _grid(n)
