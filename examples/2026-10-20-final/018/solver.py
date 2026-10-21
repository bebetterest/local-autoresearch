import math
import random
import time


def _grid(n):
    side = math.ceil(math.sqrt(n))
    radius = 0.5 / side - 1e-12
    return [
        [(c + 0.5) / side, (row + 0.5) / side, radius]
        for row in range(side)
        for c in range(side)
    ][:n]


def _pack26(seed):
    r = 0.0937
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


def _max_radius(circles):
    r = 0.5
    for x, y, _ in circles:
        r = min(r, x, 1.0 - x, y, 1.0 - y)
    n = len(circles)
    for i in range(n):
        xi, yi, _ = circles[i]
        for j in range(i + 1, n):
            xj, yj, _ = circles[j]
            d = math.hypot(xi - xj, yi - yj)
            r = min(r, d * 0.5)
    return r


def solve(n, seed, budget):
    rng = random.Random(seed)
    if n == 26:
        circles = _pack26(seed)
        best_circles = [list(c) for c in circles]
        best_r = _max_radius(best_circles)

        start = time.process_time()
        iterations = 0
        max_iter = 2000
        time_limit = min(budget * 0.5, 2.0)

        while iterations < max_iter and time.process_time() - start < time_limit:
            idx = rng.randrange(len(best_circles))
            x, y, _ = best_circles[idx]
            dx = rng.uniform(-0.003, 0.003)
            dy = rng.uniform(-0.003, 0.003)

            cand = [list(c) for c in best_circles]
            cand[idx][0] = x + dx
            cand[idx][1] = y + dy
            r_cand = _max_radius(cand)

            if r_cand > best_r + 1e-10:
                best_circles = cand
                best_r = r_cand

            iterations += 1

        final_r = best_r - 1e-9
        return [[c[0], c[1], final_r] for c in best_circles]
    else:
        return _grid(n)
