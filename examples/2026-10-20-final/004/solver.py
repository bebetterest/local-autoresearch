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


def _hex_base(seed):
    r = 0.09
    h = math.sqrt(3.0) * r
    circles = []
    y = r
    row_idx = 0
    while len(circles) < 26:
        if row_idx % 2 == 0:
            xs = [r + 2.0 * r * j for j in range(5)]
        else:
            xs = [2.0 * r + 2.0 * r * j for j in range(5)]
        for x in xs:
            if len(circles) >= 26:
                break
            circles.append([x, y, r])
        y += h
        row_idx += 1

    if seed % 2 == 1:
        circles = [[1.0 - x, y, r] for x, y, r in circles]
    return circles


def _max_radius(centers):
    m = len(centers)
    r = 0.5
    for x, y in centers:
        b = min(x, 1.0 - x, y, 1.0 - y)
        if b < r:
            r = b
    for i in range(m):
        xi, yi = centers[i]
        for j in range(i + 1, m):
            d = math.hypot(xi - centers[j][0], yi - centers[j][1])
            half = d * 0.5
            if half < r:
                r = half
    if r < 0.0:
        r = 0.0
    return r


def solve(n, seed, budget):
    if n != 26:
        return _grid(n)

    circles = _hex_base(seed)
    centers = [[c[0], c[1]] for c in circles]

    best_c = [c[:] for c in centers]
    best_r = _max_radius(best_c)
    cur_c = [c[:] for c in centers]
    cur_r = best_r

    rng = random.Random(seed)
    start = time.process_time()
    deadline = start + min(float(budget), 10.0)

    T = 0.01
    cool = 0.995
    it = 0

    while time.process_time() < deadline and it < 4000:
        it += 1
        scale = best_r * rng.uniform(0.001, 0.03)
        new_c = []
        for x, y in cur_c:
            nx = x + rng.gauss(0.0, scale)
            ny = y + rng.gauss(0.0, scale)
            nx = min(1.0, max(0.0, nx))
            ny = min(1.0, max(0.0, ny))
            new_c.append([nx, ny])

        nr = _max_radius(new_c)

        if nr > best_r + 1e-12:
            best_r = nr
            best_c = new_c
            cur_c = new_c
            cur_r = nr
        else:
            if T > 1e-12 and rng.random() < math.exp((nr - cur_r) / T):
                cur_c = new_c
                cur_r = nr

        T *= cool
        if T < 1e-12:
            T = 1e-12

    radius = best_r - 1e-9
    return [[x, y, radius] for x, y in best_c]
