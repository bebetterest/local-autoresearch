import math


def _grid(n):
    side = math.ceil(math.sqrt(n))
    radius = 0.5 / side - 1e-12
    return [
        [(c + 0.5) / side, (row + 0.5) / side, radius]
        for row in range(side)
        for c in range(side)
    ][:n]


def _pack26(seed):
    r = 0.093 - 1e-9
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

    # Small improvement: scale the packing to use spare space, increasing radii.
    s = 1.001
    circles = [[x * s, y * s, r * s] for x, y, r in circles]

    if seed % 2 == 1:
        circles = [[1.0 - x, y, r] for x, y, r in circles]

    return circles


def solve(n, seed, budget):
    if n == 26:
        return _pack26(seed)
    return _grid(n)
