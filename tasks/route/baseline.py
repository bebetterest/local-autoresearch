"""Nearest-neighbor baseline, starting at point zero."""

import math


def solve(points, seed, budget):
    route = [0]
    remaining = set(range(1, len(points)))
    while remaining:
        next_point = min(remaining, key=lambda i: (math.dist(points[route[-1]], points[i]), i))
        route.append(next_point)
        remaining.remove(next_point)
    return route
