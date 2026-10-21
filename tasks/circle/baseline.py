"""Place equal circles on a square grid."""

import math


def solve(n, seed, budget):
    side = math.ceil(math.sqrt(n))
    radius = 0.5 / side - 1e-12
    return [
        [(column + 0.5) / side, (row + 0.5) / side, radius]
        for row in range(side)
        for column in range(side)
    ][:n]
