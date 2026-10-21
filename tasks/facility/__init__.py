"""Place eight service depots to reduce travel for 100 weighted customers."""

import math
import random

_rng = random.Random(20261020)
PROBLEM = {
    "points": [[_rng.random(), _rng.random()] for _ in range(100)],
    "demand": [_rng.randint(1, 5) for _ in range(100)],
    "depots": 8,
}


def check(depots):
    points, demand = PROBLEM["points"], PROBLEM["demand"]
    if (
        not isinstance(depots, list)
        or len(depots) != PROBLEM["depots"]
        or any(type(i) is not int or not 0 <= i < len(points) for i in depots)
        or len(set(depots)) != len(depots)
    ):
        raise ValueError("Return eight distinct customer indices for the depots")
    distances = [min(math.dist(point, points[i]) for i in depots) for point in points]
    distance = math.fsum(w * d for w, d in zip(demand, distances)) / sum(demand)
    return {"score": -distance, "distance": distance, "solution": depots}
