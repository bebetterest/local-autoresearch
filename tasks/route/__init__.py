"""Visit 128 synthetic points once and return to the start."""

import math
import random

_rng = random.Random(20261020)
PROBLEM = [[_rng.random(), _rng.random()] for _ in range(128)]


def check(order):
    if (
        not isinstance(order, list)
        or any(type(i) is not int for i in order)
        or sorted(order) != list(range(len(PROBLEM)))
    ):
        raise ValueError("Return each point index exactly once")
    length = math.fsum(
        math.dist(PROBLEM[a], PROBLEM[b]) for a, b in zip(order, order[1:] + order[:1])
    )
    return {"score": -length, "length": length, "solution": order}
