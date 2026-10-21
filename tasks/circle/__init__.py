"""Pack 26 circles into a unit square without overlaps."""

import math
from fractions import Fraction

PROBLEM = 26
TOLERANCE = 1e-9


def validate(circles, n=PROBLEM):
    if not isinstance(circles, list) or len(circles) != n:
        raise ValueError(f"Expected {n} circles")
    for row in circles:
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError("Each circle must be [x, y, radius]")
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in row):
            raise ValueError("Coordinates and radii must be finite numbers")
        if row[2] <= 0:
            raise ValueError("Radii must be positive")
    walls = [gap for x, y, r in circles for gap in (x - r, y - r, 1 - x - r, 1 - y - r)]
    pairs = [
        math.hypot(x - a, y - b) - r - s
        for i, (x, y, r) in enumerate(circles)
        for a, b, s in circles[i + 1 :]
    ]
    violation = max(0.0, -min(walls + pairs))
    if violation > TOLERANCE:
        raise ValueError(f"Constraint violation {violation:.12g} exceeds {TOLERANCE}")
    return {"score": math.fsum(row[2] for row in circles), "max_violation": violation}


def certify(circles):
    """Check the decimal coordinates with exact rational arithmetic, no tolerance."""
    exact = [[Fraction(str(v)) for v in row] for row in circles]
    walls = all(r > 0 and min(x - r, y - r, 1 - x - r, 1 - y - r) >= 0 for x, y, r in exact)
    pairs = all(
        (x - a) ** 2 + (y - b) ** 2 >= (r + s) ** 2
        for i, (x, y, r) in enumerate(exact)
        for a, b, s in exact[i + 1 :]
    )
    return {"exact_feasible": walls and pairs, "rational_sum": str(sum(row[2] for row in exact))}


def check(circles):
    return {**validate(circles), "circles": circles, "certificate": certify(circles)}
