"""Assign 80 synthetic jobs to eight machines with different processing times."""

import random

_rng = random.Random(20261020)
PROBLEM = [[_rng.randint(10, 100) for _ in range(8)] for _ in range(80)]


def check(assignment):
    if (
        not isinstance(assignment, list)
        or len(assignment) != len(PROBLEM)
        or any(type(m) is not int or not 0 <= m < 8 for m in assignment)
    ):
        raise ValueError("Return one machine index (0–7) for each of the 80 jobs")
    loads = [0] * 8
    for durations, machine in zip(PROBLEM, assignment):
        loads[machine] += durations[machine]
    makespan = max(loads)
    return {"score": -makespan, "makespan": makespan, "loads": loads, "solution": assignment}
