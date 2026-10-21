"""Greedily add the depot that most reduces the current weighted travel distance."""

import numpy as np


def solve(problem, seed, budget):
    points = np.array(problem["points"])
    weights = np.array(problem["demand"])
    distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    nearest = np.full(len(points), np.inf)
    depots = []
    for _ in range(problem["depots"]):
        costs = weights @ np.minimum(nearest[:, None], distances)
        costs[depots] = np.inf
        chosen = int(np.argmin(costs))
        depots.append(chosen)
        nearest = np.minimum(nearest, distances[:, chosen])
    return depots
