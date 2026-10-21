"""Seeded random sequence followed by one greedy pass over the bits."""

import numpy as np


def solve(n, seed, budget):
    bits = np.random.default_rng(seed).choice([-1, 1], size=n)

    def energy():
        correlations = np.correlate(bits, bits, mode="full")[n:]
        return int(correlations @ correlations)

    best = energy()
    for i in range(n):
        bits[i] *= -1
        score = energy()
        if score < best:
            best = score
        else:
            bits[i] *= -1
    return bits.tolist()
