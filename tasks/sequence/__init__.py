"""Low-autocorrelation binary sequences, scored with exact integer arithmetic."""

PROBLEM = 256


def check(bits):
    if (
        not isinstance(bits, list)
        or len(bits) != PROBLEM
        or any(type(bit) is not int or bit not in (-1, 1) for bit in bits)
    ):
        raise ValueError(f"Return exactly {PROBLEM} integers, each -1 or +1")
    correlations = [sum(a * b for a, b in zip(bits, bits[lag:])) for lag in range(1, PROBLEM)]
    energy = sum(c * c for c in correlations)
    return {
        "score": -energy,
        "energy": energy,
        "merit_factor": PROBLEM**2 / (2 * energy),
        "correlations": correlations,
        "solution": bits,
    }
