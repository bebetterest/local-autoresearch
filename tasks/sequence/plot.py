"""Compare sidelobes on the same amplitude scale."""

TITLE = "Binary signal design"
METRIC = "energy"
LABEL = "Sidelobe energy"
MINIMIZE = True


def draw(axes, trials):
    limit = max(abs(v) for trial in trials for v in trial["correlations"]) + 2
    for axis, trial in zip(axes, trials):
        values = trial["correlations"]
        axis.plot(range(1, len(values) + 1), values, color="#397fba", linewidth=0.8)
        axis.set(xlabel="Lag", ylabel="Autocorrelation sidelobes", ylim=(-limit, limit))
