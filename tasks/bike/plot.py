"""Plot count calibration separately from the RMSLE search objective."""

from collections import defaultdict

from . import VALIDATION

TITLE = "Bike rental demand"
METRIC = "rmsle"
LABEL = "Validation RMSLE"
MINIMIZE = True


def draw(axes, trials):
    series = []
    for trial in trials:
        observed, predicted = defaultdict(float), defaultdict(float)
        for row, value in zip(VALIDATION, trial["solution"], strict=True):
            observed[row["dteday"]] += int(row["cnt"])
            predicted[row["dteday"]] += value
        series.append((list(observed.values()), list(predicted.values())))
    limit = max(value for pair in series for values in pair for value in values) * 1.05
    for axis, (observed, predicted) in zip(axes, series, strict=True):
        axis.scatter(observed, predicted, s=10, alpha=0.5)
        axis.plot([0, limit], [0, limit], color="#687A7F", linewidth=1, linestyle="--")
        axis.set(
            xlabel="Observed daily rentals",
            ylabel="Predicted daily rentals",
            xlim=(0, limit),
            ylim=(0, limit),
            aspect="equal",
        )
        axis.text(
            0.04,
            0.95,
            f"Total bias: {100 * (sum(predicted) / sum(observed) - 1):+.1f}%",
            transform=axis.transAxes,
            va="top",
            fontsize=9,
        )
