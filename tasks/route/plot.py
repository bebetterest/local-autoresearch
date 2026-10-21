"""Display the route using the fixed point coordinates."""

from . import PROBLEM

TITLE = "Route optimization"
METRIC = "length"
LABEL = "Route length"
MINIMIZE = True


def draw(axes, trials):
    for axis, trial in zip(axes, trials):
        closed = trial["solution"] + trial["solution"][:1]
        axis.plot(
            [PROBLEM[i][0] for i in closed],
            [PROBLEM[i][1] for i in closed],
            color="#397fba",
            linewidth=1,
            marker=".",
            markersize=3,
        )
        axis.set(aspect="equal", xticks=[], yticks=[])
