"""Show customers, their nearest depot and their travel distance."""

import math

from . import PROBLEM

TITLE = "Service depots"
METRIC = "distance"
LABEL = "Mean weighted distance"
MINIMIZE = True


def draw(axes, trials):
    points = PROBLEM["points"]
    for axis, trial in zip(axes, trials):
        depots = trial["solution"]
        for point in points:
            nearest = min(depots, key=lambda i: math.dist(point, points[i]))
            axis.plot(
                [point[0], points[nearest][0]],
                [point[1], points[nearest][1]],
                color="#b9cfe2",
                linewidth=0.7,
                zorder=1,
            )
        axis.scatter(*zip(*points), s=[8 * w for w in PROBLEM["demand"]], color="#6da4d5", zorder=2)
        axis.scatter(
            *zip(*(points[i] for i in depots)),
            marker="s",
            s=65,
            color="#df8545",
            edgecolors="white",
            zorder=3,
        )
        axis.set(aspect="equal", xlim=(-0.03, 1.03), ylim=(-0.03, 1.03), xticks=[], yticks=[])
