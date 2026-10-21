"""Compare machine timelines on the same time scale."""

from matplotlib import colormaps

from . import PROBLEM, schedule

TITLE = "Factory scheduling"
METRIC = "makespan"
LABEL = "Finishing time"
MINIMIZE = True


def draw(axes, trials):
    limit = max(trial["makespan"] for trial in trials)
    colors = colormaps["Blues"]
    for axis, trial in zip(axes, trials):
        operations = schedule(trial["solution"])
        for machine in range(PROBLEM["machines"]):
            assigned = [operation for operation in operations if operation[2] == machine]
            axis.broken_barh(
                [(start, finish - start) for _, _, _, start, finish in assigned],
                (machine + 0.65, 0.7),
                facecolors=[
                    colors(0.35 + 0.6 * job / len(PROBLEM["jobs"])) for job, *_ in assigned
                ],
            )
        axis.set(
            xlim=(0, limit),
            ylim=(0.3, PROBLEM["machines"] + 0.7),
            xlabel="Time",
            ylabel="Machine",
            yticks=[1, 5, 10, 15, 20],
        )
        axis.grid(axis="x", alpha=0.12)
