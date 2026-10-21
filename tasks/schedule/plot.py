"""Compare machine loads on the same time scale."""

from . import PROBLEM

TITLE = "Machine scheduling"
METRIC = "makespan"
LABEL = "Finishing time"
MINIMIZE = True


def draw(axes, trials):
    limit = max(trial["makespan"] for trial in trials) * 1.05
    for axis, trial in zip(axes, trials):
        starts = [0] * 8
        for durations, machine in zip(PROBLEM, trial["solution"]):
            duration = durations[machine]
            axis.barh(
                machine,
                duration,
                left=starts[machine],
                color="#6da4d5",
                edgecolor="white",
                linewidth=0.5,
            )
            starts[machine] += duration
        axis.set(xlabel="Time", ylabel="Machine", yticks=range(8), xlim=(0, limit))
