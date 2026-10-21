"""Assign each job to the machine with the earliest projected finish."""


def solve(durations, seed, budget):
    loads = [0] * len(durations[0])
    assignment = []
    for job in durations:
        machine = min(range(len(loads)), key=lambda m: loads[m] + job[m])
        assignment.append(machine)
        loads[machine] += job[machine]
    return assignment
