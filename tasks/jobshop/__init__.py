"""Schedule 800 ordered operations from 40 jobs on 20 shared machines."""

from collections import Counter
import random

_rng = random.Random(20261020)
_jobs = []
for _ in range(40):
    _machines = list(range(20))
    _rng.shuffle(_machines)
    _jobs.append([[machine, _rng.randint(1, 99)] for machine in _machines])
PROBLEM = {"machines": 20, "jobs": _jobs}


def schedule(order):
    """Append each job's next operation after its job and machine are ready."""
    jobs = PROBLEM["jobs"]
    next_operation = [0] * len(jobs)
    job_ready = [0] * len(jobs)
    machine_ready = [0] * PROBLEM["machines"]
    operations = []
    for job in order:
        index = next_operation[job]
        machine, duration = jobs[job][index]
        start = max(job_ready[job], machine_ready[machine])
        finish = start + duration
        operations.append((job, index, machine, start, finish))
        job_ready[job] = machine_ready[machine] = finish
        next_operation[job] += 1
    return operations


def check(order):
    jobs = PROBLEM["jobs"]
    if (
        not isinstance(order, list)
        or any(type(job) is not int or not 0 <= job < len(jobs) for job in order)
        or Counter(order) != Counter({job: len(operations) for job, operations in enumerate(jobs)})
    ):
        raise ValueError(
            "Return each job index once per operation, with no missing or extra entries"
        )
    makespan = max(operation[4] for operation in schedule(order))
    return {"score": -makespan, "makespan": makespan, "solution": order}
