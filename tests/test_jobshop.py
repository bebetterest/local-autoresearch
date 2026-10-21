from collections import defaultdict

import pytest

from tasks import jobshop
from tasks.jobshop.baseline import solve


@pytest.fixture
def two_jobs(monkeypatch):
    problem = {"machines": 2, "jobs": [[[0, 3], [1, 2]], [[1, 4], [0, 1]]]}
    monkeypatch.setattr(jobshop, "PROBLEM", problem)
    return problem


def test_interleaving_jobs_reduces_finishing_time(two_jobs):
    assert jobshop.check([0, 0, 1, 1])["makespan"] == 10
    result = jobshop.check([0, 1, 0, 1])
    assert result["makespan"] == 6
    assert result["score"] == -6
    assert jobshop.schedule([0, 1, 0, 1]) == [
        (0, 0, 0, 0, 3),
        (1, 0, 1, 0, 4),
        (0, 1, 1, 4, 6),
        (1, 1, 0, 4, 5),
    ]


@pytest.mark.parametrize(
    "answer",
    [
        [],
        [0, 1],
        [0, 0, 0, 1],
        [0, 1, 0, 1, 1],
        [-1, 0, 1, 1],
        [2, 0, 1, 1],
        [False, 1, 0, 1],
        [0.0, 1, 0, 1],
    ],
)
def test_rejects_missing_extra_and_invalid_operations(two_jobs, answer):
    with pytest.raises(ValueError, match="each job index"):
        jobshop.check(answer)


def test_baseline_respects_precedence_and_machine_exclusivity():
    problem = jobshop.PROBLEM
    order = solve(problem, 0, 15)
    operations = jobshop.schedule(order)
    machines = defaultdict(list)
    jobs = defaultdict(list)
    assert len(operations) == 800
    for job, index, machine, start, finish in operations:
        assert finish - start == problem["jobs"][job][index][1]
        machines[machine].append((start, finish))
        jobs[job].append((index, start, finish))
    for intervals in machines.values():
        assert all(end <= start for (_, end), (start, _) in zip(intervals, intervals[1:]))
    for steps in jobs.values():
        assert [index for index, _, _ in steps] == list(range(20))
        assert all(end <= start for (_, _, end), (_, start, _) in zip(steps, steps[1:]))
    assert jobshop.check(order)["makespan"] == max(operation[4] for operation in operations)
