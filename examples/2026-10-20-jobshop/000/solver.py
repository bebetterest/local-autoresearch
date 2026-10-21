"""Take the next operation from each job in round-robin order."""


def solve(problem, seed, budget):
    jobs = problem["jobs"]
    return [
        job
        for operation in range(max(map(len, jobs)))
        for job, operations in enumerate(jobs)
        if operation < len(operations)
    ]
