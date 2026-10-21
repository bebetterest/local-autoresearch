def solve(problem, seed, budget):
    jobs = problem["jobs"]
    n_machines = problem.get("machines", 20)

    job_ready = [0.0] * len(jobs)
    machine_ready = [0.0] * n_machines
    next_op = [0] * len(jobs)

    seq = []
    total = sum(len(ops) for ops in jobs)

    while len(seq) < total:
        best_key = None
        best_j = -1
        best_m = -1
        best_finish = 0.0

        for j, ops in enumerate(jobs):
            if next_op[j] >= len(ops):
                continue

            m, d = ops[next_op[j]]
            start = max(job_ready[j], machine_ready[m])
            finish = start + d

            # Earliest completion first, then shortest duration, then job index.
            key = (finish, d, j)
            if best_key is None or key < best_key:
                best_key = key
                best_j = j
                best_m = m
                best_finish = finish

        job_ready[best_j] = best_finish
        machine_ready[best_m] = best_finish
        next_op[best_j] += 1
        seq.append(best_j)

    return seq
