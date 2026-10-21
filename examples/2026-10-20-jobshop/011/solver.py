import random

def solve(problem, seed, budget):
    jobs = problem["jobs"]
    n_machines = problem.get("machines", 20)
    n_jobs = len(jobs)
    total = sum(len(ops) for ops in jobs)

    # Remaining processing time from each operation onward (critical path tail).
    tails = []
    for ops in jobs:
        L = len(ops)
        s = [0.0] * (L + 1)
        for k in range(L - 1, -1, -1):
            s[k] = s[k + 1] + ops[k][1]
        tails.append(s)

    def makespan(seq):
        job_ready = [0.0] * n_jobs
        machine_ready = [0.0] * n_machines
        next_op = [0] * n_jobs
        best = 0.0
        for j in seq:
            ops = jobs[j]
            k = next_op[j]
            m, d = ops[k]
            start = job_ready[j] if job_ready[j] > machine_ready[m] else machine_ready[m]
            fin = start + d
            job_ready[j] = fin
            machine_ready[m] = fin
            next_op[j] = k + 1
            if fin > best:
                best = fin
        return best

    rng = random.Random(seed)
    best_seq = None
    best_mk = float("inf")

    restarts = max(1, min(40, int(budget / 0.05) + 1))

    for r in range(restarts):
        job_ready = [0.0] * n_jobs
        machine_ready = [0.0] * n_machines
        next_op = [0] * n_jobs
        seq = []

        noise_scale = 0.0 if r == 0 else rng.random() * 2.0 + 0.01

        while len(seq) < total:
            best_key = None
            best_j = -1
            best_m = -1
            best_finish = 0.0

            for j, ops in enumerate(jobs):
                if next_op[j] >= len(ops):
                    continue

                m, d = ops[next_op[j]]
                start = job_ready[j] if job_ready[j] > machine_ready[m] else machine_ready[m]
                finish = start + d
                tail = tails[j][next_op[j]]

                # Longer remaining tail first, then earlier finish, shorter duration, lower job index.
                # Add small random noise for restart variation.
                key = (-tail - rng.random() * noise_scale, finish, d, j)

                if best_key is None or key < best_key:
                    best_key = key
                    best_j = j
                    best_m = m
                    best_finish = finish

            job_ready[best_j] = best_finish
            machine_ready[best_m] = best_finish
            next_op[best_j] += 1
            seq.append(best_j)

        mk = makespan(seq)
        if mk < best_mk:
            best_mk = mk
            best_seq = seq

    return best_seq
