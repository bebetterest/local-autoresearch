import time
import random

def solve(problem, seed, budget):
    jobs = problem["jobs"]
    n_machines = problem.get("machines", 20)
    n_jobs = len(jobs)
    ops_len = [len(ops) for ops in jobs]
    total = sum(ops_len)

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
        ms = 0.0
        for j in seq:
            ops = jobs[j]
            k = next_op[j]
            m, d = ops[k]
            start = job_ready[j] if job_ready[j] > machine_ready[m] else machine_ready[m]
            fin = start + d
            job_ready[j] = fin
            machine_ready[m] = fin
            next_op[j] = k + 1
            if fin > ms:
                ms = fin
        return ms

    rng = random.Random(seed)

    def run(pool_size, jitter_scale):
        job_ready = [0.0] * n_jobs
        machine_ready = [0.0] * n_machines
        next_op = [0] * n_jobs
        seq = []

        for _ in range(total):
            cand = []
            for j, ops in enumerate(jobs):
                k = next_op[j]
                if k >= ops_len[j]:
                    continue
                m, d = ops[k]
                jr = job_ready[j]
                mr = machine_ready[m]
                start = jr if jr > mr else mr
                fin = start + d
                tail_after = tails[j][k + 1] if k + 1 < ops_len[j] else 0.0
                base = fin + tail_after
                if jitter_scale > 0.0:
                    base = base + rng.random() * jitter_scale
                cand.append(((base, fin, d, j), j, k, m, fin))

            if pool_size <= 1:
                best = min(cand, key=lambda x: x[0])
                _, j, k, m, fin = best
            else:
                cand.sort(key=lambda x: x[0])
                chosen = cand if len(cand) <= pool_size else cand[:pool_size]
                _, j, k, m, fin = rng.choice(chosen)

            job_ready[j] = fin
            machine_ready[m] = fin
            next_op[j] = k + 1
            seq.append(j)

        return seq

    best_seq = run(1, 0.0)
    best_ms = makespan(best_seq)

    start_time = time.monotonic()
    restart = 0
    time_limit = min(3.0, max(0.0, budget if isinstance(budget, (int, float)) else 3.0))

    while time.monotonic() - start_time < time_limit and restart < 10:
        pool_size = 2 + (restart % 3)
        jitter_scale = 0.0 if restart % 2 == 0 else 0.75
        seq = run(pool_size, jitter_scale)
        ms = makespan(seq)
        if ms < best_ms - 1e-9:
            best_ms = ms
            best_seq = seq
        restart += 1

    return best_seq
