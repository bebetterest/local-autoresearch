import time
import random

def solve(problem, seed, budget):
    jobs = problem["jobs"]
    n_machines = problem.get("machines", 20)
    n_jobs = len(jobs)
    total = sum(len(ops) for ops in jobs)
    start_time = time.monotonic()
    rng = random.Random(seed)

    beam_width = 32
    deadline = start_time + 8.0

    beam = [((), [0.0] * n_jobs, [0.0] * n_machines, [0] * n_jobs, 0.0)]
    best_seq = None
    best_ms = float("inf")

    job_order = list(range(n_jobs))
    rng.shuffle(job_order)

    for _ in range(total):
        if time.monotonic() > deadline:
            break

        pool = []
        for si, (seq, jr, mr, no, ms) in enumerate(beam):
            for j in job_order:
                ops = jobs[j]
                k = no[j]
                if k >= len(ops):
                    continue
                m, d = ops[k]
                st = jr[j] if jr[j] > mr[m] else mr[m]
                fin = st + d
                pool.append((fin, d, j, si, m))

        pool.sort(key=lambda x: (x[0], x[1], x[2]))
        new_beam = []

        for fin, d, j, si, m in pool[:beam_width]:
            parent_seq, parent_jr, parent_mr, parent_no, parent_ms = beam[si]
            new_seq = parent_seq + (j,)
            new_jr = parent_jr.copy()
            new_jr[j] = fin
            new_mr = parent_mr.copy()
            new_mr[m] = fin
            new_no = parent_no.copy()
            new_no[j] = parent_no[j] + 1
            new_ms = parent_ms if parent_ms > fin else fin
            new_beam.append((new_seq, new_jr, new_mr, new_no, new_ms))

            if new_ms < best_ms:
                best_ms = new_ms
                best_seq = new_seq

        beam = new_beam
        if not beam:
            break

    if best_seq is None:
        job_ready = [0.0] * n_jobs
        machine_ready = [0.0] * n_machines
        next_op = [0] * n_jobs
        seq = []
        for _ in range(total):
            cand = []
            for j, ops in enumerate(jobs):
                if next_op[j] >= len(ops):
                    continue
                m, d = ops[next_op[j]]
                st = job_ready[j] if job_ready[j] > machine_ready[m] else machine_ready[m]
                fin = st + d
                cand.append((fin, d, j, m))
            cand.sort(key=lambda x: (x[0], x[1], x[2]))
            fin, d, j, m = cand[0]
            job_ready[j] = fin
            machine_ready[m] = fin
            next_op[j] += 1
            seq.append(j)
        best_seq = tuple(seq)

    return list(best_seq)
