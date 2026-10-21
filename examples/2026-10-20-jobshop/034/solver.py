import time
import random
import math

def solve(problem, seed, budget):
    jobs = problem["jobs"]
    n_machines = problem.get("machines", 20)
    n_jobs = len(jobs)
    total = sum(len(ops) for ops in jobs)
    start_time = time.monotonic()
    rng = random.Random(seed)

    def evaluate(seq):
        job_ready = [0.0] * n_jobs
        machine_ready = [0.0] * n_machines
        next_op = [0] * n_jobs
        makespan = 0.0
        for j in seq:
            ops = jobs[j]
            k = next_op[j]
            m, d = ops[k]
            st = job_ready[j] if job_ready[j] > machine_ready[m] else machine_ready[m]
            fin = st + d
            job_ready[j] = fin
            machine_ready[m] = fin
            next_op[j] = k + 1
            if fin > makespan:
                makespan = fin
        return makespan

    def greedy(rng, explore_prob):
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
            if rng.random() < explore_prob and len(cand) > 1:
                fin, d, j, m = cand[rng.randrange(min(3, len(cand)))]
            else:
                fin, d, j, m = cand[0]
            job_ready[j] = fin
            machine_ready[m] = fin
            next_op[j] += 1
            seq.append(j)
        return seq

    best_seq = greedy(rng, 0.0)
    best_ms = evaluate(best_seq)

    current_seq = best_seq[:]
    current_ms = best_ms
    best_seq = current_seq[:]
    best_ms = current_ms

    T = 50.0
    T_min = 0.5
    cooling = 0.999
    deadline = start_time + 11.0

    while time.monotonic() < deadline:
        cand = current_seq[:]

        # Improvement: cut-and-paste perturbation, moving an operation later.
        if total > 1:
            i = rng.randrange(total - 1)
            j = rng.randrange(i + 1, total)
            op = cand.pop(i)
            cand.insert(j, op)

        ms = evaluate(cand)

        delta = (current_ms - ms) / T
        if ms < current_ms - 1e-9 or rng.random() < math.exp(min(700.0, delta)):
            current_seq = cand
            current_ms = ms
            if ms < best_ms - 1e-9:
                best_ms = ms
                best_seq = cand[:]

        T = max(T_min, T * cooling)
        if T < T_min:
            T = T_min

    return list(best_seq)
