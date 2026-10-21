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
            start = job_ready[j] if job_ready[j] > machine_ready[m] else machine_ready[m]
            fin = start + d
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
                start = job_ready[j] if job_ready[j] > machine_ready[m] else machine_ready[m]
                fin = start + d
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

    best_seq = []
    best_ms = float("inf")

    # Improvement: beam search over partial schedules
    beam_width = 8
    beam_deadline = start_time + 3.0
    states = [(0.0, [0.0] * n_jobs, [0.0] * n_machines, [0] * n_jobs, [])]

    for _ in range(total):
        if time.monotonic() > beam_deadline or not states:
            break
        new_states = []
        for ms, job_ready, machine_ready, next_op, seq in states:
            cand = []
            for j, ops in enumerate(jobs):
                if next_op[j] >= len(ops):
                    continue
                m, d = ops[next_op[j]]
                st = job_ready[j] if job_ready[j] > machine_ready[m] else machine_ready[m]
                fin = st + d
                cand.append((fin, d, j, m))
            cand.sort(key=lambda x: (x[0], x[1], x[2]))
            for fin, d, j, m in cand:
                nj = job_ready[:]
                nm = machine_ready[:]
                nx = next_op[:]
                ns = seq + [j]
                nj[j] = fin
                nm[m] = fin
                nx[j] += 1
                new_states.append((max(nm), nj, nm, nx, ns))
        if not new_states:
            break
        new_states.sort(key=lambda x: x[0])
        states = new_states[:beam_width]

    if states and len(states[0][4]) == total:
        seq = states[0][4]
        ms = evaluate(seq)
        if ms < best_ms - 1e-9:
            best_ms = ms
            best_seq = seq[:]
    else:
        seq = greedy(rng, 0.0)
        ms = evaluate(seq)
        best_seq = seq[:]
        best_ms = ms

    # Multi-start randomized greedy
    for r in range(64):
        if time.monotonic() - start_time > 8.0:
            break
        seq = greedy(random.Random(seed + 1000 + r), 0.25)
        ms = evaluate(seq)
        if ms < best_ms - 1e-9:
            best_ms = ms
            best_seq = seq[:]

    # Simulated annealing polish
    current_seq = best_seq[:]
    current_ms = best_ms
    temp = max(20.0, best_ms / 50.0)
    ls_steps = 0
    ls_deadline = start_time + 11.0

    while time.monotonic() < ls_deadline and ls_steps < 3000:
        ls_steps += 1
        i = rng.randrange(total - 1)
        if current_seq[i] == current_seq[i + 1]:
            continue
        cand = current_seq[:]
        cand[i], cand[i + 1] = cand[i + 1], cand[i]
        ms = evaluate(cand)
        if ms < current_ms or rng.random() < math.exp((current_ms - ms) / temp):
            current_seq = cand
            current_ms = ms
            if ms < best_ms - 1e-9:
                best_ms = ms
                best_seq = cand[:]
        temp *= 0.999
        if temp < 1e-3:
            temp = 1e-3

    return best_seq
