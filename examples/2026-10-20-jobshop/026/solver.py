import time
import random

def solve(problem, seed, budget):
    jobs = problem["jobs"]
    n_machines = problem.get("machines", 20)
    n_jobs = len(jobs)
    ops_len = [len(ops) for ops in jobs]
    total = sum(ops_len)
    rng = random.Random(seed)
    t0 = time.monotonic()

    job_rem = []
    for ops in jobs:
        arr = [0.0] * (len(ops) + 1)
        for k in range(len(ops) - 1, -1, -1):
            arr[k] = arr[k + 1] + ops[k][1]
        job_rem.append(arr)

    def evaluate(seq):
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

    # Greedy fallback
    job_ready = [0.0] * n_jobs
    machine_ready = [0.0] * n_machines
    next_op = [0] * n_jobs
    greedy_seq = []
    for _ in range(total):
        best_key = None
        best_j = -1
        best_m = -1
        best_finish = 0.0
        for j, ops in enumerate(jobs):
            if next_op[j] >= len(ops):
                continue
            m, d = ops[next_op[j]]
            start = job_ready[j] if job_ready[j] > machine_ready[m] else machine_ready[m]
            fin = start + d
            key = (fin, d, j)
            if best_key is None or key < best_key:
                best_key = key
                best_j = j
                best_m = m
                best_finish = fin
        job_ready[best_j] = best_finish
        machine_ready[best_m] = best_finish
        next_op[best_j] += 1
        greedy_seq.append(best_j)

    best_ms = evaluate(greedy_seq)
    best_seq = greedy_seq[:]

    # Beam search
    beam_width = 256
    beam = [(tuple([0.0] * n_jobs), tuple([0.0] * n_machines), tuple([0] * n_jobs), ())]
    for _ in range(total):
        candidates = []
        for jr_t, mr_t, no_t, seq in beam:
            for j in range(n_jobs):
                k = no_t[j]
                if k >= ops_len[j]:
                    continue
                m, d = jobs[j][k]
                jr = jr_t[j]
                mr = mr_t[m]
                start = jr if jr > mr else mr
                fin = start + d
                new_job_ready = list(jr_t)
                new_job_ready[j] = fin
                new_machine_ready = list(mr_t)
                new_machine_ready[m] = fin
                new_next_op = list(no_t)
                new_next_op[j] = k + 1
                new_seq = seq + (j,)

                lb = max(new_machine_ready)
                max_job_lb = 0.0
                for jj in range(n_jobs):
                    kk = new_next_op[jj]
                    val = new_job_ready[jj] + job_rem[jj][kk]
                    if val > max_job_lb:
                        max_job_lb = val
                if max_job_lb > lb:
                    lb = max_job_lb
                if lb > best_ms + 1e-9:
                    continue
                candidates.append((lb, new_job_ready, new_machine_ready, new_next_op, new_seq))

        if not candidates:
            break
        candidates.sort(key=lambda x: x[0])
        new_beam = []
        for _, nj, nm, no, ns in candidates[:beam_width]:
            new_beam.append((tuple(nj), tuple(nm), tuple(no), ns))
        beam = new_beam
        if not beam:
            break

    for state in beam:
        seq = list(state[3])
        if len(seq) == total:
            ms = evaluate(seq)
            if ms < best_ms - 1e-9:
                best_ms = ms
                best_seq = seq

    # Improvement: final adjacent-swap hill climb
    seq = best_seq[:]
    cur_ms = best_ms
    local_iters = 1200
    while local_iters > 0 and time.monotonic() - t0 < 11.0:
        i = rng.randrange(total - 1)
        a = seq[i]
        b = seq[i + 1]
        if a == b:
            local_iters -= 1
            continue
        seq[i], seq[i + 1] = b, a
        new_ms = evaluate(seq)
        if new_ms < cur_ms - 1e-9:
            cur_ms = new_ms
            best_ms = new_ms
            best_seq = seq[:]
        else:
            seq[i], seq[i + 1] = a, b
        local_iters -= 1

    return best_seq
