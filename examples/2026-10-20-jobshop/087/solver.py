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

    suffix = []
    for ops in jobs:
        L = len(ops)
        suf = [0.0] * (L + 1)
        for k in range(L - 1, -1, -1):
            suf[k] = suf[k + 1] + ops[k][1]
        suffix.append(suf)

    tail_by_machine = []
    for ops in jobs:
        L = len(ops)
        arr = [[0.0] * n_machines for _ in range(L + 1)]
        for k in range(L - 1, -1, -1):
            arr[k] = arr[k + 1].copy()
            arr[k][ops[k][0]] += ops[k][1]
        tail_by_machine.append(arr)

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

    def beam_lower_bound(state):
        jr, mr, no, seq, ms = state
        lb = ms
        for j in range(n_jobs):
            v = jr[j] + suffix[j][no[j]]
            if v > lb:
                lb = v
        rem_total = [0.0] * n_machines
        for j in range(n_jobs):
            k = no[j]
            if k < len(jobs[j]):
                arr = tail_by_machine[j][k]
                for m, d in enumerate(arr):
                    if d > 0.0:
                        rem_total[m] += d
        for m in range(n_machines):
            if rem_total[m] > 0.0:
                v = mr[m] + rem_total[m]
                if v > lb:
                    lb = v
        return lb

    def beam_search(width):
        beams = [([0.0] * n_jobs, [0.0] * n_machines, [0] * n_jobs, [], 0.0)]
        for _ in range(total):
            new_beams = []
            for jr, mr, no, seq, ms in beams:
                cand = []
                for j, ops in enumerate(jobs):
                    if no[j] >= len(ops):
                        continue
                    m, d = ops[no[j]]
                    st = jr[j] if jr[j] > mr[m] else mr[m]
                    fin = st + d
                    tail = suffix[j][no[j] + 1]
                    cand.append((fin, d, j, m, tail, fin + tail, fin))
                cand.sort(key=lambda x: (x[5], x[6], x[1], x[2]))
                for c in cand[:width]:
                    fin, d, j, m, tail, _, _ = c
                    st = jr[j] if jr[j] > mr[m] else mr[m]
                    fin = st + d
                    new_jr = jr.copy()
                    new_jr[j] = fin
                    new_mr = mr.copy()
                    new_mr[m] = fin
                    new_no = no.copy()
                    new_no[j] += 1
                    new_seq = seq + [j]
                    new_ms = ms if ms > fin else fin
                    new_beams.append((new_jr, new_mr, new_no, new_seq, new_ms))
            new_beams.sort(key=beam_lower_bound)
            beams = new_beams[:width]
        best_seq = min(beams, key=lambda b: evaluate(b[3]))[3]
        return best_seq

    best_seq = beam_search(16)
    best_ms = evaluate(best_seq)

    current_seq = best_seq[:]
    current_ms = best_ms
    best_seq = current_seq[:]
    best_ms = current_ms

    T = 50.0
    T_min = 0.5
    cooling = 0.999
    deadline = start_time + 11.0
    no_improve = 0

    while time.monotonic() < deadline:
        cand = current_seq[:]
        if total > 1:
            i = rng.randrange(total)
            op = cand.pop(i)
            j = rng.randrange(total)
            cand.insert(j, op)
        ms = evaluate(cand)
        delta = (current_ms - ms) / T
        accepted = ms < current_ms - 1e-9 or rng.random() < math.exp(min(700.0, delta))
        if accepted:
            current_seq = cand
            current_ms = ms
            no_improve = 0
            if ms < best_ms - 1e-9:
                best_ms = ms
                best_seq = cand[:]
        else:
            no_improve += 1
            if no_improve >= 3000:
                T = max(T, 20.0)
                no_improve = 0
        T = max(T_min, T * cooling)
        if T < T_min:
            T = T_min

    def local_search(seq, ms, max_rounds=25):
        seq = seq[:]
        ms = ms
        improved = True
        rounds = 0
        while improved and rounds < max_rounds:
            improved = False
            rounds += 1
            n = len(seq)
            # adjacent swaps
            for i in range(n - 1):
                if seq[i] == seq[i + 1]:
                    continue
                cand = seq[:]
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                cand_ms = evaluate(cand)
                if cand_ms < ms - 1e-9:
                    seq = cand
                    ms = cand_ms
                    improved = True
                    break
            if improved:
                continue
            # 2-opt valid segment reversals
            found = False
            for i in range(n - 1):
                max_j = min(n - 1, i + 20)
                for j in range(i + 2, max_j + 1):
                    seg = seq[i:j + 1]
                    if len(set(seg)) != len(seg):
                        continue
                    cand = seq[:i] + seg[::-1] + seq[j + 1:]
                    cand_ms = evaluate(cand)
                    if cand_ms < ms - 1e-9:
                        seq = cand
                        ms = cand_ms
                        improved = True
                        found = True
                        break
                if found:
                    break
            if not found:
                break
        return seq, ms

    best_seq, best_ms = local_search(best_seq, best_ms, max_rounds=25)
    return list(best_seq)
