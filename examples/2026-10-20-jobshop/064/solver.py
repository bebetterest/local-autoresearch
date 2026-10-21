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
        return lb

    def make_state(seq):
        jr = [0.0] * n_jobs
        mr = [0.0] * n_machines
        no = [0] * n_jobs
        ms = 0.0
        for j in seq:
            ops = jobs[j]
            k = no[j]
            m, d = ops[k]
            st = jr[j] if jr[j] > mr[m] else mr[m]
            fin = st + d
            jr[j] = fin
            mr[m] = fin
            no[j] = k + 1
            if fin > ms:
                ms = fin
        return (jr, mr, no, seq, ms)

    def greedy_prefix(length):
        jr = [0.0] * n_jobs
        mr = [0.0] * n_machines
        no = [0] * n_jobs
        seq = []
        for _ in range(length):
            best = None
            best_key = None
            for j in range(n_jobs):
                if no[j] >= len(jobs[j]):
                    continue
                ops = jobs[j]
                k = no[j]
                m, d = ops[k]
                st = jr[j] if jr[j] > mr[m] else mr[m]
                fin = st + d
                tail = suffix[j][k + 1]
                key = (tail, -fin)
                if best is None or key > best_key:
                    best = (j, m, d, fin, tail)
                    best_key = key
            j, m, d, fin, tail = best
            seq.append(j)
            jr[j] = fin
            mr[m] = fin
            no[j] += 1
        return seq

    def beam_search(width, initial_beam=None, initial_len=0):
        if initial_beam is None:
            beams = [([0.0] * n_jobs, [0.0] * n_machines, [0] * n_jobs, [], 0.0)]
        else:
            beams = [initial_beam]
        steps = total - initial_len
        for _ in range(steps):
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
                    cand.append((fin, d, j, m, tail, fin + tail, fin, st))
                cand.sort(key=lambda x: (x[5], x[6], x[1], x[2]))
                for c in cand[:width]:
                    fin, d, j, m, tail, _, _, st = c
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

    init_len = total // 2
    init_seq = greedy_prefix(init_len)
    init_beam = make_state(init_seq)
    best_seq = beam_search(width=16, initial_beam=init_beam, initial_len=init_len)
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
            i = rng.randrange(total - 1)
            j = rng.randrange(i + 1, total)
            op = cand.pop(i)
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

    return list(best_seq)
