def solve(problem, seed, budget):
    jobs = problem["jobs"]
    n_machines = problem.get("machines", 20)
    n_jobs = len(jobs)
    ops_len = [len(ops) for ops in jobs]

    tails = []
    total_load = [0.0] * n_machines
    for ops in jobs:
        L = len(ops)
        s = [0.0] * (L + 1)
        for k in range(L - 1, -1, -1):
            d = ops[k][1]
            s[k] = s[k + 1] + d
            total_load[ops[k][0]] += d
        tails.append(s)

    def lower_bound(job_ready, machine_ready, next_op, rem):
        lb = 0.0
        for j in range(n_jobs):
            k = next_op[j]
            v = job_ready[j] + tails[j][k] if k < ops_len[j] else job_ready[j]
            if v > lb:
                lb = v
        for m in range(n_machines):
            v = machine_ready[m] + rem[m]
            if v > lb:
                lb = v
        return lb

    def evaluate(seq):
        job_ready = [0.0] * n_jobs
        machine_ready = [0.0] * n_machines
        next_op = [0] * n_jobs
        rem = list(total_load)
        mk = 0.0
        for j in seq:
            ops = jobs[j]
            k = next_op[j]
            m, d = ops[k]
            start = job_ready[j] if job_ready[j] > machine_ready[m] else machine_ready[m]
            fin = start + d
            job_ready[j] = fin
            machine_ready[m] = fin
            next_op[j] = k + 1
            rem[m] -= d
            if fin > mk:
                mk = fin
        return mk

    def make_greedy():
        job_ready = [0.0] * n_jobs
        machine_ready = [0.0] * n_machines
        next_op = [0] * n_jobs
        rem = list(total_load)
        seq = []
        total = sum(ops_len)
        while len(seq) < total:
            best_key = None
            best_j = -1
            best_m = -1
            best_fin = 0.0
            for j, ops in enumerate(jobs):
                k = next_op[j]
                if k >= ops_len[j]:
                    continue
                m, d = ops[k]
                jr = job_ready[j]
                mr = machine_ready[m]
                start = jr if jr > mr else mr
                fin = start + d
                key = (fin, d, j)
                if best_key is None or key < best_key:
                    best_key = key
                    best_j = j
                    best_m = m
                    best_fin = fin
            k = next_op[best_j]
            duration = jobs[best_j][k][1]
            job_ready[best_j] = best_fin
            machine_ready[best_m] = best_fin
            next_op[best_j] = k + 1
            rem[best_m] -= duration
            seq.append(best_j)
        return seq

    greedy_seq = make_greedy()
    best_seq = greedy_seq
    best_mk = evaluate(greedy_seq)

    beam_width = 64
    beam = [(tuple([0.0] * n_jobs), tuple([0.0] * n_machines), tuple([0] * n_jobs), tuple(total_load), ())]
    total = sum(ops_len)

    for _ in range(total):
        candidates = []
        for state in beam:
            job_ready, machine_ready, next_op, rem, seq = state
            job_ready = list(job_ready)
            machine_ready = list(machine_ready)
            next_op = list(next_op)
            rem = list(rem)

            for j, ops in enumerate(jobs):
                k = next_op[j]
                if k >= ops_len[j]:
                    continue
                m, d = ops[k]
                jr = job_ready[j]
                mr = machine_ready[m]
                start = jr if jr > mr else mr
                fin = start + d

                nr = list(job_ready)
                nr[j] = fin
                nm = list(machine_ready)
                nm[m] = fin
                nk = list(next_op)
                nk[j] = k + 1
                nrem = list(rem)
                nrem[m] -= d
                nseq = seq + (j,)

                lb = lower_bound(nr, nm, nk, nrem)
                candidates.append((lb, nr, nm, nk, nrem, nseq))

        candidates.sort(key=lambda x: x[0])
        new_beam = []
        for lb, nr, nm, nk, nrem, nseq in candidates[:beam_width]:
            new_beam.append((tuple(nr), tuple(nm), tuple(nk), tuple(nrem), nseq))
        beam = new_beam
        if not beam:
            break

    for state in beam:
        seq = state[4]
        mk = evaluate(seq)
        if mk < best_mk:
            best_mk = mk
            best_seq = list(seq)

    return list(best_seq)
