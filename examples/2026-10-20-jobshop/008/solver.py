def solve(problem, seed, budget):
    jobs = problem["jobs"]
    n_machines = problem.get("machines", 20)
    n_jobs = len(jobs)
    total = sum(len(ops) for ops in jobs)

    # Precompute remaining processing times for lower bounds.
    job_rem = []
    machine_rem = []
    for ops in jobs:
        L = len(ops)
        job_rem.append([0.0] * (L + 1))
        machine_rem.append([[0.0] * n_machines for _ in range(L + 1)])
        for k in range(L - 1, -1, -1):
            m, d = ops[k]
            job_rem[-1][k] = job_rem[-1][k + 1] + d
            machine_rem[-1][k] = machine_rem[-1][k + 1].copy()
            machine_rem[-1][k][m] += d

    def lower_bound(state):
        job_ready, machine_ready, next_op = state[0], state[1], state[2]
        best = 0.0

        # Job-based lower bound: remaining job work cannot overlap.
        for j in range(n_jobs):
            v = job_ready[j] + job_rem[j][next_op[j]]
            if v > best:
                best = v

        # Machine-based lower bound: remaining machine load.
        machine_lb = list(machine_ready)
        for j in range(n_jobs):
            mr = machine_rem[j][next_op[j]]
            for m in range(n_machines):
                machine_lb[m] += mr[m]
        m_best = max(machine_lb)
        if m_best > best:
            best = m_best

        return best

    beam_width = 256
    beam = [(tuple([0.0] * n_jobs), tuple([0.0] * n_machines), tuple([0] * n_jobs), ())]

    for _ in range(total):
        candidates = []

        for si, state in enumerate(beam):
            job_ready, machine_ready, next_op, seq = state
            lb = lower_bound((job_ready, machine_ready, next_op))

            for j, ops in enumerate(jobs):
                no = next_op[j]
                if no >= len(ops):
                    continue
                m, d = ops[no]
                start = job_ready[j] if job_ready[j] > machine_ready[m] else machine_ready[m]
                finish = start + d
                candidates.append((lb, finish, d, j, si, m))

        candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))
        selected = candidates[:beam_width]

        next_beam = []
        for lb, finish, d, j, si, m in selected:
            job_ready, machine_ready, next_op, seq = beam[si]

            new_job_ready = list(job_ready)
            new_job_ready[j] = finish

            new_machine_ready = list(machine_ready)
            new_machine_ready[m] = finish

            new_next_op = list(next_op)
            new_next_op[j] += 1

            new_seq = list(seq)
            new_seq.append(j)

            next_beam.append((
                tuple(new_job_ready),
                tuple(new_machine_ready),
                tuple(new_next_op),
                tuple(new_seq)
            ))

        beam = next_beam
        if not beam:
            break

    best_state = min(beam, key=lambda s: max(max(s[0]), max(s[1])))
    return list(best_state[3])
