def solve(problem, seed, budget):
    jobs = problem["jobs"]
    n_machines = problem.get("machines", 20)
    n_jobs = len(jobs)
    ops_len = [len(ops) for ops in jobs]
    total = sum(ops_len)

    job_rem = []
    for ops in jobs:
        arr = [0.0] * (len(ops) + 1)
        for k in range(len(ops) - 1, -1, -1):
            arr[k] = arr[k + 1] + ops[k][1]
        job_rem.append(arr)

    beam = [
        (
            tuple([0.0] * n_jobs),
            tuple([0.0] * n_machines),
            tuple([0] * n_jobs),
            (),
        )
    ]
    beam_width = 64

    for _ in range(total):
        candidates = []

        for job_ready, machine_ready, next_op, seq in beam:
            for j in range(n_jobs):
                k = next_op[j]
                if k >= ops_len[j]:
                    continue

                m, d = jobs[j][k]
                jr = job_ready[j]
                mr = machine_ready[m]
                start = jr if jr > mr else mr
                fin = start + d

                new_job_ready = list(job_ready)
                new_job_ready[j] = fin
                new_job_ready = tuple(new_job_ready)

                new_machine_ready = list(machine_ready)
                new_machine_ready[m] = fin
                new_machine_ready = tuple(new_machine_ready)

                new_next_op = list(next_op)
                new_next_op[j] = k + 1
                new_next_op = tuple(new_next_op)

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

                candidates.append((lb, new_job_ready, new_machine_ready, new_next_op, new_seq))

        if not candidates:
            break

        candidates.sort(key=lambda x: x[0])

        new_beam = []
        for _, nj, nm, no, ns in candidates[:beam_width]:
            new_beam.append((nj, nm, no, ns))

        beam = new_beam

    best_seq = None
    best_ms = float("inf")

    for state in beam:
        seq = state[3]

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

        if ms < best_ms - 1e-9:
            best_ms = ms
            best_seq = seq

    return list(best_seq)
