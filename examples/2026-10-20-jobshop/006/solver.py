def solve(problem, seed, budget):
    jobs = problem["jobs"]
    n_machines = problem.get("machines", 20)
    n_jobs = len(jobs)
    total = sum(len(ops) for ops in jobs)
    beam_width = 256

    def make_state():
        return {
            "job_ready": [0.0] * n_jobs,
            "machine_ready": [0.0] * n_machines,
            "next_op": [0] * n_jobs,
            "seq": [],
        }

    beam = [make_state()]

    for _ in range(total):
        candidates = []

        for si, state in enumerate(beam):
            job_ready = state["job_ready"]
            machine_ready = state["machine_ready"]
            next_op = state["next_op"]

            for j, ops in enumerate(jobs):
                if next_op[j] >= len(ops):
                    continue

                m, d = ops[next_op[j]]
                start = job_ready[j] if job_ready[j] > machine_ready[m] else machine_ready[m]
                finish = start + d
                candidates.append((finish, d, j, si, m))

        candidates.sort(key=lambda x: (x[0], x[1], x[2]))

        next_beam = []
        for finish, d, j, si, m in candidates[:beam_width]:
            state = beam[si]
            new_state = {
                "job_ready": state["job_ready"][:],
                "machine_ready": state["machine_ready"][:],
                "next_op": state["next_op"][:],
                "seq": state["seq"] + [j],
            }
            new_state["job_ready"][j] = finish
            new_state["machine_ready"][m] = finish
            new_state["next_op"][j] += 1
            next_beam.append(new_state)

        beam = next_beam

    best = min(
        beam,
        key=lambda s: max(max(s["job_ready"]), max(s["machine_ready"]))
    )

    return best["seq"]
