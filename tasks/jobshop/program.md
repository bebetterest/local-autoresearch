Can you make this factory schedule finish sooner? Try one improvement and we'll measure it.

Implement `solve(problem, seed, budget)`. `problem["jobs"]` contains 40 jobs, each with 20 ordered `[machine, duration]` operations. `problem["machines"]` is 20. Machines are numbered 0–19.

Return a list of 800 job indices (0–39), with each job appearing 20 times. Each occurrence schedules that job's next operation. Append it after both the job's previous operation and the machine's previous operation finish: `start = max(job_ready[job], machine_ready[machine])`. Update both finish times. Operations cannot be interrupted. The evaluator reconstructs this schedule and minimizes its latest finish time; the score is minus that time.

You have Python, NumPy, SciPy and 15 CPU seconds including imports, with a 25-second wall limit. `budget` is the remaining CPU time; use the supplied seed for randomness. One fixed synthetic instance, seed 0. No network or stored solutions.

Give a short explanation and the complete solver in one Python code block. I'll show you the measured result next time.
