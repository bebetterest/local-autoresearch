Can you make these jobs finish sooner? Try one improvement and we'll measure it.

Implement `solve(durations, seed, budget)`. There are 80 jobs and 8 machines. `durations[j][m]` is how long job j takes on machine m. Return one machine index per job. Every job must run once; a machine runs its assigned jobs sequentially. We minimize the largest machine load on this fixed synthetic instance. The evaluator computes the loads itself; its score is minus the finishing time, so higher is better.

You have Python, NumPy, SciPy and 15 CPU seconds including imports, with a 25-second wall limit. `budget` is the remaining CPU time; use `seed` for any randomness. No network or stored solutions. This run uses seed 0.

Give a short explanation and the complete solver in one Python code block. Keep it simple enough to finish. I'll show you the measured result next time.
