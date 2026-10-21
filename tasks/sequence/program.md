Can you improve this binary sequence search? Try one change and we'll measure it.

Implement `solve(n, seed, budget)` and return n integers, each -1 or +1. Here n is 256. For each lag k from 1 to n-1, the aperiodic autocorrelation is `C[k] = sum(s[i] * s[i+k] for i in range(n-k))`. Minimize `sum(C[k]**2)`. The evaluator recomputes this exactly; its score is minus that energy, so higher is better.

You have Python, NumPy, SciPy and 15 CPU seconds including imports, with a 25-second wall limit. `budget` is the remaining CPU time. Use the supplied seed (0) for randomness. No network or stored sequences: search for the bits yourself.

Give a short explanation and the complete solver in one Python code block. Keep it simple enough to finish. I'll show you the measured result next time.
