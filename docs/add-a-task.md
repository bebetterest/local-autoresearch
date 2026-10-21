# Add a task

## Tweak a baseline

Start with a small change to an existing solver. After [setup](running.md#setup), run this from the repository root:

```sh
uv run python -m tasks check bike
```

Try changing `max_iter=100` to `max_iter=150` in [the bike baseline](../tasks/bike/baseline.py), then run the command again. Compare the score and CPU time. For bike demand, a score closer to zero means lower prediction error. These checks work without a model server.

## Make a new task

Copy the depot-placement task; the runner discovers the new folder automatically:

```sh
cp -R tasks/facility tasks/my_task
```

Start with these three files:

| File | What you write |
| --- | --- |
| `__init__.py` | Fixed input in `PROBLEM`, and `check(solution)` to validate and score an answer |
| `baseline.py` | A simple working `solve(problem, seed, budget)` |
| `program.md` | A short request for Flint |

Use a lowercase module name such as `my_task` and relative imports such as `from . import PROBLEM`. Update the copied README with your task's description and run command.

`PROBLEM` is the JSON-serializable input passed to the solver. Use public or synthetic data with a fixed generation seed. For ML tasks, keep validation and test labels out of `PROBLEM`; see [bike demand](../tasks/bike).

The solver returns a JSON-serializable answer. `seed` controls randomness; `budget` is the remaining CPU time in seconds, after imports. Return before it expires. Each evaluation allows 15 CPU seconds and 25 wall seconds.

The checker runs separately. Reject invalid answers with `ValueError` and compute the score yourself from the answer and fixed input. Return:

```python
{
    "solution": answer,
    "score": -distance,  # Larger always wins.
    "distance": distance,
}
```

Higher scores win, so negate the measurement for minimization tasks. Keep extra measurements small; they become part of Flint's feedback.

In `program.md`, describe the input, answer, objective and available resources. Ask for one change and one complete Python code block. The runner adds the best solver and recent results.

Check that your starting solver works:

```sh
uv run python -m tasks check my_task
```

With a ready local Flint server (`rec status`), try two attempts:

```sh
uv run python research.py --task my_task --run runs/my-task-demo --experiments 2 --max-tokens 16384
```

The shared tests exercise every discovered baseline. Add a hand-computed scoring test and tests for invalid answers; [test_facility.py](../tests/test_facility.py) shows how. Run them with `uv run pytest -q`.

Start a new run after changing inputs, evaluation code, the baseline, brief, runner or dependencies. Plots and annotations can be updated separately.

## Plot and share

When you want a report, adapt the copied `plot.py`. Set `TITLE`, `METRIC` (the checker's measurement key), `LABEL` (its axis label), and `MINIMIZE`. Implement `draw(axes, trials)` for baseline and best-answer panels, using the same scales. The report adds the progress chart and original model responses.

```sh
uv run python report.py runs/my-task-demo
```

Open `runs/my-task-demo/report.html` to see the result. To save a finished run for sharing:

```sh
uv run python -m tools.export_run runs/my-task-demo examples/my-task-demo
uv run python -m tools.verify_run examples/my-task-demo
```

Before sharing, review the export and describe the baseline and limitations. Keep every attempt, including failures.
