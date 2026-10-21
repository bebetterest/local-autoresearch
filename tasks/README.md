# Tasks

Each solver returns an answer for the evaluator to score. New runs use one fixed instance and seed 0, with 15 CPU seconds and 25 wall seconds per attempt.

## Recorded examples

| Task key | Problem | Objective | Full run |
| --- | --- | --- | --- |
| `circle` | Choose positions and radii for 26 circles inside a unit square, without overlaps. | Maximize the sum of radii. | [128 attempts](../examples/2026-10-20-final) |
| `jobshop` | Order 800 operations from 40 jobs across 20 shared machines, preserving each job's operation order. | Minimize the last finishing time. | [100 attempts](../examples/2026-10-20-jobshop) |
| `bike` | Predict hourly rentals from calendar and observed weather, training on 2011 and validating on the first half of 2012. | Minimize validation RMSLE; test once on the second half of 2012. | [100 attempts](../examples/2026-10-21-bike) |

Read the [circle evaluation notes](../examples/2026-10-20-final#evaluation), [factory task](jobshop/README.md) or [bike task and dataset attribution](bike/README.md) for details. The historical circle run averaged three trials; subsequent tasks use one seed.

## Smaller task templates

These smaller problems are starting points for your own experiments; they have no full recorded runs here.

| Task key | Problem | Objective |
| --- | --- | --- |
| `route` | Visit 128 synthetic points exactly once and return to the start. | Minimize total Euclidean route length. |
| `schedule` | Assign 80 independent jobs to 8 machines. Each job's duration depends on its assigned machine. | Minimize the largest machine load. |
| `sequence` | Choose 256 values of −1 or +1 with low correlation between the sequence and shifted copies. | Minimize the sum of squared aperiodic autocorrelations. |
| `facility` | Place 8 depots at distinct sites among 100 weighted customer locations. Each customer uses its nearest depot. | Minimize demand-weighted mean distance. |

```sh
uv run python -m tasks check facility
uv run python research.py --task facility --run runs/depot-demo --experiments 2 --max-tokens 16384
uv run python report.py runs/depot-demo
```

[Add a task](../docs/add-a-task.md) by copying the facility folder and editing its checker, baseline, brief and plot. Tasks are discovered automatically.
