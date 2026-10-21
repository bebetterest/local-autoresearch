# Factory scheduling

Forty jobs share twenty machines. Each job has twenty operations that must happen in order, giving 800 operations in total. Every operation has a fixed machine and duration. A machine can process only one operation at a time.

The solver returns job indices in dispatch order. Each occurrence schedules that job's next operation, starting when both the job and its machine are ready. This representation always produces a feasible schedule when every job appears the required number of times. The evaluator computes the final finishing time itself.

The baseline takes one operation from each job in turn. Improving the order can reduce machine idle time, but moving one operation also changes when later operations become available. The timeline plot shows those interactions as work and gaps across machines.

This is a synthetic [job-shop scheduling problem](https://developers.google.com/optimization/scheduling/job_shop). The instance and evaluation seed stay fixed; candidates get 15 CPU seconds each.

```sh
uv run python -m tasks check jobshop
uv run python research.py --task jobshop --run runs/factory-demo --experiments 100 --max-tokens 16384
uv run python report.py runs/factory-demo
```
