# Service depots

Place eight depots among 100 synthetic customer locations in a unit square. Each customer has a demand weight from 1 to 5 and uses its nearest depot. Minimize the **demand-weighted mean Euclidean distance**.

This is a weighted *k*-median problem using straight-line distances, with no capacity limits or opening costs.

The baseline adds depots one at a time, choosing the site that most reduces weighted distance. It never revisits an earlier choice. Flint can try exchanging sites or reconsidering combinations.

Return eight distinct site indices. The evaluator computes the distances and returns minus the weighted mean as the score. In the plot, larger blue dots mean more demand, orange squares are depots, and lines connect customers to their nearest depot.

```sh
uv run python -m tasks check facility
uv run python research.py --task facility --run runs/depot-demo --experiments 2 --max-tokens 16384
uv run python report.py runs/depot-demo
```
