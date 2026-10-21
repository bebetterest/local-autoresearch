# Bike prediction diagnostics

The completed [bike example](../examples/2026-10-21-bike) uses elapsed days, a Saturday/Sunday weekend flag, and a time-trend plus boosted-residual baseline. Training is 2011, validation is January–June 2012, and the held-out test is July–December 2012.

## Measured accuracy and calibration

| Split and program | Hourly RMSLE | Total rental bias | Mean log error |
| --- | ---: | ---: | ---: |
| Validation baseline | 0.535904 | −5.84% | −0.0323 |
| Validation selected | 0.370806 | −4.43% | −0.0538 |
| Test baseline, refit | 0.395280 | −5.89% | +0.0255 |
| Test selected, refit | 0.359408 | −1.16% | +0.0274 |

Every value is recomputed from the saved hourly predictions. Validation improved by 30.81%; the matched held-out comparison improved by 9.08%. Both final fits used identical data, rows, seed and time budgets. The final evaluations were performed once after selecting attempt 34.

Daily totals in the overview are sums of the hourly predictions across the full validation period. A fixed vertical scale covers all retained programs. Daily-total agreement and hourly RMSLE are different measurements: aggregation can hide hourly errors. The progress curves always show hourly validation RMSLE.

## What changed

Flint explored a nonlinear trend and separated its inputs from the residual booster. The selected program combines annual features, log time and quadratic time in a Tweedie regressor fitted to **log rental counts**, followed by boosted residuals. It is not a Tweedie count-likelihood fit to raw rental counts. The annotated changes were checked against parent/child code, without attributing isolated causal gains.

Rental demand changes between years. Explicit elapsed time permits extrapolation, while the year flag alone is constant in the training period. Future-hour observed weather is supplied, so this setup differs from operational forecasting with unknown future weather. One split, one seed and repeated validation feedback limit generalization claims.

[Task specification](../tasks/bike/program.md) · [Selected program](../examples/2026-10-21-bike/034/solver.py) · [Saved measurements](../examples/2026-10-21-bike/notes.json).
