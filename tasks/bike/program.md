Can you improve this bike-rental predictor? Try one substantive change and we'll measure it.

Implement `solve(problem, seed, budget)`. You receive `features` (column names), `train_x`, `train_y` (hourly counts) and `predict_x`. Return one nonnegative count per prediction row, in order, as a Python list. Use feature names to identify columns.

The UCI Bike Sharing data is split chronologically: train on 2011; validate on January–June 2012. Calendar and observed weather are supplied. Features are season, yr, mnth, hr, holiday, weekday, workingday, weathersit, temp, atemp, hum, windspeed, weekend and elapsed_days. Sunday is weekday 0, Saturday 6. The supplied weekend flag correctly marks both. Elapsed_days counts days since January 1, 2011.

Demand can shift over time. The year flag is constant in the initial training set, so a tree cannot learn an annual growth effect from it. You may change the whole model: temporal trends, linear or additive models, count regression, residual models, calibration, trees and small ensembles are all allowed. The baseline fits a linear calendar/weather/time model in log space, then boosts its residuals. Consider chronological internal validation using only the supplied training labels. Do not use random internal splits to assess temporal generalization, or assume the validation period's labels are available.

Selection minimizes hourly RMSLE: root mean squared error between log(1 + prediction) and log(1 + count). Feedback also reports signed total-count bias (%) and mean log error; negative values mean underprediction. These diagnostics do not change the selection metric. Improving RMSLE need not improve count totals. Avoid repeating the same implementation or merely changing comments.

Python, NumPy, SciPy and scikit-learn 1.7.2 are available. Use one thread, the supplied seed, 15 CPU seconds including imports, and 25 wall seconds. `budget` is remaining CPU time. No network or outside-file access. After 100 attempts, the selected program and baseline will each be refit unchanged on training plus validation and evaluated once on July–December 2012. Use the supplied arrays rather than fixed row counts.

Give a short explanation and the complete solver in one Python code block.
