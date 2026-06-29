# timeseries-anomaly-detection

A time-series anomaly detection **bench**, not a plotting demo. It generates a synthetic
series with a **known anomaly ground truth** — point spikes, sustained level shifts, and
variance bursts injected at recorded indices — then runs seven detectors and scores each on
precision / recall / F1 and **point-adjusted F1**.

```bash
tsanomaly            # ranked F1 table
tsanomaly --json
tsanomaly --n 5000 --seed 1
```

## Detectors

| detector | idea |
|---|---|
| `rolling_zscore` | distance from a rolling mean in rolling-std units |
| `iqr` | Tukey fences on a rolling window (robust to spikes) |
| `ewma` | residual from an exponentially weighted moving average |
| `stl_residual` | STL seasonal decomposition, flag a large remainder |
| `isolation_forest` | unsupervised tree ensemble on lagged features |
| `rolling_zscore_deseason` · `iqr_deseason` | the naive detectors run **on the STL remainder** |

## Measured results

`tsanomaly` on the default series (n=2000, 165 anomalous points, 8.2% rate):

| detector | P | R | F1 | adj-P | adj-R | **adj-F1** |
|---|---|---|---|---|---|---|
| **iqr_deseason** | 0.817 | 0.351 | 0.491 | 0.926 | 0.988 | **0.956** |
| rolling_zscore_deseason | 0.903 | 0.170 | 0.286 | 0.977 | 0.788 | 0.873 |
| ewma | 0.526 | 0.242 | 0.332 | 0.775 | 0.751 | 0.763 |
| isolation_forest | 0.680 | 0.412 | 0.513 | 0.784 | 0.703 | 0.741 |
| stl_residual | 1.000 | 0.261 | 0.413 | 1.000 | 0.558 | 0.716 |
| rolling_zscore | 1.000 | 0.012 | 0.024 | 1.000 | 0.012 | 0.024 |
| iqr | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## The headline: seasonality is what breaks naive detectors

On a series with a daily + weekly cycle, the textbook rolling detectors **collapse**:

- raw **IQR → adjusted-F1 0.000** (the seasonal swing widens the rolling fences so far that
  nothing trips them),
- raw **rolling z-score → 0.024** (same cause — the rolling std is dominated by the season, not the noise).

Strip the seasonality with STL first and run the *exact same detectors* on the remainder:

- **IQR: 0.000 → 0.956** adjusted-F1,
- **rolling z-score: 0.024 → 0.873**.

That ~0.95 jump is the entire lesson of practical anomaly detection: **a simple detector on a
properly deseasonalized signal beats a fancy detector on the raw signal.** `isolation_forest`
(0.741) and even `stl_residual`'s own remainder thresholding (0.716) lose to a deseasonalized
IQR. `stl_residual` gets perfect precision but low recall — it only fires on the sharpest
remainders.

## Point-adjusted F1

The `adj-*` columns use standard point-adjustment: flagging *any* point inside a true
anomalous segment counts as catching that segment. A detector shouldn't be punished for
flagging a 30-step level shift one step late. Both pointwise and adjusted numbers are
reported so neither metric can flatter a detector alone.

## Install & test

```bash
pip install -e ".[dev]"
pytest -q          # 6 passed
```

## Stack

NumPy / pandas rolling stats, statsmodels STL, scikit-learn IsolationForest. Ground-truth
labels + point-adjusted scoring throughout.

## License

MIT
