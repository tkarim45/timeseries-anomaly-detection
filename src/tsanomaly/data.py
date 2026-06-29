"""Synthetic time series with a KNOWN anomaly ground truth.

Base signal = trend + daily/weekly seasonality + gaussian noise. Three kinds of anomaly are
injected with recorded indices so every detector can be scored against the same truth:

  * spikes    — isolated point outliers (sudden single-step jumps)
  * level shifts — a sustained change in the mean over a window
  * variance bursts — a window of inflated noise

This is what separates a real anomaly toolkit from a plotting demo: the labels exist.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Series:
    values: np.ndarray
    labels: np.ndarray   # 1 = anomalous timestamp, 0 = normal


def make_series(n: int = 2000, seed: int = 7, n_spikes: int = 20,
                n_level_shifts: int = 3, n_variance_bursts: int = 3) -> Series:
    rng = np.random.default_rng(seed)
    t = np.arange(n)

    trend = 0.002 * t
    daily = 3.0 * np.sin(2 * np.pi * t / 24)
    weekly = 1.5 * np.sin(2 * np.pi * t / (24 * 7))
    noise = rng.normal(0, 0.6, n)
    values = 20 + trend + daily + weekly + noise
    labels = np.zeros(n, dtype=int)

    # avoid the warmup region so rolling detectors have history
    valid = np.arange(72, n - 48)

    # 1) point spikes
    spike_idx = rng.choice(valid, size=n_spikes, replace=False)
    signs = rng.choice([-1, 1], size=n_spikes)
    mags = rng.uniform(6, 10, size=n_spikes)        # well outside the ~±1.8 noise band
    values[spike_idx] += signs * mags
    labels[spike_idx] = 1

    # 2) sustained level shifts
    for _ in range(n_level_shifts):
        start = int(rng.choice(valid))
        width = int(rng.integers(20, 40))
        end = min(start + width, n)
        values[start:end] += rng.choice([-1, 1]) * rng.uniform(4, 6)
        labels[start:end] = 1

    # 3) variance bursts
    for _ in range(n_variance_bursts):
        start = int(rng.choice(valid))
        width = int(rng.integers(15, 30))
        end = min(start + width, n)
        values[start:end] += rng.normal(0, 4, end - start)
        labels[start:end] = 1

    return Series(values=values, labels=labels)
