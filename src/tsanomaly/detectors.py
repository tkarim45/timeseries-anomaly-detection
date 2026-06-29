"""Anomaly detectors. Each takes a 1-D series and returns a 0/1 flag per timestamp.

Span the spectrum from cheap-and-transparent to model-based:
  * rolling_zscore  — distance from a rolling mean in rolling-std units
  * ewma            — residual from an exponentially weighted moving average
  * iqr             — Tukey fences on a rolling window (robust to spikes)
  * stl_residual    — STL seasonal decomposition, flag large remainder (best for seasonal data)
  * isolation_forest— unsupervised tree ensemble on lagged features
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def deseasonalize(x: np.ndarray, period: int = 24) -> np.ndarray:
    """Remove seasonality + trend via STL, returning the remainder. Simple distance
    detectors are only valid on a series whose seasonal swing has been stripped out."""
    from statsmodels.tsa.seasonal import STL
    res = STL(x, period=period, robust=True).fit()
    return np.asarray(res.resid)


def rolling_zscore(x: np.ndarray, window: int = 48, z: float = 3.5) -> np.ndarray:
    s = pd.Series(x)
    mean = s.rolling(window, min_periods=window // 2).mean()
    std = s.rolling(window, min_periods=window // 2).std().replace(0, np.nan)
    score = (s - mean).abs() / std
    return (score > z).fillna(False).to_numpy().astype(int)


def ewma(x: np.ndarray, span: int = 24, k: float = 3.5) -> np.ndarray:
    s = pd.Series(x)
    fit = s.ewm(span=span).mean()
    resid = (s - fit).abs()
    thresh = resid.rolling(span * 4, min_periods=span).std() * k
    return (resid > thresh).fillna(False).to_numpy().astype(int)


def iqr(x: np.ndarray, window: int = 48, k: float = 3.0) -> np.ndarray:
    s = pd.Series(x)
    q1 = s.rolling(window, min_periods=window // 2).quantile(0.25)
    q3 = s.rolling(window, min_periods=window // 2).quantile(0.75)
    span = (q3 - q1)
    lo, hi = q1 - k * span, q3 + k * span
    return ((s < lo) | (s > hi)).fillna(False).to_numpy().astype(int)


def stl_residual(x: np.ndarray, period: int = 24, k: float = 4.0) -> np.ndarray:
    from statsmodels.tsa.seasonal import STL
    res = STL(x, period=period, robust=True).fit()
    r = pd.Series(res.resid)
    thresh = k * r.std()
    return (r.abs() > thresh).to_numpy().astype(int)


def isolation_forest(x: np.ndarray, lags: int = 6, contamination: float = 0.05,
                     seed: int = 7) -> np.ndarray:
    from sklearn.ensemble import IsolationForest
    s = pd.Series(x)
    feats = pd.DataFrame({f"lag{i}": s.shift(i) for i in range(lags)})
    feats["diff"] = s.diff()
    feats["roll_mean"] = s.rolling(24, min_periods=1).mean()
    feats = feats.bfill().fillna(0)
    model = IsolationForest(contamination=contamination, random_state=seed, n_estimators=200)
    pred = model.fit_predict(feats.to_numpy())
    return (pred == -1).astype(int)


def rolling_zscore_deseason(x: np.ndarray, period: int = 24, **kw) -> np.ndarray:
    """The same z-score detector, but run on the STL remainder. Demonstrates the lift from
    accounting for seasonality before flagging distance outliers."""
    return rolling_zscore(deseasonalize(x, period), **kw)


def iqr_deseason(x: np.ndarray, period: int = 24, **kw) -> np.ndarray:
    return iqr(deseasonalize(x, period), **kw)


DETECTORS = {
    "rolling_zscore": rolling_zscore,
    "iqr": iqr,
    "ewma": ewma,
    "rolling_zscore_deseason": rolling_zscore_deseason,
    "iqr_deseason": iqr_deseason,
    "stl_residual": stl_residual,
    "isolation_forest": isolation_forest,
}
