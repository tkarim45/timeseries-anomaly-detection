"""Scoring. Plain point-wise precision/recall/F1, plus point-adjusted F1 — the standard
time-series metric where detecting any point inside a true anomalous segment counts as
catching the whole segment (a detector shouldn't be punished for flagging a level shift one
step late)."""
from __future__ import annotations

import numpy as np


def _prf(pred: np.ndarray, truth: np.ndarray) -> dict:
    tp = int(((pred == 1) & (truth == 1)).sum())
    fp = int(((pred == 1) & (truth == 0)).sum())
    fn = int(((pred == 0) & (truth == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn}


def _point_adjust(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """If any point inside a contiguous true-anomaly segment is flagged, mark the whole
    segment as detected (standard point-adjustment for time-series anomaly detection)."""
    adj = pred.copy()
    in_seg = False
    start = 0
    for i in range(len(truth)):
        if truth[i] == 1 and not in_seg:
            in_seg, start = True, i
        elif truth[i] == 0 and in_seg:
            if pred[start:i].any():
                adj[start:i] = 1
            in_seg = False
    if in_seg and pred[start:].any():
        adj[start:] = 1
    return adj


def score(pred: np.ndarray, truth: np.ndarray) -> dict:
    point = _prf(pred, truth)
    adjusted = _prf(_point_adjust(pred, truth), truth)
    return {"point": point, "point_adjusted": adjusted}
