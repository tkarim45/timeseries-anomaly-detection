import numpy as np

from tsanomaly.cli import run
from tsanomaly.data import make_series
from tsanomaly.detectors import DETECTORS
from tsanomaly.evaluate import score


def test_series_has_labeled_anomalies():
    s = make_series(n=2000)
    assert s.values.shape == s.labels.shape == (2000,)
    assert 0.02 < s.labels.mean() < 0.20      # a sensible anomaly rate
    assert set(np.unique(s.labels)) <= {0, 1}


def test_detectors_return_binary_same_length():
    s = make_series(n=1000)
    for name, fn in DETECTORS.items():
        flags = fn(s.values)
        assert flags.shape == (1000,), name
        assert set(np.unique(flags)) <= {0, 1}, name


def test_point_adjusted_never_below_pointwise():
    s = make_series(n=1500)
    for fn in DETECTORS.values():
        sc = score(fn(s.values), s.labels)
        assert sc["point_adjusted"]["f1"] >= sc["point"]["f1"] - 1e-9


def test_deseasonalizing_rescues_naive_detectors():
    res = run()["detectors"]
    # raw rolling detectors collapse on seasonal data; deseasonalizing rescues them
    assert res["iqr"]["point_adjusted"]["f1"] < 0.1
    assert res["iqr_deseason"]["point_adjusted"]["f1"] > 0.8
    assert (res["rolling_zscore_deseason"]["point_adjusted"]["f1"]
            > res["rolling_zscore"]["point_adjusted"]["f1"] + 0.5)


def test_stl_residual_high_precision():
    res = run()["detectors"]
    assert res["stl_residual"]["point"]["precision"] >= 0.9


def test_best_detector_strong():
    res = run()["detectors"]
    best = max(d["point_adjusted"]["f1"] for d in res.values())
    assert best > 0.9
