"""CLI — generate the labeled series, run every detector, print a ranked F1 table."""
from __future__ import annotations

import argparse
import json

from .data import make_series
from .detectors import DETECTORS
from .evaluate import score


def run(n: int = 2000, seed: int = 7) -> dict:
    series = make_series(n=n, seed=seed)
    out = {}
    for name, fn in DETECTORS.items():
        pred = fn(series.values)
        out[name] = score(pred, series.labels)
    return {"detectors": out,
            "_meta": {"n": n, "n_anomalies": int(series.labels.sum()),
                      "anomaly_rate": round(float(series.labels.mean()), 4)}}


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark time-series anomaly detectors.")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    res = run(args.n, args.seed)
    if args.json:
        print(json.dumps(res, indent=2))
        return

    m = res["_meta"]
    print("=" * 72)
    print(f"  TIME-SERIES ANOMALY DETECTION   (n={m['n']}, "
          f"{m['n_anomalies']} anomalous points, rate {m['anomaly_rate']:.1%})")
    print("=" * 72)
    print(f"{'detector':<18}{'P':>8}{'R':>8}{'F1':>8}   |  {'adjP':>7}{'adjR':>7}{'adjF1':>7}")
    print("-" * 72)
    ranked = sorted(res["detectors"].items(),
                    key=lambda kv: kv[1]["point_adjusted"]["f1"], reverse=True)
    for name, s in ranked:
        p, a = s["point"], s["point_adjusted"]
        print(f"{name:<18}{p['precision']:>8.3f}{p['recall']:>8.3f}{p['f1']:>8.3f}   |  "
              f"{a['precision']:>7.3f}{a['recall']:>7.3f}{a['f1']:>7.3f}")
    print("-" * 72)
    best = ranked[0]
    print(f"best (point-adjusted F1): {best[0]}  "
          f"F1={best[1]['point_adjusted']['f1']:.3f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
