"""Compare feature distributions and raise alerts when drift exceeds a threshold."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from drift_or_shift.drift_monitor import feature_drift_summary, univariate_feature_stats
from drift_or_shift.reporting import DRIFT_ALERT_THRESHOLDS


def _load_features(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"input file {path} is empty")
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        raise ValueError(f"no numeric features found in {path}")
    return numeric.to_numpy()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute drift stats between reference and target data."
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument(
        "--mean-threshold",
        type=float,
        default=DRIFT_ALERT_THRESHOLDS["feature_max_mean_diff"],
    )
    parser.add_argument(
        "--std-threshold",
        type=float,
        default=DRIFT_ALERT_THRESHOLDS["feature_max_std_diff"],
    )
    parser.add_argument(
        "--ks-threshold", type=float, default=DRIFT_ALERT_THRESHOLDS["feature_max_ks"]
    )
    return parser.parse_args()


def _thresholds_from_args(args: argparse.Namespace) -> dict[str, float]:
    return {
        "feature_max_mean_diff": args.mean_threshold,
        "feature_max_std_diff": args.std_threshold,
        "feature_max_ks": args.ks_threshold,
    }


def main() -> None:
    args = _parse_args()
    reference = _load_features(args.reference)
    target = _load_features(args.target)
    stats = univariate_feature_stats(reference, target)
    summary = feature_drift_summary(stats)
    thresholds = _thresholds_from_args(args)
    alerts = [
        (metric, summary.get(metric, 0.0), threshold)
        for metric, threshold in thresholds.items()
        if summary.get(metric, 0.0) >= threshold
    ]
    print("Feature drift summary:")
    for name, value in summary.items():
        print(f"- {name}: {value:.4f}")
    if alerts:
        print("Drift alerts:")
        for metric, value, threshold in alerts:
            print(f"- {metric}: {value:.4f} exceeds threshold {threshold:.4f}")
        raise SystemExit(1)
    print("No drift alerts detected.")


if __name__ == "__main__":
    main()
