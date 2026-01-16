"""Shared configuration and helpers for DriftOrShift experiments."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from drift_or_shift.drift_monitor import (
    covariance_frobenius_diff,
    correlation_mean_diff,
    feature_drift_summary,
    multivariate_projection_ks,
    univariate_feature_stats,
)

SEEDS = (0, 1, 2, 3, 4)
PI_TEST_GRID = (0.5, 0.2, 0.1, 0.05, 0.01)
PI_TRAIN = 0.2
DEFAULT_COSTS = {"c10": 1.0, "c01": 1.0}
DRIFT_FEATURE_METRICS = (
    "feature_max_mean_diff",
    "feature_max_std_diff",
    "feature_max_ks",
    "feature_mean_ks",
    "feature_covariance_fro_diff",
    "feature_correlation_mean_diff",
    "feature_projection_max_ks",
    "feature_projection_mean_ks",
)


@dataclass(frozen=True)
class ExperimentConfig:
    exp_name: str
    n_train: int
    n_test: int
    d: int
    pi_train: float = PI_TRAIN
    pi_tests: Sequence[float] = PI_TEST_GRID
    seeds: Sequence[int] = SEEDS
    c10: float = DEFAULT_COSTS["c10"]
    c01: float = DEFAULT_COSTS["c01"]

    def metadata(self) -> dict[str, object]:
        """Return serializable metadata for the run."""
        return {
            "exp_name": self.exp_name,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "d": self.d,
            "pi_train": self.pi_train,
            "pi_tests": list(self.pi_tests),
            "seeds": list(self.seeds),
            "c10": self.c10,
            "c01": self.c01,
            "timestamp": datetime.now().isoformat(),
        }


def _std_ddof0(series: pd.Series) -> float:
    return float(series.std(ddof=0))


_std_ddof0.__name__ = "std"

def aggregate_mean_std(
    df: pd.DataFrame,
    metrics: Sequence[str],
    groupby: Sequence[str] | str = ("pi_test",),
) -> pd.DataFrame:
    """Return mean/std statistics for metrics grouped by the provided keys."""
    if isinstance(groupby, str):
        groupby = (groupby,)
    if groupby:
        grouped = df.groupby(list(groupby))
        agg = grouped[list(metrics)].agg(["mean", _std_ddof0])
        agg.columns = [f"{metric}_{suffix}" for metric, suffix in agg.columns]
        return agg.reset_index()

    agg = df[list(metrics)].agg(["mean", _std_ddof0])
    data = {}
    for metric in metrics:
        data[f"{metric}_mean"] = float(agg.at["mean", metric])
        data[f"{metric}_std"] = float(agg.at["std", metric])
    return pd.DataFrame([data])


def feature_drift_metrics(X_ref: np.ndarray, X_target: np.ndarray) -> dict[str, float]:
    stats = univariate_feature_stats(X_ref, X_target)
    summary = feature_drift_summary(stats)
    summary["feature_covariance_fro_diff"] = covariance_frobenius_diff(X_ref, X_target)
    summary["feature_correlation_mean_diff"] = correlation_mean_diff(X_ref, X_target)
    projection_stats = multivariate_projection_ks(X_ref, X_target)
    summary["feature_projection_max_ks"] = float(max(projection_stats))
    summary["feature_projection_mean_ks"] = float(np.mean(projection_stats))
    return summary
