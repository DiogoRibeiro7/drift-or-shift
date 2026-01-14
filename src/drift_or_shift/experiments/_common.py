"""Shared configuration and helpers for DriftOrShift experiments."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Sequence

import pandas as pd

SEEDS = (0, 1, 2, 3, 4)
PI_TEST_GRID = (0.5, 0.2, 0.1, 0.05, 0.01)
PI_TRAIN = 0.2
DEFAULT_COSTS = {"c10": 1.0, "c01": 1.0}


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
    grouped = df.groupby(list(groupby))
    agg = grouped[list(metrics)].agg(["mean", _std_ddof0])
    agg.columns = [f"{metric}_{suffix}" for metric, suffix in agg.columns]
    return agg.reset_index()
