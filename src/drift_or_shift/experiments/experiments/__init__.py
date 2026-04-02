"""Legacy compatibility package for `drift_or_shift.experiments.experiments`.

This package exists only as a shim so old import paths do not break when the
flattened layout under `drift_or_shift.experiments` is used.
"""

from __future__ import annotations

from pathlib import Path

from .._common import (
    ExperimentConfig,
    aggregate_mean_std,
    DEFAULT_COSTS,
    PI_TEST_GRID,
    PI_TRAIN,
    SEEDS,
)  # noqa: F401

# Make `drift_or_shift.experiments.experiments.<experiment_module>` resolve in
# `drift_or_shift.experiments` without copying data files.
__path__.insert(0, str(Path(__file__).resolve().parents[1]))

__all__ = [
    "ExperimentConfig",
    "aggregate_mean_std",
    "DEFAULT_COSTS",
    "PI_TEST_GRID",
    "PI_TRAIN",
    "SEEDS",
]
