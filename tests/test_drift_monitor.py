"""Tests for the feature drift monitoring helpers."""

from __future__ import annotations

import numpy as np
import pytest

from drift_or_shift.drift_monitor import feature_drift_summary, univariate_feature_stats


def test_univariate_feature_stats_simple_case() -> None:
    X_ref = np.array([[0.0, 1.0], [2.0, 3.0]])
    X_target = np.array([[1.0, 0.5], [3.0, 2.0]])
    stats = univariate_feature_stats(X_ref, X_target)
    assert len(stats) == 2
    assert stats[0]["feature"] == 0
    assert stats[1]["feature"] == 1
    assert stats[0]["mean_diff"] == pytest.approx(1.0)


def test_feature_drift_summary_aggregates_metrics() -> None:
    stats = [
        {"feature": 0, "mean_diff": 0.1, "std_diff": 0.2, "ks_stat": 0.1},
        {"feature": 1, "mean_diff": 0.5, "std_diff": 0.1, "ks_stat": 0.4},
    ]
    summary = feature_drift_summary(stats)
    assert summary["feature_max_mean_diff"] == pytest.approx(0.5)
    assert summary["feature_max_std_diff"] == pytest.approx(0.2)
    assert summary["feature_max_ks"] == pytest.approx(0.4)
    assert summary["feature_mean_ks"] == pytest.approx(0.25)
