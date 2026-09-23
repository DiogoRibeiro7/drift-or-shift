"""Lightweight helpers for monitoring feature drift."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike


def _ks_statistic(reference: np.ndarray, target: np.ndarray) -> float:
    if reference.size == 0 or target.size == 0:
        return 0.0
    reference_sorted = np.sort(reference)
    target_sorted = np.sort(target)
    combined = np.sort(np.concatenate([reference_sorted, target_sorted]))
    cdf_ref = (
        np.searchsorted(reference_sorted, combined, side="right")
        / reference_sorted.size
    )
    cdf_target = (
        np.searchsorted(target_sorted, combined, side="right") / target_sorted.size
    )
    return float(np.max(np.abs(cdf_ref - cdf_target)))


def _covariance_matrix(data: np.ndarray) -> np.ndarray:
    return np.atleast_2d(np.cov(data, rowvar=False))


def _correlation_matrix(data: np.ndarray) -> np.ndarray:
    if data.shape[1] == 0:
        return np.zeros((0, 0))
    corr = np.corrcoef(data, rowvar=False)
    if corr.ndim == 0:
        corr = np.array([[corr]])
    return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)


def covariance_frobenius_diff(reference: np.ndarray, target: np.ndarray) -> float:
    """Return the Frobenius difference between training and target covariance matrices."""
    cov_ref = _covariance_matrix(reference)
    cov_target = _covariance_matrix(target)
    return float(np.linalg.norm(cov_ref - cov_target, ord="fro"))


def correlation_mean_diff(reference: np.ndarray, target: np.ndarray) -> float:
    """Compute average absolute deviation between correlation matrices (off-diagonal)."""
    d = reference.shape[1]
    if d <= 1:
        return 0.0
    corr_ref = _correlation_matrix(reference)
    corr_target = _correlation_matrix(target)
    mask = ~np.eye(d, dtype=bool)
    diffs = np.abs(corr_ref - corr_target)
    return float(np.mean(diffs[mask]))


def multivariate_projection_ks(
    reference: np.ndarray,
    target: np.ndarray,
    *,
    projections: int = 12,
    rng: np.random.Generator | None = None,
) -> list[float]:
    """Approximate multivariate drift via KS on random projections."""
    rng = rng or np.random.default_rng(0)
    d = reference.shape[1]
    if d == 0 or projections <= 0:
        return [0.0]
    stats: list[float] = []
    for _ in range(projections):
        direction = np.asarray(rng.normal(size=d), dtype=float)
        norm = np.linalg.norm(direction)
        if norm == 0:
            continue
        direction /= norm
        ref_proj = reference @ direction
        target_proj = target @ direction
        stats.append(_ks_statistic(ref_proj, target_proj))
    return stats or [0.0]


def univariate_feature_stats(
    X_ref: ArrayLike,
    X_target: ArrayLike,
) -> list[dict]:
    """Return per-feature drift statistics between reference and target arrays."""
    ref_arr = np.asarray(X_ref, dtype=float)
    target_arr = np.asarray(X_target, dtype=float)
    if ref_arr.ndim != 2 or target_arr.ndim != 2:
        raise ValueError("input arrays must be two-dimensional")
    if ref_arr.shape[1] != target_arr.shape[1]:
        raise ValueError("feature dimensionality must match")
    if ref_arr.shape[0] == 0 or target_arr.shape[0] == 0:
        raise ValueError("input datasets must contain samples")
    stats: list[dict] = []
    for idx in range(ref_arr.shape[1]):
        ref_col = ref_arr[:, idx]
        target_col = target_arr[:, idx]
        stats.append(
            {
                "feature": idx,
                "mean_diff": float(np.abs(ref_col.mean() - target_col.mean())),
                "std_diff": float(np.abs(ref_col.std(ddof=0) - target_col.std(ddof=0))),
                "ks_stat": _ks_statistic(ref_col, target_col),
            }
        )
    return stats


def feature_drift_summary(stats: Sequence[dict]) -> dict[str, float]:
    """Aggregate univariate stats into summary metrics suitable for tracking."""
    if not stats:
        raise ValueError("stats cannot be empty")
    mean_diffs = [entry["mean_diff"] for entry in stats]
    std_diffs = [entry["std_diff"] for entry in stats]
    ks_stats = [entry["ks_stat"] for entry in stats]
    return {
        "feature_max_mean_diff": float(max(mean_diffs)),
        "feature_max_std_diff": float(max(std_diffs)),
        "feature_max_ks": float(max(ks_stats)),
        "feature_mean_ks": float(np.mean(ks_stats)),
    }
