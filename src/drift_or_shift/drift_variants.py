"""Utilities for introducing and detecting practical drift patterns."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.neighbors import KernelDensity


def apply_covariance_shift(covariance: ArrayLike, scale: float) -> np.ndarray:
    """Scale a covariance matrix to simulate changing feature variance."""
    cov = np.asarray(covariance, dtype=float)
    if scale <= 0:
        raise ValueError("scale must be positive")
    return cov * float(scale)


def shift_mean_vector(mean: ArrayLike, delta: ArrayLike) -> np.ndarray:
    """Shift a mean vector to create feature-space drift."""
    mean_arr = np.asarray(mean, dtype=float)
    delta_arr = np.asarray(delta, dtype=float)
    if mean_arr.shape != delta_arr.shape:
        raise ValueError("mean and delta must share shape")
    return mean_arr + delta_arr


def inject_label_noise(
    labels: ArrayLike, flip_prob: float, rng: np.random.Generator
) -> np.ndarray:
    """Flip a fraction of binary labels to simulate annotation noise."""
    if not 0 <= flip_prob <= 1:
        raise ValueError("flip_prob must be between 0 and 1")
    label_arr = np.asarray(labels, dtype=np.int64).copy()
    if label_arr.size == 0:
        return label_arr
    flips = rng.random(label_arr.shape[0]) < flip_prob
    label_arr[flips] = 1 - label_arr[flips]
    return label_arr


def density_ratio_shift(
    X_source: ArrayLike,
    X_target: ArrayLike,
    bandwidth: float = 1.0,
) -> np.ndarray:
    """Estimate density ratios using kernel density on source and target."""
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    X_source_arr = np.asarray(X_source, dtype=float)
    X_target_arr = np.asarray(X_target, dtype=float)
    if X_source_arr.ndim != 2 or X_target_arr.ndim != 2:
        raise ValueError("inputs must be two-dimensional")
    kde_source = KernelDensity(bandwidth=bandwidth).fit(X_source_arr)
    kde_target = KernelDensity(bandwidth=bandwidth).fit(X_target_arr)
    log_source = kde_source.score_samples(X_target_arr)
    log_target = kde_target.score_samples(X_target_arr)
    ratio = np.exp(log_target - log_source)
    return np.maximum(ratio, 0.0)


def drift_score_from_ratio(ratio: ArrayLike) -> float:
    """Return a simple drift score from density ratios."""
    arr = np.asarray(ratio, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr))
