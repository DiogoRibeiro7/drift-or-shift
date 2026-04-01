"""Synthetic data and resampling helpers for label shift experiments."""

from __future__ import annotations

from typing import Sequence, cast

import numpy as np


def default_mu0(d: int) -> np.ndarray:
    """Return the default negative-class mean (zeros)."""
    return np.zeros(d)


def default_mu1(d: int) -> np.ndarray:
    """Return the default positive-class mean (first five dims at 1)."""
    mu = np.zeros(d)
    mu[: min(5, d)] = 1.0
    return mu


def concept_drift_mu1(mu1: Sequence[float], shift_dim: int = 5, delta: float = 0.5) -> np.ndarray:
    """Return mu1 shifted along one dimension to simulate concept drift."""
    mu_arr = np.asarray(mu1, dtype=float)
    if shift_dim >= mu_arr.shape[0]:
        raise IndexError("shift_dim must be within the dimensionality of mu1.")
    mu_arr = mu_arr.copy()
    mu_arr[shift_dim] += delta
    return mu_arr


def _validate_prevalence(pi: float) -> None:
    if not 0 < pi < 1:
        raise ValueError("pi must lie in the open interval (0, 1).")


def _build_covariance(sigma: float | np.ndarray, d: int) -> np.ndarray:
    if np.isscalar(sigma):
        scalar = float(cast(float, np.asarray(sigma, dtype=float)))
        return np.eye(d, dtype=float) * scalar
    cov = np.asarray(sigma, dtype=float)
    if cov.shape != (d, d):
        raise ValueError("Covariance must be scalar or a d-by-d array.")
    return cov


def make_gaussian_binary(
    n: int,
    d: int,
    pi: float,
    mu0: Sequence[float],
    mu1: Sequence[float],
    sigma: float | np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate binary Gaussian data for label shift experiments."""
    if n <= 0:
        raise ValueError("n must be positive.")
    if d <= 0:
        raise ValueError("d must be positive.")
    _validate_prevalence(pi)
    mu0_arr = np.asarray(mu0, dtype=float)
    mu1_arr = np.asarray(mu1, dtype=float)
    if mu0_arr.shape != (d,) or mu1_arr.shape != (d,):
        raise ValueError("Mean vectors must match dimensionality d.")
    cov = _build_covariance(sigma, d)

    probabilities = [1 - pi, pi]
    labels = rng.choice([0, 1], size=n, p=probabilities)
    X = np.empty((n, d), dtype=float)
    if np.any(labels == 0):
        count0 = np.sum(labels == 0)
        X[labels == 0] = rng.multivariate_normal(mu0_arr, cov, size=count0)
    if np.any(labels == 1):
        count1 = np.sum(labels == 1)
        X[labels == 1] = rng.multivariate_normal(mu1_arr, cov, size=count1)
    return X, labels


def resample_to_prevalence(
    X: np.ndarray, y: np.ndarray, pi_target: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a fixed pool to a target prevalence via stratified sampling."""
    _validate_prevalence(pi_target)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    n = len(y)
    target_pos = int(round(pi_target * n))
    target_neg = n - target_pos
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    replace_pos = target_pos > len(pos_idx)
    replace_neg = target_neg > len(neg_idx)
    chosen_pos = rng.choice(pos_idx, size=target_pos, replace=replace_pos)
    chosen_neg = rng.choice(neg_idx, size=target_neg, replace=replace_neg)
    indices = np.concatenate([chosen_pos, chosen_neg])
    rng.shuffle(indices)
    return X[indices], y[indices]


def make_multimodal_binary(
    n: int,
    d: int,
    pi: float,
    components_neg: Sequence[tuple[Sequence[float], float]],
    components_pos: Sequence[tuple[Sequence[float], float]],
    sigma: float | np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a binary dataset where each class is a mixture of Gaussians."""
    if n <= 0:
        raise ValueError("n must be positive.")
    if d <= 0:
        raise ValueError("d must be positive.")
    _validate_prevalence(pi)
    neg_means, neg_weights = _prep_mixture(components_neg, d)
    pos_means, pos_weights = _prep_mixture(components_pos, d)
    cov = _build_covariance(sigma, d)
    labels = rng.choice([0, 1], size=n, p=[1 - pi, pi])
    X = np.empty((n, d), dtype=float)
    if np.any(labels == 0):
        X[labels == 0] = _sample_mixture(cov, neg_means, neg_weights, rng, size=np.sum(labels == 0))
    if np.any(labels == 1):
        X[labels == 1] = _sample_mixture(cov, pos_means, pos_weights, rng, size=np.sum(labels == 1))
    return X, labels


def _prep_mixture(
    components: Sequence[tuple[Sequence[float], float]], d: int
) -> tuple[list[np.ndarray], np.ndarray]:
    if not components:
        raise ValueError("At least one mixture component is required.")
    means: list[np.ndarray] = []
    weights = []
    for mean, weight in components:
        if weight < 0:
            raise ValueError("Mixture weights must be non-negative.")
        arr = np.asarray(mean, dtype=float)
        if arr.shape != (d,):
            raise ValueError("Component means must match dimensionality d.")
        means.append(arr)
        weights.append(weight)
    weights_arr = np.asarray(weights, dtype=float)
    total = float(weights_arr.sum())
    if total <= 0:
        raise ValueError("Mixture weights must sum to a positive value.")
    return means, weights_arr / total


def _sample_mixture(
    cov: np.ndarray,
    means: Sequence[np.ndarray],
    weights: np.ndarray,
    rng: np.random.Generator,
    *,
    size: int,
) -> np.ndarray:
    if size == 0:
        return np.empty((0, cov.shape[0]))
    indices = rng.choice(len(means), size=size, p=weights)
    samples = [rng.multivariate_normal(means[idx], cov, size=1)[0] for idx in indices]
    return np.vstack(samples)
