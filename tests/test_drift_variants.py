"""Tests for drift variant helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from drift_or_shift.drift_variants import (
    apply_covariance_shift,
    density_ratio_shift,
    drift_score_from_ratio,
    inject_label_noise,
    shift_mean_vector,
)


def test_shift_mean_vector() -> None:
    mean = [0.0, 0.5]
    delta = [0.2, -0.3]
    shifted = shift_mean_vector(mean, delta)
    assert np.allclose(shifted, [0.2, 0.2])


def test_inject_label_noise() -> None:
    rng = np.random.default_rng(0)
    y = np.zeros(10, dtype=int)
    noisy = inject_label_noise(y, 0.5, rng)
    assert noisy.shape == y.shape
    assert noisy.dtype == np.int64
    assert 0 <= noisy.sum() <= 10


def test_density_ratio_shift_detects_drift(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    source = rng.normal(0, 1, size=(200, 2))
    target = source + 2.0
    ratio = density_ratio_shift(source, target, bandwidth=1.0)
    assert np.all(ratio >= 0)
    score = drift_score_from_ratio(ratio)
    assert score > 1.0


def test_apply_covariance_shift() -> None:
    cov = np.eye(2)
    scaled = apply_covariance_shift(cov, 2.0)
    assert np.allclose(scaled, np.eye(2) * 2)
