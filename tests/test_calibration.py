"""Tests for label-shift calibration utilities."""

from __future__ import annotations

import numpy as np
import pytest

from drift_or_shift.calibration import (
    apply_calibrator,
    find_best_temperature,
    fit_isotonic_calibrator,
    temperature_scale,
)


def test_temperature_scale_roundtrip() -> None:
    values = np.array([-2.0, 0.0, 2.0])
    scaled = temperature_scale(values, temperature=2.0)
    assert np.allclose(scaled, values / 2.0)
    with pytest.raises(ValueError):
        temperature_scale(values, temperature=0.0)


def test_find_best_temperature_uses_grid() -> None:
    scores = np.array([1.0, -1.0, 0.0])
    targets = np.array([1, 0, 1])
    best = find_best_temperature(scores, targets, grid=[0.5, 1.0])
    assert best == 0.5


def test_isotonic_calibration_increases_probs() -> None:
    scores = np.linspace(-2, 2, 5)
    targets = np.array([0, 0, 1, 1, 1])
    calibrator = fit_isotonic_calibrator(scores, targets)
    preds = apply_calibrator(calibrator, scores * 1.5)
    assert preds.shape == scores.shape
    assert np.all(preds >= 0) and np.all(preds <= 1)
    diffs = np.diff(preds)
    assert np.all(diffs >= -1e-8)
