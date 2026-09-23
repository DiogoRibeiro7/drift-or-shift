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


def test_sigmoid_does_not_warn_on_large_negative_logits() -> None:
    """`1 / (1 + exp(-x))` overflows for large negative x.

    The saturated result is correct, but NumPy warns on the way there, and that
    noise can mask a real warning. The piecewise form avoids it.
    """
    import warnings

    from drift_or_shift.calibration import _sigmoid

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        probs = _sigmoid(np.array([-1000.0, -745.0, -50.0, 0.0, 50.0, 1000.0]))

    overflow = [str(w.message) for w in caught if "overflow" in str(w.message)]
    assert not overflow, f"sigmoid still overflows: {overflow}"
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert np.all(np.diff(probs) >= 0), "sigmoid must stay monotonic"


def test_sigmoid_matches_the_closed_form_where_it_is_safe() -> None:
    from drift_or_shift.calibration import _sigmoid

    x = np.linspace(-30.0, 30.0, 201)

    assert _sigmoid(x) == pytest.approx(1.0 / (1.0 + np.exp(-x)), rel=1e-12)


def test_sigmoid_is_symmetric_about_zero() -> None:
    from drift_or_shift.calibration import _sigmoid

    x = np.array([0.5, 2.0, 7.5, 40.0])

    assert _sigmoid(x) == pytest.approx(1.0 - _sigmoid(-x), abs=1e-12)
