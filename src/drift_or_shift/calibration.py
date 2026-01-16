"""Calibration helpers for logistic outputs under label shift."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike
from sklearn.isotonic import IsotonicRegression


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    """Return probability after applying the logistic sigmoid function."""
    return 1.0 / (1.0 + np.exp(-logits))


def temperature_scale(logits: ArrayLike, temperature: float) -> np.ndarray:
    """
    Adjust logits by a temperature factor.

    Higher temperatures soften the distribution, while temperatures below 1 boost confidence.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits_arr = np.asarray(logits, dtype=float)
    return logits_arr / float(temperature)


def find_best_temperature(
    logits: ArrayLike,
    targets: ArrayLike,
    *,
    grid: Iterable[float] | None = None,
) -> float:
    """
    Return the temperature that minimizes logistic cross-entropy on the provided data.

    A default grid spans from 0.05 to 3.0.
    """
    if grid is None:
        grid = list(np.concatenate([np.logspace(-2, -0.5, 5), np.linspace(0.1, 3.0, 50)]))
    logits_arr = np.asarray(logits, dtype=float)
    targets_arr = np.asarray(targets, dtype=float)
    if logits_arr.shape != targets_arr.shape:
        raise ValueError("logits and targets must share shape")
    if not (np.all((targets_arr == 0) | (targets_arr == 1))):
        raise ValueError("targets must be binary (0 or 1)")
    losses: list[tuple[float, float]] = []
    eps = 1e-12
    for temperature in grid:
        if temperature <= 0:
            continue
        scaled = _sigmoid(logits_arr / temperature)
        loss = -(
            targets_arr * np.log(scaled + eps) + (1 - targets_arr) * np.log(1 - scaled + eps)
        ).mean()
        losses.append((temperature, float(loss)))
    if not losses:
        raise ValueError("no valid temperatures found in the grid")
    return min(losses, key=lambda item: item[1])[0]


def fit_isotonic_calibrator(scores: ArrayLike, targets: ArrayLike, *, out_of_bounds: str = "clip") -> IsotonicRegression:
    """
    Fit an isotonic regression calibrator mapping scores to calibrated probabilities.
    """
    calib = IsotonicRegression(out_of_bounds=out_of_bounds)
    calib.fit(np.asarray(scores, dtype=float), np.asarray(targets, dtype=float))
    return calib


def apply_calibrator(calibrator: IsotonicRegression, scores: ArrayLike) -> np.ndarray:
    """Apply the fitted calibrator to a new set of scores."""
    return calibrator.predict(np.asarray(scores, dtype=float))
