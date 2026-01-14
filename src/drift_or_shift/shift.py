"""Label shift utilities and offset correction helpers."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def validate_prevalence(pi: float) -> None:
    """Raise if prevalence is not strictly between 0 and 1."""
    if not 0 < pi < 1:
        raise ValueError("pi must lie in the open interval (0, 1).")


def odds(pi: float) -> float:
    """Return odds for a positive-prevalence value."""
    validate_prevalence(pi)
    return pi / (1 - pi)


def logit_offset(pi_train: float, pi_test: float) -> float:
    """Return an additive logit offset that adjusts from train to test prevalences.

    The logits are assumed to be posterior log-odds computed with the train
    prevalence; adding ``logit_offset(pi_train, pi_test)`` pushes them to the
    effective log-odds for ``pi_test``.
    """
    validate_prevalence(pi_train)
    validate_prevalence(pi_test)
    return float(np.log(pi_test * (1 - pi_train) / (pi_train * (1 - pi_test))))


def apply_logit_offset(logits: np.ndarray, offset: float) -> np.ndarray:
    """Shift logits by a fixed constant offset (additive update)."""
    return np.asarray(logits) + offset


def threshold_from_costs(pi: float, c10: float, c01: float) -> float:
    """Return the log-odds threshold derived from cost ratio.

    When logits represent log-posteriors, the optimal decision boundary for a
    cost-sensitive binary decision is ``log(c10 / c01)`` (i.e., favor the class
    with the lower misclassification cost).
    """
    validate_prevalence(pi)
    if c10 <= 0 or c01 <= 0:
        raise ValueError("Costs must be strictly positive.")
    return float(np.log(c10 / c01))


def decision_from_logits(logits: np.ndarray, threshold: float) -> np.ndarray:
    """Return binary decisions based on whether logits exceed a threshold."""
    logits_arr = np.asarray(logits)
    return (logits_arr >= threshold).astype(np.int64)
