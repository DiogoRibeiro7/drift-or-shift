"""Model training helpers for drift-or-shift experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.linear_model import LogisticRegression


def _rng_seed(rng: np.random.Generator | None) -> int | None:
    if rng is None:
        return None
    return int(rng.integers(0, 2**31 - 1))


def fit_logistic_regression(
    X: ArrayLike,
    y: ArrayLike,
    *,
    class_weight: dict[int, float] | None = None,
    C: float = 1.0,
    max_iter: int = 1000,
    rng: np.random.Generator | None = None,
    **kwargs: Any,
) -> LogisticRegression:
    """Train a logistic regression model and return it."""
    model = LogisticRegression(
        class_weight=class_weight,
        C=C,
        max_iter=max_iter,
        random_state=_rng_seed(rng),
        solver="lbfgs",
        **kwargs,
    )
    model.fit(X, y)
    return model


def predict_logits(model: LogisticRegression, X: ArrayLike) -> np.ndarray:
    """Return decision scores (logits) from the fitted model."""
    return model.decision_function(X)
