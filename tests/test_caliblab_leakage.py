"""Verify the measurement that justifies `caliblab`'s calibration default.

`benchmark` fits the calibration map on the same data the base estimator was
trained on. Textbook practice is to calibrate on held-out data, because a
model's predictions on data it has already seen are over-confident and the map
learned from them need not transfer.

The docstring on `benchmark` claims that on the shipped breast-cancer
benchmark the effect is within noise. A claim like that does not belong in a
docstring unless something checks it, so this module checks it.

The construction matters. A naive comparison confounds two effects:

* the held-out arm trains its base model on less data, and
* its calibration set is a different size.

Both are controlled here: one base model is shared by both arms, and the leaky
calibration set is drawn to the same size as the held-out one, so the only
difference is whether the calibrator has seen the base model's training data.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from caliblab import FrozenEstimator

pytestmark = pytest.mark.skipif(
    FrozenEstimator is None, reason="needs scikit-learn >= 1.6 for FrozenEstimator"
)

# The isolated effect measured over 10 repeats was between -0.0014 and +0.0006
# Brier, with standard errors around 0.001. This bound is comfortably above that
# and far below any difference that would change a conclusion.
MAX_DEFENSIBLE_DELTA = 0.01


def _base_model() -> object:
    return make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, random_state=0)
    )


def _leaky_and_holdout_brier(method: str, repeats: int) -> tuple[float, float]:
    X, y = load_breast_cancer(return_X_y=True)
    leaky: list[float] = []
    holdout: list[float] = []

    for rep in range(repeats):
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=rep)
        for train_idx, test_idx in cv.split(X, y):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            X_fit, X_calib, y_fit, y_calib = train_test_split(
                X_train, y_train, test_size=0.3, stratify=y_train, random_state=rep
            )
            # One base model, shared by both arms.
            base = _base_model().fit(X_fit, y_fit)

            # Leaky calibration set: same size as the held-out one, but drawn
            # from data the base model was trained on.
            X_leak, _, y_leak, _ = train_test_split(
                X_fit, y_fit, train_size=len(y_calib), stratify=y_fit, random_state=rep
            )

            leaky_model = CalibratedClassifierCV(
                FrozenEstimator(base), method=method
            ).fit(X_leak, y_leak)
            holdout_model = CalibratedClassifierCV(
                FrozenEstimator(base), method=method
            ).fit(X_calib, y_calib)

            leaky.append(
                brier_score_loss(y_test, leaky_model.predict_proba(X_test)[:, 1])
            )
            holdout.append(
                brier_score_loss(y_test, holdout_model.predict_proba(X_test)[:, 1])
            )

    return float(np.mean(leaky)), float(np.mean(holdout))


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_calibrating_on_seen_data_costs_little_here(method: str) -> None:
    """The measurement behind `benchmark`'s default.

    If this ever fails, the default is no longer defensible and
    `calibration_holdout` should become the recommended path.
    """
    leaky, holdout = _leaky_and_holdout_brier(method, repeats=3)

    assert abs(leaky - holdout) < MAX_DEFENSIBLE_DELTA, (
        f"calibrating on the base model's own training data now shifts Brier by "
        f"{leaky - holdout:+.4f} for {method} (leaky={leaky:.4f}, "
        f"held-out={holdout:.4f}); revisit caliblab's default"
    )


def test_both_arms_produce_usable_probabilities() -> None:
    leaky, holdout = _leaky_and_holdout_brier("sigmoid", repeats=1)

    assert 0.0 < leaky < 0.25
    assert 0.0 < holdout < 0.25
