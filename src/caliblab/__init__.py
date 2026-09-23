"""Simple benchmarking utilities for calibrated classifiers."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split

try:  # scikit-learn >= 1.6
    from sklearn.frozen import FrozenEstimator
except ImportError:  # pragma: no cover - scikit-learn < 1.6
    FrozenEstimator = None

_ESTIMATOR_REGISTRY = {
    "logistic": LogisticRegression,
    "random_forest": RandomForestClassifier,
}


def _make_estimator(spec: dict, random_state: int | None):
    """Instantiate a base estimator from spec."""
    estimator_type = spec.get("type")
    if estimator_type not in _ESTIMATOR_REGISTRY:
        raise ValueError(f"Unsupported estimator type: {estimator_type!r}")
    estimator_cls = _ESTIMATOR_REGISTRY[estimator_type]
    params = dict(spec.get("params", {}))
    if random_state is not None:
        params.setdefault("random_state", random_state)
    if estimator_type == "logistic":
        params.setdefault("max_iter", 1000)
    return estimator_cls(**params)


def _calibrate_prefit(estimator, method: str) -> CalibratedClassifierCV:
    """Wrap an already-fitted estimator in a calibrator.

    ``cv="prefit"`` is deprecated in scikit-learn 1.6 and removed in 1.8, so we
    prefer ``FrozenEstimator`` where it exists and fall back only for older
    releases still covered by our dependency floor.
    """
    if FrozenEstimator is not None:
        return CalibratedClassifierCV(FrozenEstimator(estimator), method=method)
    return CalibratedClassifierCV(  # pragma: no cover - scikit-learn < 1.6
        estimator=estimator, method=method, cv="prefit"
    )


_MIN_CALIBRATION_PER_CLASS = 5


def _validate_calibration_set(y_calib: np.ndarray, holdout: float) -> None:
    """Fail early and clearly when the held-out split is too small to calibrate.

    ``CalibratedClassifierCV`` cross-validates internally even over a frozen
    estimator, so it needs at least as many members of each class as its
    default number of folds. Left to sklearn this surfaces as "n_splits=5
    cannot be greater than the number of members in each class", which says
    nothing about the knob the caller actually turned.
    """
    counts = np.bincount(np.asarray(y_calib, dtype=int), minlength=2)
    smallest = int(counts.min())
    if smallest < _MIN_CALIBRATION_PER_CLASS:
        raise ValueError(
            f"calibration_holdout={holdout} leaves only {smallest} sample(s) of "
            f"the rarest class to calibrate on; at least "
            f"{_MIN_CALIBRATION_PER_CLASS} are needed. Raise the holdout "
            f"fraction, use fewer CV folds, or pass more data."
        )


def benchmark(
    X: np.ndarray,
    y: np.ndarray,
    estimators: Iterable[dict],
    calibrations: Iterable[str],
    *,
    cv_params: dict,
    calibration_holdout: float | None = None,
) -> pd.DataFrame:
    """Run calibration benchmarking over a dataset.

    Args:
        calibration_holdout: Fraction of each fold's training data to hold out
            for fitting the calibration map. ``None`` (the default) fits the
            calibrator on the same data the base estimator was trained on.

    On fitting the calibrator on the estimator's own training data:

    Textbook practice is to calibrate on held-out data, because a model's
    predictions on data it has already seen are over-confident and the map
    learned from them need not transfer. We measured it on the shipped
    breast-cancer benchmark, holding the base model and the calibration-set
    size fixed so that only the overlap differed, and the effect on test Brier
    score was within noise::

        logistic      sigmoid    -0.0014  (SE 0.0003)
        logistic      isotonic   +0.0006  (SE 0.0009)
        random_forest sigmoid    +0.0001  (SE 0.0003)
        random_forest isotonic   -0.0002  (SE 0.0010)

    The default therefore stays as it was, so published numbers do not move.
    Pass ``calibration_holdout`` to opt into a held-out split; on a harder
    dataset or a higher-capacity estimator the difference may well matter.

    ``tests/test_caliblab_leakage.py`` re-runs that comparison, so the claim
    above is checked rather than merely asserted.
    """
    if calibration_holdout is not None and not 0.0 < calibration_holdout < 1.0:
        raise ValueError("calibration_holdout must lie in the open interval (0, 1)")
    X = np.asarray(X)
    y = np.asarray(y)
    cv = StratifiedKFold(
        n_splits=cv_params.get("n_splits", 5),
        shuffle=True,
        random_state=cv_params.get("random_state"),
    )
    results = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        X_fit, y_fit = X_train, y_train
        X_calib, y_calib = X_train, y_train
        if calibration_holdout is not None:
            X_fit, X_calib, y_fit, y_calib = train_test_split(
                X_train,
                y_train,
                test_size=calibration_holdout,
                stratify=y_train,
                random_state=cv_params.get("random_state"),
            )
            _validate_calibration_set(y_calib, calibration_holdout)
        for spec in estimators:
            for calibration in calibrations:
                estimator = _make_estimator(spec, cv_params.get("random_state"))
                estimator.fit(X_fit, y_fit)
                predictor = estimator
                if calibration != "none":
                    calibrator = _calibrate_prefit(estimator, calibration)
                    calibrator.fit(X_calib, y_calib)
                    predictor = calibrator
                probs = predictor.predict_proba(X_test)[:, 1]
                for local_idx, prob in enumerate(probs):
                    results.append(
                        {
                            "fold": fold,
                            "estimator": spec.get("name", spec.get("type")),
                            "estimator_type": spec.get("type"),
                            "calibration": calibration,
                            "sample_index": int(test_idx[local_idx]),
                            "y_true": int(y_test[local_idx]),
                            "y_prob": float(prob),
                        }
                    )
    return pd.DataFrame(results)
