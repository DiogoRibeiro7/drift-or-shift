"""Simple benchmarking utilities for calibrated classifiers."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

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


def benchmark(
    X: np.ndarray,
    y: np.ndarray,
    estimators: Iterable[dict],
    calibrations: Iterable[str],
    *,
    cv_params: dict,
) -> pd.DataFrame:
    """Run calibration benchmarking over a dataset."""
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
        for spec in estimators:
            for calibration in calibrations:
                estimator = _make_estimator(spec, cv_params.get("random_state"))
                estimator.fit(X_train, y_train)
                predictor = estimator
                if calibration != "none":
                    calibrator = CalibratedClassifierCV(
                        base_estimator=estimator,
                        method=calibration,
                        cv="prefit",
                    )
                    calibrator.fit(X_train, y_train)
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
