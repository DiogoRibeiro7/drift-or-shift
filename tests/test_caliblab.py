"""Tests for the calibration benchmarking helpers.

The calibrated code path regressed silently once before: ``caliblab`` passed
``base_estimator=`` to ``CalibratedClassifierCV``, which scikit-learn removed in
1.2, while the only test in the suite exercised ``calibration="none"``. These
tests cover the calibrated branches so that cannot recur.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from caliblab import benchmark

ESTIMATORS = [
    {"name": "logistic", "type": "logistic", "params": {"solver": "liblinear"}},
]


@pytest.fixture
def dataset() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 4))
    y = (X[:, 0] + rng.normal(scale=0.2, size=80) > 0).astype(int)
    return X, y


@pytest.mark.parametrize("calibration", ["none", "sigmoid", "isotonic"])
def test_benchmark_supports_each_calibration(dataset, calibration) -> None:
    X, y = dataset
    frame = benchmark(
        X, y, ESTIMATORS, [calibration], cv_params={"n_splits": 2, "random_state": 0}
    )

    assert not frame.empty
    assert set(frame["calibration"].unique()) == {calibration}
    assert len(frame) == len(y)  # every sample is predicted exactly once


def test_benchmark_emits_probabilities_and_labels(dataset) -> None:
    X, y = dataset
    frame = benchmark(
        X, y, ESTIMATORS, ["sigmoid"], cv_params={"n_splits": 2, "random_state": 0}
    )

    assert frame["y_prob"].between(0.0, 1.0).all()
    assert set(frame["y_true"].unique()) <= {0, 1}
    assert sorted(frame["sample_index"]) == list(range(len(y)))


def test_calibrated_path_raises_no_deprecation_warnings(dataset) -> None:
    """``cv="prefit"`` is removed in scikit-learn 1.8; we must not rely on it."""
    X, y = dataset
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        benchmark(
            X, y, ESTIMATORS, ["isotonic"], cv_params={"n_splits": 2, "random_state": 0}
        )

    offenders = [
        str(w.message)
        for w in caught
        if issubclass(w.category, (DeprecationWarning, FutureWarning, UserWarning))
        and "prefit" in str(w.message)
    ]
    assert not offenders, f"deprecated calibration API in use: {offenders}"


def test_unknown_estimator_type_is_rejected(dataset) -> None:
    X, y = dataset
    with pytest.raises(ValueError, match="Unsupported estimator type"):
        benchmark(
            X,
            y,
            [{"name": "nope", "type": "not_a_model"}],
            ["none"],
            cv_params={"n_splits": 2, "random_state": 0},
        )
