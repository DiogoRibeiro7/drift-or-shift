"""Tests for the calibration benchmarking helpers.

The calibrated code path regressed silently once before: ``caliblab`` passed
``base_estimator=`` to ``CalibratedClassifierCV``, which scikit-learn removed in
1.2, while the only test in the suite exercised ``calibration="none"``. These
tests cover the calibrated branches so that cannot recur.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from caliblab import benchmark

ESTIMATORS = [
    {"name": "logistic", "type": "logistic", "params": {"solver": "liblinear"}},
]


@pytest.fixture
def larger_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Big enough that a held-out calibration split still has both classes."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    y = (X[:, 0] + rng.normal(scale=0.2, size=300) > 0).astype(int)
    return X, y


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


# ---------------------------------------------------------------------------
# calibration_holdout
# ---------------------------------------------------------------------------


def test_holdout_is_opt_in_and_changes_nothing_by_default(dataset) -> None:
    """The default must stay bit-for-bit what it was, so published numbers hold."""
    X, y = dataset
    params = {"n_splits": 2, "random_state": 0}

    default = benchmark(X, y, ESTIMATORS, ["sigmoid"], cv_params=params)
    explicit_none = benchmark(
        X, y, ESTIMATORS, ["sigmoid"], cv_params=params, calibration_holdout=None
    )

    pd.testing.assert_frame_equal(default, explicit_none)


def test_holdout_fits_the_calibrator_on_unseen_data(larger_dataset) -> None:
    X, y = larger_dataset
    params = {"n_splits": 2, "random_state": 0}

    default = benchmark(X, y, ESTIMATORS, ["sigmoid"], cv_params=params)
    holdout = benchmark(
        X, y, ESTIMATORS, ["sigmoid"], cv_params=params, calibration_holdout=0.3
    )

    # Same coverage: every test sample still gets exactly one probability.
    assert sorted(holdout["sample_index"]) == sorted(default["sample_index"])
    # But a different model, so different probabilities.
    assert not np.allclose(
        holdout.sort_values("sample_index")["y_prob"].to_numpy(),
        default.sort_values("sample_index")["y_prob"].to_numpy(),
    )


@pytest.mark.parametrize("holdout", [0.4, 0.5])
def test_holdout_still_produces_valid_probabilities(larger_dataset, holdout) -> None:
    X, y = larger_dataset
    frame = benchmark(
        X,
        y,
        ESTIMATORS,
        ["none", "sigmoid", "isotonic"],
        cv_params={"n_splits": 2, "random_state": 0},
        calibration_holdout=holdout,
    )

    assert frame["y_prob"].between(0.0, 1.0).all()
    assert len(frame) == 3 * len(y)


def test_too_small_a_holdout_is_rejected_with_a_useful_message(dataset) -> None:
    """sklearn's own error names n_splits, not the knob the caller turned."""
    X, y = dataset

    with pytest.raises(ValueError, match="rarest class"):
        benchmark(
            X,
            y,
            ESTIMATORS,
            ["sigmoid"],
            cv_params={"n_splits": 2, "random_state": 0},
            calibration_holdout=0.05,
        )


@pytest.mark.parametrize("holdout", [0.0, 1.0, -0.2, 1.5])
def test_holdout_fraction_must_be_a_proper_fraction(dataset, holdout) -> None:
    X, y = dataset

    with pytest.raises(ValueError, match="open interval"):
        benchmark(
            X,
            y,
            ESTIMATORS,
            ["sigmoid"],
            cv_params={"n_splits": 2, "random_state": 0},
            calibration_holdout=holdout,
        )


def test_uncalibrated_arm_sees_the_same_training_data_as_the_calibrated_one(
    larger_dataset,
) -> None:
    """With a holdout, the base model must be trained on the reduced set in
    both arms, or `none` and `sigmoid` would not be comparable."""
    X, y = larger_dataset
    frame = benchmark(
        X,
        y,
        ESTIMATORS,
        ["none", "sigmoid"],
        cv_params={"n_splits": 2, "random_state": 0},
        calibration_holdout=0.3,
    )

    counts = frame.groupby("calibration")["sample_index"].count()
    assert counts["none"] == counts["sigmoid"] == len(y)
