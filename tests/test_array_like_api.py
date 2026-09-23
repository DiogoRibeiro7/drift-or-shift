"""The public API must accept the array types it is actually called with.

Every public helper was annotated `Sequence[float]` / `Sequence[int]` while
being called throughout the experiments with NumPy arrays and pandas Series.
Because the package ships `py.typed`, that narrowness also reached downstream
users: passing a NumPy array to a NumPy library produced a type error.

These tests pin the runtime half of the contract. The static half is enforced
by mypy now covering `drift_or_shift.experiments`, which passes arrays into
every one of these functions.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from drift_or_shift.data_synth import (
    concept_drift_mu1,
    make_gaussian_binary,
    make_multimodal_binary,
)
from drift_or_shift.drift_variants import (
    drift_score_from_ratio,
    inject_label_noise,
    shift_mean_vector,
)
from drift_or_shift.metrics import (
    oracle_threshold_min_risk,
    pr_auc,
    risk_cost_sensitive,
    roc_auc,
)
from drift_or_shift.models import fit_logistic_regression, predict_logits
from drift_or_shift.plotting import (
    plot_auc_pr_vs_prevalence,
    plot_ess_vs_alpha,
    plot_risk_vs_prevalence,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def labelled(rng):
    X = rng.normal(size=(60, 3))
    y = (X[:, 0] > 0).astype(int)
    return X, y


def _as_variants(values: list[float]):
    """The same data as a list, a NumPy array, and a pandas Series."""
    return [values, np.asarray(values), pd.Series(values)]


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("truth", _as_variants([0, 1, 1, 0]))
@pytest.mark.parametrize("pred", _as_variants([0, 1, 0, 0]))
def test_risk_accepts_every_array_flavour(truth, pred) -> None:
    assert risk_cost_sensitive(truth, pred, 1.0, 1.0) == pytest.approx(0.25)


@pytest.mark.parametrize("container", [np.asarray, pd.Series, list])
def test_auc_helpers_accept_every_array_flavour(container) -> None:
    truth = container([0, 0, 1, 1])
    scores = container([0.1, 0.4, 0.35, 0.8])

    assert 0.0 <= roc_auc(truth, scores) <= 1.0
    assert 0.0 <= pr_auc(truth, scores) <= 1.0


@pytest.mark.parametrize("container", [np.asarray, pd.Series, list])
def test_oracle_threshold_accepts_every_array_flavour(container) -> None:
    threshold, risk = oracle_threshold_min_risk(
        container([0, 0, 1, 1]), container([-2.0, -1.0, 0.5, 2.0]), 1.0, 1.0
    )

    assert np.isfinite(threshold)
    assert risk >= 0.0


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------


def test_models_accept_arrays_and_frames(labelled) -> None:
    X, y = labelled

    from_arrays = fit_logistic_regression(X, y)
    from_pandas = fit_logistic_regression(pd.DataFrame(X), pd.Series(y))

    assert predict_logits(from_arrays, X).shape == (len(y),)
    assert predict_logits(from_pandas, pd.DataFrame(X)).shape == (len(y),)


# ---------------------------------------------------------------------------
# data_synth
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("container", [np.asarray, list])
def test_generators_accept_sequence_or_array_means(rng, container) -> None:
    mu0 = container([0.0, 0.0])
    mu1 = container([1.0, 1.0])

    X, y = make_gaussian_binary(40, 2, 0.3, mu0, mu1, sigma=1.0, rng=rng)

    assert X.shape == (40, 2)
    assert set(np.unique(y)) <= {0, 1}


def test_concept_drift_accepts_an_array(rng) -> None:
    shifted = concept_drift_mu1(np.zeros(4), shift_dim=2, delta=0.5)

    assert shifted[2] == pytest.approx(0.5)


def test_multimodal_accepts_array_component_means(rng) -> None:
    X, y = make_multimodal_binary(
        40,
        2,
        0.4,
        [(np.array([0.0, 0.0]), 1.0)],
        [(np.array([2.0, 2.0]), 1.0)],
        sigma=1.0,
        rng=rng,
    )

    assert X.shape == (40, 2)
    assert set(np.unique(y)) <= {0, 1}


# ---------------------------------------------------------------------------
# drift_variants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("container", [np.asarray, pd.Series, list])
def test_drift_variants_accept_every_array_flavour(rng, container) -> None:
    assert inject_label_noise(container([0, 1, 0]), 0.0, rng).tolist() == [0, 1, 0]
    assert drift_score_from_ratio(container([1.0, 3.0])) == pytest.approx(2.0)


def test_shift_mean_vector_accepts_arrays() -> None:
    shifted = shift_mean_vector(np.zeros(3), np.ones(3))

    assert shifted.tolist() == [1.0, 1.0, 1.0]


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("container", [np.asarray, pd.Series, list])
def test_plot_helpers_accept_every_array_flavour(container) -> None:
    """The experiments pass pandas Series straight out of a summary frame."""
    prevalences = container([0.1, 0.5])
    a = container([0.3, 0.2])
    b = container([0.25, 0.15])
    c = container([0.2, 0.1])
    # plot_ess_vs_alpha is log-scaled on class-weight multipliers, not
    # prevalences; feed it alphas so the axis is meaningful.
    alphas = container([1.0, 10.0])
    ess_fractions = container([1.0, 0.4])

    for figure in (
        plot_risk_vs_prevalence(prevalences, a, b, c),
        plot_auc_pr_vs_prevalence(prevalences, a, b),
        plot_ess_vs_alpha(alphas, ess_fractions),
    ):
        assert figure is not None
        figure.clf()


def test_plot_helpers_emit_no_matplotlib_warnings() -> None:
    """A log-scaled shared x-axis used to warn on every draw."""
    prevalences = [0.01, 0.05, 0.1, 0.3, 0.5]
    series = [0.9, 0.8, 0.7, 0.6, 0.5]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for figure in (
            plot_risk_vs_prevalence(prevalences, series, series, series),
            plot_auc_pr_vs_prevalence(prevalences, series, series),
            plot_ess_vs_alpha([1.0, 5.0, 10.0], [1.0, 0.7, 0.4]),
        ):
            figure.canvas.draw()
            figure.clf()

    offenders = [str(w.message) for w in caught if "log-scaled" in str(w.message)]
    assert not offenders, f"plot helpers warn on draw: {offenders}"
