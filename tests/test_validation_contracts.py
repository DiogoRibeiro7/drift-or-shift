"""Tests for input validation and degenerate-input handling.

These guards are the contract this library offers its callers: they decide
whether a bad prevalence, a mismatched shape, or an empty array fails loudly
or silently produces a meaningless number. They were entirely untested.
"""

from __future__ import annotations

import numpy as np
import pytest

from drift_or_shift.calibration import find_best_temperature, temperature_scale
from drift_or_shift.data_synth import (
    concept_drift_mu1,
    make_gaussian_binary,
    make_multimodal_binary,
    resample_to_prevalence,
)
from drift_or_shift.drift_monitor import (
    correlation_mean_diff,
    covariance_frobenius_diff,
    feature_drift_summary,
    multivariate_projection_ks,
    univariate_feature_stats,
)
from drift_or_shift.drift_variants import (
    apply_covariance_shift,
    density_ratio_shift,
    drift_score_from_ratio,
    inject_label_noise,
    shift_mean_vector,
)
from drift_or_shift.ess import effective_sample_size, ess_fraction
from drift_or_shift.metrics import risk_cost_sensitive


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


# ---------------------------------------------------------------------------
# data_synth
# ---------------------------------------------------------------------------


def test_concept_drift_rejects_out_of_range_dimension() -> None:
    with pytest.raises(IndexError, match="shift_dim"):
        concept_drift_mu1([0.0, 1.0], shift_dim=5)


def test_concept_drift_does_not_mutate_its_input() -> None:
    original = np.array([0.0, 0.0, 0.0])
    shifted = concept_drift_mu1(original, shift_dim=1, delta=0.5)

    assert shifted[1] == pytest.approx(0.5)
    assert original[1] == pytest.approx(0.0), "input was mutated in place"


@pytest.mark.parametrize("pi", [0.0, 1.0, -0.1, 1.5])
def test_gaussian_generator_rejects_degenerate_prevalence(rng, pi) -> None:
    with pytest.raises(ValueError, match="open interval"):
        make_gaussian_binary(10, 2, pi, np.zeros(2), np.ones(2), sigma=1.0, rng=rng)


@pytest.mark.parametrize(("n", "d"), [(0, 2), (-1, 2), (10, 0), (10, -3)])
def test_gaussian_generator_rejects_non_positive_sizes(rng, n, d) -> None:
    width = max(d, 1)
    with pytest.raises(ValueError, match="must be positive"):
        make_gaussian_binary(
            n, d, 0.3, np.zeros(width), np.ones(width), sigma=1.0, rng=rng
        )


def test_gaussian_generator_rejects_mismatched_means(rng) -> None:
    with pytest.raises(ValueError, match="dimensionality"):
        make_gaussian_binary(10, 3, 0.3, np.zeros(2), np.ones(3), sigma=1.0, rng=rng)


def test_gaussian_generator_rejects_bad_covariance_shape(rng) -> None:
    with pytest.raises(ValueError, match="Covariance"):
        make_gaussian_binary(
            10, 3, 0.3, np.zeros(3), np.ones(3), sigma=np.eye(2), rng=rng
        )


def test_resample_rejects_mismatched_lengths(rng) -> None:
    with pytest.raises(ValueError, match="same number of samples"):
        resample_to_prevalence(np.zeros((5, 2)), np.zeros(4), 0.5, rng)


def test_multimodal_rejects_empty_components(rng) -> None:
    with pytest.raises(ValueError, match="At least one mixture component"):
        make_multimodal_binary(10, 2, 0.3, [], [([0.0, 0.0], 1.0)], sigma=1.0, rng=rng)


def test_multimodal_rejects_negative_weights(rng) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        make_multimodal_binary(
            10,
            2,
            0.3,
            [([0.0, 0.0], -1.0)],
            [([1.0, 1.0], 1.0)],
            sigma=1.0,
            rng=rng,
        )


def test_multimodal_rejects_zero_weight_sum(rng) -> None:
    with pytest.raises(ValueError, match="sum to a positive value"):
        make_multimodal_binary(
            10,
            2,
            0.3,
            [([0.0, 0.0], 0.0)],
            [([1.0, 1.0], 1.0)],
            sigma=1.0,
            rng=rng,
        )


def test_multimodal_rejects_component_dimension_mismatch(rng) -> None:
    with pytest.raises(ValueError, match="dimensionality"):
        make_multimodal_binary(
            10,
            3,
            0.3,
            [([0.0, 0.0], 1.0)],
            [([1.0, 1.0, 1.0], 1.0)],
            sigma=1.0,
            rng=rng,
        )


# ---------------------------------------------------------------------------
# ess
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "pi", "alpha"),
    [(0, 0.3, 1.0), (-5, 0.3, 1.0), (100, 0.3, 0.0), (100, 0.3, -2.0)],
)
def test_ess_rejects_non_positive_arguments(n, pi, alpha) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        effective_sample_size(n, pi, alpha)


@pytest.mark.parametrize(
    ("pi", "message"),
    [
        # pi <= 0 trips the positivity guard; pi >= 1 trips the interval guard.
        (0.0, "pi must be positive"),
        (-0.2, "pi must be positive"),
        (1.0, "open interval"),
        (2.0, "open interval"),
    ],
)
def test_ess_rejects_degenerate_prevalence(pi, message) -> None:
    with pytest.raises(ValueError, match=message):
        effective_sample_size(100, pi, 1.0)


def test_ess_fraction_is_one_at_unit_weight() -> None:
    """With alpha = 1 there is no reweighting, so no sample size is lost."""
    assert ess_fraction(1000, 0.3, 1.0) == pytest.approx(1.0)


def test_ess_fraction_shrinks_as_weight_grows() -> None:
    fractions = [ess_fraction(1000, 0.2, alpha) for alpha in (1.0, 5.0, 20.0)]

    assert fractions == sorted(fractions, reverse=True)
    assert all(0.0 < value <= 1.0 for value in fractions)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("c10", "c01"), [(-1.0, 1.0), (1.0, -1.0)])
def test_risk_rejects_negative_costs(c10, c01) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        risk_cost_sensitive([0, 1], [0, 1], c10, c01)


def test_risk_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="Shapes"):
        risk_cost_sensitive([0, 1, 1], [0, 1], 1.0, 1.0)


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------


def test_find_best_temperature_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="share shape"):
        find_best_temperature(np.zeros(5), np.zeros(4))


def test_find_best_temperature_rejects_non_binary_targets() -> None:
    with pytest.raises(ValueError, match="binary"):
        find_best_temperature(np.zeros(3), np.array([0.0, 0.5, 1.0]))


def test_find_best_temperature_skips_non_positive_grid_entries() -> None:
    """Non-positive temperatures are undefined and must be skipped, not used."""
    best = find_best_temperature(
        np.array([-1.0, 1.0]), np.array([0.0, 1.0]), grid=[-1.0, 0.0, 0.5, 2.0]
    )

    assert best > 0


def test_find_best_temperature_rejects_grid_with_no_valid_entry() -> None:
    with pytest.raises(ValueError, match="no valid temperatures"):
        find_best_temperature(np.zeros(2), np.array([0.0, 1.0]), grid=[0.0, -3.0])


def test_temperature_scale_divides_logits() -> None:
    assert temperature_scale(np.array([2.0, -4.0]), 2.0) == pytest.approx([1.0, -2.0])


# ---------------------------------------------------------------------------
# drift_variants
# ---------------------------------------------------------------------------


def test_covariance_shift_rejects_non_positive_scale() -> None:
    with pytest.raises(ValueError, match="scale must be positive"):
        apply_covariance_shift(np.eye(2), 0.0)


def test_shift_mean_vector_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="share shape"):
        shift_mean_vector([0.0, 1.0], [1.0])


@pytest.mark.parametrize("flip_prob", [-0.1, 1.1])
def test_inject_label_noise_rejects_out_of_range_probability(rng, flip_prob) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        inject_label_noise([0, 1], flip_prob, rng)


def test_inject_label_noise_handles_empty_input(rng) -> None:
    assert inject_label_noise([], 0.5, rng).size == 0


def test_inject_label_noise_flips_everything_at_probability_one(rng) -> None:
    assert inject_label_noise([0, 1, 0, 1], 1.0, rng).tolist() == [1, 0, 1, 0]


def test_density_ratio_rejects_non_positive_bandwidth() -> None:
    with pytest.raises(ValueError, match="bandwidth must be positive"):
        density_ratio_shift(np.zeros((3, 2)), np.zeros((3, 2)), bandwidth=0.0)


def test_density_ratio_rejects_one_dimensional_input() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        density_ratio_shift(np.zeros(3), np.zeros(3))


def test_drift_score_of_empty_ratio_is_zero() -> None:
    assert drift_score_from_ratio([]) == 0.0


# ---------------------------------------------------------------------------
# drift_monitor
# ---------------------------------------------------------------------------


def test_univariate_stats_reject_one_dimensional_input() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        univariate_feature_stats(np.zeros(4), np.zeros((4, 2)))


def test_univariate_stats_reject_feature_mismatch() -> None:
    with pytest.raises(ValueError, match="dimensionality must match"):
        univariate_feature_stats(np.zeros((4, 2)), np.zeros((4, 3)))


def test_univariate_stats_reject_empty_datasets() -> None:
    with pytest.raises(ValueError, match="must contain samples"):
        univariate_feature_stats(np.zeros((0, 2)), np.zeros((4, 2)))


def test_feature_drift_summary_rejects_empty_stats() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        feature_drift_summary([])


def test_projection_ks_is_zero_for_identical_inputs(rng) -> None:
    data = rng.normal(size=(40, 3))

    assert max(multivariate_projection_ks(data, data, projections=4)) == 0.0


def test_projection_ks_detects_a_shift(rng) -> None:
    reference = rng.normal(size=(60, 3))
    target = rng.normal(size=(60, 3)) + 3.0

    assert max(multivariate_projection_ks(reference, target, projections=6)) > 0.5


def test_projection_ks_handles_degenerate_requests(rng) -> None:
    """Zero features or zero projections yield a neutral score, not a crash."""
    data = rng.normal(size=(10, 2))

    assert multivariate_projection_ks(np.zeros((10, 0)), np.zeros((10, 0))) == [0.0]
    assert multivariate_projection_ks(data, data, projections=0) == [0.0]


def test_covariance_and_correlation_diffs_are_zero_for_identical_inputs(rng) -> None:
    data = rng.normal(size=(50, 3))

    assert covariance_frobenius_diff(data, data) == pytest.approx(0.0, abs=1e-9)
    assert correlation_mean_diff(data, data) == pytest.approx(0.0, abs=1e-9)


def test_correlation_diff_handles_single_feature(rng) -> None:
    """A single-feature correlation matrix is scalar; it must still compare."""
    reference = rng.normal(size=(30, 1))
    target = rng.normal(size=(30, 1)) + 1.0

    assert correlation_mean_diff(reference, target) == pytest.approx(0.0, abs=1e-9)
