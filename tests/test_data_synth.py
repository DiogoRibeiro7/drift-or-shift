import numpy as np

from drift_or_shift import (
    concept_drift_mu1,
    default_mu0,
    default_mu1,
    make_gaussian_binary,
    resample_to_prevalence,
)


def test_make_gaussian_binary_shapes_and_prevalence() -> None:
    rng = np.random.default_rng(0)
    d = 6
    mu0 = default_mu0(d)
    mu1 = default_mu1(d)
    X, y = make_gaussian_binary(500, d, 0.2, mu0, mu1, sigma=1.0, rng=rng)
    assert X.shape == (500, d)
    assert y.shape == (500,)
    empirical_pi = y.mean()
    assert 0.15 <= empirical_pi <= 0.25


def test_make_gaussian_binary_deterministic_for_seed() -> None:
    seed = 1
    kwargs = {
        "n": 200,
        "d": 4,
        "pi": 0.3,
        "mu0": default_mu0(4),
        "mu1": default_mu1(4),
        "sigma": 1.2,
    }
    rng1 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed)
    X1, y1 = make_gaussian_binary(rng=rng1, **kwargs)
    X2, y2 = make_gaussian_binary(rng=rng2, **kwargs)
    np.testing.assert_allclose(X1, X2)
    np.testing.assert_array_equal(y1, y2)


def test_resample_to_prevalence_near_target() -> None:
    rng = np.random.default_rng(2)
    mu0 = default_mu0(3)
    mu1 = default_mu1(3)
    X, y = make_gaussian_binary(500, 3, 0.2, mu0, mu1, sigma=0.5, rng=rng)
    rng2 = np.random.default_rng(3)
    X_resampled, y_resampled = resample_to_prevalence(X, y, 0.1, rng2)
    assert X_resampled.shape == (500, 3)
    assert y_resampled.shape == (500,)
    assert 0.08 <= y_resampled.mean() <= 0.12


def test_concept_drift_mu1_increases_dimension() -> None:
    base = default_mu1(7)
    drifted = concept_drift_mu1(base, shift_dim=5, delta=0.7)
    assert drifted[5] == base[5] + 0.7
