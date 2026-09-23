import numpy as np

from drift_or_shift import effective_sample_size, ess_fraction


def test_effective_sample_size_identity_at_alpha_one() -> None:
    n = 1000
    pi = 0.3
    alpha = 1.0
    assert effective_sample_size(n, pi, alpha) == n


def test_effective_sample_size_decreases_with_alpha() -> None:
    n = 500
    pi = 0.2
    low = effective_sample_size(n, pi, 1.0)
    high = effective_sample_size(n, pi, 10.0)
    assert high < low


def test_effective_sample_size_scales_with_n() -> None:
    pi = 0.4
    alpha = 5.0
    base = effective_sample_size(100, pi, alpha)
    doubled = effective_sample_size(200, pi, alpha)
    assert np.isclose(doubled, 2 * base)


def test_ess_fraction_matches_ratio() -> None:
    n = 120
    pi = 0.25
    alpha = 3.0
    assert np.isclose(
        ess_fraction(n, pi, alpha), effective_sample_size(n, pi, alpha) / n
    )
