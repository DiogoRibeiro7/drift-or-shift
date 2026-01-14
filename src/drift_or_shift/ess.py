"""Effective sample size calculations for reweighted samples."""

from __future__ import annotations

def _validate_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def effective_sample_size(n: int, pi: float, alpha: float) -> float:
    """Return Kish-style effective sample size after reweighting."""
    _validate_positive(n, "n")
    _validate_positive(pi, "pi")
    if pi >= 1:
        raise ValueError("pi must lie in the open interval (0, 1).")
    _validate_positive(alpha, "alpha")
    numerator = (pi * alpha + (1 - pi)) ** 2
    denominator = pi * alpha ** 2 + (1 - pi)
    return n * numerator / denominator


def ess_fraction(n: int, pi: float, alpha: float) -> float:
    """Return the effective sample size as a fraction of the original sample."""
    ess = effective_sample_size(n, pi, alpha)
    return ess / n
