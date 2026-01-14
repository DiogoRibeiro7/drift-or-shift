import math

import numpy as np
import pytest

from drift_or_shift import (
    apply_logit_offset,
    decision_from_logits,
    logit_offset,
    odds,
    threshold_from_costs,
    validate_prevalence,
)


def test_validate_prevalence_rejects_bounds() -> None:
    for invalid in (0.0, 1.0, -1e-6, 1.0 + 1e-6):
        with pytest.raises(ValueError):
            validate_prevalence(invalid)


def test_odds_matches_definition() -> None:
    assert math.isclose(odds(0.25), 0.25 / 0.75, rel_tol=1e-12)


def test_logit_offset_identity_and_reversibility() -> None:
    base = 0.2
    assert math.isclose(logit_offset(base, base), 0.0, abs_tol=1e-12)

    offset = logit_offset(base, 0.8)
    rng = np.random.default_rng(0)
    scores = rng.standard_normal(128)
    shifted = apply_logit_offset(scores, offset)
    restored = apply_logit_offset(shifted, -offset)
    np.testing.assert_allclose(restored, scores)


def test_offset_does_not_change_ranking() -> None:
    rng = np.random.default_rng(1)
    logits = rng.uniform(-2, 2, size=50)
    offset = 1.5
    shifted = apply_logit_offset(logits, offset)
    np.testing.assert_array_equal(np.argsort(logits), np.argsort(shifted))


def test_threshold_from_costs_and_decision() -> None:
    thr = threshold_from_costs(pi=0.3, c10=2.0, c01=1.0)
    assert thr > 0

    logits = np.array([-5.0, 0.0, 5.0])
    decisions = decision_from_logits(logits, thr)
    assert decisions.tolist() == [0, 0, 1]

    with pytest.raises(ValueError):
        threshold_from_costs(pi=0.5, c10=0.0, c01=1.0)
