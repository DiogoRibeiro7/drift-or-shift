import numpy as np
import pytest
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

from drift_or_shift import (
    fit_logistic_regression,
    oracle_threshold_min_risk,
    pr_auc,
    predict_logits,
    risk_cost_sensitive,
    roc_auc,
)


def test_fit_logistic_regression_produces_logits() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 3))
    y = (rng.random(80) > 0.4).astype(int)
    model = fit_logistic_regression(X, y, rng=rng)
    logits = predict_logits(model, X)
    assert logits.shape == (80,)
    assert np.isfinite(logits).all()


def test_risk_cost_sensitive_zero_for_perfect_preds() -> None:
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 0])
    assert risk_cost_sensitive(y_true, y_pred, 2.0, 1.0) == 0.0


def test_roc_pr_auc_match_sklearn() -> None:
    rng = np.random.default_rng(1)
    scores = rng.standard_normal(50)
    y_true = (rng.random(50) > 0.3).astype(int)
    assert roc_auc(y_true, scores) == pytest.approx(roc_auc_score(y_true, scores))
    precision, recall, _ = precision_recall_curve(y_true, scores)
    expected_pr_auc = auc(recall, precision)
    assert pr_auc(y_true, scores) == pytest.approx(expected_pr_auc)


def test_oracle_threshold_improves_over_fixed_decision() -> None:
    scores = np.array([-2.0, -1.0, 0.5, 1.5])
    y_true = np.array([0, 0, 1, 1])
    naive_decisions = (scores >= 0.0).astype(int)
    naive_risk = risk_cost_sensitive(y_true, naive_decisions, 2.0, 1.0)
    _, oracle_risk = oracle_threshold_min_risk(y_true, scores, 2.0, 1.0)
    assert oracle_risk <= naive_risk
