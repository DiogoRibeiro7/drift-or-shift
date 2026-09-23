"""Metrics used for comparing label shift adjustments."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score


def risk_cost_sensitive(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    c10: float,
    c01: float,
) -> float:
    """Return the average cost-sensitive risk given predictions."""
    if c10 < 0 or c01 < 0:
        raise ValueError("Costs must be non-negative.")
    y_true_arr = np.asarray(y_true, int)
    y_pred_arr = np.asarray(y_pred, int)
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("Shapes of y_true and y_pred must match.")
    fn = np.sum((y_true_arr == 1) & (y_pred_arr == 0))
    fp = np.sum((y_true_arr == 0) & (y_pred_arr == 1))
    return float((c10 * fn + c01 * fp) / len(y_true_arr))


def roc_auc(y_true: Sequence[int], scores: Sequence[float]) -> float:
    """Wrap sklearn ROC AUC with a consistent signature."""
    return float(roc_auc_score(y_true, scores))


def pr_auc(y_true: Sequence[int], scores: Sequence[float]) -> float:
    """Compute PR AUC using the precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, scores)
    return float(auc(recall, precision))


def oracle_threshold_min_risk(
    y_true: Sequence[int],
    scores: Sequence[float],
    c10: float,
    c01: float,
) -> tuple[float, float]:
    """Return the threshold minimizing risk (operates on logits)."""
    score_arr = np.asarray(scores, float)
    thresholds = np.concatenate(([np.inf], np.unique(score_arr), [-np.inf]))
    best_threshold = thresholds[0]
    best_risk = float("inf")
    for thr in thresholds:
        decisions = (score_arr >= thr).astype(int)
        risk = risk_cost_sensitive(y_true, decisions, c10, c01)
        if risk < best_risk:
            best_risk = risk
            best_threshold = thr
    return best_threshold, best_risk
