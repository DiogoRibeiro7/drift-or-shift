"""DriftOrShift: label shift and offset correction utilities."""

from .data_synth import (
    concept_drift_mu1,
    default_mu0,
    default_mu1,
    make_gaussian_binary,
    resample_to_prevalence,
)
from .ess import ess_fraction, effective_sample_size
from .metrics import (
    oracle_threshold_min_risk,
    pr_auc,
    roc_auc,
    risk_cost_sensitive,
)
from .models import fit_logistic_regression, predict_logits
from .shift import (
    apply_logit_offset,
    decision_from_logits,
    logit_offset,
    odds,
    threshold_from_costs,
    validate_prevalence,
)

__all__ = [
    "fit_logistic_regression",
    "predict_logits",
    "concept_drift_mu1",
    "default_mu0",
    "default_mu1",
    "make_gaussian_binary",
    "resample_to_prevalence",
    "ess_fraction",
    "effective_sample_size",
    "odds",
    "logit_offset",
    "apply_logit_offset",
    "threshold_from_costs",
    "decision_from_logits",
    "validate_prevalence",
    "oracle_threshold_min_risk",
    "pr_auc",
    "roc_auc",
    "risk_cost_sensitive",
]
