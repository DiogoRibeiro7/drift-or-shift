"""DriftOrShift: label shift and offset correction utilities."""

from .data_synth import (
    concept_drift_mu1,
    default_mu0,
    default_mu1,
    make_gaussian_binary,
    resample_to_prevalence,
)
from .ess import ess_fraction, effective_sample_size
from .experiments._common import (
    ExperimentConfig,
    DEFAULT_COSTS,
    PI_TEST_GRID,
    PI_TRAIN,
    SEEDS,
    aggregate_mean_std,
)
from .io_utils import ensure_dir, save_json, save_table, timestamped_run_dir
from .metrics import (
    oracle_threshold_min_risk,
    pr_auc,
    roc_auc,
    risk_cost_sensitive,
)
from .models import fit_logistic_regression, predict_logits
from .plotting import (
    plot_auc_pr_vs_prevalence,
    plot_ess_vs_alpha,
    plot_risk_vs_prevalence,
)
from .shift import (
    apply_logit_offset,
    decision_from_logits,
    logit_offset,
    odds,
    threshold_from_costs,
    validate_prevalence,
)

__all__ = [
    "ExperimentConfig",
    "DEFAULT_COSTS",
    "PI_TEST_GRID",
    "PI_TRAIN",
    "SEEDS",
    "aggregate_mean_std",
    "concept_drift_mu1",
    "default_mu0",
    "default_mu1",
    "effective_sample_size",
    "ensure_dir",
    "ess_fraction",
    "fit_logistic_regression",
    "make_gaussian_binary",
    "odds",
    "oracle_threshold_min_risk",
    "plot_auc_pr_vs_prevalence",
    "plot_ess_vs_alpha",
    "plot_risk_vs_prevalence",
    "predict_logits",
    "pr_auc",
    "resample_to_prevalence",
    "roc_auc",
    "risk_cost_sensitive",
    "save_json",
    "save_table",
    "threshold_from_costs",
    "timestamped_run_dir",
    "validate_prevalence",
]
