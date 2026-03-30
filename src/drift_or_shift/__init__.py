"""DriftOrShift: label shift and offset correction utilities."""

from .calibration import (
    apply_calibrator,
    find_best_temperature,
    fit_isotonic_calibrator,
    temperature_scale,
)
from .data_synth import (
    concept_drift_mu1,
    default_mu0,
    default_mu1,
    make_gaussian_binary,
    make_multimodal_binary,
    resample_to_prevalence,
)
from .ess import ess_fraction, effective_sample_size
from .drift_monitor import feature_drift_summary, univariate_feature_stats
from .drift_variants import (
    apply_covariance_shift,
    density_ratio_shift,
    drift_score_from_ratio,
    inject_label_noise,
    shift_mean_vector,
)
from .experiments._common import (
    ExperimentConfig,
    DEFAULT_COSTS,
    PI_TEST_GRID,
    PI_TRAIN,
    SEEDS,
    aggregate_mean_std,
    feature_drift_metrics,
    DRIFT_FEATURE_METRICS,
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
from .reporting import (
    collect_summary_jsons,
    format_results_overview,
    select_latest_summaries,
    write_results_overview,
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
    "feature_drift_metrics",
    "concept_drift_mu1",
    "default_mu0",
    "default_mu1",
    "effective_sample_size",
    "ensure_dir",
    "ess_fraction",
    "fit_logistic_regression",
    "make_gaussian_binary",
    "make_multimodal_binary",
    "odds",
    "apply_covariance_shift",
    "density_ratio_shift",
    "drift_score_from_ratio",
    "inject_label_noise",
    "shift_mean_vector",
    "feature_drift_summary",
    "univariate_feature_stats",
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
    "collect_summary_jsons",
    "format_results_overview",
    "select_latest_summaries",
    "write_results_overview",
    "apply_calibrator",
    "find_best_temperature",
    "fit_isotonic_calibrator",
    "temperature_scale",
    "apply_logit_offset",
    "decision_from_logits",
    "logit_offset",
    "DRIFT_FEATURE_METRICS",
]
