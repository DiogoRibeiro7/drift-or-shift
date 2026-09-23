"""drift-or-shift: label shift and offset correction utilities."""

__version__ = "0.1.0"

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
from .drift_monitor import feature_drift_summary, univariate_feature_stats
from .drift_variants import (
    apply_covariance_shift,
    density_ratio_shift,
    drift_score_from_ratio,
    inject_label_noise,
    shift_mean_vector,
)
from .ess import effective_sample_size, ess_fraction
from .experiments._common import (
    DEFAULT_COSTS,
    DRIFT_FEATURE_METRICS,
    PI_TEST_GRID,
    PI_TRAIN,
    SEEDS,
    ExperimentConfig,
    aggregate_mean_std,
    feature_drift_metrics,
)
from .io_utils import ensure_dir, save_json, save_table, timestamped_run_dir
from .metrics import oracle_threshold_min_risk, pr_auc, risk_cost_sensitive, roc_auc
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
    "DEFAULT_COSTS",
    "DRIFT_FEATURE_METRICS",
    "PI_TEST_GRID",
    "PI_TRAIN",
    "SEEDS",
    "ExperimentConfig",
    "__version__",
    "aggregate_mean_std",
    "apply_calibrator",
    "apply_covariance_shift",
    "apply_logit_offset",
    "collect_summary_jsons",
    "concept_drift_mu1",
    "decision_from_logits",
    "default_mu0",
    "default_mu1",
    "density_ratio_shift",
    "drift_score_from_ratio",
    "effective_sample_size",
    "ensure_dir",
    "ess_fraction",
    "feature_drift_metrics",
    "feature_drift_summary",
    "find_best_temperature",
    "fit_isotonic_calibrator",
    "fit_logistic_regression",
    "format_results_overview",
    "inject_label_noise",
    "logit_offset",
    "make_gaussian_binary",
    "make_multimodal_binary",
    "odds",
    "oracle_threshold_min_risk",
    "plot_auc_pr_vs_prevalence",
    "plot_ess_vs_alpha",
    "plot_risk_vs_prevalence",
    "pr_auc",
    "predict_logits",
    "resample_to_prevalence",
    "risk_cost_sensitive",
    "roc_auc",
    "save_json",
    "save_table",
    "select_latest_summaries",
    "shift_mean_vector",
    "temperature_scale",
    "threshold_from_costs",
    "timestamped_run_dir",
    "univariate_feature_stats",
    "validate_prevalence",
    "write_results_overview",
]
