# API reference

Everything below is exported from the top-level package:

```python
from drift_or_shift import logit_offset, apply_logit_offset, threshold_from_costs
```

The package ships `py.typed`, so these annotations are visible to your type
checker. Array arguments accept anything array-like — lists, NumPy arrays, or
pandas Series.

For running the experiments see [REPORT.md](../REPORT.md); for the helper
scripts see [CLI_USAGE.md](CLI_USAGE.md).

---

## Correcting label shift

The core of the library. When only the class prior moves, the correct posterior
is recovered by adding a constant to the logits — no retraining.

| Function | Purpose |
| --- | --- |
| `logit_offset(pi_train, pi_test)` | The additive correction, `log(pi_test (1-pi_train) / (pi_train (1-pi_test)))` |
| `apply_logit_offset(logits, offset)` | Add the offset to a score vector |
| `threshold_from_costs(pi, c10, c01)` | Decision threshold in logit space from misclassification costs |
| `decision_from_logits(logits, threshold)` | Binarise scores against a threshold |
| `odds(pi)` | `pi / (1 - pi)` |
| `validate_prevalence(pi)` | Raise if `pi` is outside the open interval `(0, 1)` |

```python
from drift_or_shift import apply_logit_offset, logit_offset, threshold_from_costs

offset = logit_offset(pi_train=0.2, pi_test=0.01)
threshold = threshold_from_costs(0.2, c10=1.0, c01=1.0)
decisions = (apply_logit_offset(scores, offset) >= threshold).astype(int)
```

The offset is inert when the class-conditionals move rather than the prior. That
is the distinction the repository exists to demonstrate; see Exp4 and Exp7.

## Measuring cost and ranking

| Function | Purpose |
| --- | --- |
| `risk_cost_sensitive(y_true, y_pred, c10, c01)` | Mean cost-sensitive risk of a set of decisions |
| `oracle_threshold_min_risk(y_true, scores, c10, c01)` | The risk-minimising threshold and its risk — a lower bound |
| `roc_auc(y_true, scores)` | ROC AUC, invariant to a constant offset |
| `pr_auc(y_true, scores)` | PR-AUC, which depends on prevalence |

`roc_auc` and `pr_auc` together are the point of Exp2: a prevalence-invariant
metric can hide a shift that a prevalence-sensitive one exposes.

## Effective sample size

| Function | Purpose |
| --- | --- |
| `effective_sample_size(n, pi, alpha)` | Kish-style ESS after reweighting the positive class by `alpha` |
| `ess_fraction(n, pi, alpha)` | The same as a fraction of `n` |

Answers how much data aggressive class weighting actually costs. At `alpha = 1`
the fraction is exactly 1; it falls monotonically from there.

## Generating data

| Function | Purpose |
| --- | --- |
| `make_gaussian_binary(n, d, pi, mu0, mu1, sigma, rng)` | Two-class Gaussian data at a given prevalence |
| `make_multimodal_binary(n, d, pi, components_neg, components_pos, sigma, rng)` | Each class a mixture of Gaussians |
| `resample_to_prevalence(X, y, pi_target, rng)` | Stratified resample of an existing set to a target prevalence |
| `default_mu0(d)`, `default_mu1(d)` | The mean vectors the experiments use |
| `concept_drift_mu1(mu1, shift_dim, delta)` | Move one coordinate of `mu1` to simulate concept drift |

Every generator takes an explicit `rng`. Pass `numpy.random.default_rng(seed)`
rather than relying on global state.

## Injecting drift

| Function | Purpose |
| --- | --- |
| `apply_covariance_shift(covariance, scale)` | Scale a covariance matrix |
| `shift_mean_vector(mean, delta)` | Translate a mean vector |
| `inject_label_noise(labels, flip_prob, rng)` | Flip a fraction of binary labels |
| `density_ratio_shift(X_source, X_target, bandwidth)` | Kernel-density ratio estimates |
| `drift_score_from_ratio(ratio)` | Reduce those ratios to a single score |

## Detecting drift

| Function | Purpose |
| --- | --- |
| `univariate_feature_stats(X_ref, X_target)` | Per-feature mean, std and KS differences |
| `feature_drift_summary(stats)` | Reduce those to the `feature_max_*` and `feature_mean_*` scalars |
| `feature_drift_metrics(X_ref, X_target)` | Every statistic in `DRIFT_FEATURE_METRICS` in one call |

`feature_drift_metrics` produces the keys that
`reporting.detect_drift_alerts` thresholds against. If you are only after drift
statistics and not label-shift correction, the sibling `drift_control` package
has a much fuller set (PSI, JS, KL, MMD, Wasserstein, energy, chi2, TVD).

## Calibration

| Function | Purpose |
| --- | --- |
| `find_best_temperature(logits, targets, grid=None)` | Temperature minimising logistic loss |
| `temperature_scale(logits, temperature)` | Divide logits by a temperature |
| `fit_isotonic_calibrator(scores, targets)` | Fit isotonic regression |
| `apply_calibrator(calibrator, scores)` | Apply a fitted calibrator |

Exp6 shows calibration adds nothing over the plain offset under pure label
shift: a correctly specified model is already calibrated, so scaling only adds
variance. Reach for these when the model is misspecified.

## Models

| Function | Purpose |
| --- | --- |
| `fit_logistic_regression(X, y, *, class_weight, C, max_iter, rng)` | Fit a logistic regression |
| `predict_logits(model, X)` | Decision scores in log-odds space |

Standardise real-world features before fitting. Exp5 and Exp9 originally did
not, LBFGS never converged, and their results differed by platform.

## Running experiments

| Object | Purpose |
| --- | --- |
| `ExperimentConfig(...)` | Dataclass holding sizes, prevalences, seeds and costs; `.metadata()` is written into each summary |
| `aggregate_mean_std(df, metrics, groupby)` | Mean and std per group, emitting `<metric>_mean` and `<metric>_std` |
| `PI_TRAIN`, `PI_TEST_GRID`, `SEEDS`, `DEFAULT_COSTS` | The defaults every experiment shares |
| `DRIFT_FEATURE_METRICS` | Names of the drift statistics recorded in summaries |

## Artifacts and reporting

| Function | Purpose |
| --- | --- |
| `timestamped_run_dir(base, name)` | Create `base/name/<utc-timestamp>/` with `tables/` and `figures/` |
| `save_table(df, path)` | Write a DataFrame to CSV, creating parents |
| `save_json(obj, path, indent=2)` | Write JSON, creating parents |
| `ensure_dir(path)` | Create a directory and return it |
| `collect_summary_jsons(results_root)` | Load every `*_summary.json` under a results tree |
| `select_latest_summaries(summaries)` | Newest summary per experiment name |
| `format_results_overview(summaries)` | Render a markdown overview |
| `write_results_overview(summaries, output_path)` | Write that overview to disk |

Run directories and summary timestamps are UTC and timezone-aware.
`select_latest_summaries` normalises older naive timestamps, so a results tree
containing both still sorts.

## Plotting

| Function | Purpose |
| --- | --- |
| `plot_risk_vs_prevalence(pi_tests, risk_none, risk_offset, risk_oracle)` | Risk curves for the three strategies |
| `plot_auc_pr_vs_prevalence(pi_tests, auc_scores, pr_auc_scores)` | Two-panel ROC AUC and PR-AUC |
| `plot_ess_vs_alpha(alpha, ess_fraction)` | ESS fraction against class weight, log x-axis |

Each returns a Matplotlib figure without writing it, so the caller chooses the
path. Select a non-interactive backend (`matplotlib.use("Agg")`) before
importing pyplot in a headless environment.

## Version

`drift_or_shift.__version__` is the single source of truth; `pyproject.toml`
reads it, and the release workflow refuses a tag that disagrees with it.

---

`tests/test_api_reference.py` checks that every name in `__all__` appears on
this page, so the two cannot drift apart.
