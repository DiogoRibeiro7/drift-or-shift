# drift-shift-pipeline

![CI](https://github.com/DiogoRibeiro7/drift-shift-pipeline/actions/workflows/ci.yml/badge.svg) ![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green)

`drift-shift-pipeline` packages a lean research pipeline for reproducing the label shift insights from `ssrn-6052514`. Start with synthetic two-class Gaussian data, apply Bayes-ideal offset correction without retraining, and then inspect how invariances (ROC AUC) and dependencies (PR-AUC, ESS) emerge across multiple experiments. When the class-conditionals drift, we rerun logistic regression (concept drift experiment) and show why the offset-only strategy fails.

Use the CLI entry points or the experiment scripts to rerun every figure/table, and rely on the smoke runner + tests to stay release-ready as you iterate.

## Quickstart

- `pip install -e .` (also installs the `dos-exp1`...`dos-exp5` scripts).
- `python -c "import drift_or_shift"` should succeed to ensure the package is discoverable.
- During development run `pytest`, `ruff .`, and `mypy` as listed in `pyproject.toml` to keep invariants tight.

## Key concepts

### Label shift vs. concept drift

- **Label shift**: class-conditionals remain stable; only `pi_test` differs from `pi_train`. An additive logit offset pushes the scoring function to the test prevalence.
- **Concept drift**: decision boundaries move (e.g., drifted `mu1`). Offset correction can't recover the new posterior, so retraining becomes necessary, as shown in Exp4.

### Offset correction & thresholds

- Logits are log-odds, and the offset is `log(pi_test*(1-pi_train) / (pi_train*(1-pi_test)))`.
- Apply the offset before comparing to the threshold derived from `log(c10/c01)`; default costs (`c10=c01=1`) give a zero threshold.
- The ranking of scores remains invariant under constant offsets, so ROC AUC stays stable across prevalences.

## Experiments snapshot

- `dos-exp1`: synthetic label shift with offset correction vs. oracle risk (`src/drift_or_shift/experiments/experiments/exp1_label_shift_synth.py`).
- `dos-exp2`: ROC AUC invariance and PR-AUC dependence (`exp2_auc_pr_invariance.py`).
- `dos-exp3`: ESS fraction as class weight α grows (`exp3_ess_vs_weight.py`).
- `dos-exp4`: concept drift demonstration (standard vs. offset vs. retrained).
- `dos-exp5`: breast cancer label shift replication with resampling.
- `dos-exp6`: calibration vs. offset risks (temperature scaling + isotonic calibrators under label shift).
- `dos-exp7`: drift-type sweep showing how covariance, feature, and label shifts defeat the single offset (`src/drift_or_shift/experiments/experiments/exp7_drift_types.py`).
- `dos-exp8`: multimodal label shift with mixture components to stress-test offsets (`src/drift_or_shift/experiments/experiments/exp8_multimodal_label_shift.py`).
- `dos-exp9`: Covertype label shift replication for a real-world, high-dimensional dataset (`src/drift_or_shift/experiments/experiments/exp9_covtype_label_shift.py`).
- `dos-exp10`: credit card fraud label shift benchmark with standardized features and sample-reweighted prevalences (`src/drift_or_shift/experiments/experiments/exp10_credit_card_fraud.py`).
- `dos-exp11`: high-variance medical-style benchmark that injects class-specific covariances and nonlinear test shifts to stress offset correction (`src/drift_or_shift/experiments/experiments/exp11_high_variance_medical.py`).
- Each run writes `tables/`, `figures/`, and a `_summary.json` into `results/<exp>/<timestamp>/`.

## Reproducibility practices

- All data generation accepts `seed`, and experiments record their seeds, sizes, costs, and prevalence grids in the summary JSON.
- Use `ExperimentConfig` in `drift_or_shift.experiments._common` or the CLI args to consistently reproduce runs.
- The repository ships with smoke runners and regression tests, so rerunning `python scripts/smoke_exp1.py --results-dir results/smoke` gives you a fast sanity check.

## Extension utilities

- `drift_or_shift.drift_variants` lets you inject covariance/feature shifts, label noise, and density-ratio scoring for new drift scenarios without rebuilding the generators.
- `scripts/aggregate_results.py` (uses `drift_or_shift.reporting`) collects every `results/*/*_summary.json` and emits a markdown overview in `reports/` so you can surface orientation-ready summaries for reviewers.
- `scripts/drift_alerts.py` scans the latest summaries for `feature_max_*` stats that exceeded the configured thresholds and writes `reports/drift_alerts.csv` (or a supplied fallback) for quick drift triage; `scripts/aggregate_results.py` now accepts `--fail-on-alerts` to stop the pipeline (exit 1) when any alert exists.
- `scripts/watch_results.py` keeps a lightweight poller around the `results/` tree, rerunning the drift alerts/dashboard pair whenever new `_summary.json` files appear so dashboards stay aligned with fresh batches; use `python scripts/watch_results.py --once` for a single refresh or run it long-lived with `--poll-interval` tuned to your workflow.

## Reports & lessons

- `RESULTS_DIGEST.md` summarizes the latest metrics and artifact paths so reviewers can understand what each experiment demonstrates without rerunning it.
- `REPORT.md` lists the reproduction steps and highlights what each figure/table claims about label shift vs. concept drift.
- `notes/lessons.md` captures counter-intuitive behaviors (e.g., PR-AUC dependence, ESS shrinkage, solver warnings) for future exploration.

## Maintenance & readiness

- Follow `RELEASE.md` when preparing versioned candidates: bump metadata, run the smoke script, capture results, and tag.
- `FUTURE.md` holds candidate extensions (new datasets, drift types, calibration studies, automation dashboards) plus their learning goals.
- `tests/test_smoke_exp1_script.py` invokes the smoke runner every pytest session to guard against regressions.
