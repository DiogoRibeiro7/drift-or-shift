# Experiment Report

This report walks through rerunning the experiments end-to-end and explains what
each artifact says about label shift and concept drift. The numeric summary lives
in [RESULTS_DIGEST.md](RESULTS_DIGEST.md); use that for a quick glance at metrics.

## Reproducing every experiment

1. Install the project, which also installs the `dos-expN` console scripts:

   ```sh
   python -m pip install -e ".[dev]"
   ```

2. Run any experiment. Each writes `tables/`, `figures/`, and a `_summary.json`
   into `results/<exp_name>/<utc-timestamp>/`:

   | Command | Experiment |
   | --- | --- |
   | `dos-exp1 --results-dir results` | Synthetic label shift with offset correction |
   | `dos-exp2 --results-dir results` | ROC AUC invariance, PR-AUC dependence |
   | `dos-exp3 --results-dir results` | ESS against the class-weight multiplier |
   | `dos-exp4 --results-dir results` | Concept drift: offset vs. retraining |
   | `dos-exp5 --results-dir results` | Breast cancer label shift replication |
   | `dos-exp6 --results-dir results` | Calibration vs. offset under label shift |
   | `dos-exp7 --results-dir results` | Drift-type sweep (covariance, feature, label) |
   | `dos-exp8 --results-dir results` | Multimodal label shift |
   | `dos-exp9 --results-dir results` | Covertype label shift (downloads a dataset) |
   | `dos-exp10 --results-dir results` | Credit-card fraud benchmark (downloads a dataset) |
   | `dos-exp11 --results-dir results` | High-variance medical-style benchmark |

   Equivalently, run a module directly, which needs no install:
   `python src/drift_or_shift/experiments/exp1_label_shift_synth.py --results-dir results`.

3. Every script accepts `--help`. Shared flags (`--n-train`, `--n-test`, `--d`,
   `--pi-tests`, `--seeds`, `--c10`, `--c01`) let you cut a run down for a quick
   check. Point `--results-dir` at the same base each time so artifacts group
   under `results/<exp>/<timestamp>/`.

4. Aggregate everything into a reviewable dashboard:

   ```sh
   python scripts/aggregate_results.py --results-dir results \
       --output reports/dashboard.md --figure reports/best_risk.png
   ```

`results/` is regenerated output and is not committed.

## What each artifact shows

**Exp1 — synthetic label shift.** The table gives mean/std risk for *no
correction*, *offset*, and *oracle* across the `pi_test` grid; the figure plots
risk against prevalence. An additive logit offset recovers the test posterior,
pulling the risk curve close to the oracle. The gain is largest where the shift
is largest, and exactly zero at `pi_test = pi_train`, where no correction is
called for.

**Exp2 — ROC AUC invariance.** Mean/std `roc_auc` and `pr_auc` across the grid,
as a two-panel figure. A constant offset preserves the ranking of scores, so ROC
AUC barely moves; PR-AUC depends on the positive rate and climbs steeply with it.
This is the cleanest statement of why a prevalence-invariant metric can hide a
problem that a prevalence-sensitive one exposes.

**Exp3 — effective sample size.** ESS fraction for each `alpha`, on a log-scale
figure. The Kish-style formula predicts the decay, and the curve shows how much
effective data aggressive class weighting throws away.

**Exp4 — concept drift.** Aggregated risk for no correction, offset, and
retraining, as a bar chart. When the class-conditionals move, the offset is
*inert*: it produces exactly the same risk as doing nothing, because it only
shifts a prior that is no longer the problem. Retraining on the drifted data is
what lowers risk.

**Exp5 — breast cancer.** Real-data risk per strategy across the prevalence
grid. Features are standardized on the training split before fitting, so the
model converges and the results reproduce across platforms. Offset correction
helps at every prevalence, most under the largest shift, and is inert at
`pi_test = pi_train`. See the takeaway below for why this experiment used to
say something different.

**Exp6–Exp11.** Calibration under label shift (temperature and isotonic), a
sweep over drift types, multimodal mixtures, and three higher-dimensional or
real-world benchmarks. Each follows the same artifact layout, and each records
its configuration in `_summary.json`.

## Conceptual takeaways

- **Label shift is correctable without retraining.** Exp1 shows the offset
  closing most of the gap to the oracle while preserving ranking.
- **Concept drift is not.** Exp4 shows the offset making no difference at all,
  because the quantity it corrects is not the one that moved. This is the
  distinction the whole repository is built around.
- **Metric choice decides what you can see.** Exp2 shows ROC AUC surviving a
  prevalence shift that PR-AUC registers strongly. Reporting only the invariant
  metric would hide the shift entirely.
- **An unconverged fit is not a result.** Exp5 and Exp9 used to fit on raw,
  unscaled features, so LBFGS stopped at `max_iter` rather than at an optimum and
  the coefficients depended on the platform's BLAS. That produced a finding —
  that offset correction *hurts* at `pi_test=0.5` on real data — which looked like
  an interesting limitation of the method and was in fact an artifact. With the
  features standardized, both experiments converge and the offset helps at every
  prevalence. When a real-data result contradicts the theory, check that the
  optimizer finished before believing it.

## Verification

`tests/test_documented_results.py` re-runs the configurations behind
`RESULTS_DIGEST.md` on every CI build and asserts both the published figures and
the qualitative claims above, on Linux, macOS and Windows.
`test_exp5_fit_converges` and `test_exp5_run_emits_no_convergence_warnings` guard
the convergence fix, so the experiment cannot silently regress to an unconverged
state.
