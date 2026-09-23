# Results Digest

This digest summarizes a reference run of each experiment and highlights how the
numbers support (or nuance) the claims in _ssrn-6052514_.

**Reproducing these numbers.** `results/` is regenerated output and is not
committed, so the artifact paths below are illustrative of a run's layout
rather than files you will find in a fresh clone. Each entry records the exact
configuration; run the matching `dos-expN` command with those flags to
regenerate it.

**These figures are enforced.** `tests/test_documented_results.py` re-runs the
configurations below on every CI build and checks both the published figures
and the qualitative claims, so this digest cannot silently go stale.

**Experiment 5 is the exception, and it is a known defect.** Its figures
reproduce exactly on Windows and differ on Linux and macOS, because it fits
logistic regression on raw, unscaled features whose means span `0.004` to
`880`. LBFGS exhausts `max_iter` without converging, so the coefficients --
and every number derived from them -- depend on the platform's BLAS.
Standardizing the features makes the same fit converge in about twenty
iterations. That would make Exp5 reproducible but would also change its
published numbers, so it is left as a maintainer decision; the Exp5
assertions are skipped until then. Exp1-Exp4 are stable on every platform
CI covers.

## Experiment 1 – Synthetic label shift with offset correction

- **Config & provenance:** `n_train=2000`, `n_test=2000`, `d=6`, `pi_train=0.2`, `seeds=[0,1,2,3,4]`, symmetric costs, timestamp `2026-01-14T13:26:26.879652`. Table: `results/exp1_label_shift_synth/20260114_132626/tables/exp1_label_shift_synth.csv` (per-prevalence mean/std risk). Figure: `results/exp1_label_shift_synth/20260114_132626/figures/exp1_label_shift_synth.png` (risk vs. prevalence curves).
- **Key metrics:** At `pi_test=0.01`, `risk_none=0.0469`, `risk_offset=0.0084`, `risk_oracle=0.0078`; at `pi_test=0.5`, `risk_none=0.1699`, `risk_offset=0.1299`, `risk_oracle=0.1257`. Offset correction closes most of the gap to the oracle for both extremes while preserving ordering, validating the paper's label-shift claim.
- **Observation/deviation:** No notable deviation--the empirical risk curves match the expectation that offsetting logits recovers the test posterior, confirming label-shift adaptation without retraining.

## Experiment 2 – AUC invariance and PR-AUC dependence

- **Config & provenance:** Same data configuration as Exp1, timestamp `2026-01-14T13:26:41.496639`. Table: `results/exp2_auc_pr_invariance/20260114_132641/tables/exp2_auc_pr_invariance.csv`. Figure: `results/exp2_auc_pr_invariance/20260114_132641/figures/exp2_auc_pr_invariance.png` (two-panel plot of ROC AUC and PR AUC).
- **Key metrics:** ROC AUC stays in `[0.93, 0.96]` across the `pi_test` grid (observed range
  `0.938`-`0.957`; an earlier revision of this digest quoted `[0.94, 0.96]`, which the
  `0.9382` at `pi_test=0.1` falls outside), while PR AUC climbs from ~0.35 (pi=0.01) to ~0.94 (pi=0.5). This mirrors the claim that ROC AUC is prevalence-invariant while PR AUC depends on the positive rate.
- **Observation/deviation:** Prevalence dependence of PR AUC is even steeper than the paper's toy curve, hinting that highly imbalanced scenarios amplify the precision drop faster.

## Experiment 3 – ESS vs. class-weight multiplier

- **Config & provenance:** `n=1000`, `pi=0.2`, `alphas=[1,5,10,20,50]`, timestamp `2026-01-14T13:26:51.334537`. Table: `results/exp3_ess_vs_weight/20260114_132651/tables/exp3_ess_vs_weight.csv`. Figure: `results/exp3_ess_vs_weight/20260114_132651/figures/exp3_ess_vs_weight.png`.
- **Key metrics:** ESS fraction drops from `1.00` at `alpha=1` to `0.23` at `alpha=50`, mirroring the Kish-style formula and explaining why aggressive class weighting diminishes effective training data.
- **Observation/deviation:** No deviation--dropping ESS confirms the paper's warning about variance inflation under strong weights.

## Experiment 4 – Concept drift comparison

- **Config & provenance:** `n_train=2000`, `n_test=2000`, `d=6`, `pi_train=0.2`, `seeds=[0,1,2]`, timestamp `2026-01-14T13:28:05.720822`. Table: `results/exp4_concept_drift/20260114_132805/tables/exp4_concept_drift.csv`. Figure: `results/exp4_concept_drift/20260114_132805/figures/exp4_concept_drift.png` (bar chart of risks).
- **Key metrics:** `risk_none=0.0995`, `risk_offset=0.0995`, `risk_retrain=0.0968`. Offset correction fails to improve risk when class-conditionals drift, while retraining on the drifted distribution recovers better performance, matching the paper's cautionary example.
- **Observation/deviation:** None; concept drift nullifies the offset-only strategy as expected, reinforcing the need to retrain.

## Experiment 5 – Breast cancer label shift replication

- **Config & provenance:** Stratified train/test split of `sklearn.datasets.load_breast_cancer`, resampled to `pi_train=0.2`, `pi_tests` as above, `seeds=[0,1,2,3,4]`, timestamp `2026-01-14T13:28:15.064888`. Table: `results/exp5_realdata_breast_cancer/20260114_132814/tables/exp5_realdata_breast_cancer.csv`. Figure: `results/exp5_realdata_breast_cancer/20260114_132814/figures/exp5_realdata_breast_cancer.png`.
- **Key metrics (Windows; see the caveat above):** At `pi_test=0.01`,
  `risk_none=0.0246`, `risk_offset=0.0091`, `risk_oracle=0.0035`; at `pi_test=0.5`,
  `risk_none=0.0484`, `risk_offset=0.0519`, `risk_oracle=0.0421`. Offset correction
  helps most under the strongest shift (`pi_test=0.01`), which holds on every
  platform. The mid-grid and `pi_test=0.5` figures do **not** reproduce elsewhere:
  on Linux the offset wins at `pi_test=0.5` rather than losing. Treat these numbers
  as one platform's run of a non-converged fit, not as a result.
- **Observation/deviation:** The LBFGS convergence warnings that earlier revisions of
  this digest noted in passing are not cosmetic -- they are the reason this experiment
  is not reproducible. Until the features are scaled, no conclusion should be drawn
  from Exp5 about whether offset correction helps or hurts at a given prevalence;
  different platforms disagree on the sign. `test_exp5_does_not_converge_on_raw_features`
  pins the cause so it stays visible.
