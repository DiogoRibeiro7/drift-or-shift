# Results Digest

This digest summarizes a reference run of each experiment and highlights how the
numbers support (or nuance) the claims in [SSRN 6052514](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6052514).

**Reproducing these numbers.** `results/` is regenerated output and is not
committed, so the artifact paths below are illustrative of a run's layout
rather than files you will find in a fresh clone. Each entry records the exact
configuration; run the matching `dos-expN` command with those flags to
regenerate it.

**These figures are enforced.** `tests/test_documented_results.py` re-runs the
configurations below on every CI build and checks both the published figures
and the qualitative claims, so this digest cannot silently go stale.

**Experiments 5 and 9 were re-derived after a convergence fix.** Both fit
logistic regression on raw, unscaled real-world features -- breast-cancer
means span `0.004` to `880`, Covertype's span `0` to `2959` -- so LBFGS
exhausted `max_iter` without converging and the coefficients depended on the
platform's BLAS. Their previous figures were one machine's snapshot of a
non-converged optimizer and did not reproduce on Linux or macOS. Both now
standardize features on the training split (as Exp10 already did), converge
in tens of iterations, and the figures below are the corrected values.

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
- **Key metrics:** At `pi_test=0.01`, `risk_none=0.0449`, `risk_offset=0.0105`,
  `risk_oracle=0.0042`; at `pi_test=0.5`, `risk_none=0.0540`, `risk_offset=0.0421`,
  `risk_oracle=0.0351`. Offset correction helps at every prevalence, most strongly
  under the largest shift, and is exactly inert at `pi_test=pi_train=0.2`.
- **Observation/deviation:** An earlier revision of this digest reported that offset
  correction was *worse than doing nothing* at `pi_test=0.5` (`0.0519` vs `0.0484`)
  and read that as a real limitation of the method on finite real data. It was not:
  it was an artifact of the unconverged fit. With the fit converged, the offset helps
  at every prevalence, exactly as the theory predicts. At five seeds the difference at
  `pi_test=0.1` is smaller than the seed-to-seed spread (`0.0007` against a standard
  deviation of `0.0152`); the qualitative claim is checked over twenty seeds in
  `test_exp5_offset_helps_at_every_prevalence`. The LBFGS convergence warnings earlier
  revisions mentioned in passing were the symptom of the underlying defect.

## Experiment 6 – Calibration vs. offset under label shift

- **Config & provenance:** `n_train=2000`, `n_test=2000`, `d=6`, `pi_train=0.2`, `seeds=[0..4]`,
  symmetric costs. Compares the plain offset against temperature scaling and isotonic
  regression applied before the same offset.
- **Key metrics:** At `pi_test=0.01`, `risk_none=0.0469`, `risk_offset=0.0084`,
  `risk_temp_offset=0.0085`, `risk_isotonic_offset=0.0091`; at `pi_test=0.5`, `0.1699`,
  `0.1299`, `0.1295`, `0.1308` respectively.
- **Observation/deviation:** Calibration buys nothing here, and isotonic regression is
  consistently the worst of the three. That is the expected result rather than a
  disappointment: under pure label shift a correctly specified logistic model is already
  calibrated, so there is no miscalibration left for temperature or isotonic scaling to
  remove — only estimation variance for them to add. Reach for calibration when the model
  is misspecified, not when the prior moved.

## Experiment 7 – Drift-type sweep

- **Config & provenance:** `n_train=2000`, `n_test=2000`, `d=6`, `pi_train=0.2`,
  `seeds=[0..4]`. Sweeps covariance shift, feature shift, and label noise at fixed prevalence.
- **Key metrics:**

  | drift type | `risk_none` | `risk_offset` | `risk_retrain` | `drift_score` |
  | --- | --- | --- | --- | --- |
  | covariance shift | 0.1663 | 0.1663 | 0.1432 | 3.7540 |
  | feature shift | 0.0959 | 0.0959 | 0.0864 | 1.0548 |
  | label noise | 0.3358 | 0.3358 | 0.3351 | 1.0514 |

- **Observation/deviation:** `risk_offset` equals `risk_none` to the last digit in all three
  rows, which is the cleanest statement of the repository's central point: none of these
  scenarios moves the class prior, so the offset has nothing to correct and is exactly
  inert. Retraining recovers something for covariance and feature shift. Under label noise
  it recovers almost nothing (`0.3358` to `0.3351`) — noise destroys information that no
  amount of refitting restores.

## Experiment 8 – Multimodal label shift

- **Config & provenance:** `n_train=2000`, `n_test=2000`, `d=6`, `pi_train=0.2`, `seeds=[0..4]`,
  each class a mixture of Gaussian components.
- **Key metrics:** At `pi_test=0.01`, `risk_none=0.0403`, `risk_offset=0.0077`,
  `risk_oracle=0.0064`; at `pi_test=0.5`, `0.1360`, `0.1059`, `0.1023`.
- **Observation/deviation:** None. The offset correction does not require unimodal
  class-conditionals — only that they stay fixed while the prior moves.

## Experiment 9 – Covertype label shift

- **Config & provenance:** `n_train=2000`, `n_test=2000`, `seeds=[0..4]`, features standardized
  on the training split. Downloads the Covertype dataset.
- **Key metrics:** At `pi_test=0.01`, `risk_none=0.0646`, `risk_offset=0.0101`,
  `risk_oracle=0.0097`; at `pi_test=0.5`, `0.3595`, `0.2488`, `0.2409`.
- **Observation/deviation:** The offset helps at every prevalence on a real, high-dimensional
  dataset. These figures post-date the convergence fix; before it this experiment fitted on
  raw features spanning `0` to `2959` and never converged.

## Experiment 10 – Credit-card fraud

- **Config & provenance:** `n_train=2000`, `n_test=2000`, `seeds=[0..4]`, features standardized
  on the training split. Downloads the `creditcard` dataset from OpenML.
- **Key metrics:** At `pi_test=0.01`, `risk_none=0.0041`, `risk_offset=0.0033`,
  `risk_oracle=0.0016`; at `pi_test=0.5`, `0.1332`, `0.1139`, `0.0801`.
- **Observation/deviation:** At `pi_test=0.05` the offset is marginally worse than no
  correction (`0.0173` vs `0.0153`). With five seeds on a severely imbalanced dataset this is
  within run-to-run variation, and it is **not** treated as a finding — the mistake made
  earlier with Exp5, where a similar inversion was written up as a property of the method
  before it turned out to be an artifact. The gap between `risk_offset` and `risk_oracle` is
  wider here than anywhere else, which says the decision threshold, not the prior, is what
  limits performance under extreme imbalance.

## Experiment 11 – High-variance medical benchmark

- **Config & provenance:** `n_train=2000`, `n_test=2000`, `d=6`, `pi_train=0.2`, `seeds=[0..4]`,
  class-specific covariances and nonlinear test-time shifts.
- **Key metrics:** At `pi_test=0.01`, `risk_none=0.1019`, `risk_offset=0.0165`,
  `risk_retrain=0.0092`; at `pi_test=0.5`, `0.1184`, `0.1332`, `0.1095`.
- **Observation/deviation:** At `pi_test=0.5` the offset is worse than doing nothing. Unlike
  the superficially identical result Exp5 used to report, this one is real: the model
  converges cleanly, and the experiment deliberately violates the label-shift assumption by
  giving each class its own covariance. When the class-conditionals differ, a single additive
  logit offset is the wrong correction, and retraining wins at every prevalence.
