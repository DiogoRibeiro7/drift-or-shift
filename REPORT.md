% Experiment Report for drift-shift-pipeline

This report walks through rerunning the experiments end-to-end and explains what each artifact says about label shift and concept drift. The numeric summary lives in `RESULTS_DIGEST.md`, so use that for a quick glance at metrics.

## Reproducing every experiment
1. Create an editable install (so entry points recognize the package):
   ```sh
   python -m pip install -e .
   ```
2. Each experiment writes to `results/<exp_name>/<timestamp>/` with `tables/`, `figures/`, and a `_summary.json`. Run them with the provided script files:
   - **Exp1 (label shift)**:
     ```sh
     python src/drift_or_shift/experiments/experiments/exp1_label_shift_synth.py --results-dir results
     ```
   - **Exp2 (ROC/PR invariance)**:
     ```sh
     python src/drift_or_shift/experiments/experiments/exp2_auc_pr_invariance.py --results-dir results
     ```
   - **Exp3 (ESS vs. class weight)**:
     ```sh
     python src/drift_or_shift/experiments/experiments/exp3_ess_vs_weight.py --results-dir results
     ```
   - **Exp4 (concept drift)**:
     ```sh
     python src/drift_or_shift/experiments/experiments/exp4_concept_drift.py --results-dir results
     ```
   - **Exp5 (breast cancer label shift)**:
     ```sh
     python src/drift_or_shift/experiments/experiments/exp5_realdata_breast_cancer.py --results-dir results
     ```
3. Each script honors CLI flags shown in its `_parse_args` function (`--n-train`, `--pi-tests`, `--c10`, etc.), so you can create smaller runs (fewer seeds, fewer prevalences) for quick checks. Always point `--results-dir` to the same base so artifacts group under `results/<exp>/<timestamp>/`.

## Artifact narrative
- **Exp1 table & figure** (`results/exp1_label_shift_synth/...`):
  - Table: mean/std risk for *no correction*, *offset*, and *oracle* across `pi_test`. It proves that an additive logit offset recovers the test posterior, driving the risk curve close to the oracle line.
  - Figure: risk vs. prevalence; visualizes how offset correction dramatically lowers risk for rare positives and maintains alignment for balanced cases.
- **Exp2 table & figure** (`results/exp2_auc_pr_invariance/...`):
  - Table: mean/std `roc_auc` and `pr_auc`. Shows that ROC AUC is stable while PR AUC rises with prevalence as expected.
  - Figure: two-panel plot highlighting the invariance/dependence trade-off, reinforcing the paper’s claim that ranking (ROC) is prevalence agnostic but precision–recall shifts under skew.
- **Exp3 table & figure** (`results/exp3_ess_vs_weight/...`):
  - Table: ESS fraction for each `alpha`.
  - Figure: log-scale plot of ESS fraction vs. `alpha`, illustrating the same variance inflation the paper predicts when class weights grow.
- **Exp4 table & figure** (`results/exp4_concept_drift/...`):
  - Table: aggregated risk for no correction, offset, and retrain.
  - Figure: bar chart showing that offset correction sticks to the stale decision boundary while retraining on the drifted data lowers risk, typifying concept drift failure modes.
- **Exp5 table & figure** (`results/exp5_realdata_breast_cancer/...`):
  - Table: real-data risk averages per strategy across prevalence grid.
  - Figure: risk vs. prevalence on breast-cancer data, demonstrating that offset correction still helps but that practical solvers may issue LBFGS convergence warnings (scale/max_iter if you rerun).

## Conceptual takeaways
- Label shift only (Exp1 & Exp5) keeps the curves ordered as predicted: offset correction closes the gap to the oracle while preserving ranking.
- Aggregated ROC and PR metrics (Exp2) give a first-principles view of why ROC curves survive prevalence shifts but precision–recall does not.
- Concept drift (Exp4) breaks the offset strategy, so retraining is the only path to lower risk even when prevalences match training.
