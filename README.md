# DriftOrShift

## Quickstart
- `pip install -e .`
- `pip install -e .` exposes the `dos-exp1`...`dos-exp5` entry points for running each experiment from the CLI.
- `python -c "import drift_or_shift"` should complete without errors.
- Use `pytest`, `ruff`, and `mypy` while developing to keep the codebase healthy.

## Concepts
### Label shift vs. concept drift
- **Label shift** assumes stable class-conditionals and only changes the prevalence; offset correction handles this case without retraining.
- **Concept drift** modifies the class-conditionals themselves (e.g., moving `mu1`), and offset correction alone cannot recover the optimal decision boundary—retraining becomes necessary.

### Offset correction
- Logits are treated as log-odds, so correcting for a new prevalence is simply an additive shift: `offset = log(pi_test*(1-pi_train) / (pi_train*(1-pi_test)))`.
- Apply the offset to the logits before thresholding to recover posteriors for the target prevalence.
- Thresholds are derived in log-odds space as `log(c10/c01)` when costs differ; otherwise 0 for symmetric costs.

## Experiments
- `dos-exp1`: synthetic label shift with offset correction vs. oracle risk.
- `dos-exp2`: AUC invariance and PR-AUC dependence on prevalence.
- `dos-exp3`: ESS fraction degradation as class weight alpha increases.
- `dos-exp4`: concept drift comparison (no correction, offset, retrained).
- `dos-exp5`: breast cancer dataset label shift replication.
- Each experiment saves:
  - `results/<exp>/<timestamp>/tables/<exp>.csv`
  - `results/<exp>/<timestamp>/figures/<exp>.png`
  - `results/<exp>/<timestamp>/<exp>_summary.json`

## Reproducibility
- Data generation always accepts an explicit `seed`, and every experiment records which seeds were used.
- Outputs include dataset sizes, prevalence grids, costs, and timestamps so you can trace each figure/table back to its configuration.
- Use the provided entry points rather than editing scripts to guarantee seeded, deterministic runs (see `ExperimentConfig` in `drift_or_shift.experiments._common`).
