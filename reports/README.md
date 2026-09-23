# reports/

Output directory for the aggregation and drift-alert scripts.

```sh
# dashboard + best-risk figure across every run in results/
python scripts/aggregate_results.py --results-dir results \
    --output reports/dashboard.md --figure reports/best_risk.png

# drift triage
python scripts/drift_alerts.py --results-dir results \
    --output reports/drift_alerts.csv
```

## What belongs here

Artifacts someone chose to keep: a dashboard attached to a release, a figure
referenced from a write-up. Add those deliberately.

Auto-generated, timestamped output does **not** belong here. The default
filenames the scripts produce — `results_dashboard_<timestamp>.md`,
`best_risk_<timestamp>.png`, `drift_alerts.csv` — are gitignored, so an
ordinary run cannot accidentally add to the repository. Pass `--output` and
`--figure` explicitly if you mean to commit something.

## Why the previous contents were removed

This directory held 30 files: fifteen dashboards and fifteen figures, all
generated over two days in January 2026. They were removed because they had
become actively misleading rather than merely stale:

- Every one of them published Exp5's pre-convergence-fix figures, including
  the `risk_offset` of `0.0519` at `pi_test=0.5` that the repository has since
  established was an artifact of an optimizer that never converged.
- Each linked artifacts under `results/`, which is regenerated output and is
  not committed, so none of the paths resolved in a fresh clone.
- The paths were written with Windows separators, so they did not resolve on
  Linux or macOS either.

They remain in the git history if any of them is ever needed.
