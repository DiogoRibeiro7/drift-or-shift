# CLI Usage Guide

This document describes the command-line interface for the drift-shift-pipeline scripts and common usage patterns.

## 1) `scripts/drift_alerts.py`

Purpose:
- Scan summary JSON output from experiments and emit `drift_alerts.csv` for features that exceeded drift thresholds.

Usage:
```bash
python scripts/drift_alerts.py --results-dir results --output reports/drift_alerts.csv --threshold-config path/to/thresholds.yaml
```

Options:
- `--results-dir`: root of experiment result folders (default `results`)
- `--output`: output CSV file
- `--threshold-config`: YAML/JSON file overriding default thresholds

## 2) `scripts/aggregate_results.py`

Purpose:
- Aggregate all summaries, choose best per experiment, make a dashboard, and optionally enforce drift alerts.

Usage:
```bash
python scripts/aggregate_results.py --results-dir results --output reports/dashboard.md --figure reports/fig.png --drift-output reports/drift_alerts.csv --fail-on-alerts --threshold-config path/to/thresholds.yaml
```

Options:
- `--results-dir`: root folder storing `*/<timestamp>/*_summary.json`
- `--output`: markdown dashboard output path
- `--figure`: summary figure output path
- `--drift-output`: CSV path for drift alerts
- `--fail-on-alerts`: return exit code 1 when alerts non-empty
- `--threshold-config`: drift threshold override file

## 3) Experiment script conventions

For each `dos-expN` script, common CLI flags include:
- `--seed`
- `--pi_test`
- `--results-dir`
- `--no-plot`

Example:
```bash
python src/drift_or_shift/experiments/experiments/exp1_label_shift_synth.py --seed 42 --pi_test 0.2 --results-dir results/exp1
```

## 4) Recommended debug workflow

1. Run one experiment with short config:
   - `python src/drift_or_shift/experiments/experiments/exp1_label_shift_synth.py --seed 42 --pi_test 0.2 --results-dir results/exp1`.
2. Run drift alerts:
   - `python scripts/drift_alerts.py --results-dir results --output reports/drift_alerts.csv`.
3. Run dashboard aggregation:
   - `python scripts/aggregate_results.py --results-dir results --output reports/dashboard.md --figure reports/best.png --drift-output reports/drift_alerts.csv --fail-on-alerts`.
4. Inspect outputs: `reports/dashboard.md`, `reports/drift_alerts.csv`, `reports/best.png`.

## 5) Troubleshooting

- If CLI reports no summaries found:
  - check `results` folder structure and the presence of `*_summary.json` files.
  - check `timestamp` string format in summary payload.
- If drift alerts never fire, verify thresholds, or adjust `threshold-config`.
