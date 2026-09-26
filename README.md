# drift-or-shift

[![CI](https://github.com/DiogoRibeiro7/drift-or-shift/actions/workflows/ci.yml/badge.svg)](https://github.com/DiogoRibeiro7/drift-or-shift/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000)](https://github.com/psf/black)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-2a6db2)](https://mypy-lang.org/)

A reproducible research pipeline for distinguishing **label shift** from **concept
drift**, and for showing exactly when a training-free logit offset is sufficient —
and when it is not.

The experiments reproduce and extend the analysis in
[SSRN 6052514](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6052514): start from synthetic two-class Gaussian
data, apply the Bayes-optimal offset correction without retraining, then observe
which metrics are invariant (ROC AUC) and which are not (PR-AUC, effective sample
size). When the class-conditionals themselves move, the offset provably cannot
recover the new posterior, and retraining becomes necessary.

---

## Installation

```bash
# From source, for development
git clone https://github.com/DiogoRibeiro7/drift-or-shift.git
cd drift-or-shift
make install          # editable install + dev extras + pre-commit hooks
```

Without `make`:

```bash
python -m pip install -e ".[dev]"
python -m pre_commit install
```

Runtime install only:

```bash
python -m pip install .
```

Requires Python 3.10–3.14. This installs the `drift_or_shift` package, the
`drift_shift_pipeline` compatibility alias, the `caliblab` calibration
benchmarking helpers, and the eleven `dos-expN` console scripts.

## Quickstart

```bash
# Fast end-to-end sanity check (seconds)
python scripts/smoke_exp1.py --results-dir results/smoke

# A full experiment
dos-exp1 --pi-train 0.2 --pi-tests 0.05 0.2 0.5 --results-dir results

# Aggregate every run into a reviewable dashboard
python scripts/aggregate_results.py --results-dir results \
    --output reports/dashboard.md --figure reports/best_risk.png
```

Each run writes `tables/`, `figures/`, and a `_summary.json` into
`results/<experiment>/<utc-timestamp>/`.

## Core concepts

### Label shift vs. concept drift

| | Label shift | Concept drift |
| --- | --- | --- |
| What moves | Class prior `π` only | The class-conditionals `p(x\|y)` |
| Posterior recoverable without retraining? | **Yes** — additive logit offset | **No** |
| Demonstrated by | Exp 1, 2, 5, 8–11 | Exp 4, 7 |

### Offset correction

Logits are log-odds, so shifting the prior from `π_train` to `π_test` is a
constant additive term:

```
offset = log( π_test · (1 − π_train) / (π_train · (1 − π_test)) )
```

Apply the offset **before** comparing against the cost-derived threshold
`log(c10 / c01)`; with default costs (`c10 = c01 = 1`) that threshold is zero.
Because a constant offset preserves the ranking of scores, ROC AUC is invariant
under label shift — while PR-AUC, which depends on prevalence, is not.

## Experiments

| Command | What it demonstrates |
| --- | --- |
| `dos-exp1` | Synthetic label shift: offset correction vs. oracle risk |
| `dos-exp2` | ROC AUC invariance and PR-AUC prevalence dependence |
| `dos-exp3` | ESS fraction shrinking as class weight α grows |
| `dos-exp4` | Concept drift: standard vs. offset vs. retrained |
| `dos-exp5` | Breast cancer label shift replication with resampling |
| `dos-exp6` | Calibration (temperature, isotonic) vs. offset under label shift |
| `dos-exp7` | Drift-type sweep: covariance, feature, and label shift |
| `dos-exp8` | Multimodal label shift with mixture components |
| `dos-exp9` | Covertype: real, high-dimensional label shift |
| `dos-exp10` | Credit-card fraud benchmark with reweighted prevalences |
| `dos-exp11` | High-variance medical-style benchmark with nonlinear test shifts |

Sources live in [src/drift_or_shift/experiments/](src/drift_or_shift/experiments/).
Every script accepts `--help`.

## Reproducibility

- All generators take an explicit `seed`; summaries record seeds, sizes, costs,
  and prevalence grids.
- Run-directory names and summary timestamps are **UTC** and timezone-aware.
- `ExperimentConfig` in `drift_or_shift.experiments._common` (or the CLI flags)
  pins a run's configuration.
- `reproduce/` holds a separate calibration-benchmark replication driven by
  [reproduce/config/default.yaml](reproduce/config/default.yaml).

## Tooling utilities

- `drift_or_shift.drift_variants` — inject covariance/feature shifts, label
  noise, and density-ratio scoring without rebuilding the generators.
- `scripts/aggregate_results.py` — collect every `results/*/*_summary.json` into
  a markdown dashboard; `--fail-on-alerts` exits non-zero when drift is detected.
- `scripts/drift_alerts.py` — flag `feature_max_*` statistics above threshold
  into `reports/drift_alerts.csv`.
- `scripts/watch_results.py` — poll `results/` and refresh alerts + dashboard as
  new summaries land (`--once` for a single pass).

See [docs/CLI_USAGE.md](docs/CLI_USAGE.md) for full flag documentation.

## Development

```bash
make check      # ruff + black + mypy + pytest, exactly what CI runs
make test-cov   # tests with an HTML coverage report
make build      # sdist + wheel, validated with twine
make help       # all targets
```

CI runs the suite on Python 3.10–3.13 on Linux, plus Windows and macOS spot
checks, and verifies that the built wheel installs and runs in a clean
environment.

Artifact and configuration I/O uses DataExcept: failed CSV/YAML reads raise
`DataLoadingError` with the source path and original exception; failed CSV,
JSON, text, directory, and figure writes raise `FileWriteError` with the
destination path and original exception. Invalid experiment settings and
non-numeric drift inputs keep their validation errors. The dashboard continues
to skip unreadable summaries and optional experiment tables.

See [CONTRIBUTING.md](CONTRIBUTING.md) for branching and PR conventions, and
[CHANGELOG.md](CHANGELOG.md) for release history.

## Project documents

| File | Contents |
| --- | --- |
| [RESULTS_DIGEST.md](RESULTS_DIGEST.md) | Latest metrics and artifact paths |
| [REPORT.md](REPORT.md) | Reproduction steps and what each figure claims |
| [notes/lessons.md](notes/lessons.md) | Counter-intuitive findings worth revisiting |
| [ROADMAP.md](ROADMAP.md) | Milestones |
| [FUTURE.md](FUTURE.md) | Candidate extensions |
| [RELEASE.md](RELEASE.md) | Release checklist |
| [docs/API.md](docs/API.md) | API reference for the public package |
| [docs/CLI_USAGE.md](docs/CLI_USAGE.md) | Flags for the helper scripts |

## Citation

If this code supports academic work, please cite the underlying paper
([SSRN 6052514](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6052514)) and reference this repository:

```bibtex
@software{ribeiro_drift_shift_pipeline,
  author  = {Ribeiro, Diogo},
  title   = {drift-or-shift: label shift, offset correction and drift diagnostics},
  url     = {https://github.com/DiogoRibeiro7/drift-or-shift},
  version = {0.1.0a1}
}
```

## License

Released under the [MIT License](LICENSE).
