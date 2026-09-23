# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **Experiments 5 and 9 fitted an unconverged model, and their published
  numbers are re-derived.** Both fit logistic regression on raw, unscaled
  real-world features -- breast-cancer means span `0.004` to `880`, Covertype's
  span `0` to `2959` -- so LBFGS exhausted `max_iter` without ever converging.
  The coefficients, and every risk derived from them, depended on the platform's
  BLAS: the figures reproduced exactly on Windows and differed on Linux and
  macOS, where at `pi_test=0.5` even the sign of the offset's effect flipped.
  Both now standardize features on the training split only, matching what Exp10
  already did, and converge in tens of iterations.

  **This changes published results.** Exp5's reference figures move (for example
  `risk_offset` at `pi_test=0.01` from `0.0091` to `0.0105`, and at `pi_test=0.5`
  from `0.0519` to `0.0421`), and one previously documented finding is
  withdrawn: that offset correction is *worse than doing nothing* at
  `pi_test=0.5` on real data. That was an artifact of the unconverged fit, not a
  limitation of the method. With the fit converged the offset helps at every
  prevalence, as the theory predicts.
- **`RESULTS_DIGEST.md` quoted a ROC AUC band that excluded its own data.** It
  claimed `[0.94, 0.96]`, but Exp2 produces `0.9382` at `pi_test=0.1`. The
  observed range is `0.938`-`0.957`.
- `RESULTS_DIGEST.md` linked artifacts under `results/`, which is regenerated
  output and never committed, so every path was dead in a fresh clone. It now
  explains how to regenerate a run instead.
- `REPORT.md` documented five of the eleven experiments and invoked them by file
  path rather than the installed `dos-expN` console scripts.
- **The distribution could not be built at all.** `tool.setuptools.packages` was
  set to the literal string `"find:"`, which modern setuptools rejects as an
  invalid package name. `pip install .` failed outright and `pip install -e .`
  produced a distribution containing only `drift_or_shift/__init__.py`, so every
  `dos-expN` console script pointed at a module that was never shipped.
  Replaced with a proper `[tool.setuptools.packages.find]` section.
- **CI never ran on the advertised Python versions.** The matrix used unquoted
  `3.10`, which YAML parses as the float `3.1`, so the job resolved to a
  non-existent interpreter. All versions are now quoted, and a guard step
  asserts the running interpreter matches the matrix entry.
- **`caliblab` was broken on every supported scikit-learn.** It passed
  `base_estimator=` to `CalibratedClassifierCV`, an argument removed in
  scikit-learn 1.2, so the shipped `reproduce/config/default.yaml` (which
  requests `sigmoid` and `isotonic`) raised `TypeError`. The test suite missed
  this because it only exercised `calibration: none`.
- Calibration now uses `FrozenEstimator` where available instead of
  `cv="prefit"`, which is deprecated in scikit-learn 1.6 and removed in 1.8.
- Summary timestamps are timezone-aware (UTC). `reporting._as_utc` normalises
  older naive timestamps so mixed `results/` trees do not raise `TypeError`
  during comparison.

### Added

- The published results are now enforced. `tests/test_documented_results.py`
  re-runs the configurations behind `RESULTS_DIGEST.md` on every CI build and
  checks the published figures as well as the qualitative claims, so the digest
  cannot silently go stale. Exp1, Exp3, Exp4, and Exp5 all reproduce their
  documented numbers exactly.
- Automated releases. Pushing a `v*` tag builds and smoke-tests the
  distribution and publishes a GitHub Release with notes taken from this file.
  The workflow refuses to publish a tag that does not match
  `drift_or_shift.__version__`, or a version with no `CHANGELOG.md` section --
  the two mistakes a manual `twine upload` checklist is most likely to make.
  PyPI publishing uses trusted publishing (no stored token) and stays inert
  until a `pypi` environment is created.
- `tools/print_version.py` and `tools/changelog_section.py`, the release
  workflow's logic kept as testable scripts rather than inline YAML, covered by
  `tests/test_release_tools.py`.
- A security workflow: CodeQL, a `pip-audit` dependency audit, and `zizmor`
  linting of the workflows themselves, on push, pull request, and weekly.
  `tools/check_sarif.py` gates the build on the report's contents, because
  zizmor exits 0 when asked for SARIF output even when it found problems.
- Workflow hardening found by zizmor: tag names are no longer interpolated
  into shell (a code-injection vector), and checkouts no longer persist
  credentials.
- Test coverage raised from 44% to 89% (48 tests to 153).
  - `tests/test_experiments_cli.py` drives all eleven experiments through
    their real command line. The nine offline experiments were previously at
    12-24% coverage: the suite imported them and never ran them, so nothing
    verified that an experiment completes, writes the artifacts it
    advertises, or emits a well-formed summary.
  - `tests/test_validation_contracts.py` covers the input-validation guards
    and degenerate-input paths across `data_synth`, `ess`, `metrics`,
    `calibration`, `drift_variants`, and `drift_monitor`.
  - `tests/test_reporting_edge_cases.py` covers the reporting fallbacks:
    missing results directory, truncated summary JSON, unparseable
    timestamps with mtime fallback, and threshold-config validation.
- A `network` pytest marker for the two dataset-downloading experiments
  (exp9, exp10), deselected by default so the standard run stays offline.
- A coverage floor of 85% (`fail_under`) to guard against regression.
- `LICENSE` (MIT) — the project declared MIT in metadata but shipped no licence text.
- PEP 561 `py.typed` markers for `drift_or_shift`, `drift_shift_pipeline`, and `caliblab`.
- `CHANGELOG.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, issue/PR templates, and Dependabot config.
- `Makefile` with the standard development loop (`make install`, `lint`, `test`, `build`).
- Regression tests: `tests/test_caliblab.py` (calibrated paths, deprecation guard)
  and `tests/test_packaging.py` (console-script targets, `py.typed`, licence).
- CI jobs for distribution build + clean-environment install, and `pre-commit`.
- Explicit Ruff rule selection and exact dev-tool pins so local and CI agree.

### Changed

- CI test matrix broadened to Python 3.10–3.13, plus Windows and macOS spot checks.
- `pyproject.toml` metadata completed: real author, project URLs, classifiers,
  keywords, and a `dynamic` version sourced from `drift_or_shift.__version__`.
- Pytest, coverage, Black, and Mypy configuration consolidated into `pyproject.toml`.

### Removed

- `restore_experiments.py`, a one-off migration script left at the repository root.
- Tracked build artifacts: `src/drift_or_shift.egg-info/` and committed `.pyc` files.
- `.github/workflows/coverage-command.txt`, a stray note in the workflows directory.

## [0.1.0] - 2026-01-13

### Added

- Initial research pipeline: synthetic label shift, offset correction, ESS,
  calibration, drift variants, and experiments 1–11.

[Unreleased]: https://github.com/DiogoRibeiro7/drift-shift-pipeline/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/DiogoRibeiro7/drift-shift-pipeline/releases/tag/v0.1.0
