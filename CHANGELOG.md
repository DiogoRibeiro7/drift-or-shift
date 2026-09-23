# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0a1] - 2026-09-23

First published artifact. It reserves the `drift-or-shift` name on PyPI and
exercises the release pipeline end to end; the API is not yet stable.
Everything below was previously listed as unreleased.

### Fixed

- **Drift alerting never fired.** `detect_drift_alerts` looked up thresholds by
  the bare metric name (`feature_max_ks`), but experiments aggregate through
  `aggregate_mean_std`, which writes `feature_max_ks_mean`. The two key sets did
  not overlap at all, so every real summary produced zero alerts:
  `scripts/drift_alerts.py` always printed "No drift alerts detected",
  `--fail-on-alerts` could never trip, and `scripts/watch_results.py` polled for
  something that could not happen. The existing tests passed because each one
  hand-built a summary using the bare key that no experiment writes. The lookup
  now accepts `<metric>_mean`, and a regression test builds its summary through
  the real aggregation function instead of asserting against a hand-written
  shape.
- **`reproduce/scripts/make_tables.py` could not run.** It passed `eps=` to
  `log_loss`, which scikit-learn deprecated in 1.3 and removed in 1.5. The
  module was at 0% coverage, so nothing noticed.
- `make_tables` no longer triggers the pandas 2.2 deprecation about
  `DataFrameGroupBy.apply` operating on grouping columns.

- The pre-commit hooks and the `dev` extra had drifted apart. Dependabot
  watches `pyproject.toml` but not `.pre-commit-config.yaml`, so a dependency
  PR moved `mypy` from 1.20.2 to 2.3.1, `ruff` to 0.16.8 and the PyYAML stubs
  forward in one file and left the hooks behind. Nothing went red, because
  each file is internally consistent -- the hooks and CI had simply stopped
  checking the same thing. The hooks are resynced, Dependabot now watches the
  `pre-commit` ecosystem, and `tests/test_tooling_pins.py` fails if the two
  ever disagree again, so the rule `CONTRIBUTING.md` states is now enforced
  rather than merely written down.

- `calibration._sigmoid` overflowed for large negative logits. The saturated
  result was correct, but NumPy warned on the way there, and that noise can
  mask a real warning. Evaluated piecewise now; the values are unchanged to
  within `2.4e-38` and are the correctly rounded ones where they differ.

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
- **The public API rejected the array types it is actually called with.**
  Every helper was annotated `Sequence[float]` / `Sequence[int]` while being
  called throughout the experiments with NumPy arrays and pandas Series. Since
  the package ships `py.typed`, that narrowness reached downstream users too:
  passing a NumPy array to a NumPy library produced a type error. Widened to
  `numpy.typing.ArrayLike`, already the convention in `calibration` and
  `drift_variants`. This removed 165 type errors, all of them `arg-type`.
- **`plot_auc_pr_vs_prevalence` warned on every draw.** It built its panels
  with `sharex=True` and then switched them to a log scale, which makes
  matplotlib attempt a non-positive xlim. Both panels plot the same
  prevalences, so their limits are identical without sharing.
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

- `RESULTS_DIGEST.md` documents all eleven experiments rather than five, with
  measured figures and an observation for each, and
  `tests/test_documented_results.py` enforces the four offline ones that were
  previously undocumented (Exp6, Exp7, Exp8, Exp11) alongside Exp1-Exp5.

- `caliblab.benchmark` takes an optional `calibration_holdout`, fitting the
  calibration map on a held-out slice of each fold's training data instead of
  on the data the base estimator already saw. The default is unchanged and
  byte-for-byte identical to before, so no published number moves.

  Earlier notes in this project described the existing behaviour as a leakage
  problem that would bias calibration optimistically. Measured on the shipped
  breast-cancer benchmark -- holding the base model and the calibration-set
  size fixed so only the overlap differed -- the effect on test Brier score is
  within noise (`-0.0014` to `+0.0006`, standard errors around `0.001`). That
  is why it is an opt-in rather than a new default.
  `tests/test_caliblab_leakage.py` re-runs the comparison so the claim stays
  checked.
- The published results are now enforced. `tests/test_documented_results.py`
  re-runs the configurations behind `RESULTS_DIGEST.md` on every CI build and
  checks the published figures as well as the qualitative claims, so the digest
  cannot silently go stale. Exp1-Exp4 reproduce their documented numbers
  exactly; Exp5's were re-derived by the convergence fix above.
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
- `drift_or_shift.experiments` is now type-checked. The package was excluded
  from mypy entirely, leaving eleven modules unchecked; with the annotations
  corrected the exclusion is gone and mypy covers 28 files instead of 14.
- `tests/test_array_like_api.py` pins the runtime half of the array-like
  contract, plus a guard that the plot helpers draw without warnings.
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

- Coverage measures `scripts/` and `reproduce/` as well as the two packages.
  It previously reported 89% while leaving 439 statements of documented entry
  points entirely unmeasured -- four of them at 0%. The headline figure is
  still about 89%, but now over 1858 statements rather than 1419.

- **The distribution is renamed from `drift-shift-pipeline` to `drift-or-shift`**,
  so it matches the import package `drift_or_shift`. The project had five names
  for what is essentially two things; the distribution was the odd one out. It
  has never been published to PyPI, so nothing downstream breaks. Repository
  URLs, the PyPI trusted-publishing target, badges, and the citation block all
  follow. The import package, the console scripts, and `caliblab` are unchanged.
- `drift_shift_pipeline` remains importable but now emits a `DeprecationWarning`.
  It existed so the old distribution name worked as an import; with the
  distribution renamed it is the last remnant of that name, kept only so
  existing local scripts keep working.

- CI test matrix broadened to Python 3.10–3.13, plus Windows and macOS spot checks.
- `pyproject.toml` metadata completed: real author, project URLs, classifiers,
  keywords, and a `dynamic` version sourced from `drift_or_shift.__version__`.
- Pytest, coverage, Black, and Mypy configuration consolidated into `pyproject.toml`.

### Removed

- `paper/ssrn-6052514.pdf`. Redistributing the PDF from a public MIT-licensed
  repository is a licensing question the repository cannot answer for itself,
  and it is not needed to run or understand the code. `README.md`,
  `ROADMAP.md` and `RESULTS_DIGEST.md` now link to the paper on SSRN instead.

- The 30 generated artifacts committed under `reports/` -- fifteen dashboards
  and fifteen figures, all produced over two days in January 2026. They had
  become actively misleading rather than merely stale: every one published
  Exp5's pre-convergence-fix figures, including the `risk_offset` of `0.0519`
  at `pi_test=0.5` that this project has since established was an artifact of
  an optimizer that never converged. They also linked artifacts under
  `results/`, which is not committed, using Windows path separators, so the
  paths resolved on no platform. `reports/README.md` records what the
  directory is for, and the scripts' default output filenames are now
  gitignored so an ordinary run cannot re-add them. The files remain in git
  history.

- `restore_experiments.py`, a one-off migration script left at the repository root.
- Tracked build artifacts: `src/drift_or_shift.egg-info/` and committed `.pyc` files.
- `.github/workflows/coverage-command.txt`, a stray note in the workflows directory.

## [0.1.0] - 2026-01-13

### Added

- Initial research pipeline: synthetic label shift, offset correction, ESS,
  calibration, drift variants, and experiments 1–11.

[Unreleased]: https://github.com/DiogoRibeiro7/drift-or-shift/compare/v0.1.0a1...HEAD
[0.1.0a1]: https://github.com/DiogoRibeiro7/drift-or-shift/releases/tag/v0.1.0a1
[0.1.0]: https://github.com/DiogoRibeiro7/drift-or-shift/releases/tag/v0.1.0
