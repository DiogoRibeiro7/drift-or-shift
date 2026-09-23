# ROADMAP.md

## Project goal
Implement, in Python, the methods and experiments from [SSRN 6052514](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6052514), focusing on:

- Label-shift adaptation via a **single scalar logit/threshold offset** (no retraining).
- Demonstrations of:
  - Offset correction ≈ oracle under label shift.
  - **AUC invariance** to prevalence; **PR-AUC dependence** on prevalence.
  - **Effective sample size (ESS)** degradation under class weighting.
  - Offset correction failure under **concept drift**, where retraining helps.
- A small, reusable library with clean APIs + reproducible experiments.

Non-goals:
- Building a large ML framework.
- Heavy dependency stack (keep minimal; prefer numpy + sklearn + matplotlib).

---

## Status (September 2026)

Milestones 0-15 are complete. The project outgrew this plan in two directions:
eleven experiments rather than five, and a much larger engineering surface than
Milestone 13's "add a CI workflow" anticipated. Milestones 16-18 below record
that work; the sections above are kept as the original plan.

Current state:

- 11 experiments, all runnable via `dos-expN` console scripts.
- 292 tests, ~89% coverage measured across `src/`, `scripts/` and `reproduce/`.
- CI on Python 3.10-3.13 plus Windows and macOS, with lint, format, types,
  a build-and-install check, security scanning, and pre-commit.
- Tag-triggered releases with PyPI trusted publishing.
- `RESULTS_DIGEST.md` figures are re-run and asserted on every CI build.

---

## Success criteria (high level)
1. `pip install -e .` works and imports succeed.
2. Running each experiment script produces:
   - Results CSV (or JSON) in `results/`
   - A plot (PNG) in `results/figures/`
   - A terminal summary with mean ± std over seeds
3. Unit tests cover all math primitives and key invariances.
4. Reproducibility: fixed seeds, deterministic dataset construction per seed.
5. Documentation: minimal README describing how to reproduce figures/tables.

---

## Repository structure (actual)

Experiments live inside the package rather than at the top level, so they ship
with the wheel and back the `dos-expN` console scripts.

```
.
├─ pyproject.toml
├─ Makefile                 # make check / build / security
├─ src/
│  ├─ drift_or_shift/       # the library
│  │  ├─ shift.py  ess.py  metrics.py  models.py
│  │  ├─ data_synth.py  calibration.py  drift_monitor.py
│  │  ├─ drift_variants.py  plotting.py  reporting.py  io_utils.py
│  │  └─ experiments/       # exp1..exp11 + _common.py
│  ├─ drift_shift_pipeline/ # deprecated import alias
│  └─ caliblab/             # calibration benchmarking helpers
├─ scripts/                 # aggregate_results, drift_alerts, watch_results, ...
├─ reproduce/               # separate calibration-benchmark replication
├─ tools/                   # release helpers used by the workflow
├─ tests/
├─ docs/  notes/  reports/
└─ results/                 # generated, not committed
```

## Dependencies (minimal)
Required:
- Python >= 3.10
- numpy
- scikit-learn
- pandas (for results tables; optional but recommended)
- matplotlib
- pytest

Optional:
- scipy (only if needed for stable numerics)

Dev:
- ruff
- mypy

---

## Milestone 0 — Project scaffolding (complete)
### Tasks
- Create `pyproject.toml` with:
  - project metadata
  - dependencies
  - optional dev dependencies
  - `src/` layout config
- Create package folder `src/drift_or_shift/` with `__init__.py`
- Add `README.md` skeleton with “Quickstart” and “Reproducibility”
- Add `results/` folders

### Acceptance criteria
- `python -c "import drift_or_shift"` passes
- `pytest` runs (even if 0 tests initially)
- `ruff check .` and `mypy` can be run

---

## Milestone 1 — Core math primitives (label shift + thresholds) (complete)
This milestone builds the core library functionality.

### 1.1 `shift.py`
Implement:
- `odds(pi: float) -> float`
- `logit_offset(pi_train: float, pi_test: float) -> float`
  - offset = log( pi_test*(1-pi_train) / (pi_train*(1-pi_test)) )
- `apply_logit_offset(logits: np.ndarray, offset: float) -> np.ndarray`
- `threshold_from_costs(pi: float, c10: float, c01: float) -> float`
  - return threshold in **log-likelihood ratio** or **log-odds space** (be explicit).
- `decision_from_logits(logits: np.ndarray, threshold: float) -> np.ndarray`
- `validate_prevalence(pi: float) -> None` (0 < pi < 1)

Notes:
- Decide and document a single convention:
  - If logits are posterior log-odds, then the natural decision threshold is `log(c10/c01)` after applying the proper prior offset.
  - If logits represent LLR + log-odds(train), then the offset pushes to test prior.
- Add docstrings and type hints.

### 1.2 Tests for `shift.py`
- Edge cases:
  - pi near 0/1 should raise a `ValueError` (or be clipped with warning; pick one).
- Invariance checks:
  - For any fixed scores, adding a constant offset does not change ranking.
- Consistency:
  - `logit_offset(pi, pi) == 0`
  - applying offset + inverse offset recovers original.

### Acceptance criteria
- `pytest -q` passes for all shift tests.
- `shift.py` has stable numerical behavior and clear conventions.

---

## Milestone 2 — ESS formula module (complete)
### 2.1 `ess.py`
Implement:
- `effective_sample_size(n: int, pi: float, alpha: float) -> float`
  - Kish-style formula:
    - Neff = N * [pi*alpha + (1-pi)]^2 / [pi*alpha^2 + (1-pi)]
- `ess_fraction(...) -> float` returning `Neff / N`

### 2.2 Tests for `ess.py`
- `alpha=1` => `Neff == N`
- For fixed `pi` and `N`, `Neff` decreases as `alpha` increases.
- `Neff` increases with N linearly.

### Acceptance criteria
- `pytest -q tests/test_ess.py` passes
- Functions are documented, typed, and validated.

---

## Milestone 3 — Synthetic data generator (label shift and drift) (complete)
### 3.1 `data_synth.py`
Implement synthetic generator matching the paper setup:

- `make_gaussian_binary(
    n: int,
    d: int,
    pi: float,
    mu0: np.ndarray,
    mu1: np.ndarray,
    sigma: np.ndarray | float,
    rng: np.random.Generator
  ) -> tuple[np.ndarray, np.ndarray]`

- Provide helper constructors:
  - `default_mu0(d)`: all zeros
  - `default_mu1(d)`: first 5 dims = 1, rest 0
  - `concept_drift_mu1(mu1, shift_dim=5, delta=0.5)` (e6 is index 5)

- `resample_to_prevalence(X, y, pi_target, rng)`:
  - stratified resampling from a fixed pool.

### 3.2 Tests
- Shape checks.
- Prevalence check: empirical pi close to requested (within tolerance).
- Determinism: same seed yields identical output.

### Acceptance criteria
- Deterministic generation for a given seed.
- Prevalence control works reliably.

---

## Milestone 4 — Models and metrics (complete)
### 4.1 `models.py`
Implement minimal training/prediction wrappers:
- `fit_logistic_regression(X_train, y_train, *, class_weight=None, C=1.0, max_iter=1000, rng=None)`
- `predict_logits(model, X) -> np.ndarray`

### 4.2 `metrics.py`
Implement:
- `risk_cost_sensitive(y_true, y_pred, c10, c01) -> float`
- `roc_auc(y_true, scores) -> float`
- `pr_auc(y_true, scores) -> float`
- `oracle_threshold_min_risk(y_true, scores, c10, c01) -> tuple[float, float]`
  - returns (best_threshold, min_risk)
  - threshold in score space (operate on logits)

### 4.3 Tests
- Risk sanity: perfect predictions => risk 0
- AUC sanity: random scores ~0.5
- Oracle threshold reduces risk versus a fixed naive threshold in a controlled toy case.

### Acceptance criteria
- Model training produces logits with expected shape.
- Metrics match sklearn for AUC/PR-AUC.

---

## Milestone 5 — Experiment harness (reproducible runner + IO) (complete)
### 5.1 `io_utils.py`
- `ensure_dir(path) -> None`
- `save_table(df, path)`
- `save_json(obj, path)`
- `timestamped_run_dir(base="results", name="exp1") -> Path`

### 5.2 `plotting.py`
- functions to generate:
  - risk vs prevalence plot
  - AUC/PR-AUC vs prevalence
  - ESS vs alpha

### 5.3 Common experiment config/utilities
Create `experiments/_common.py`:
- seeds list
- prevalence grid: `[0.5, 0.2, 0.1, 0.05, 0.01]`
- train prevalence `pi_train = 0.2`
- costs (default symmetric, allow override)
- result aggregation: mean/std tables.

### Acceptance criteria
- Each experiment saves:
  - `results/tables/<exp_name>.csv`
  - `results/figures/<exp_name>.png`
  - `results/<exp_name>_summary.json`

---

## Milestone 6 — Experiment 1 (Synthetic label shift + offset correction) (complete)
Script: `experiments/exp1_label_shift_synth.py`

### Method
For each seed:
1. Generate training set with `pi_train=0.2` from class-conditionals.
2. Train logistic regression (unweighted).
3. For each `pi_test`:
   - generate test set with that prevalence from *same* class-conditionals (label shift).
   - compute:
     - **No correction** risk.
     - **Offset correction** risk:
       - offset = logit_offset(pi_train, pi_test)
       - scores_shifted = scores + offset
     - **Oracle** min risk by threshold sweep on test labels.

Aggregate over seeds.

### Outputs
- Table with columns:
  - `pi_test`, `risk_none_mean`, `risk_offset_mean`, `risk_oracle_mean`, plus std
- Plot: risk vs pi_test with three curves.

### Acceptance criteria
- Offset correction improves over no correction for extreme pi_test (0.01 / 0.5).
- Offset correction is close to oracle qualitatively.

---

## Milestone 7 — Experiment 2 (AUC invariance + PR-AUC dependence) (complete)
Script: `experiments/exp2_auc_pr_invariance.py`

### Method
- For each `pi_test`, generate test set with that prevalence.
- Compute AUC and PR-AUC using raw logits (and optionally shifted logits).

### Outputs
- Table with `pi_test`, `auc_mean/std`, `prauc_mean/std`
- Plot AUC vs pi_test
- Plot PR-AUC vs pi_test

### Acceptance criteria
- AUC remains approximately constant across pi_test (small variation).
- PR-AUC changes noticeably with pi_test.

---

## Milestone 8 — Experiment 3 (ESS vs class weighting) (complete)
Script: `experiments/exp3_ess_vs_weight.py`

### Method
- Compute ESS for alpha in `{1, 5, 10, 20, 50}`.
- Output table + plot ESS fraction vs alpha.

### Acceptance criteria
- ESS fraction is 1 at alpha=1 and decreases as alpha grows.

---

## Milestone 9 — Experiment 4 (Concept drift) (complete)
Script: `experiments/exp4_concept_drift.py`

### Method
- Train on base distribution (mu1 baseline).
- Test on drifted distribution: shift mu1 along dimension e6 (index 5) by +0.5.
- Compare:
  - no correction
  - offset correction (using prevalence only)
  - retrained model on drifted distribution

### Outputs
- Table of risk for the three approaches (mean/std)
- Plot: bar chart or point plot.

### Acceptance criteria
- Offset correction does not reliably improve under concept drift.
- Retraining improves over no correction.

---

## Milestone 10 — Experiment 5 (Real dataset replication: Breast Cancer) (complete)
Script: `experiments/exp5_realdata_breast_cancer.py`

### Method
- Load `sklearn.datasets.load_breast_cancer`.
- Split train/test (stratified).
- Force label shift in test by subsampling to target `pi_test` values.
- Train on a `pi_train=0.2` subsample (or record actual pi_train and use it).
- Apply:
  - no correction
  - offset correction
  - oracle threshold on shifted test

### Outputs
- Tables/plots analogous to Experiment 1–2 for real data.

### Acceptance criteria
- Offset correction improves over no correction when prevalence shifts materially.

---

## Milestone 11 — CLI entrypoints (optional but useful) (complete)
Add console scripts:
- `dos-exp1`, `dos-exp2`, ...

Arguments:
- `--seeds`
- `--n-train`, `--n-test`
- `--d`
- `--pi-train`
- `--pi-tests 0.5 0.2 0.1 0.05 0.01`
- `--c10`, `--c01`
- `--outdir results`

Acceptance criteria:
- Running `dos-exp1 --help` works.
- Running `dos-exp1` produces outputs in a new run directory.

---

## Milestone 12 — Documentation polish (complete)
### Tasks
- Update `README.md`:
  - Label shift vs concept drift (short explanation)
  - How offset correction works (formula + explanation)
  - Reproduction commands (one per experiment)
  - Output locations
- Add “Reproducibility” section: seeds, determinism, versions.

### Acceptance criteria
- New user can reproduce all experiments with copy-paste commands.

---

## Milestone 13 - CI + quality gates (recommended) (complete)
### Tasks
- Add GitHub Actions workflow:
  - `ruff check`
  - `mypy`
  - `pytest`

### Acceptance criteria
- CI passes on main branch.

---

## Milestone 14 - Result synthesis & write-up (complete)
### Tasks
 - [x] Document how each experiment validates the paper's claims and call out any deviations (`RESULTS_DIGEST.md`).
 - [x] Generate a single "results digest" with metrics, artifact paths, and metadata (`RESULTS_DIGEST.md`).
 - [x] Provide a narrative rerun guide plus commentary on the figures/tables (`REPORT.md`, `README.md` updates).
 - [x] Capture surprising behaviors and hypotheses for follow-up (`notes/lessons.md`).

### Acceptance criteria
- A reader can explain what each artifact shows without rerunning experiments.
- Narrative doc references the exact figure/table filenames produced under `results/`.
- Lessons note lists at least three insights or hypotheses drawn from the experiments.

---

## Milestone 15 - Maintenance & ecosystem readiness (complete)
### Tasks
 - [x] Publish a release checklist covering version bumps, tests, reports, tagging (`RELEASE.md`, README maintenance section).
 - [x] Keep packaging info and docs locked (no unused dependencies or extra keywords; README references release docs).
 - [x] Add a smoke script plus test verifying it (`scripts/smoke_exp1.py`, `tests/test_smoke_exp1_script.py`).
 - [x] Document future work & learning goals (`FUTURE.md`).

### Acceptance criteria
- Versioned release notes exist and a maintainers checklist (e.g., `make release`) captures verification steps.
- A minimal sanity test script runs in under a minute and exits cleanly; document how to run it.
- Future work appendix lists at least three concrete extensions and their expected learning goals.

---

## Milestone 16 - Engineering hardening (complete)

Milestone 13 asked for a CI workflow. What the project actually needed was
considerably more, most of it prompted by defects found along the way.

### Tasks
 - [x] Fix the distribution: `tool.setuptools.packages` was the literal string
       `"find:"`, so `pip install .` failed and the wheel shipped one file.
 - [x] Fix the CI matrix: unquoted `3.10` is the YAML float `3.1`, so no job had
       ever run on a real 3.10 interpreter.
 - [x] Widen CI to 3.10-3.13 plus Windows and macOS, with a build-and-install
       job that installs the wheel into a clean virtualenv and runs a console script.
 - [x] Add security scanning: CodeQL, `pip-audit`, and `zizmor` over the workflows.
 - [x] Add tag-triggered releases with PyPI trusted publishing, gated on a
       `pypi` environment so they cannot fire by accident.
 - [x] Pin dev tooling exactly and keep `pyproject.toml` and
       `.pre-commit-config.yaml` in lockstep, enforced by `tests/test_tooling_pins.py`.
 - [x] Raise coverage from 44% to ~89%, measured across `scripts/` and
       `reproduce/` as well as the packages.
 - [x] Rename the distribution to `drift-or-shift` so it matches the import name.
 - [x] Add LICENSE, CHANGELOG, SECURITY, CODE_OF_CONDUCT, issue and PR
       templates, Dependabot, and a Makefile.

### Acceptance criteria
- `make check` reproduces what CI runs, and a green local run means a green pipeline.
- A clean `pip install` of the built wheel imports every package and runs `dos-exp1`.

---

## Milestone 17 - Verified results (complete)

The digest published specific risks and AUCs and nothing checked them.

### Tasks
 - [x] Re-run every documented configuration in CI and assert both the published
       figures and the qualitative claims (`tests/test_documented_results.py`).
 - [x] Document all eleven experiments in `RESULTS_DIGEST.md`, not five.
 - [x] Correct two claims that contradicted their own numbers: the ROC AUC band,
       and Exp5's "offset correction closes the gap".
 - [x] Fix Exp5 and Exp9, which fitted on raw unscaled features so LBFGS never
       converged and results differed by platform. Their figures were re-derived.
 - [x] Fix drift alerting, which never fired: thresholds were keyed on
       `feature_max_ks` while experiments write `feature_max_ks_mean`.

### Acceptance criteria
- The digest cannot go stale silently; CI fails if a published figure moves.
- Qualitative claims are asserted strictly, exact figures with a tolerance that
  absorbs platform arithmetic.

---

## Milestone 18 - Open work

Nothing here is blocking; these are the known remaining items.

### Tasks
 - [ ] Publish `0.1.0a1` to PyPI to reserve the name and give the release
       workflow its first real run. Needs a pending publisher on PyPI and a
       `pypi` environment on the repository.
 - [ ] Enable branch protection on `main` requiring the CI checks. A direct push
       has already broken `main` once.
 - [ ] Decide the relationship with `drift_control` (see below).
 - [ ] Consider whether `caliblab` belongs in this repository or its own.
 - [ ] `calibration_holdout` in `caliblab` is opt-in; revisit the default if a
       harder dataset shows the leakage matters.

---

## Relationship with `drift_control`

`DiogoRibeiro7/drift_control` is a production drift-monitoring library. The two
projects are complementary, with one overlap and one gap:

- **Duplicated.** `drift_or_shift.drift_monitor._ks_statistic` is the same
  algorithm as `drift_control.distances.ks.ks_statistic`, written twice.
  `drift_control` also has PSI, JS, KL, MMD, Wasserstein, energy, chi2 and TVD,
  where this repository has an ad-hoc subset. Either depend on it for drift
  statistics, or keep the copy deliberately rather than by accident.
- **Missing there.** `drift_control` has no label-shift handling at all: no
  prior correction, no logit offset, no prevalence. `shift.py` and `ess.py` are
  roughly 150 lines that would fill a real gap in a library whose job is
  controlling drift.

---

## Implementation notes (conventions)
- Keep score semantics explicit:
  - logits are in log-odds space; offset is additive.
- Keep label encoding consistent:
  - class 1 = positive, class 0 = negative.
- Use `numpy.random.default_rng(seed)` everywhere and pass `rng` explicitly.
- Save result artifacts with metadata:
  - seeds, N, d, costs, pi settings, timestamp.

---

## Suggested development order (agent-friendly)
1. Milestone 0 (scaffold)
2. Milestone 1 (shift primitives + tests)
3. Milestone 2 (ESS + tests)
4. Milestone 3 (synthetic data + tests)
5. Milestone 4 (models + metrics + tests)
6. Milestone 5 (IO + plotting + harness)
7. Milestones 6–10 (experiments)
8. Milestones 12–13 (docs + CI)

---

## Definition of done
- All tests pass.
- Running the experiment scripts generates tables and figures without manual edits.
- Results qualitatively match the paper’s key conclusions.
- Codebase is small, readable, typed, and documented.
