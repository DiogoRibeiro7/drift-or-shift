# Contributing to drift-shift-pipeline

Thanks for taking the time to contribute.

## Getting set up

```bash
git clone https://github.com/DiogoRibeiro7/drift-shift-pipeline.git
cd drift-shift-pipeline
make install     # editable install, dev extras, and pre-commit hooks
```

`make install` also registers the git hooks, so formatting and linting run
before each commit.

## The development loop

```bash
make check       # ruff + black + mypy + pytest — exactly what CI runs
make format      # auto-fix lint and formatting
make test-cov    # tests with an HTML coverage report in htmlcov/
make hooks       # run every pre-commit hook over all files
```

If `make check` is green, CI should be green. Tool versions are pinned in both
`pyproject.toml` (`dev` extra) and `.pre-commit-config.yaml` — **bump them
together**, or local hooks and CI will disagree.

## Branching strategy

- `main` — stable release branch; protected
- `develop` — active integration branch
- `feature/<desc>` — new capability
- `bugfix/<desc>` — fixes
- `release/<version>` — release preparation
- `hotfix/<version>` — urgent fixes on top of `main`

## Pull request process

1. Branch from `develop` for features and fixes.
2. Keep PRs small and focused, with a clear title and a linked issue
   (`Fixes #123`).
3. Add or update tests for the behaviour you change. A bug fix without a
   regression test will usually be sent back.
4. Add an entry to `CHANGELOG.md` under **Unreleased**.
5. Ensure `make check` passes.
6. Target `develop`; `main` takes release PRs only.

## Code style

- `ruff` for linting and import order, `black` for formatting — both enforced
  in CI. Do not hand-format around them.
- Public functions carry type annotations; `mypy` runs over `src/` and must stay
  clean. The packages ship `py.typed`, so annotations are part of the API.
- Docstrings on public functions: what it computes, and any non-obvious
  statistical assumption.

## Tests

- Tests live in `tests/` and run with `pytest`.
- Experiments are slow; prefer small `n` and fixed seeds in tests.
- When you fix a bug, write the test that fails before the fix. Two long-lived
  bugs in this repo survived precisely because the suite exercised only the
  happy path.

## Experiments and results

- `results/` is generated output and is **not** committed.
- Curated artifacts belong in `reports/` and are added deliberately.
- When results change, update `RESULTS_DIGEST.md` and, if the finding is
  surprising, `notes/lessons.md`.

## Releases

See [RELEASE.md](RELEASE.md).
