# Contributing to drift-shift-pipeline

## Branching strategy

- `main`: stable release branch; protected
- `develop`: active integration branch
- feature branches: `feature/<desc>`
- bugfix branches: `bugfix/<desc>`
- release branches: `release/<version>`
- hotfix branches: `hotfix/<version>`

## Pull request process

1. Create branch from `develop` for features/fixes.
2. Keep PRs small and focused with a clear title and linked issue.
3. Ensure CI passes: `ruff check .`, `mypy src`, `pytest -q`, `pre-commit run --all-files`.
4. Target `develop` for regular work, `main` for release PRs only.

## Code style

- Use `ruff` and `black` for formatting.
- Add new tests for relevant behavior in `tests/`.
- Include doc updates for API or user-facing changes.

## Release workflow

1. Merge tested features into `develop`.
2. Create `release/<x.y.z>` branch from `develop`; update `pyproject.toml` version.
3. Run full CI and smoke experiments.
4. Merge release branch into `main` and `develop`.
5. Tag `main` with `vX.Y.Z` and push.
6. Publish package if required using GitHub Actions or `python -m twine upload`.

## Issue and PR metadata

- Add labels for severity/priority and story points.
- Link PR to issues with `Fixes #`.
- Add changes to `CHANGELOG.md`.
