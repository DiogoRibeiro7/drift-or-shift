# Release Checklist

Releases are automated. Pushing a `v*` tag runs `.github/workflows/release.yml`,
which verifies the tag, builds and smoke-tests the distribution, and publishes a
GitHub Release with the artifacts attached.

## 1. Prepare on `main`

- Bump `__version__` in `src/drift_or_shift/__init__.py`. This is the single
  source of truth: `pyproject.toml` reads it via `[tool.setuptools.dynamic]`,
  and `tests/test_packaging.py` asserts it matches the installed metadata.
- Move the `## [Unreleased]` entries in `CHANGELOG.md` under a new
  `## [X.Y.Z] - YYYY-MM-DD` heading, and update the link references at the
  bottom of the file.
- Reflect the version in the `README.md` citation block.
- Record milestone progress in `ROADMAP.md`, and refresh `RESULTS_DIGEST.md`
  and `notes/lessons.md` if the metrics changed.

Both of the first two steps are enforced: the release workflow refuses to
publish a tag that does not match the packaged version, or a version with no
matching `CHANGELOG.md` section.

## 2. Check locally

```bash
make check     # ruff, black, mypy, pytest
make build     # sdist + wheel, validated with twine
make security  # dependency audit + workflow lint
```

For a fuller check before tagging:

```bash
python scripts/smoke_exp1.py --results-dir results/smoke
make test-network   # the dataset-backed experiments, skipped by default
```

## 3. Tag

```bash
git tag -a v0.1.0 -m "Release 0.1.0"
git push origin v0.1.0
```

The workflow then:

1. **verify** — tag matches `drift_or_shift.__version__`, and `CHANGELOG.md`
   documents that version.
2. **build** — builds the sdist and wheel, runs `twine check`, installs the
   wheel into a clean virtualenv, asserts the installed version, and runs a
   console script.
3. **github-release** — creates the GitHub Release, using that version's
   `CHANGELOG.md` section as the release notes, with the artifacts attached.
4. **pypi** — publishes to PyPI. See below.

If the tag was wrong, delete it (`git push --delete origin vX.Y.Z`), fix, and
re-tag. `workflow_dispatch` can re-run the workflow against an existing tag.

## PyPI publishing

The `pypi` job is **inert until a `pypi` environment exists** on the repository,
so it cannot fire by accident. To enable it:

1. Configure [trusted publishing](https://docs.pypi.org/trusted-publishers/) on
   PyPI for this repository, workflow `release.yml`, environment `pypi`.
2. Create a `pypi` environment under **Settings → Environments**, ideally with a
   required reviewer.

No API token is stored anywhere; publishing uses OIDC.
