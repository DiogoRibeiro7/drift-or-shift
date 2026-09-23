# Release Checklist

Use this checklist whenever preparing a release candidate to keep the repo ready for publication.

1. **Bump metadata.**
   - Bump `__version__` in `src/drift_or_shift/__init__.py`. This is the single
     source of truth; `pyproject.toml` reads it via `[tool.setuptools.dynamic]`,
     and `tests/test_packaging.py` asserts it matches the installed metadata.
   - Reflect the same version in the `README.md` citation block.
   - Record the milestone progress in `ROADMAP.md` (e.g., mark Milestone 15 complete) and move the **Unreleased** entries in `CHANGELOG.md` under the new version heading.
2. **Sanity checks.**
   - Run `python -m pip install -e .` once to refresh editable metadata.
   - Run the full `pytest` suite; if time is tight, run the smoke runner (`python scripts/smoke_exp1.py --results-dir results/smoke`) to verify the core pipeline works end-to-end with minimal data.
   - Confirm the wheel installs cleanly: `python -m pip install dist/*.whl` in a fresh virtualenv, then `dos-exp1 --help`.
   - Run `make check` (ruff, black, mypy, pytest) and `make build` (sdist + wheel, validated with `twine check`).
   - Optionally run a full experiment before tagging: `dos-exp1 --results-dir results`.
3. **Capture artifacts.**
   - Confirm `results/` contains the latest timestamped tables/figures requested by Milestone 14; update `RESULTS_DIGEST.md` and `notes/lessons.md` if the metrics changed.
   - Review `REPORT.md` and `README.md` to make sure the documentation references the latest figure/table filenames.
4. **Tag and push.**
   - Commit the version bump, docs updates, and smoke-test results.
   - Tag the commit (e.g., `git tag -a v0.1.0-rc.1 -m "Release candidate 0.1.0-rc.1"`) and push the tag.
   - After CI passes, push the tag and publish the release from `main`.
