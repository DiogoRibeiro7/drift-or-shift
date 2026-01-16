# Release Checklist

Use this checklist whenever preparing a release candidate to keep the repo ready for publication.

1. **Bump metadata.**
   - Update `pyproject.toml` `version` to the next candidate (e.g., `0.1.0-rc.1`), and reflect the same string wherever you mention the package name or version in `README.md`.
   - Record the milestone progress in `ROADMAP.md` (e.g., mark Milestone 15 complete) and append a short entry to your changelog or release notes section (create `CHANGELOG.md` if needed).
2. **Sanity checks.**
   - Run `python -m pip install -e .` once to refresh editable metadata.
   - Run the full `pytest` suite; if time is tight, run the smoke runner (`python scripts/smoke_exp1.py --results-dir results/smoke`) to verify the core pipeline works end-to-end with minimal data.
   - Run `ruff check .` and `mypy` (per `pyproject.toml`), plus `python -m drift_or_shift.experiments.exp1_label_shift_synth --results-dir results` if you want a full experiment run before tagging.
3. **Capture artifacts.**
   - Confirm `results/` contains the latest timestamped tables/figures requested by Milestone 14; update `RESULTS_DIGEST.md` and `notes/lessons.md` if the metrics changed.
   - Review `REPORT.md` and `README.md` to make sure the documentation references the latest figure/table filenames.
4. **Tag and push.**
   - Commit the version bump, docs updates, and smoke-test results.
   - Tag the commit (e.g., `git tag -a v0.1.0-rc.1 -m "Release candidate 0.1.0-rc.1"`) and push the tag.
   - After CI passes, merge to `main`/`develop` as needed and push the tag/reference per your workflow.
