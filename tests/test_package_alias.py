"""Tests for alias package import compatibility."""

from __future__ import annotations

import drift_shift_pipeline
import drift_or_shift


def test_import_alias_modules_are_equivalent() -> None:
    assert hasattr(drift_shift_pipeline, "fit_logistic_regression")
    assert hasattr(drift_or_shift, "fit_logistic_regression")


def test_import_alias_sane_version() -> None:
    assert hasattr(drift_shift_pipeline, "__version__")
