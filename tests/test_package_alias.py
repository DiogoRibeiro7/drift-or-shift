"""Tests for the deprecated `drift_shift_pipeline` alias package."""

from __future__ import annotations

import importlib
import warnings

import drift_or_shift
import drift_shift_pipeline


def test_import_alias_modules_are_equivalent() -> None:
    assert hasattr(drift_shift_pipeline, "fit_logistic_regression")
    assert hasattr(drift_or_shift, "fit_logistic_regression")


def test_import_alias_sane_version() -> None:
    assert hasattr(drift_shift_pipeline, "__version__")


def test_alias_version_tracks_the_renamed_distribution() -> None:
    """The distribution is `drift-or-shift` now, not `drift-shift-pipeline`."""
    assert drift_shift_pipeline.__version__ == drift_or_shift.__version__


def test_importing_the_alias_warns() -> None:
    """It still works, but it should tell you to stop using it."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.reload(drift_shift_pipeline)

    messages = [
        str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert any("deprecated alias" in m for m in messages), messages
    assert any("drift_or_shift" in m for m in messages), messages


def test_experiments_path_compatibility() -> None:
    import importlib

    topo = importlib.import_module("drift_or_shift.experiments.exp1_label_shift_synth")
    legacy = importlib.import_module(
        "drift_or_shift.experiments.experiments.exp1_label_shift_synth"
    )

    assert topo.__file__ == legacy.__file__
    assert topo.__name__.endswith("exp1_label_shift_synth")
    assert legacy.__name__.endswith("exp1_label_shift_synth")
