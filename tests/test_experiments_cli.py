"""End-to-end tests driving each experiment through its command line.

Before this file the eleven experiment modules sat at 12-24% coverage: the
suite imported them and never ran them. Nothing verified that an experiment
completes, writes the artifacts it advertises, or emits a well-formed summary
-- which is the entire product of this repository.

Each test runs the real `main()` with a deliberately tiny configuration, so
the argument parser, the experiment body, and the artifact writer are all
exercised on the path a user actually takes.
"""

from __future__ import annotations

import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Minimal arguments per experiment. Kept small enough to stay fast while still
# covering more than one prevalence and more than one seed where the
# experiment supports them.
OFFLINE_EXPERIMENTS: dict[str, list[str]] = {
    "exp1_label_shift_synth": [
        "--n-train",
        "200",
        "--n-test",
        "200",
        "--d",
        "3",
        "--pi-tests",
        "0.2",
        "0.5",
        "--seeds",
        "0",
        "1",
    ],
    "exp2_auc_pr_invariance": [
        "--n-train",
        "200",
        "--n-test",
        "200",
        "--d",
        "3",
        "--pi-tests",
        "0.2",
        "0.5",
        "--seeds",
        "0",
    ],
    "exp3_ess_vs_weight": [
        "--n",
        "200",
        "--pi",
        "0.2",
        "--alphas",
        "1.0",
        "2.0",
    ],
    "exp4_concept_drift": [
        "--n-train",
        "200",
        "--n-test",
        "200",
        "--d",
        "3",
        "--seeds",
        "0",
    ],
    "exp5_realdata_breast_cancer": [
        "--pi-tests",
        "0.2",
        "0.5",
        "--seeds",
        "0",
    ],
    "exp6_calibration_label_shift": [
        "--n-train",
        "200",
        "--n-test",
        "200",
        "--d",
        "3",
        "--pi-tests",
        "0.2",
        "0.5",
        "--seeds",
        "0",
    ],
    "exp7_drift_types": [
        "--n-train",
        "200",
        "--n-test",
        "200",
        "--d",
        "3",
        "--seeds",
        "0",
    ],
    "exp8_multimodal_label_shift": [
        "--n-train",
        "200",
        "--n-test",
        "200",
        "--d",
        "3",
        "--pi-tests",
        "0.2",
        "0.5",
        "--seeds",
        "0",
    ],
    "exp11_high_variance_medical": [
        "--n-train",
        "200",
        "--n-test",
        "200",
        "--d",
        "3",
        "--pi-tests",
        "0.2",
        "0.5",
        "--seeds",
        "0",
    ],
}

# These fetch datasets over the network, so they are deselected by default.
NETWORK_EXPERIMENTS: dict[str, list[str]] = {
    "exp9_covtype_label_shift": [
        "--n-train",
        "500",
        "--n-test",
        "500",
        "--pi-tests",
        "0.2",
        "--seeds",
        "0",
    ],
    "exp10_credit_card_fraud": [
        "--n-train",
        "500",
        "--n-test",
        "500",
        "--pi-tests",
        "0.05",
        "--seeds",
        "0",
    ],
}


def _run_experiment(monkeypatch, tmp_path: Path, name: str, args: list[str]) -> dict:
    """Run one experiment's CLI and return its parsed summary payload."""
    module = importlib.import_module(f"drift_or_shift.experiments.{name}")
    results_dir = tmp_path / "results"
    monkeypatch.setattr(sys, "argv", [name, *args, "--results-dir", str(results_dir)])

    module.main()

    summaries = list(results_dir.rglob("*_summary.json"))
    assert len(summaries) == 1, f"expected exactly one summary, got {summaries}"
    return json.loads(summaries[0].read_text(encoding="utf-8"))


def _assert_summary_is_well_formed(payload: dict, name: str) -> None:
    assert payload["exp_name"] == name

    table = Path(payload["table"])
    figure = Path(payload["figure"])
    assert table.is_file() and table.stat().st_size > 0, f"empty table: {table}"
    assert figure.is_file() and figure.stat().st_size > 0, f"empty figure: {figure}"

    # Timestamps must be timezone-aware UTC, or mixed results trees cannot be
    # ordered. See reporting._as_utc.
    moment = datetime.fromisoformat(payload["timestamp"])
    assert moment.tzinfo is not None
    assert moment.utcoffset() == timezone.utc.utcoffset(None)

    # Most experiments emit an "aggregated" mapping of metric -> series. exp3
    # instead emits its own alpha/ess_fraction series, which reporting handles
    # via `.get("aggregated", {}) or {}`. Accept either, but require that
    # whichever shape is used carries equal-length columns.
    series = payload.get("aggregated") or {
        key: value
        for key, value in payload.items()
        if isinstance(value, list) and key not in {"pi_tests", "seeds"}
    }
    assert series, f"summary for {name} carries no result series"
    lengths = {len(values) for values in series.values()}
    assert len(lengths) == 1, f"ragged result columns in {name}: {lengths}"


@pytest.mark.parametrize(
    ("name", "args"),
    sorted(OFFLINE_EXPERIMENTS.items()),
    ids=sorted(OFFLINE_EXPERIMENTS),
)
def test_experiment_cli_writes_expected_artifacts(monkeypatch, tmp_path, name, args):
    payload = _run_experiment(monkeypatch, tmp_path, name, args)
    _assert_summary_is_well_formed(payload, name)


@pytest.mark.parametrize(
    ("name", "args"),
    sorted(OFFLINE_EXPERIMENTS.items()),
    ids=sorted(OFFLINE_EXPERIMENTS),
)
def test_experiment_records_its_configuration(monkeypatch, tmp_path, name, args):
    """The summary must pin down enough to reproduce the run."""
    payload = _run_experiment(monkeypatch, tmp_path, name, args)

    assert payload["seeds"], "no seeds recorded"
    assert payload["c10"] > 0 and payload["c01"] > 0
    assert 0.0 < payload["pi_train"] < 1.0
    # Dataset-backed experiments (exp5) record 0 here: their sizes come from
    # the data, not from CLI flags, so only require the field to be present
    # and non-negative.
    assert payload["n_train"] >= 0 and payload["n_test"] >= 0


@pytest.mark.network
@pytest.mark.slow
@pytest.mark.parametrize(
    ("name", "args"),
    sorted(NETWORK_EXPERIMENTS.items()),
    ids=sorted(NETWORK_EXPERIMENTS),
)
def test_dataset_backed_experiment_cli(monkeypatch, tmp_path, name, args):
    payload = _run_experiment(monkeypatch, tmp_path, name, args)
    _assert_summary_is_well_formed(payload, name)
