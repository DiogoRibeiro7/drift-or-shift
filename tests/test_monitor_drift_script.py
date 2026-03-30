"""Regression tests for scripts/monitor_drift.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _monitor_drift_script_path() -> Path:
    return Path(__file__).resolve().parents[1] / "scripts" / "monitor_drift.py"


def test_monitor_drift_no_alert(tmp_path) -> None:
    reference = tmp_path / "reference.csv"
    target = tmp_path / "target.csv"
    reference.write_text("x,y\n0,0\n1,1\n", encoding="utf-8")
    target.write_text("x,y\n0,0\n1,1\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(_monitor_drift_script_path()),
            "--reference",
            str(reference),
            "--target",
            str(target),
            "--mean-threshold",
            "0.1",
            "--std-threshold",
            "0.1",
            "--ks-threshold",
            "0.1",
        ],
        check=False,
    )
    assert result.returncode == 0


def test_monitor_drift_fails_on_alert(tmp_path) -> None:
    reference = tmp_path / "reference.csv"
    target = tmp_path / "target.csv"
    reference.write_text("x,y\n0,0\n1,1\n", encoding="utf-8")
    target.write_text("x,y\n100,100\n101,101\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(_monitor_drift_script_path()),
            "--reference",
            str(reference),
            "--target",
            str(target),
            "--mean-threshold",
            "0.1",
            "--std-threshold",
            "0.1",
            "--ks-threshold",
            "0.1",
        ],
        check=False,
    )
    assert result.returncode == 1
