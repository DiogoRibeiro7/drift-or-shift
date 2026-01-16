"""Smoke test that ensures `scripts/smoke_exp1.py` completes."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_smoke_exp1_script(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    subprocess.run(
        [sys.executable, "scripts/smoke_exp1.py", "--results-dir", str(results_dir)],
        check=True,
    )

    exp_dir = results_dir / "smoke_exp1"
    assert exp_dir.exists(), "Smoke experiment directory was not created."
    runs = list(exp_dir.iterdir())
    assert runs, "No timestamped run created."
    run_dir = runs[-1]
    assert (run_dir / "tables" / "smoke_exp1.csv").exists(), "Table missing."
    assert (run_dir / "figures" / "smoke_exp1.png").exists(), "Figure missing."
    assert (run_dir / "smoke_exp1_summary.json").exists()
