"""Summarize experiments that triggered feature-drift alerts."""

from __future__ import annotations

import argparse
from pathlib import Path

from drift_or_shift.reporting import collect_summary_jsons, write_drift_alerts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize drift alerts across experiments.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("reports") / "drift_alerts.csv")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summaries = collect_summary_jsons(args.results_dir)
    path = write_drift_alerts(summaries, args.output)
    if not path.exists():
        print("No drift alerts detected.")
        return
    print(f"drift alerts written to {path}")


if __name__ == "__main__":
    main()
