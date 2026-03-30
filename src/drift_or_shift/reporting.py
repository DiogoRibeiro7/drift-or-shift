"""Helpers for summarizing experiment artifacts."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Sequence

import pandas as pd


def collect_summary_jsons(results_root: Path | str = "results") -> list[dict]:
    """Collect available experiment summary JSON payloads."""
    root_path = Path(results_root)
    if not root_path.exists():
        return []
    entries: list[dict] = []
    for summary in sorted(root_path.rglob("*_summary.json")):
        try:
            data = json.loads(summary.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        data["_path"] = str(summary)
        entries.append(data)
    return entries


def _summary_timestamp(payload: dict) -> datetime | None:
    """Return the timestamp associated with a summary (fallback to file mtime)."""
    iso_ts = payload.get("timestamp")
    if isinstance(iso_ts, str):
        try:
            return datetime.fromisoformat(iso_ts)
        except ValueError:
            pass
    fallback_path = payload.get("_path")
    if isinstance(fallback_path, str):
        try:
            return datetime.fromtimestamp(Path(fallback_path).stat().st_mtime)
        except (OSError, ValueError):
            pass
    return None


def select_latest_summaries(summaries: Sequence[dict]) -> list[dict]:
    """Return the newest summary entry per experiment name."""
    latest: dict[str, tuple[datetime, dict]] = {}
    for payload in summaries:
        name = payload.get("exp_name")
        if not isinstance(name, str):
            continue
        timestamp = _summary_timestamp(payload)
        if timestamp is None:
            continue
        current = latest.get(name)
        if current is None or timestamp > current[0]:
            latest[name] = (timestamp, payload)
    return [payload for name, payload in sorted((name, entry[1]) for name, entry in latest.items())]


DRIFT_ALERT_THRESHOLDS: dict[str, float] = {
    "feature_max_mean_diff": 0.25,
    "feature_max_std_diff": 0.25,
    "feature_max_ks": 0.2,
    "feature_covariance_fro_diff": 0.8,
    "feature_correlation_mean_diff": 0.12,
    "feature_projection_max_ks": 0.18,
    "feature_projection_mean_ks": 0.11,
}


def _safe_mean(values: Sequence | None) -> float | None:
    if not values:
        return None
    try:
        numbers = [float(value) for value in values]
    except (TypeError, ValueError):
        return None
    if not numbers:
        return None
    return float(sum(numbers) / len(numbers))


def detect_drift_alerts(
    summary: dict,
    *,
    thresholds: dict[str, float] | None = None,
) -> list[tuple[str, float, float]]:
    """Return (metric, value, threshold) for metrics exceeding thresholds."""
    thresholds = thresholds or DRIFT_ALERT_THRESHOLDS
    aggregated = summary.get("aggregated", {}) or {}
    alerts: list[tuple[str, float, float]] = []
    for metric, threshold in thresholds.items():
        values = aggregated.get(metric)
        mean_value = _safe_mean(values)
        if mean_value is not None and mean_value >= threshold:
            alerts.append((metric, mean_value, threshold))
    return alerts


def collect_drift_alert_records(
    summaries: Sequence[dict],
    *,
    thresholds: dict[str, float] | None = None,
) -> list[dict[str, object]]:
    """Return summary records for summaries whose alerts exceed thresholds."""
    thresholds = thresholds or DRIFT_ALERT_THRESHOLDS
    records: list[dict[str, object]] = []
    for summary in summaries:
        alerts = detect_drift_alerts(summary, thresholds=thresholds)
        if not alerts:
            continue
        record = {
            "exp_name": summary.get("exp_name", "unknown"),
            "timestamp": summary.get("timestamp", "unknown"),
        }
        for idx, (metric, value, threshold) in enumerate(alerts, start=1):
            record[f"alert_{idx}_name"] = metric
            record[f"alert_{idx}_value"] = f"{value:.4f}"
            record[f"alert_{idx}_threshold"] = f"{threshold:.4f}"
        records.append(record)
    return records


def load_drift_alert_thresholds(path: Path | str | None = None) -> dict[str, float]:
    """Load drift alert thresholds from YAML/JSON file or return defaults."""
    if path is None:
        return DRIFT_ALERT_THRESHOLDS
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Threshold config not found: {source}")
    text = source.read_text(encoding="utf-8")
    try:
        import yaml

        thresholds = yaml.safe_load(text)
    except ImportError:
        import json

        thresholds = json.loads(text)

    if not isinstance(thresholds, dict):
        raise ValueError("Threshold config must be a mapping")
    parsed: dict[str, float] = {}
    for key, value in thresholds.items():
        if not isinstance(key, str):
            raise ValueError("Threshold keys must be strings")
        parsed[key] = float(value)
    return {**DRIFT_ALERT_THRESHOLDS, **parsed}


def write_drift_alerts(
    summaries: Sequence[dict],
    output_path: Path | str,
    *,
    thresholds: dict[str, float] | None = None,
) -> Path:
    """Write detected drift alerts to CSV and return the path."""
    path = Path(output_path)
    records = collect_drift_alert_records(summaries, thresholds=thresholds)
    df = pd.DataFrame(records)
    if df.empty:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def format_results_overview(summaries: Sequence[dict]) -> str:
    """Return a markdown report that lists experiment artifacts."""
    lines = ["# Results overview", ""]
    if not summaries:
        lines.append("No experiment summaries found")
        return "\n".join(lines)
    for payload in summaries:
        name = payload.get("exp_name", "unknown")
        timestamp = payload.get("timestamp")
        lines.append(f"## {name}")
        if timestamp:
            lines.append(f"- timestamp: {timestamp}")
        table = payload.get("table")
        if table:
            lines.append(f"- table: {table}")
        figure = payload.get("figure")
        if figure:
            lines.append(f"- figure: {figure}")
        if "aggregated" in payload:
            lines.append("- metrics:")
            for metric, values in payload["aggregated"].items():
                lines.append(f"  - {metric}: {values}")
        lines.append(f"- raw summary path: {payload.get('_path')}")
        lines.append("")
    lines.append(f"Generated on {datetime.now().isoformat()}")
    return "\n".join(lines)


def write_results_overview(
    summaries: Sequence[dict],
    output_path: Path | str,
) -> Path:
    """Write the overview text to disk and return the path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_results_overview(summaries), encoding="utf-8")
    return path
