"""I/O helpers for experiment artifact management."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: Path | str) -> Path:
    """Ensure a directory exists and return its path object."""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def save_table(df: pd.DataFrame, path: Path | str) -> None:
    """Persist a DataFrame to CSV, creating parent directories as needed."""
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    df.to_csv(path_obj, index=False)


def save_json(obj: Any, path: Path | str, *, indent: int = 2) -> None:
    """Write an object to disk in JSON format."""
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    with path_obj.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=indent)


def timestamped_run_dir(base: Path | str = "results", name: str = "exp1") -> Path:
    """Create a new timestamped run directory and return its path."""
    base_path = ensure_dir(Path(base) / name)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(base_path / timestamp)
    ensure_dir(run_dir / "tables")
    ensure_dir(run_dir / "figures")
    return run_dir
