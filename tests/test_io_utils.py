import json
from pathlib import Path

import pandas as pd

from drift_or_shift import io_utils


def test_io_helpers_create_dirs_and_files(tmp_path: Path) -> None:
    output_dir = io_utils.ensure_dir(tmp_path / "runs")
    assert output_dir.exists()

    df = pd.DataFrame({"x": [1, 2]})
    table_path = output_dir / "table.csv"
    io_utils.save_table(df, table_path)
    assert table_path.exists()
    read_back = pd.read_csv(table_path)
    assert read_back.equals(df)

    json_path = output_dir / "meta.json"
    io_utils.save_json({"exp": "test"}, json_path)
    assert json_path.exists()
    assert json.loads(json_path.read_text())["exp"] == "test"

    run_dir = io_utils.timestamped_run_dir(base=tmp_path, name="exp5")
    assert (run_dir / "tables").exists()
    assert (run_dir / "figures").exists()
