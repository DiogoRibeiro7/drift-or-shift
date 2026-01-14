import yaml

from reproduce.scripts.run_benchmark import run_benchmark


def test_reproduce_smoke(tmp_path):
    config = {
        "dataset": {"name": "breast_cancer", "max_samples": 50, "random_state": 0},
        "estimators": [
            {
                "name": "logistic",
                "type": "logistic",
                "params": {"C": 1.0, "max_iter": 1000, "solver": "liblinear"},
            }
        ],
        "calibration_methods": ["none"],
        "cv": {"n_splits": 2, "random_state": 0},
        "outputs": {
            "raw": str(tmp_path / "results" / "raw"),
            "tables": str(tmp_path / "results" / "tables"),
            "figures": str(tmp_path / "results" / "figures"),
        },
        "figures": {"reliability": {"methods": ["logistic|none"]}},
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)
    output_path = run_benchmark(config_path)
    assert output_path.exists()
