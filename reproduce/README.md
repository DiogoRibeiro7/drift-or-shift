# Reproduce the Calibration Benchmark

## Running the benchmark suite
- `poetry run python reproduce/scripts/run_benchmark.py --config reproduce/config/default.yaml`
- `poetry run python reproduce/scripts/make_tables.py --config reproduce/config/default.yaml`
- `poetry run python reproduce/scripts/make_figures.py --config reproduce/config/default.yaml`

## Outputs
- Raw fold-level predictions: `reproduce/results/raw/*.csv`
- Aggregated metric tables: `reproduce/results/tables/*.csv`
- Reliability diagrams: `reproduce/results/figures/*.png`

## Configuring your run
- Update `reproduce/config/default.yaml` to swap datasets, estimators, calibration methods, or CV settings.
- Adjust `outputs` paths to redirect artifacts to another base.
- Keep `cv.random_state` and `dataset.random_state` fixed for deterministic runs.
