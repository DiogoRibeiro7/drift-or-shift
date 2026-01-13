# DriftOrShift

## Quickstart
- `pip install -e .`
- `python -c "import drift_or_shift"` should run without errors once the package exists.
- Use `pytest`, `ruff`, and `mypy` to validate the codebase during development.

## Reproducibility
- All data generation relies on `numpy.random.default_rng(seed)` with fixed seeds when provided.
- Experiments record the seed, dataset size, prevalence settings, and costs in their output artifacts.
- Results and figures are stored in `results/`, with summaries in `results/tables/` and plots in `results/figures/`.
