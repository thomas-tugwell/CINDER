# Contributing

1. `pip install -e ".[dev]"` and `pre-commit install`.
2. `make check` must pass (ruff, mypy, CPU tests). TMol tests: `make test-tmol`.
3. Keep component boundaries: conformation models and sequence policies must not import TMol;
   only the `energy/tmol*.py` modules may. Interfaces live in `src/cinder/types.py`.
4. New mathematics needs a test against a closed form (see `tests/unit/test_langevin_math.py`).
