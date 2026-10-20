PY ?= python

.PHONY: install lint format typecheck test test-tmol check
install:
	$(PY) -m pip install -e ".[dev]"
lint:
	ruff check src tests
format:
	ruff format src tests
typecheck:
	mypy
test:
	pytest
test-tmol:
	pytest -m tmol
check: lint typecheck test
