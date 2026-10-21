PY ?= python

.PHONY: install lint format typecheck test test-tmol deadcode check
install:
	$(PY) -m pip install -e ".[dev]"
lint:
	ruff check src tests scripts
format:
	ruff format src tests scripts
typecheck:
	mypy
test:
	pytest
test-tmol:
	pytest -m tmol
deadcode:
	vulture
check: lint typecheck test
