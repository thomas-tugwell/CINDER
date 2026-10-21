# Development

```bash
conda activate cinder
pip install -e ".[dev,tmol]"
make check          # ruff, mypy, CPU tests (default markers exclude tmol/gpu/slow)
make test-tmol      # needs TMol
```

Markers: `tmol`, `gpu`, `slow`. Default CI installs neither TMol nor CUDA.

Run Python from the repo root, not from `src/cinder/` (the package's `types.py` would shadow
the standard-library `types` module).

Adding a conformation model: implement the `ConformationModel` protocol, add a branch in
`experiments/runner.build_sampler`, add an analytic test.

## Keeping the code lean

`make deadcode` (vulture, configured in `pyproject.toml`) lists code that nothing in `src`,
`scripts`, `tests` or `examples` uses. Entries about dataclass fields and public API that scripts
call are expected at 60% confidence; remove anything genuinely unused together with its references,
and prefer one shared implementation over copies (for example `experiments/tmol_study.py` holds the
set-up the study scripts used to repeat, and `rigid_body_perturb` wraps the jump parametrization).
