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
