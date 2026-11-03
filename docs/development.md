# Development

```bash
python -m venv .venv && .venv/bin/pip install -e ".[envs,test,dev]"
.venv/bin/python -m pytest -q          # about 30 s
.venv/bin/ruff check src tests scripts examples
.venv/bin/mkdocs build --strict        # the documentation site
```

The core tests need only numpy and scipy. The environment tests are skipped
when gymnasium or pettingzoo is missing. Continuous integration
(`.github/workflows/ci.yml`) runs four jobs:
- the core tests on Python 3.10 to 3.13;
- the full tests with the environments;
- lint;
- the docs build, plus a wheel check that every bundled data file is
  packaged.

On macOS, a file-sync service can set the `hidden` flag on the virtual
environment's `.pth` files, and Python then skips them. The tests don't
depend on them (`pythonpath = ["src"]`), and
`chflags nohidden .venv/lib/*/site-packages/*.pth` restores the editable
install.

## Release checklist

1. Run the tests, lint, and docs build. Run `python -m build` and check the
   wheel in a fresh environment.
2. Update `CHANGELOG.md` and the version in `pyproject.toml` and
   `src/icgym/__init__.py`.
3. Regenerate `benchmarks/RESULTS.md` (`icgym-bench --seeds 3 --json
   benchmarks/results.json`) if the engine or the baselines changed.
4. For double-blind review, build the anonymized copy with
   `python scripts/make_anonymous.py OUT_DIR` and check its report.
5. Tag the release.
