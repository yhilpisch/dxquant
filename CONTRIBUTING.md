# Contributing to DXQuant

Work from the repository root. Install the development, documentation, and
notebook dependencies before running the checks:

```bash
python -m pip install -e ".[dev,docs,notebooks]"
python -m ruff check src tests docs/conf.py tools notebooks
python -m mypy src tests docs/conf.py tools
python -m pytest --cov=src/dxquant --cov-report=term-missing
python tools/check_conventions.py
python tools/validate_colab.py
python tools/validate_colab.py --execute
python -m sphinx -W --keep-going -b html docs docs/_build
pre-commit install
pre-commit run --all-files
```

The pre-commit hooks check Ruff, mypy, pytest, and Python conventions without
rewriting files. Python lines, including notebook code, must fit within 80
characters. Use exactly two spaces before inline comments. Annotated function
defaults use `value: int=5`. This intentionally differs from Ruff format;
`tools/check_conventions.py` is the final convention check.

Preserve documented numerical behavior and process/payoff/pricer separation.
Public library callables need useful docstrings and annotations. Numerical
changes need focused regression tests with explicit tolerances and seeded
simulations or independent analytical references. Consult the
[numerical contracts](docs/numerical_contracts.rst) for supported domains.

The pre-1.0 compatibility policy covers `dxquant.__all__`. Intentional breaking
changes need a changelog entry and a minor-version increase. Examples use
`import dxquant as dx`. Additional module helpers do not carry the same
compatibility promise merely because they appear in the API reference.

Keep notebooks self-contained, outputs cleared, setup cells consistent, and
Colab links aligned with filenames. The headless validator performs formal
checks by default; `--execute` installs this checkout and runs all notebooks
with plotting in fresh kernels. JSON reports and positive per-cell timeouts
support hosted validation. See [tutorial instructions](notebooks/README.md).
CI also removes DXQuant before each tutorial to exercise its installation cell.

The `parity` extra supplies optional dependencies for comparisons with a
separate legacy `dx` checkout outside this repository. These comparisons skip
when that reference is unavailable and do not establish full legacy parity or
legacy parallel execution. Do not bundle legacy snapshots, local environments,
credentials, generated binaries, executed notebook copies, or build outputs.

Inspect the diff and commit coherent changes with concise messages. Publishing,
tagging, and pushing are separate explicit actions. To prepare local artifacts:

```bash
python -m build
python -m twine check --strict dist/*
```

Inspect the sdist/wheel contents and test the installed wheel outside the source
tree. Runtime matrices and hosted Colab checks need their own observed results;
a local pass is not evidence for every supported platform/dependency version.
