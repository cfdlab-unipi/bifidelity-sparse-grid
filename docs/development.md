# Development and packaging

## Install and test

From the repository directory, install an editable copy:

```bash
python -m pip install -e .
```

Run the checks with the same Python interpreter:

```bash
python tests/check_numerics.py
python tests/check_workflow.py
python tests/check_cli.py
python -m pip check
```

The numerical checks cover multi-indices, polynomial/cardinal interpolation,
a direct Lagrange reference, all Leja prefixes, quadrature, vector indicators,
and selection ties. Workflow checks cover vector file I/O, zero/full/partial
budgets, automatic modes, and missing selected evaluations. CLI checks run
installed commands from a temporary directory, including both project templates.

The numerical and workflow scripts import the checkout. The CLI script checks
the installed package outside the checkout. If using a regular installation,
reinstall with `python -m pip install .` after changing code or packaged data.

## Package resources

Runtime code is in `bifidelity_sparse_grid/`. Its `data/` and `templates/`
subdirectories are installed with the package, so node loading and project
initialization work independently of the checkout.

`config.example.json` is the public copy of `templates/scalar.json`. Keep their
settings identical when updating the default configuration. The `init` command
uses the chosen project directory's name as `project_name`.

Dependencies and installed resources are declared in `pyproject.toml`.
`MANIFEST.in` additionally includes documentation, examples, tests, and the
source launcher in source distributions. Generated simulation files,
example results, caches, and build artifacts are excluded from Git.

## Build distributions

Build a wheel with:

```bash
python -m pip wheel . --no-deps --wheel-dir dist
```

To build both a wheel and a source archive:

```bash
python -m pip install build
python -m build
```

The wheel contains the runtime package, Leja CSV, configuration templates,
project README template, and license. The source archive additionally contains
the repository documentation, examples, tests, and source launcher.
Distributions are written to `dist/`.
