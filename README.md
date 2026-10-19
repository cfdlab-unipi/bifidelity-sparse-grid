# Bifidelity Sparse Grid Toolkit

Python tools for building sparse-grid surrogate models from simulation or
experimental data. The method uses the discrepancy between a baseline
interpolant and a coarser auxiliary interpolant to select additional evaluation
points on a target grid. Values at unselected target nodes are interpolated
from the baseline model.

The workflow supports scalar and vector outputs, Clenshaw–Curtis and Leja
nodes, independent validation, response-surface plots, and statistics weighted
by the input probability distributions.

## Installation

Requires Python 3.10 or later. NumPy, SciPy, pandas, and Matplotlib are installed
as dependencies.

```bash
git clone https://github.com/cfdlab-unipi/bifidelity-sparse-grid.git
cd bifidelity-sparse-grid
python -m pip install .
```

## Usage

Create a project directory:

```bash
sparse-grid init ../my-study
cd ../my-study
```

The directory contains `config.json`, `data/`, `output/`, and a project README.
Edit `config.json` to define the input parameters, their domains, the output
quantities, the grid levels, and the refinement budget. Then check the settings:

```bash
sparse-grid check
```

### 1. Evaluate the baseline grid

```bash
sparse-grid --phase 1
```

Copy `data/my-study_phase1_to_fill.dat` to `data/my-study_phase1.dat`.
Run the simulation at every listed point and fill all output columns.
Keep the input coordinates and column names unchanged.

### 2. Evaluate the selected refinement points

```bash
sparse-grid --phase 2
```

Copy `data/my-study_phase2_to_fill.dat` to `data/my-study_phase2.dat`.
Fill all automatically selected points. The second section lists optional
additional points; their output values may be left empty. Provide the file
also when no points have been selected.

### 3. Analyze the surrogate models

```bash
sparse-grid --phase 3
```

This builds the baseline and refined models and writes plots, selection
summaries, and any requested validation metrics or statistics to `output/`.
Use the same configuration throughout the three phases. The toolkit writes
point files; run your external simulation separately to supply their values.
See [the data workflow](docs/workflow.md) for file formats and repeated runs.

Paths in `config.json` are relative to the configuration file. To run a project
from another directory, specify its configuration:

```bash
sparse-grid --phase 1 --config /path/to/my-study/config.json
```

`python -m bifidelity_sparse_grid` accepts the same arguments as `sparse-grid`.

## Configuration

The default configuration defines a two-input scalar problem.
[`config.example.json`](config.example.json) provides the same settings for
inspection or manual setup: copy it to `config.json` in your project directory
and edit it for your study. A template with scalar and vector outputs is also
available:

```bash
sparse-grid init ../vector-study --template vector
```

| Setting | Description |
| --- | --- |
| `grid_type` | `CC`, `leja_line`, `leja_sym`, or `auto` for PDF-based node selection |
| `base_level`, `target_level` | Baseline and target sparse-grid levels |
| `indicator_type` | `absolute` or `symmetric_relative` discrepancy |
| `budget_percent` | Nominal percentage of new target points to evaluate |
| `max_budget` | Fixed point count, or nominal cap for automatic selection |
| `include_ties` | Include points with nearly equal indicators at the selection cutoff |

With `include_ties=true`, the actual count may exceed the nominal budget.
Set it to `false` for a strict count limit. Equal levels for CC and Leja do
not imply equal evaluation counts.

The classical and symmetric Leja sequences contain 31 one-dimensional nodes
each. Larger multidimensional grids can be constructed from these prefixes;
a rule requiring more than 31 nodes in one direction is not supported.
PDF-weighted Leja nodes are generated numerically from the specified distribution.

See the [configuration reference](docs/configuration.md) for PDF definitions,
vector outputs, automatic budget selection, validation, and statistics.

## Examples

Run from the repository directory after installation:

```bash
python examples/run_example.py --case ishigami --grid CC
python examples/run_example.py --case oscillatory --grid leja_sym
```

The examples execute all three phases using analytical functions and validate
the models on 1024 scrambled-Sobol points. Results are saved in
`examples/output/<case>_<grid>/`. Use `--budget-percent` to change the budget
or `--output-dir` to choose another output directory.

Function definitions and parameters are documented in [examples/README.md](examples/README.md).

## Output files

| Output | Contents |
| --- | --- |
| `<project>_selection.csv` | Candidate coordinates, indicators, and selected flags |
| `<project>_selection.json` | Grid settings and nominal and actual evaluation counts |
| `<project>_validation_metrics.csv` | RMSE, mean and maximum absolute error, 95th percentile absolute error, and bias |
| `*.png` | Selection, model-slice, grid-point, and validation plots |
| `statistics/` | Optional weighted statistics |
| `polynomials/` | Optional polynomial coefficients |

To enable validation, set `validation.validation_data_file` to a CSV or
whitespace-separated file with the configured input and output columns.
Absolute errors have the output's physical units. Percentage plots are
normalized by the reference range over the validation set. The CSV column
`mae` is mean absolute error; `max_absolute_error` is maximum absolute error.
Refinement need not reduce every error metric at every budget.

## Repository structure

```text
bifidelity_sparse_grid/   Python package, node data, and configuration templates
docs/                    Workflow, configuration, method, and development guides
examples/                Analytical examples
tests/                   Numerical and workflow checks
config.example.json      Scalar configuration for reference or manual setup
main.py                  Launcher from the source checkout
pyproject.toml           Package metadata and dependencies
```

## Tests

From the repository directory, after installation:

```bash
python tests/check_numerics.py
python tests/check_workflow.py
python tests/check_cli.py
```

These check interpolation, Leja rules, quadrature, point selection, file
handling, and installed commands. See [development and packaging](docs/development.md)
for editable installation, test coverage, and distribution builds.

## Documentation

- [Workflow](docs/workflow.md): simulation files, validation, paths, and repeated runs.
- [Configuration](docs/configuration.md): all settings, defaults, and numerical options.
- [Method](docs/method.md): grids, discrepancy indicators, selection, and reconstruction.
- [Development](docs/development.md): tests, package resources, and distribution builds.

## License

[MIT](LICENSE).

## Reference

M. Rosellini, F. Fruzza, A. Mariotti, M. V. Salvetti, and L. Tamellini.
*A bifidelity sparse-grid surrogate model driven by a discrepancy indicator*.
Manuscript, 2026.

## Contact

[cfdlab.unipi@gmail.com](mailto:cfdlab.unipi@gmail.com)
