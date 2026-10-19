# Analytical examples

From the repository directory, install with `python -m pip install .` if needed. Then run:

```bash
python examples/run_example.py --case ishigami --grid CC
python examples/run_example.py --case oscillatory --grid leja_sym
```

The script runs the three phases using analytical function values instead of
an external solver. It writes a configuration, fills baseline and selected
point files, and validates on 1024 scrambled-Sobol points with seed 42.

## Functions

| Case | Function | Domain |
| --- | --- | --- |
| `ishigami` | `sin(x1) + 7*sin(x2)**2 + 0.1*x3**4*sin(x1)` | `[-pi, pi]^3` |
| `oscillatory` | `sum_i cos(theta_i*x_i)`, `theta/pi = [1.5, 3, 0.5, 4.5]` | `[0, 1]^4` |

## Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--case` | `ishigami` | `ishigami` or `oscillatory` |
| `--grid` | `CC` | `CC`, `leja_line`, or `leja_sym` |
| `--budget-percent` | `20` | Nominal percentage of new target nodes |
| `--output-dir` | Case/grid folder | Alternative output location |

Other settings are in the configuration dictionary in `run_example.py`.
Auxiliary/baseline/target levels are `1/2/3` for CC and `2/3/5` for Leja.
The indicator is symmetric relative with `epsilon=1e-5`; the cutoff tie band
is `1e-3`. Equal grid levels do not imply equal evaluation counts.

## Results

By default, files go to `examples/output/<case>_<grid>/`:

- `config.json`: complete settings with resolved output paths;
- `data/`: baseline, refinement, and independent validation values;
- `analysis/`: selections, figures, validation metrics, and weighted statistics.

Read `analysis/<case>_selection.json` for nominal and actual evaluation counts.
Read `analysis/<case>_validation_metrics.csv` for accuracy: `mae` is mean
absolute error; `max_absolute_error` is the maximum. Error metrics have the
function's units. Percentage errors use the reference range over the validation set.

Repeating a command replaces generated files in that case/grid directory.
Use `--output-dir` to preserve a different run. These examples demonstrate
execution and outputs; refinement need not improve every metric at every
budget. They are not a matched-cost benchmark campaign.
