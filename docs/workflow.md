# Simulation workflow and data files

## Create a project

After installing the toolkit, run:

```bash
sparse-grid init /path/to/my-study
cd /path/to/my-study
```

This creates `config.json`, `data/`, `output/`, and a README. The initial
configuration has two inputs and one scalar output. Use `--template vector`
when creating a new project to start with scalar and vector outputs.
Initialization refuses to overwrite an existing `config.json`.

Edit the inputs, domains, outputs, levels, and budget, then run:

```bash
sparse-grid check
```

The check validates settings and does not execute simulations or create output
files. Input and output names must be unique and contain no whitespace.

## Phase 1: baseline evaluations

```bash
sparse-grid --phase 1
```

The toolkit writes `data/my-study_phase1_to_fill.dat`. Copy it to
`data/my-study_phase1.dat`, run your own solver at every listed point, and fill
all output columns. The toolkit does not launch the external solver.

For a scalar response, a completed file has this form:

```text
x1       x2       response
0.0      0.0      1.23
0.5      0.5      2.34
1.0      1.0      3.45
```

These rows illustrate the format only. Use every row and the exact coordinates
from your generated template. Tabs and spaces are accepted. Preserve the
column names and input precision. Do not add input units or comments inside
numeric cells.

For vector outputs, each component has its own column, such as
`velocity_t0000`, `velocity_t0001`, and so on. Every component must be filled.

## Phase 2: refinement evaluations

```bash
sparse-grid --phase 2
```

The algorithm compares baseline and auxiliary interpolants on target nodes
that are absent from the baseline grid. Auxiliary nodes are already contained
in the baseline; no separate auxiliary simulation file is needed.

Review `output/my-study_selection.csv` and `output/my-study_selection.json`.
The JSON distinguishes the nominal budget from the actual selected count.
When ties are enabled, a tied group can exceed the nominal budget.

Copy `data/my-study_phase2_to_fill.dat` to `data/my-study_phase2.dat`. The file
has two sections:

1. Automatically selected points: fill every output at every point.
2. Optional remaining target points: fill any additional evaluations you want
   to use, or leave their output columns empty.

Keep the section headers and coordinates. The reader accepts comment lines
starting with `#`, repeated column headers, and blank lines. Optional rows with
any missing output are ignored. Provide a Phase 2 file with its header even
when zero points are selected.

## Phase 3: models and analysis

```bash
sparse-grid --phase 3
```

The refined model uses measured values at supplied target nodes and baseline
predictions at unmeasured target nodes. Baseline values take priority if a
point appears in both input files. Points outside the target grid are not used.
Coordinates are matched using 12 decimal places, with a nearest-node fallback
of `1e-10` in physical input coordinates.

Keep the selection CSV produced by Phase 2. When it exists, Phase 3 checks
that all automatically selected evaluations have been supplied. Without this
CSV, that completeness check is unavailable.

The toolkit writes model slices, selection and grid plots, validation metrics,
and any enabled statistics or polynomial exports to `output/`. A full target
model is available only if you evaluate every target node; it is not required
for the selective workflow.

## Independent validation

Create a CSV, `.dat`, or `.tsv` file containing the configured input columns
and reference output values at independent test points:

```json
"validation": {"validation_data_file": "validation.csv"}
```

CSV files use commas; `.dat` and `.tsv` files use whitespace. Metrics are
reported for the configured output components present in the validation file.
Include every output if you want all quantities checked. An empty path disables
validation. A nonexistent path is reported and validation is skipped.

The metrics CSV contains baseline and informed model rows. `mae` is mean
absolute error; `max_absolute_error` is the maximum. RMSE and absolute errors
use the output's units. Percentage plots divide absolute errors by the range
of reference values over the test set. A constant reference has zero range,
so its percentage plots are skipped while absolute metrics remain available.

## Paths, repeated runs, and reproducibility

Relative data, output, and validation paths are resolved from `config.json`,
not from the shell's working directory. You can run any phase with:

```bash
sparse-grid --phase 2 --config /path/to/my-study/config.json
```

Do not change domains, grid family, levels, output definitions, or criterion
between phases of one study. Use a new project directory for a different case.
Re-running a phase replaces generated files with the same names; it does not
replace the completed simulation files with the `_to_fill` suffix removed.

The selection summary records planned baseline and selected evaluations.
Optional extra evaluations supplied later are not included in that planned
count. Archive the configuration, completed input files, selection files, and
validation data together when reporting results.
