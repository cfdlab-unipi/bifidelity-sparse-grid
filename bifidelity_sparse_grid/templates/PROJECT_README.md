# {project_name}

## Configuration

Edit `config.json` to set input names and domains, output quantities, grid
levels, and refinement budget. Paths are relative to this file.

```bash
sparse-grid check
```

## Baseline evaluations

```bash
sparse-grid --phase 1
```

Copy `{data_dir}/{project_name}_phase1_to_fill.dat` to
`{data_dir}/{project_name}_phase1.dat`. Run your simulation at every point and
fill all output columns. Keep the input coordinates and column names.

## Refinement evaluations

```bash
sparse-grid --phase 2
```

Review the actual selected count in `{output_dir}/{project_name}_selection.json`.
Copy `{data_dir}/{project_name}_phase2_to_fill.dat` to
`{data_dir}/{project_name}_phase2.dat`. Fill every automatically selected point.
Additional target points in the second section are optional. Leave all output
values empty for optional points you do not evaluate.

## Analysis

```bash
sparse-grid --phase 3
```

Plots, selection summaries, and any enabled validation metrics or statistics
are written to `{output_dir}/`. Set `validation.validation_data_file` to your
independent validation file to obtain error metrics.

Use the same configuration throughout the three phases and keep the selection
CSV. Repeated runs replace generated files with the same names. Create another
project folder when changing the problem or grid settings.

Use `python -m bifidelity_sparse_grid` with the same arguments if the console
command is unavailable. Run `sparse-grid --help` for command options.

Full instructions: https://github.com/cfdlab-unipi/bifidelity-sparse-grid
