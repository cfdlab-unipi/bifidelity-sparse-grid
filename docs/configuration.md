# Configuration reference

The CLI reads `config.json`. Create one with `sparse-grid init PROJECT`, or
copy [`config.example.json`](../config.example.json) to `PROJECT/config.json`
for manual setup. The scalar template is the default;
`sparse-grid init PROJECT --template vector` creates the complete scalar/vector
template. Both templates are included in the installed package.

Run `sparse-grid check --config PATH` before generating points. Relative paths
are resolved from the configuration file. The defaults below describe CLI
usage; direct Python phase calls use the supplied dictionary without adding
paths or plotting defaults.

## Project settings

| Field | Required | Meaning |
| --- | --- | --- |
| `project_name` | Yes | Name used in generated filenames; no directory separators |
| `data_dir` | No | Simulation input/output directory; default `data` |
| `output_dir` | No | Analysis directory; default `output` |
| `problem_definition` | Yes | Inputs, domains, distributions, outputs |
| `sparse_grid_settings` | Yes | Grid family and levels |
| `refinement_settings` | Yes | Indicator and selection budget |
| `validation` | No | Independent validation file |
| `plotting` | No | Response slices and quantities to plot |
| `statistics` | No | Weighted statistics and polynomial exports |

## Inputs and outputs: `problem_definition`

| Field | Required | Meaning |
| --- | --- | --- |
| `dim_names` | Yes | Ordered, unique input names without whitespace |
| `domains` | Yes | One finite `[min, max]` interval per input, with `max > min` |
| `output_quantities` | Yes | Scalar names or vector definitions |
| `pdfs` | No | One distribution per input; defaults to uniform distributions |

A scalar output is a string, for example `"pressure"`. A vector is an object:

```json
{
    "name": "velocity",
    "size": 100,
    "label": "t",
    "pad": 4,
    "plot_indices": [0, 25, 99]
}
```

`name` and positive `size` are required. `label` defaults to `t`; `pad` defaults
to `4`. This definition produces `velocity_t0000` through `velocity_t0099`.
`plot_indices` selects zero-based components for individual plots. If omitted,
no individual vector components are plotted by default. All components still
participate in the vector refinement criterion and available validation metrics.
Input names, output names, and expanded component columns must not conflict.

### Probability distributions

Each `pdfs` entry can be a type string or an object:

```json
{"type": "normal", "params": {"mu": 0.0, "sigma": 1.0}, "leja_variant": "symmetric"}
```

| Type | Parameters and defaults | Density shape |
| --- | --- | --- |
| `uniform` | No shape parameters | Constant |
| `normal` | `mu=0`, `sigma=1`, with `sigma > 0` | Gaussian |
| `exponential` | `lam=1`, positive; `lambda` is an alias | `exp(-lam*x)` for `x >= 0` |
| `gamma` | `alpha=0`, `beta=1`; `alpha > -1`, `beta > 0` | `x**alpha * exp(-beta*x)` for `x >= 0` |
| `beta` | `alpha=0`, `beta=0`, both greater than `-1` | `z**alpha * (1-z)**beta`, with `z` scaled to the domain |
| `triangular` | No shape parameters | Linearly decreasing from the left domain endpoint |

For gamma, `beta` is a rate and the usual shape parameter is `alpha + 1`.
For beta, the usual shape parameters are `alpha + 1` and `beta + 1`.
Exponential and gamma intervals must overlap their nonnegative support.

All workflow inputs have finite domains. Normal, exponential, and gamma
probabilities are truncated to those intervals and integration weights are
normalized over the finite domain. Put interval bounds in `domains`.
`params.a` and `params.b` override those bounds when supplied; omit them or
keep them equal to `domains` to avoid conflicting definitions. The low-level
`trunc` option does not replace explicit workflow bounds.

`leja_variant="symmetric"` requests symmetric PDF-weighted Leja when the PDF
is symmetric on its interval. Normal requires `mu` at the midpoint; beta
requires equal exponents. In `auto` mode, uniform inputs use CC regardless of
this option. Tabulated `leja_sym` is selected through `grid_type` instead.

The PDF-weighted Leja implementation maximizes its score on 2001 candidate
locations per direction. It requires finite density values at those locations;
PDFs with endpoint singularities need an explicit `CC`, `leja_line`, or
`leja_sym` interpolation grid. They can still be integrated with the configured
PDF. Check quadrature convergence for sharply concentrated or singular PDFs.

## Grid family and levels: `sparse_grid_settings`

| Field | Default | Meaning |
| --- | --- | --- |
| `base_level` | Required | Positive integer baseline level |
| `target_level` | Required | Integer strictly above `base_level` |
| `auxiliary_level` | `base_level - 1` | Nonnegative integer below `base_level` |
| `grid_type` | `auto` | `CC`, `leja_line`, `leja_sym`, or `auto` |

`CC` uses Clenshaw-Curtis nodes. `leja_line` and `leja_sym` use the packaged
classical and symmetric real Leja sequences. The 31-node limit is per direction,
not the total number of sparse-grid points. Symmetric odd-length prefixes have
complete mirror pairs; even prefixes may have one unmatched member.

The total-degree rule is `sum(i_j - 1) <= level`, where each one-dimensional
level `i_j` starts at one. The largest one-dimensional rule therefore has
level `level + 1`. CC grows as `1, 3, 5, 9, ...` nodes. Tabulated Leja grows as
`1, 2, 3, 4, ...`, so `target_level <= 30` is required.

In `auto`, uniform inputs use CC and other PDFs use weighted Leja. The common
growth rule is doubling if any input uses CC, otherwise odd growth
`1, 3, 5, ...` if a symmetric weighted rule is requested, otherwise linear.
Explicit grid families choose interpolation nodes; the configured PDFs still
control statistics. Equal grid levels across families do not give equal costs.

## Indicator and budget: `refinement_settings`

| Field | Default | Meaning |
| --- | --- | --- |
| `refinement_criterion` | Required | Scalar output, vector name, or vector component |
| `indicator_type` | `legacy` | `absolute`, `symmetric_relative`, `relative` alias, or `legacy` |
| `relative_epsilon` | `1e-5` | Positive finite relative stabilizer |
| `budget_percent` | Not set | Percentage of new target candidates, from 0 to 100 |
| `max_budget` | Mode dependent | Fixed count or nominal cap; nonnegative integer |
| `use_auto_budget` | `false` | Enable automatic count selection |
| `auto_budget_method` | `elbow` | `elbow` or `threshold` |
| `auto_budget_threshold` | `0.10` | Fraction of the maximum indicator, from 0 to 1 |
| `include_ties` | `true` | Retain near-equal cutoff indicators |
| `tie_tolerance` | `1e-3` | Relative cutoff band, from 0 included to 1 excluded |

The absolute indicator is `|I_base - I_aux|`. The symmetric relative indicator
is `2*|I_base-I_aux| / (|I_base| + |I_aux| + relative_epsilon*S)`, where `S` is
the larger maximum magnitude of the two predictions over all candidates.
Vector outputs are interpolated component by component; absolute values in
these formulas become Euclidean norms. Identically zero predictions give
zero indicators. Legacy mode divides absolute discrepancy by `|I_base|` where
that magnitude exceeds `1e-9`, and uses absolute discrepancy elsewhere.

Selection follows this precedence:

1. If `use_auto_budget=true`, elbow or threshold selects a count. Threshold
   retains indicators at least the specified fraction of their maximum.
   `max_budget` caps the nominal count when supplied.
2. Otherwise, if `budget_percent` exists, it determines the count. The count
   is `floor(q*N_candidates)`, where `q=budget_percent/100`. A small positive
   fraction may therefore select zero points. `max_budget`, if present, caps it.
3. Otherwise, `max_budget` is the requested count, defaulting to zero.

A zero requested count selects no points. With a positive count and identical
zero indicators, tie handling retains the whole tied set. Ranking is stable.
With ties enabled, indicators at least `cutoff - tie_tolerance*abs(cutoff)`
are included, so the actual count can exceed the nominal count and
`max_budget`. Use `include_ties=false` for a strict cap. Selection CSV/JSON
files record the actual choice; preserve them for Phase 3.

## Validation: `validation`

`validation_data_file` is an optional CSV or whitespace-separated `.dat`/`.tsv`
path. An empty or absent path disables validation; an unavailable file is
reported and skipped. Include the input columns and reference outputs at
independent test points. Metrics are computed for available configured output
components, separately for baseline and informed models.

The output metrics are `rmse`, `mae` (mean absolute error),
`max_absolute_error`, `p95_absolute_error`, and signed `bias`. They use the
output's units. Percentage plots divide by the reference range over the whole
test set. Constant references receive absolute metrics but no percentage plots.

## Plots: `plotting`

| Field | Default | Meaning |
| --- | --- | --- |
| `plot_resolution` | `40` | Response-grid resolution, integer at least two |
| `slices_2d` | `[]` | List of response slices |
| `plot_quantities` | Automatic | Output names or component columns to plot |

A slice varies exactly two different known inputs:

```json
{"varying": ["x1", "x2"], "fixed": {"x3": 0.5}}
```

Missing `fixed` defaults to `{}`. Any unspecified remaining input is fixed
at its domain midpoint; this is a slice, not a projection or average.
Fixed coordinates must lie inside their physical domains. One-dimensional
problems can omit response slices entirely.

Automatic quantity selection, including an empty `plot_quantities` list,
uses all scalar outputs and vector components listed in `plot_indices`.
A vector name in `plot_quantities` expands to its `plot_indices`; explicit
component names can also be used. This controls individual plots, not which
components receive validation metrics or statistics.

## Statistics and polynomial export: `statistics`

| Field | Default | Meaning |
| --- | --- | --- |
| `enabled` | `true` | Compute PDF-weighted statistics of the informed model |
| `quantiles` | `[0.05, 0.5, 0.95]` | Requested weighted quantile fractions |
| `quad_order` | `200` | Gauss-Legendre order for one-dimensional basis integration |
| `export_lagrange_coeffs` | `true` | Export baseline and target-grid polynomial coefficients |
| `plot_vector_mean_std` | `true` | Plot vector component means and standard deviations |
| `vector_metrics` | `["l2", "mean", "rms", "max"]` | Vector aggregates to summarize |
| `exceedance` | `[]` | Output thresholds for estimated exceedance probabilities |

The starter scalar configuration explicitly disables statistics and polynomial
export. Polynomial export is controlled separately: setting `enabled=false`
alone does not disable `export_lagrange_coeffs`.

Exports include mean, variance, standard deviation, skewness, excess kurtosis,
and requested quantiles. Vector output files contain per-component summaries
and aggregate summaries. `l2` is the Euclidean norm; `mean` is the component
mean; `rms` is the root mean square; `max` is the signed component maximum.

An exceedance specification for a scalar or each vector component is:

```json
{"qoi": "pressure", "threshold": 0.8}
```

For an aggregate vector metric:

```json
{"qoi": "velocity", "metric": "l2", "threshold": 2.0}
```

Omitting `metric` means component-wise exceedance. Values are estimated from
the informed model at target nodes using sparse-grid quadrature weights.
These weights can be negative: nonlinear moments, quantiles, and probabilities
are numerical estimates and require convergence checks. They are not a
replacement for independent validation. Coefficient files are in `polynomials/`;
weighted summaries are in `statistics/`.
