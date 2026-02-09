# Adaptive Sparse Grid Toolkit — Configuration Reference

This document lists **all supported `config.json` fields** and explains what they do.
For mathematical background (sparse‑grid interpolation and quadrature), see `theory.tex`.

---

## 1) Top‑level keys

| Key | Type | Required | Description |
| --- | --- | --- | --- |
| `project_name` | string | yes | Project identifier used to name generated files. |
| `output_dir` | string | yes | Directory where plots, statistics, and polynomial outputs are written. |
| `data_dir` | string | yes | Directory for input/output `.dat` files. |
| `problem_definition` | object | yes | Input space definition and QoIs. |
| `sparse_grid_settings` | object | yes | Baseline/target sparse‑grid levels. |
| `refinement_settings` | object | yes | Refinement criterion and budget settings. |
| `validation` | object | no | Optional validation file settings. |
| `plotting` | object | yes | Plots configuration (required in Phase 3). |
| `statistics` | object | no | Weighted statistics and polynomial export settings. |

---

## 2) `problem_definition`

### Required fields

| Key | Type | Required | Description |
| --- | --- | --- | --- |
| `dim_names` | string[] | yes | Names of input parameters. Order matters everywhere. |
| `domains` | number[][] | yes | List of `[min, max]` per dimension, same order as `dim_names`. |
| `output_quantities` | (string \| object)[] | yes | Scalar or vector QoI definitions. |

### Optional fields

| Key | Type | Required | Description |
| --- | --- | --- | --- |
| `pdfs` | (string \| object)[] | no | Per‑dimension PDF specs used to generate collocation knots. If omitted, all dimensions are `uniform`. |

### `pdfs` specification

You must provide **one entry per dimension**. Each entry can be:

- **String**: e.g. `"uniform"`, `"normal"`, `"exponential"`, `"gamma"`, `"beta"`, `"triangular"`.
- **Object**: `{ "type": "normal", "params": {"mu": 0.0, "sigma": 1.0}, "leja_variant": "symmetric" }`.

Supported `type` values:
- `uniform`  → nested Clenshaw‑Curtis nodes
- `normal`   → weighted Leja
- `exponential` → weighted Leja
- `gamma`    → weighted Leja
- `beta`     → weighted Leja
- `triangular` → weighted Leja

**Important:** in the current implementation, `domains` are always used as the finite support for all PDFs.
That means unbounded PDFs (normal/exponential/gamma) are **truncated to your `domains`**. The optional
`trunc` field is accepted but effectively ignored because `a,b` are always set from `domains`.

#### PDF parameters and constraints

| PDF | Parameters in `params` | Constraints |
| --- | --- | --- |
| `uniform` | `a`, `b` (optional; taken from `domains`) | `b > a` |
| `normal` | `mu`, `sigma` | `sigma > 0` |
| `exponential` | `lam` (or `lambda`) | `lam > 0` |
| `gamma` | `alpha`, `beta` (rate) | `alpha > -1`, `beta > 0` |
| `beta` | `alpha`, `beta`, `a`, `b` | `alpha > -1`, `beta > -1` |
| `triangular` | `a`, `b` (optional; taken from `domains`) | `b > a` |

**Note on exponential:** the code uses `lam` internally; use `"lam"` in `params` to be safe.

#### `leja_variant`

Optional: `"symmetric"` (or `"sym"`, `"symmetrized"`).

Symmetric Leja is allowed only if the PDF is symmetric on the chosen domain:
- `normal`: `mu` must be exactly at the domain center `(a+b)/2`
- `beta`: `alpha == beta`
- `uniform`: allowed

### `output_quantities` specification

Each entry can be:

1) **Scalar QoI** (string)
```json
"qoi_scalar"
```
This expects a column named `qoi_scalar` in the `.dat` files.

2) **Vector QoI** (object)
```json
{
  "name": "qoi_vector",
  "size": 1000,
  "label": "t",
  "pad": 4,
  "plot_indices": [0, 10, 999]
}
```
- `name` (required): base name of the vector.
- `size` (required): number of components.
- `label` (optional, default: `"t"`): label inserted before the component index.
- `pad` (optional, default: `4`): zero‑padding for indices.
- `plot_indices` (optional): list of component indices to plot.

This generates component columns like:
```
qoi_vector_t0000, qoi_vector_t0001, ..., qoi_vector_t0999
```

---

## 3) `sparse_grid_settings`

| Key | Type | Required | Description |
| --- | --- | --- | --- |
| `base_level` | integer | yes | Sparse‑grid level for the baseline model (Phase 1). |
| `target_level` | integer | yes | Sparse‑grid level for the target grid (Phase 2/3). |

---

## 4) `refinement_settings`

| Key | Type | Required | Description |
| --- | --- | --- | --- |
| `refinement_criterion` | string | yes | QoI name used to compute the refinement indicator. For vector QoIs, L2 norm over components is used. |
| `use_auto_budget` | boolean | no | If true, budget is determined automatically. Default: `false`. |
| `auto_budget_method` | string | no | `"elbow"` or `"threshold"`. Default: `"elbow"`. |
| `auto_budget_threshold` | number | no | Relative threshold for `"threshold"` (e.g. `0.10`). Default: `0.10`. |
| `max_budget` | integer | no | Upper bound on refinement points. If `use_auto_budget` is false, this becomes the fixed budget. Default: `0`. |

**Behavior notes:**
- If `use_auto_budget` is `false` and `max_budget` is missing, the budget defaults to `0`.
- If `use_auto_budget` is `true`, the budget is capped by `max_budget` if provided.

---

## 5) `validation` (optional)

| Key | Type | Required | Description |
| --- | --- | --- | --- |
| `validation_data_file` | string | no | Path to validation data file. If missing or not found, validation plots are skipped. |

**File parsing:**
- `.dat` or `.tsv` → whitespace‑separated
- otherwise → CSV

---

## 6) `plotting`

This section is required for Phase 3 plotting.

| Key | Type | Required | Description |
| --- | --- | --- | --- |
| `plot_resolution` | integer | yes | Grid resolution for response surfaces. |
| `slices_2d` | object[] | yes | List of 2D slices to plot. |
| `plot_quantities` | string[] | no | Optional list of QoIs (or component columns) to plot. If omitted, all scalars and selected vector components are used. |

### `slices_2d`

Each entry must include:
```json
{"fixed": {"x3": 0.5}, "varying": ["x1", "x2"]}
```

- `varying`: list of exactly two dimensions.
- `fixed`: mapping of remaining dimensions to fixed values. Can be `{}` for 2D problems.

**Note:** If `fixed` is empty and `N > 2`, the plots become projections (not true slices). Provide
explicit fixed values if you want an actual slice through higher‑dimensional space.

---

## 7) `statistics` (optional)

If omitted, statistics are **enabled by default** with default parameters.

| Key | Type | Required | Description |
| --- | --- | --- | --- |
| `enabled` | boolean | no | Enable/disable statistics. Default: `true`. |
| `quantiles` | number[] | no | Quantiles to compute. Default: `[0.05, 0.5, 0.95]`. |
| `quad_order` | integer | no | 1D Gauss‑Legendre order for integrating Lagrange bases. Default: `200`. |
| `export_lagrange_coeffs` | boolean | no | Export Lagrange coefficients for levels `w` and `w+1`. Default: `true`. |
| `plot_vector_mean_std` | boolean | no | Plot mean ± std across vector components. Default: `true`. |
| `vector_metrics` | string[] | no | Vector aggregate metrics: any of `l2`, `mean`, `rms`, `max`. Default: `['l2','mean','rms','max']`. |
| `exceedance` | object[] | no | Exceedance probability settings (see below). |

### `exceedance`

Each entry is:
```json
{"qoi": "qoi_scalar", "threshold": 0.8}
```
or (for vector metrics):
```json
{"qoi": "qoi_vector", "metric": "l2", "threshold": 2.0}
```

- `qoi` (required): QoI name.
- `threshold` (required): threshold value.
- `metric` (optional): for vector QoIs, which metric to use. If omitted, component‑wise exceedance is used.

---

## 8) Example: full configuration template

Use `config.example.json` as a starting point. It includes:
- scalar + vector QoI definitions
- PDF specifications
- plotting and statistics sections

---

## 9) Notes on generated columns and files

- Vector QoIs generate component columns according to `name`, `label`, and `pad`.
- Phase templates are written as `*_phase1_to_fill.dat` and `*_phase2_to_fill.dat`.
- You must copy each template to a file **without** the `_to_fill` suffix and fill the new file.

