"""
Core workflow logic for the adaptive sparse-grid toolkit.

Phases:
  1) Generate baseline sparse-grid points.
  2) Select refinement points via a discrepancy indicator and write a "to_fill" file
     (including an optional section for extra user points).
  3) Build baseline + informed models, validate (optional), and generate plots.

Robust I/O features:
  - to_fill files accept tabs or spaces, blank lines, and repeated headers.
  - Optional rows may be left empty and are ignored safely.
  - Merging base + refinement data de-duplicates by point-key; baseline wins.
"""

import os
import json
from .nodes import MAX_LEJA_NODES
import io
import re
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# --- Numerical and plotting tools of this package ---
from .sparse_grid_toolbox import (
    PdfSpec,
    create_sparse_grid_structure, reduce_sparse_grid_knots,
    evaluate_sparse_grid_interpolant, compute_discrepancy_indicator,
    select_refinement_points, knots_cc_1d, knots_leja_line_1d, knots_leja_sym_line_1d,
    lev2knots_doubling, lev2knots_lin, lev2knots_2step,
    build_1d_knots_and_domains_from_pdfspecs,
    pdfspec_to_domain_and_rho,
    compute_sparse_grid_quadrature_weights
)
from .plotting_tools import (
    plot_error_comparison_barchart, plot_combined_response_surfaces,
    plot_quadrature_points_slice, plot_discrepancy_by_point,
    plot_error_distribution, plot_vector_mean_std
)

# =============================================================================
# Global settings
# =============================================================================

ROUND_DECIMALS = 12  # for deterministic point keys (set operations, dedupe)
DEFAULT_TOL = 1e-10   # matching tolerance in physical space (snap-to-grid)
# Coordinates read from text files are replaced by the exact generated grid
# nodes when every coordinate lies within this fraction of its domain width.
# Text round trips can change the last binary digit of a coordinate; for
# inputs of order 1e4 or larger, 12-decimal keys cannot absorb that change.
# Distinct sparse-grid nodes are much farther apart than this tolerance.
SNAP_RELATIVE_TOL = 1e-6

# =============================================================================
# QoI helpers (scalar + vector support)
# =============================================================================

def _expand_qoi_specs(p_def):
    """
    Expand output_quantities into a list of specs with resolved column names.
    Supports:
      - "qoi_scalar" (scalar)
      - {"name": "qoi_vector", "size": 1000, "label": "t", "pad": 4, "plot_indices": [0, 10, 999]}
    """
    specs = []
    out_qois = p_def.get("output_quantities", [])
    for item in out_qois:
        if isinstance(item, str):
            specs.append({
                "name": item,
                "columns": [item],
                "is_vector": False,
                "size": 1,
                "plot_indices": [],
            })
            continue
        if isinstance(item, dict):
            name = item.get("name")
            size = int(item.get("size", 0))
            if not name or size <= 0:
                raise ValueError("Vector QoI spec requires 'name' and positive 'size'.")
            label = str(item.get("label", "t"))
            pad = int(item.get("pad", 4))
            columns = [f"{name}_{label}{i:0{pad}d}" for i in range(size)]
            specs.append({
                "name": name,
                "columns": columns,
                "is_vector": True,
                "size": size,
                "plot_indices": item.get("plot_indices", []),
            })
            continue
        raise ValueError(f"Invalid output_quantities entry: {item}")
    return specs

def _qoi_columns_from_specs(qoi_specs):
    """All output column names, in configuration order."""
    columns = []
    for spec in qoi_specs:
        columns.extend(spec["columns"])
    return columns

def _describe_qois_for_msg(qoi_specs):
    parts = []
    for spec in qoi_specs:
        if spec["is_vector"]:
            parts.append(f"{spec['name']} (vector, {spec['size']} components)")
        else:
            parts.append(spec["name"])
    return ", ".join(parts)

def _resolve_plot_columns(plot_cfg, qoi_specs):
    """
    Decide which QoI columns to plot.
    Default: all scalar quantities + vector quantities limited to plot_indices (if provided).
    If plot_cfg["plot_quantities"] is provided, only those are used.
    """
    name_to_spec = {}
    for spec in qoi_specs:
        name_to_spec[spec["name"]] = spec
    requested = plot_cfg.get("plot_quantities")

    # Collect the specs to plot. An explicit component column is added directly.
    cols = []
    specs_to_plot = []
    if requested:
        for name in requested:
            spec = name_to_spec.get(name)
            if spec is not None:
                specs_to_plot.append(spec)
            elif name in _qoi_columns_from_specs(qoi_specs):
                specs_to_plot.append({"is_vector": False, "columns": [name]})
            else:
                raise ValueError(f"plot_quantities contains unknown QoI '{name}'.")
    else:
        specs_to_plot = qoi_specs

    # A scalar gives its column; a vector gives only its plot_indices components.
    for spec in specs_to_plot:
        if not spec["is_vector"]:
            cols.extend(spec["columns"])
        else:
            for i in spec["plot_indices"]:
                if 0 <= int(i) < spec["size"]:
                    cols.append(spec["columns"][int(i)])

    seen = set()
    ordered = []
    for c in cols:
        if c not in seen:
            ordered.append(c)
            seen.add(c)
    return ordered

def _refinement_values_from_qoi(df, ref_name, qoi_specs):
    """
    Return scalar values or component rows for the refinement criterion.
    - If ref_name is a scalar QoI column -> use it directly.
    - If ref_name is a vector QoI name -> retain component rows for interpolation.
    """
    if ref_name in df.columns:
        return df[ref_name].values.astype(float)

    for spec in qoi_specs:
        if spec["name"] == ref_name and spec["is_vector"]:
            vals = df[spec["columns"]].values.astype(float)
            return vals.T

    raise ValueError(
        f"Unknown refinement_criterion '{ref_name}'. "
        f"Expected a scalar QoI column or a vector QoI name."
    )

def _parse_pdf_specs(p_def):
    """
    Build PdfSpec list from config.
    If pdfs is omitted, defaults to uniform per dimension.
    Domains are always taken from config and used as finite supports (truncation)
    for unbounded PDFs unless explicit a,b are provided in params.
    """
    dim_names = p_def["dim_names"]
    domains = p_def["domains"]
    pdfs = p_def.get("pdfs", None)
    if pdfs is None:
        pdfs = ["uniform"] * len(dim_names)
    if len(pdfs) != len(dim_names):
        raise ValueError("problem_definition.pdfs must match dim_names length.")

    specs = []
    for i, spec in enumerate(pdfs):
        if isinstance(spec, str):
            pdf_type = spec.lower()
            params = {}
            trunc = None
            leja_variant = "standard"
        elif isinstance(spec, dict):
            pdf_type = str(spec.get("type", "")).lower()
            params = dict(spec.get("params", {}) or {})
            trunc = spec.get("trunc")
            leja_variant = str(spec.get("leja_variant", "standard")).lower()
        else:
            raise ValueError(f"Invalid pdf spec at index {i}: {spec}")

        a, b = domains[i][0], domains[i][1]

        # Provide domain bounds when not explicitly included in params.
        # For unbounded PDFs these bounds are the truncation interval.
        params.setdefault("a", a)
        params.setdefault("b", b)

        # Exponential and gamma densities are zero for x < 0: an interval
        # entirely on the negative axis would have no probability at all.
        if pdf_type in ("exponential", "gamma") and float(params["b"]) <= 0.0:
            raise ValueError(
                f"{pdf_type} PDF needs an interval overlapping x > 0 (dim '{dim_names[i]}')."
            )

        # Basic parameter validation
        if pdf_type == "normal":
            sigma = float(params.get("sigma", 1.0))
            if sigma <= 0.0:
                raise ValueError(f"Normal PDF requires sigma > 0 (dim '{dim_names[i]}').")
        elif pdf_type == "exponential":
            lam = float(params.get("lam", params.get("lambda", 1.0)))
            params["lam"] = lam
            if lam <= 0.0:
                raise ValueError(f"Exponential PDF requires lambda > 0 (dim '{dim_names[i]}').")
        elif pdf_type == "gamma":
            alpha = float(params.get("alpha", 0.0))
            beta = float(params.get("beta", 1.0))
            if alpha <= -1.0 or beta <= 0.0:
                raise ValueError(f"Gamma PDF requires alpha > -1 and beta > 0 (dim '{dim_names[i]}').")
        elif pdf_type == "beta":
            alpha = float(params.setdefault("alpha", 0.0))
            beta = float(params.setdefault("beta", 0.0))
            if alpha <= -1.0 or beta <= -1.0:
                raise ValueError(f"Beta PDF requires alpha > -1 and beta > -1 (dim '{dim_names[i]}').")

        symmetric = leja_variant in ("symmetric", "sym", "symmetrized")
        if symmetric:
            if pdf_type == "normal":
                mu = float(params.get("mu", 0.0))
                center = 0.5 * (a + b)
                if abs(mu - center) > 1e-8:
                    raise ValueError(
                        f"Symmetric Leja for normal requires mu at domain center (dim '{dim_names[i]}')."
                    )
            elif pdf_type == "beta":
                alpha = float(params.get("alpha", 0.0))
                beta = float(params.get("beta", 0.0))
                if abs(alpha - beta) > 1e-8:
                    raise ValueError(
                        f"Symmetric Leja for beta requires alpha == beta (dim '{dim_names[i]}')."
                    )
            elif pdf_type not in ("uniform",):
                raise ValueError(
                    f"Symmetric Leja is not supported for pdf '{pdf_type}' (dim '{dim_names[i]}')."
                )
        specs.append(PdfSpec(name=pdf_type, params=params, trunc=trunc, symmetric=symmetric))

    return specs

def _choose_level_to_knots(pdf_specs):
    """
    Select a global level->knots function.
    - If any dimension is uniform (Clenshaw-Curtis), use doubling for nestedness.
    - Else, if any symmetric Leja is requested, use 2-step (odd counts).
    - Otherwise, linear is fine for Leja.
    """
    names = []
    symmetric_flags = []
    for spec in pdf_specs:
        names.append(spec.name.lower())
        symmetric_flags.append(bool(spec.symmetric))
    if "uniform" in names:
        return lev2knots_doubling
    if True in symmetric_flags:
        return lev2knots_2step
    return lev2knots_lin

def _grid_settings(p_def, sg_settings):
    """Resolve the grid family once, consistently across the three phases."""
    pdf_specs = _parse_pdf_specs(p_def)
    knots_funcs, domains = build_1d_knots_and_domains_from_pdfspecs(pdf_specs)
    level_to_knots = _choose_level_to_knots(pdf_specs)
    grid_type = str(sg_settings.get("grid_type", "auto")).lower()

    # The automatic setting preserves the original PDF-based rules. Explicit
    # CC and tabulated Leja grids also work with weighted PDF integration.
    if grid_type == "cc":
        knots_funcs = [knots_cc_1d] * len(domains)
        level_to_knots = lev2knots_doubling
    elif grid_type == "leja_line":
        knots_funcs = [knots_leja_line_1d] * len(domains)
        level_to_knots = lev2knots_lin
    elif grid_type == "leja_sym":
        knots_funcs = [knots_leja_sym_line_1d] * len(domains)
        level_to_knots = lev2knots_lin
    elif grid_type != "auto":
        raise ValueError("grid_type must be auto, CC, leja_line, or leja_sym.")

    base_level = sg_settings["base_level"]
    target_level = sg_settings["target_level"]
    auxiliary_level = sg_settings.get("auxiliary_level", base_level - 1)
    if int(base_level) != base_level or base_level < 1:
        raise ValueError("base_level must be a positive integer.")
    if int(target_level) != target_level or target_level <= base_level:
        raise ValueError("target_level must be an integer above base_level.")
    if int(auxiliary_level) != auxiliary_level or not 0 <= auxiliary_level < base_level:
        raise ValueError("auxiliary_level must be an integer below base_level.")
    if grid_type in ("leja_line", "leja_sym") and target_level + 1 > MAX_LEJA_NODES:
        raise ValueError(f"The target grid needs more than the {MAX_LEJA_NODES} available Leja nodes.")
    for a, b in domains:
        if not np.isfinite(a) or not np.isfinite(b) or b <= a:
            raise ValueError("Every domain must have finite bounds with max > min.")
    return pdf_specs, knots_funcs, domains, level_to_knots


def _build_pdf_functions(pdf_specs):
    """
    Build per-dimension PDF callables for weighted integration.
    Returns (pdf_funcs, pdf_keys, domains).
    """
    funcs = []
    keys = []
    domains = []

    for spec in pdf_specs:
        dom, rho = pdfspec_to_domain_and_rho(spec)
        domains.append(dom)
        funcs.append(rho)
        keys.append(f"{spec.name}:{spec.params}:{spec.trunc}:{spec.symmetric}")

    return funcs, keys, domains

def _has_negative_weights(weights):
    """True if a quadrature rule contains clearly negative weights.

    Smolyak combination weights are often negative in two or more dimensions.
    They then do not define a probability distribution: their cumulative sum
    is not monotone and can leave the interval [0, 1]. Values below a tiny
    fraction of the largest weight are treated as rounding noise.
    """
    weights = np.asarray(weights, dtype=float)
    if weights.size == 0:
        return False
    return bool(np.any(weights < -1e-12 * np.max(np.abs(weights))))

def _exceedance_probability(values, weights, threshold):
    """P(value > threshold) from quadrature weights, or NaN if not meaningful.

    With negative weights the sum can be negative or larger than one, so it is
    not reported as a probability.
    """
    if _has_negative_weights(weights):
        return np.nan
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    return float(np.sum(weights[values > threshold]))

def _weighted_quantile(values, weights, quantiles):
    """Compute weighted quantiles for 1D values.

    Returns NaN when the weights contain negative values: the cumulative
    weight is then not a distribution function and np.interp, which needs an
    increasing abscissa, would return meaningless numbers.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.size == 0:
        return np.array([])
    if _has_negative_weights(weights):
        return np.full(len(quantiles), np.nan)
    idx = np.argsort(values)
    v = values[idx]
    w = weights[idx]
    cdf = np.cumsum(w)
    if cdf[-1] <= 0:
        return np.full(len(quantiles), np.nan)
    cdf /= cdf[-1]
    return np.interp(quantiles, cdf, v)

def _weighted_stats(values, weights, quantiles):
    """Compute mean, var, std, skewness, and excess kurtosis with weights."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.size == 0:
        return {}
    wsum = np.sum(weights)
    if wsum <= 0:
        return {}
    w = weights / wsum
    mean = np.sum(w * values)
    diff = values - mean
    var = np.sum(w * diff ** 2)
    # With negative quadrature weights the estimated variance can itself be
    # negative; then no standard deviation or higher moment is reported.
    if var < 0:
        std = np.nan
    else:
        std = np.sqrt(var)
    if std > 0:
        skew = np.sum(w * diff ** 3) / (std ** 3)
        kurt = np.sum(w * diff ** 4) / (std ** 4) - 3.0
    else:
        skew = np.nan
        kurt = np.nan
    qs = _weighted_quantile(values, w, quantiles)
    return {
        "mean": mean,
        "var": var,
        "std": std,
        "skew": skew,
        "kurtosis_excess": kurt,
        "quantiles": qs,
    }

def _statistics_row(first_columns, values, weights, quantiles, thresholds):
    """
    One row of a statistics table: the identifying columns, the weighted
    moments, the requested quantiles, and P(value > threshold) for each
    threshold. Columns keep this order in the CSV files.
    """
    stats = _weighted_stats(values, weights, quantiles)
    row = dict(first_columns)
    for key in ["mean", "var", "std", "skew", "kurtosis_excess"]:
        row[key] = stats.get(key)
    for qv, qn in zip(stats.get("quantiles", []), quantiles):
        row[f"q{qn:.3f}"] = qv
    for thr in thresholds:
        row[f"P_gt_{thr}"] = _exceedance_probability(values, weights, thr)
    return row

def _infer_vector_label(component_name):
    """
    Extract a label from a component name like 'qoi_t0003' -> 't'.
    Returns empty string if not found.
    """
    parts = component_name.split("_")
    if len(parts) < 2:
        return ""
    token = parts[-1]
    label = token.rstrip("0123456789")
    return label

def _dump_lagrange_coefficients(model, S, dim_names, level_label, qty, output_dir):
    """
    Dump Lagrange coefficients for each tensor grid in S.
    For each node y_j of each tensor grid, we export:
      - grid multi-index
      - grid coefficient (Smolyak)
      - node index
      - node coordinates
      - f(node)
      - combined coefficient = grid_coeff * f(node)
    """
    unique_knots = model["unique_knots"]
    f_values = model["f_on_unique_knots"]
    # Lookup from rounded node coordinates to the nodal value.
    f_map = {}
    for node, value in zip(unique_knots.T, f_values):
        f_map[tuple(np.round(node, 12))] = float(value)

    rows = []
    for grid in S:
        grid_coeff = float(grid.get("coeff", 0.0))
        idx_vec = grid.get("idx")
        # Multi-index written as text, e.g. "1,2,1".
        idx_str = ""
        if idx_vec is not None:
            idx_parts = []
            for x in idx_vec:
                idx_parts.append(str(int(x)))
            idx_str = ",".join(idx_parts)

        for j in range(int(grid["size"])):
            node = grid["knots"][:, j]
            node_key = tuple(np.round(node, 12))
            f_val = f_map.get(node_key, np.nan)
            # NaN times a number is NaN, so a missing value stays missing.
            combined = grid_coeff * f_val

            row = {
                "grid_idx": idx_str,
                "grid_coeff": grid_coeff,
                "node_index": j,
                "f_value": f_val,
                "combined_coeff": combined,
            }
            for d, name in enumerate(dim_names):
                row[name] = node[d]
            rows.append(row)

    if not rows:
        return

    poly_dir = os.path.join(output_dir, "polynomials")
    os.makedirs(poly_dir, exist_ok=True)
    out_file = os.path.join(poly_dir, f"lagrange_coeffs_w{level_label}_{qty}.csv")
    pd.DataFrame(rows).to_csv(out_file, index=False)

# =============================================================================
# Robust I/O + data hygiene
# =============================================================================

def _normalize_header_tokens(line):
    """Normalize a header line by collapsing whitespace and joining with tabs."""
    tokens = re.split(r"\s+", line.strip())
    return "\t".join(tokens)

def _read_to_fill_file(path, expected_columns, sep_regex=r"\s+"):
    """
    Robust reader for *_to_fill.dat files that may contain:
      - tabs OR spaces as separators
      - comment lines starting with '#'
      - blank lines
      - repeated header lines (e.g. multiple sections)

    It returns ONLY the expected columns (in the given order),
    coercing everything to numeric (non-numeric -> NaN).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r") as f:
        raw_lines = f.readlines()

    # 1) strip, drop comments and blank lines
    lines = []
    for ln in raw_lines:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("#"):
            continue
        lines.append(s)

    if not lines:
        raise ValueError(f"File '{path}' contains no data lines after removing comments/blank lines.")

    # 2) find a header that matches expected columns (in exact order)
    expected_header = "\t".join(expected_columns)
    norm_lines = [_normalize_header_tokens(ln) for ln in lines]

    header_idx = None
    for i, nh in enumerate(norm_lines):
        if nh == expected_header:
            header_idx = i
            break
    if header_idx is None:
        # fallback: first non-comment line is assumed header
        header_idx = 0

    header_norm = norm_lines[header_idx]
    header_tokens = header_norm.split("\t")

    missing = []
    for c in expected_columns:
        if c not in header_tokens:
            missing.append(c)
    if missing:
        raise ValueError(
            f"File '{path}': header missing columns {missing}. Found columns: {header_tokens}"
        )

    # 3) rebuild content with exactly one header, and remove repeated headers later
    out_lines = [header_norm]
    for ln, nh in zip(lines[header_idx + 1:], norm_lines[header_idx + 1:]):
        if nh == header_norm:
            continue
        out_lines.append(ln)

    # 4) parse with whitespace regex -> accepts TAB or spaces
    buf = io.StringIO("\n".join(out_lines))
    df = pd.read_csv(buf, sep=sep_regex, engine="python")

    # 5) keep only expected columns and coerce to numeric
    df = df[expected_columns].copy()
    for c in expected_columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def _filter_fully_filled_rows(df, dim_names, qoi_columns):
    """Keep only rows where all dimensions and all QoI columns are present (not NaN)."""
    required = dim_names + qoi_columns
    mask = ~df[required].isnull().any(axis=1)
    return df.loc[mask].copy()

def _deduplicate_by_point(df, dim_names):
    """
    Deduplicate rows by a rounded coordinate key (ROUND_DECIMALS).
    The first occurrence is kept, so the baseline wins when
    df = concat([base, ref]).
    """
    seen_keys = set()
    keep_row = []
    for point in df[dim_names].values.astype(float):
        key = tuple(np.round(point, ROUND_DECIMALS))
        keep_row.append(key not in seen_keys)
        seen_keys.add(key)
    return df.loc[np.array(keep_row, dtype=bool)].copy()

def _point_key_array(knots_T):
    """knots_T shape: (npts, ndim) -> set of rounded tuple keys."""
    keys = set()
    for point in np.asarray(knots_T, dtype=float):
        keys.add(tuple(np.round(point, ROUND_DECIMALS)))
    return keys

def _require_filled_file(path_filled, path_template):
    """
    Enforce the "copy then fill" workflow.
    Users must copy *_to_fill.dat to a new file without the suffix and fill that file.
    """
    if os.path.exists(path_filled):
        return path_filled
    if os.path.exists(path_template):
        raise FileNotFoundError(
            f"Filled file not found: '{path_filled}'.\n"
            f"Please copy the template '{path_template}' to '{path_filled}' and fill the QoI columns."
        )
    raise FileNotFoundError(
        f"Neither the filled file nor the template exists.\n"
        f"Expected filled file: '{path_filled}'.\n"
        f"Expected template: '{path_template}'."
    )

def _snap_points_to_grid(points, grid_nodes, domains, relative_tolerance=SNAP_RELATIVE_TOL):
    """
    Replace coordinates read from a file with the exact generated grid nodes.

    points:     (n_points, N) coordinates read from a text file
    grid_nodes: (N, n_nodes) nodes generated by the sparse-grid construction
    domains:    one (min, max) interval per dimension

    A point is matched when its largest coordinate difference from a node,
    measured as a fraction of each domain width, is at most relative_tolerance.
    Returns (snapped_points, matched), where matched is a boolean mask.
    Unmatched points keep their original coordinates.
    """
    points = np.array(points, dtype=float).reshape(-1, len(domains))
    matched = np.zeros(points.shape[0], dtype=bool)
    if points.shape[0] == 0 or grid_nodes.shape[1] == 0:
        return points, matched

    # Scale every dimension to [0, 1], so one tolerance fits all physical units.
    lower = np.array([d[0] for d in domains], dtype=float)
    width = np.array([d[1] - d[0] for d in domains], dtype=float)
    scaled_nodes = (grid_nodes.T - lower) / width
    scaled_points = (points - lower) / width

    # Nearest node in the maximum norm (p=inf): every coordinate must be close.
    distance, nearest = cKDTree(scaled_nodes).query(scaled_points, k=1, p=np.inf)
    matched = distance <= relative_tolerance
    points[matched] = grid_nodes.T[nearest[matched]]
    return points, matched

def _snap_baseline_rows(df_base, dim_names, grid_nodes, domains, path):
    """
    Put the exact baseline-grid coordinates into the baseline table.

    Every row must correspond to one baseline node and every baseline node must
    be present exactly once, because the baseline interpolant needs all of them.
    """
    snapped, matched = _snap_points_to_grid(df_base[dim_names].values, grid_nodes, domains)
    if not np.all(matched):
        first_rows = df_base.loc[~matched, dim_names].head(3).values.tolist()
        raise ValueError(
            f"File '{path}': {int(np.sum(~matched))} rows are not on the baseline grid "
            f"(for example {first_rows}). Keep the coordinates of the generated template."
        )
    present = _point_key_array(snapped)
    if len(present) != len(snapped):
        raise ValueError(f"File '{path}': some baseline points appear more than once.")
    if len(present) != grid_nodes.shape[1]:
        raise ValueError(
            f"File '{path}': {grid_nodes.shape[1] - len(present)} baseline grid points are missing."
        )
    df_base = df_base.copy()
    df_base[dim_names] = snapped
    return df_base

# =============================================================================
# Workflow helpers
# =============================================================================

def _find_elbow_point(sorted_values):
    """Find the elbow point in a descending-sorted array."""
    if len(sorted_values) < 3:
        return len(sorted_values)
    n_points = len(sorted_values)
    all_coords = np.vstack((range(n_points), sorted_values)).T
    first_point, last_point = all_coords[0], all_coords[-1]
    line_vec = last_point - first_point
    line_vec_norm = line_vec / (np.sqrt(np.sum(line_vec**2)) + 1e-12)
    vec_from_first = all_coords - first_point
    scalar_product = np.sum(vec_from_first * np.tile(line_vec_norm, (n_points, 1)), axis=1)
    vec_from_first_parallel = np.outer(scalar_product, line_vec_norm)
    distances = np.sqrt(np.sum((vec_from_first - vec_from_first_parallel)**2, axis=1))
    return int(np.argmax(distances) + 1)

def _build_informed_model_values(
    base_knots,
    base_values,
    all_target_knots,
    all_known_data_df,
    quantity_name,
    dim_names,
    base_grid_structure,
    nn_tolerance=DEFAULT_TOL,
):
    """
    Build nodal values on the TARGET grid:
      1) exact match by rounded key (stable, no tolerance dependence)
      2) fallback: nearest-neighbor snap within nn_tolerance (physical units)
      3) remaining nodes are interpolated from the BASE model
    """
    # Target nodes, one row per node, and a lookup from rounded key to row.
    canonical = all_target_knots.T.astype(float)  # (M, N)
    canon_index = {}
    for i, point in enumerate(canonical):
        canon_index[tuple(np.round(point, ROUND_DECIMALS))] = i

    dfk = all_known_data_df[dim_names + [quantity_name]].copy()
    dfk = dfk.dropna(axis=0, how="any")
    if dfk.empty:
        raise ValueError(f"No valid known rows found for quantity '{quantity_name}'.")

    data_points = dfk[dim_names].values.astype(float)
    data_values = dfk[quantity_name].values.astype(float)

    final_values = np.full(canonical.shape[0], np.nan, dtype=float)

    # 1) exact match by key
    unmatched = []
    for p, val in zip(data_points, data_values):
        k = tuple(np.round(p, ROUND_DECIMALS))
        idx = canon_index.get(k, None)
        if idx is None:
            unmatched.append((p, val))
        else:
            final_values[idx] = val

    # 2) fallback nearest-neighbor snap
    if unmatched:
        tol2 = float(nn_tolerance) ** 2
        snapped = 0
        rejected = 0

        for p, val in unmatched:
            dist2 = np.sum((canonical - p) ** 2, axis=1)
            j = int(np.argmin(dist2))
            if dist2[j] <= tol2:
                final_values[j] = val
                snapped += 1
            else:
                rejected += 1
                print(
                    f"[WARN] '{quantity_name}': point {p} not on target grid "
                    f"(min_dist={np.sqrt(dist2[j]):.3e} > tol={nn_tolerance:.3e}). Ignored."
                )

        print(
            f"[INFO] '{quantity_name}': exact-matched={len(data_points)-len(unmatched)}, "
            f"nn-snapped={snapped}, rejected={rejected}"
        )

    # 3) interpolate missing target nodes from base model
    missing = np.where(np.isnan(final_values))[0]
    if missing.size > 0:
        pts = all_target_knots[:, missing]

        est = evaluate_sparse_grid_interpolant(
            pts,
            base_grid_structure,
            base_knots,
            base_values
        )

        final_values[missing] = np.asarray(est, dtype=float)

        # Diagnostic: values inherited from the base model
        print(f"\n[INFO] '{quantity_name}': values inherited from BASE model "
              f"({missing.size} target nodes)")

    return final_values



# =============================================================================
# Phase 1: generate baseline grid
# =============================================================================

def run_phase_1(config):
    """
    Generate baseline sparse-grid points and save a to_fill file.
    """
    proj_name = config["project_name"]
    p_def = config["problem_definition"]
    sg_settings = config["sparse_grid_settings"]

    dim_names = p_def["dim_names"]
    qoi_specs = _expand_qoi_specs(p_def)
    qoi_columns = _qoi_columns_from_specs(qoi_specs)

    N = len(dim_names)
    w_start = sg_settings["base_level"]
    pdf_specs, knots_funcs, domains, level_to_knots = _grid_settings(p_def, sg_settings)

    S_start = create_sparse_grid_structure(N, w_start, level_to_knots, knots_funcs, domains)
    knots_base = reduce_sparse_grid_knots(S_start)

    print(f"Generated {knots_base.shape[1]} unique points for the baseline grid (level w={w_start}).")

    df_out = pd.DataFrame(knots_base.T, columns=dim_names)
    if qoi_columns:
        df_out = pd.concat([df_out, pd.DataFrame(np.nan, index=df_out.index, columns=qoi_columns)], axis=1)

    output_path = os.path.join(config["data_dir"], f"{proj_name}_phase1_to_fill.dat")
    df_out.to_csv(output_path, sep="\t", index=False)

    print("\nACTION REQUIRED:")
    filled_path = os.path.join(config["data_dir"], f"{proj_name}_phase1.dat")
    print(f"  -> Copy '{output_path}' to '{filled_path}'")
    print(f"  -> Fill the output columns for: {_describe_qois_for_msg(qoi_specs)}")
    print("  -> Then run the script again with --phase 2.")

# =============================================================================
# Phase 2: select refinement points
# =============================================================================

def run_phase_2(config):
    """
    Read filled baseline data, select refinement points, and write phase2_to_fill.dat:
      - automatically selected points (must fill)
      - optional additional target points (you may add/fill as you wish)

    Notes:
      - optional points may be left empty
      - file includes comments and section headers (robust reader handles this)
    """
    proj_name = config["project_name"]
    p_def = config["problem_definition"]
    sg_settings = config["sparse_grid_settings"]
    ref_settings = config["refinement_settings"]
    output_dir = config["output_dir"]

    dim_names = p_def["dim_names"]
    qoi_specs = _expand_qoi_specs(p_def)
    qoi_columns = _qoi_columns_from_specs(qoi_specs)

    input_template = os.path.join(config["data_dir"], f"{proj_name}_phase1_to_fill.dat")
    input_path = os.path.join(config["data_dir"], f"{proj_name}_phase1.dat")
    input_path = _require_filled_file(input_path, input_template)

    # Robust read (accepts spaces, tabs, and comments)
    cols = dim_names + qoi_columns
    df_base = _read_to_fill_file(input_path, expected_columns=cols)

    if df_base[qoi_columns].isnull().values.any():
        raise ValueError(f"The file '{input_path}' contains missing values. Fill all baseline rows before Phase 2.")

    # Grid settings are needed first: file coordinates are matched to the grid.
    N = len(dim_names)
    w_start, w_target = sg_settings["base_level"], sg_settings["target_level"]
    pdf_specs, knots_funcs, domains, level_to_knots = _grid_settings(p_def, sg_settings)

    # Replace the coordinates read from the file with the exact baseline nodes.
    S_start = create_sparse_grid_structure(N, w_start, level_to_knots, knots_funcs, domains)
    df_base = _snap_baseline_rows(df_base, dim_names, reduce_sparse_grid_knots(S_start), domains, input_path)

    knots_base = df_base[dim_names].values.T
    ref_criterion_name = ref_settings["refinement_criterion"]
    values_base_criterion = _refinement_values_from_qoi(df_base, ref_criterion_name, qoi_specs)

    # Candidate points = target grid minus base grid
    S_target = create_sparse_grid_structure(N, w_target, level_to_knots, knots_funcs, domains)
    knots_target_total = reduce_sparse_grid_knots(S_target)

    base_set = _point_key_array(knots_base.T)

    candidate_indices = []
    for index, point in enumerate(knots_target_total.T):
        if tuple(np.round(point, ROUND_DECIMALS)) not in base_set:
            candidate_indices.append(index)
    candidate_knots = knots_target_total[:, candidate_indices]
    candidate_tuples = []
    for point in candidate_knots.T:
        candidate_tuples.append(tuple(point))

    if candidate_knots.shape[1] == 0:
        print("No new candidate points found for refinement.")

    # Compute discrepancy indicator
    print(f"Analyzing {candidate_knots.shape[1]} candidate points for refinement based on '{ref_criterion_name}'...")
    indicators = compute_discrepancy_indicator(
        candidate_knots,
        w_start,
        N,
        domains,
        knots_base,
        values_base_criterion,
        knots_func_1d=knots_funcs,
        level_to_knots_func=level_to_knots,
        indicator_type=ref_settings.get("indicator_type", "legacy"),
        relative_epsilon=float(ref_settings.get("relative_epsilon", 1e-5)),
        auxiliary_level=sg_settings.get("auxiliary_level", w_start - 1)
    )

    # Determine the requested number of new target evaluations.
    candidate_count = len(indicators)
    indicator_cutoff = None
    # Without max_budget: automatic modes are uncapped, fixed-count mode selects nothing.
    if ref_settings.get("use_auto_budget", False):
        default_max_budget = candidate_count
    else:
        default_max_budget = 0
    max_budget = ref_settings.get("max_budget", default_max_budget)
    if not np.isfinite(max_budget) or int(max_budget) != max_budget or max_budget < 0:
        raise ValueError("max_budget must be a non-negative integer.")
    final_budget = min(int(max_budget), candidate_count)
    if ref_settings.get("use_auto_budget", False):
        method = ref_settings.get("auto_budget_method", "elbow")
        if method == "elbow":
            final_budget = min(_find_elbow_point(np.sort(indicators)[::-1]), final_budget)
        elif method == "threshold":
            fraction = float(ref_settings.get("auto_budget_threshold", 0.1))
            if not np.isfinite(fraction) or not 0 <= fraction <= 1:
                raise ValueError("auto_budget_threshold must lie between zero and one.")
            cutoff = 0.0
            if candidate_count:
                cutoff = fraction * np.max(indicators)
            above_count = int(np.sum(indicators >= cutoff))
            if above_count <= final_budget:
                indicator_cutoff = cutoff
            # If a separate count cap is active, use the capped rank cutoff.
            final_budget = min(above_count, final_budget)
        else:
            raise ValueError("auto_budget_method must be elbow or threshold.")
    elif "budget_percent" in ref_settings:
        percent = float(ref_settings["budget_percent"])
        if not np.isfinite(percent) or not 0 <= percent <= 100:
            raise ValueError("budget_percent must lie between zero and 100.")
        # Round the requested fraction down to a whole number of evaluations.
        requested = int(np.floor(candidate_count * percent / 100.0))
        final_budget = min(requested, int(ref_settings.get("max_budget", candidate_count)))

    nominal_budget = final_budget
    best_local_indices = select_refinement_points(
        indicators, nominal_budget,
        include_ties=ref_settings.get("include_ties", True),
        tie_tolerance=float(ref_settings.get("tie_tolerance", 1e-3)),
        indicator_cutoff=indicator_cutoff,
    )
    final_budget = len(best_local_indices)
    print(f"Nominal budget: {nominal_budget}; actual selected points: {final_budget}.")
    if best_local_indices.size:
        selected_knots_arr = candidate_knots[:, best_local_indices]
    else:
        selected_knots_arr = np.empty((N, 0), dtype=float)

    print(f"\nSelected {selected_knots_arr.shape[1]} new points for refinement.")

    # Save both the candidate ranking and actual selection. These files also
    # let Phase 3 check that every required evaluation has been supplied.
    os.makedirs(output_dir, exist_ok=True)
    selected_mask = np.zeros(candidate_count, dtype=bool)
    selected_mask[best_local_indices] = True
    diagnostics = pd.DataFrame(candidate_knots.T, columns=dim_names)
    diagnostics["indicator"] = indicators
    diagnostics["selected"] = selected_mask
    diagnostics = diagnostics.sort_values("indicator", ascending=False, kind="stable")
    diagnostics.to_csv(os.path.join(output_dir, f"{proj_name}_selection.csv"), index=False)
    selection_cutoff = 0.0
    if nominal_budget > 0:
        if indicator_cutoff is not None:
            selection_cutoff = float(indicator_cutoff)
        else:
            selection_cutoff = float(np.sort(indicators)[-nominal_budget])
    summary = {
        "grid_type": sg_settings.get("grid_type", "auto"),
        "indicator_type": ref_settings.get("indicator_type", "legacy"),
        "relative_epsilon": float(ref_settings.get("relative_epsilon", 1e-5)),
        "include_ties": bool(ref_settings.get("include_ties", True)),
        "tie_tolerance": float(ref_settings.get("tie_tolerance", 1e-3)),
        "base_level": w_start,
        "auxiliary_level": sg_settings.get("auxiliary_level", w_start - 1),
        "target_level": w_target,
        "baseline_count": len(df_base),
        "candidate_count": candidate_count,
        "nominal_budget": nominal_budget,
        "selection_cutoff": selection_cutoff,
        "selected_count": final_budget,
        "total_evaluations": len(df_base) + final_budget,
    }
    with open(os.path.join(output_dir, f"{proj_name}_selection.json"), "w") as file:
        json.dump(summary, file, indent=2)

    # Discrepancy plot
    print("\nGenerating discrepancy plot to visualize the selection...")
    os.makedirs(output_dir, exist_ok=True)
    plot_discrepancy_by_point(
        candidate_tuples=candidate_tuples,
        indicators=indicators,
        budget=final_budget,
        refinement_criterion=ref_criterion_name,
        save_path=os.path.join(output_dir, f"{proj_name}_phase2_discrepancy_plot.png")
    )

    # Build output file sections
    df_selected = pd.DataFrame(selected_knots_arr.T, columns=dim_names)
    if qoi_columns:
        df_selected = pd.concat([df_selected, pd.DataFrame(np.nan, index=df_selected.index, columns=qoi_columns)], axis=1)

    # OPTIONAL points: everything in target that is not in base and not in selected
    remaining_knots = candidate_knots[:, ~selected_mask].T
    if remaining_knots.size:
        df_remaining = pd.DataFrame(remaining_knots, columns=dim_names)
    else:
        df_remaining = pd.DataFrame(columns=dim_names)
    if qoi_columns:
        df_remaining = pd.concat([df_remaining, pd.DataFrame(np.nan, index=df_remaining.index, columns=qoi_columns)], axis=1)

    output_path = os.path.join(config["data_dir"], f"{proj_name}_phase2_to_fill.dat")

    with open(output_path, "w") as f:
        f.write("# ==============================================================\n")
        f.write("# AUTOMATICALLY SELECTED REFINEMENT POINTS\n")
        f.write("# These points were selected by the adaptive refinement algorithm.\n")
        f.write("# You should fill ALL QoIs for these points.\n")
        f.write("# ==============================================================\n")
        df_selected.to_csv(f, sep="\t", index=False)
        f.write("\n")
        f.write("# ==============================================================\n")
        f.write("# OPTIONAL ADDITIONAL TARGET GRID POINTS\n")
        f.write("# You may add and/or fill any points below at your discretion.\n")
        f.write("# Leave QoIs empty for points you do not want to use.\n")
        f.write("# ==============================================================\n")
        df_remaining.to_csv(f, sep="\t", index=False)

    print("\nACTION REQUIRED:")
    filled_path = os.path.join(config["data_dir"], f"{proj_name}_phase2.dat")
    print(f"  -> Copy '{output_path}' to '{filled_path}'")
    print(f"  -> Fill all automatically selected points for: {_describe_qois_for_msg(qoi_specs)}")
    print("  -> Optionally, add/fill any additional points below.")
    print("  -> Then run the script again with --phase 3.")

# =============================================================================
# Phase 3: final analysis and plotting
# =============================================================================

def run_phase_3(config):
    """
    Load data (phase 1 + phase 2), build base + informed surrogates,
    optional validation, and plotting.
    """
    print("\n--- Starting Final Analysis and Plotting ---")

    proj_name = config["project_name"]
    p_def = config["problem_definition"]
    sg_settings = config["sparse_grid_settings"]
    output_dir = config["output_dir"]

    dim_names = p_def["dim_names"]
    qoi_specs = _expand_qoi_specs(p_def)
    qoi_columns = _qoi_columns_from_specs(qoi_specs)
    cols = dim_names + qoi_columns

    base_template = os.path.join(config["data_dir"], f"{proj_name}_phase1_to_fill.dat")
    ref_template  = os.path.join(config["data_dir"], f"{proj_name}_phase2_to_fill.dat")
    path_base = os.path.join(config["data_dir"], f"{proj_name}_phase1.dat")
    path_ref  = os.path.join(config["data_dir"], f"{proj_name}_phase2.dat")
    path_base = _require_filled_file(path_base, base_template)
    path_ref  = _require_filled_file(path_ref, ref_template)

    df_base_raw = _read_to_fill_file(path_base, expected_columns=cols)
    df_ref_raw  = _read_to_fill_file(path_ref,  expected_columns=cols)

    if df_base_raw[qoi_columns].isnull().values.any():
        raise ValueError(f"Phase 3: baseline file '{path_base}' has missing QoIs. Fill it completely.")

    df_base = _filter_fully_filled_rows(df_base_raw, dim_names, qoi_columns)
    df_ref  = _filter_fully_filled_rows(df_ref_raw,  dim_names, qoi_columns)

    if len(df_ref_raw) > len(df_ref):
        print(f"[INFO] Ignored {len(df_ref_raw)-len(df_ref)} Phase-2 rows with missing QoIs (optional/unfilled).")

    # Grid structures (needed before matching file coordinates to grid nodes)
    N = len(dim_names)
    w_start, w_target = sg_settings["base_level"], sg_settings["target_level"]
    pdf_specs, knots_funcs, domains, level_to_knots = _grid_settings(p_def, sg_settings)

    S_start  = create_sparse_grid_structure(N, w_start,  level_to_knots, knots_funcs, domains)
    S_target = create_sparse_grid_structure(N, w_target, level_to_knots, knots_funcs, domains)
    knots_target_total = reduce_sparse_grid_knots(S_target)

    # Replace file coordinates with the exact grid nodes. Baseline rows must
    # cover the baseline grid. Phase-2 rows away from the target grid keep
    # their coordinates and are reported and ignored when the model is built.
    df_base = _snap_baseline_rows(df_base, dim_names, reduce_sparse_grid_knots(S_start), domains, path_base)
    snapped_ref, _ = _snap_points_to_grid(df_ref[dim_names].values, knots_target_total, domains)
    df_ref = df_ref.copy()
    df_ref[dim_names] = snapped_ref

    selection_path = os.path.join(output_dir, f"{proj_name}_selection.csv")
    if os.path.exists(selection_path):
        selection = pd.read_csv(selection_path)
        # The CSV parser can also change the last binary digit: snap these too.
        selected_points, _ = _snap_points_to_grid(
            selection.loc[selection["selected"], dim_names].values, knots_target_total, domains
        )
        required_points = _point_key_array(selected_points)
        filled_points = _point_key_array(df_ref[dim_names].values)
        if required_points - filled_points:
            raise ValueError("Phase 3: some automatically selected refinement points are missing or unfilled.")

    # Merge data; baseline wins on duplicates
    df_all = pd.concat([df_base, df_ref], ignore_index=True)
    df_all = _deduplicate_by_point(df_all, dim_names)

    print(f"Loaded {len(df_base)} baseline points and {len(df_ref)} refinement points (fully filled).")

    knots_base         = df_base[dim_names].values.T

    # Build models
    models = {}
    for qty in qoi_columns:
        print(f"Building models for quantity: '{qty}'")

        values_base = df_base[qty].values.astype(float)

        values_informed = _build_informed_model_values(
            base_knots=knots_base,
            base_values=values_base,
            all_target_knots=knots_target_total,
            all_known_data_df=df_all,
            quantity_name=qty,
            dim_names=dim_names,
            base_grid_structure=S_start,
        )

        models[qty] = {
            "base":     {"S": S_start,  "unique_knots": knots_base,         "f_on_unique_knots": values_base},
            "informed": {"S": S_target, "unique_knots": knots_target_total, "f_on_unique_knots": values_informed},
        }

        # Export Lagrange coefficients for polynomial interpolants (w and w+1)
        export_lagrange = bool(config.get("statistics", {}).get("export_lagrange_coeffs", True))
        if export_lagrange:
            _dump_lagrange_coefficients(models[qty]["base"], S_start, dim_names, level_label=w_start, qty=qty, output_dir=output_dir)
            _dump_lagrange_coefficients(models[qty]["informed"], S_target, dim_names, level_label=w_target, qty=qty, output_dir=output_dir)

    # Optional validation
    validation_file = config.get("validation", {}).get("validation_data_file")
    if validation_file and os.path.exists(validation_file):
        print(f"\nValidation file found: '{validation_file}'. Calculating model errors...")

        try:
            if validation_file.endswith((".dat", ".tsv")):
                df_test = pd.read_csv(validation_file, sep=r"\s+", engine="python")
            else:
                df_test = pd.read_csv(validation_file)
        except Exception as e:
            raise ValueError(f"Could not read validation file '{validation_file}': {e}")

        test_points = df_test[dim_names].values.T.astype(float)

        validation_metrics = []
        for qty in qoi_columns:
            if qty not in df_test.columns:
                continue
            truth = df_test[qty].values.astype(float)
            if not np.all(np.isfinite(truth)):
                raise ValueError(f"Validation values for '{qty}' must be finite.")
            for model_name in ("base", "informed"):
                prediction = evaluate_sparse_grid_interpolant(test_points, **models[qty][model_name])
                error = prediction - truth
                validation_metrics.append({
                    "quantity": qty, "model": model_name, "point_count": len(truth),
                    "rmse": float(np.sqrt(np.mean(error ** 2))),
                    "mae": float(np.mean(np.abs(error))),
                    "max_absolute_error": float(np.max(np.abs(error))),
                    "p95_absolute_error": float(np.quantile(np.abs(error), 0.95)),
                    "bias": float(np.mean(error)),
                })
        os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame(validation_metrics).to_csv(
            os.path.join(output_dir, f"{proj_name}_validation_metrics.csv"), index=False
        )

        plot_columns = _resolve_plot_columns(config.get("plotting", {}), qoi_specs)
        for qty in plot_columns:
            if qty not in df_test.columns:
                print(f"  - WARNING: Quantity '{qty}' not found in validation file. Skipping error plots for it.")
                continue

            true_values = df_test[qty].values.astype(float)

            pred_base = evaluate_sparse_grid_interpolant(test_points, **models[qty]["base"])
            pred_inf  = evaluate_sparse_grid_interpolant(test_points, **models[qty]["informed"])

            # Normalize percentage errors by the reference range over the test set.
            # A constant reference has no such range; absolute metrics remain valid.
            reference_range = float(np.max(true_values) - np.min(true_values))
            if reference_range == 0.0:
                print(f"  - Constant reference for '{qty}': percentage plots skipped; absolute metrics are available.")
                continue
            error_base = 100.0 * np.abs(pred_base - true_values) / reference_range
            error_inf = 100.0 * np.abs(pred_inf - true_values) / reference_range

            print(f"  - Average Error for '{qty}': Baseline={error_base.mean():.2f}%, Informed={error_inf.mean():.2f}%")

            df_errors = pd.DataFrame({
                "TestID": df_test.index,
                "Baseline Model Error [%]": error_base,
                "Informed Model Error [%]": error_inf
            })

            os.makedirs(output_dir, exist_ok=True)
            plot_error_comparison_barchart(df_errors, qty, save_path=os.path.join(output_dir, f"error_barchart_{qty}.png"))
            plot_error_distribution(error_base, error_inf, qty, save_path=os.path.join(output_dir, f"error_distribution_{qty}.png"))
    else:
        print("\nNo validation file specified or found. Skipping error analysis plots.")

    # Final plots
    print("\nGenerating final plots...")
    os.makedirs(output_dir, exist_ok=True)

    # Selected knots for plotting: only filled refinement rows
    if len(df_ref):
        selected_knots_arr = df_ref[dim_names].values.T.astype(float)
    else:
        selected_knots_arr = np.empty((N, 0), dtype=float)

    plot_columns = _resolve_plot_columns(config.get("plotting", {}), qoi_specs)
    for slice_info in config["plotting"]["slices_2d"]:
        # File-name label of the slice, e.g. {"x3": 0.5} -> "x30p5".
        slice_parts = []
        for k, v in slice_info["fixed"].items():
            slice_parts.append(f"{k}{str(v).replace('.', 'p')}")
        slice_id = "_".join(slice_parts)

        for qty in plot_columns:
            plot_combined_response_surfaces(
                slice_info,
                domains,
                dim_names,
                config["plotting"]["plot_resolution"],
                base_model=models[qty]["base"],
                informed_model=models[qty]["informed"],
                val_name=qty,
                save_path=os.path.join(output_dir, f"response_surface_{qty}_{slice_id}.png")
            )

        # Use the same slice as the response surfaces: inputs that are neither
        # varying nor fixed are set to their domain midpoint (not projected).
        fixed_for_points = dict(slice_info["fixed"])
        for i, dim in enumerate(dim_names):
            if dim not in slice_info["varying"] and dim not in fixed_for_points:
                fixed_for_points[dim] = 0.5 * (domains[i][0] + domains[i][1])

        plot_quadrature_points_slice(
            base_knots=knots_base,
            target_knots=knots_target_total,
            selected_knots=selected_knots_arr,
            dim_names=dim_names,
            varying_dims=slice_info["varying"],
            fixed_dims=fixed_for_points,
            save_path=os.path.join(output_dir, f"quadrature_points_{slice_id}.png")
        )

    # ------------------------------------------------------------
    # Weighted statistics (optional)
    # ------------------------------------------------------------
    stats_cfg = config.get("statistics", {})
    if stats_cfg.get("enabled", True):
        print("\nComputing weighted statistics...")
        quantiles = stats_cfg.get("quantiles", [0.05, 0.5, 0.95])
        quad_order = int(stats_cfg.get("quad_order", 200))
        vector_metrics = stats_cfg.get("vector_metrics", ["l2", "mean", "rms", "max"])
        plot_vector_mean_std_enabled = bool(stats_cfg.get("plot_vector_mean_std", True))
        exceedance_cfg = stats_cfg.get("exceedance", [])

        pdf_funcs, pdf_keys, pdf_domains = _build_pdf_functions(pdf_specs)
        weights = compute_sparse_grid_quadrature_weights(
            S_target,
            knots_target_total,
            pdf_funcs,
            pdf_domains,
            quad_order=quad_order,
            pdf_keys=pdf_keys,
            normalize=True,
        )

        # Mean and moments are quadrature estimates and remain available.
        # Quantiles and exceedance probabilities need a probability
        # distribution, which signed Smolyak weights do not provide.
        if _has_negative_weights(weights):
            print(
                f"[WARN] {int(np.sum(weights < 0))} of {weights.size} sparse-grid quadrature weights "
                "are negative: quantiles and exceedance probabilities are written as NaN."
            )

        stats_dir = os.path.join(output_dir, "statistics")
        os.makedirs(stats_dir, exist_ok=True)

        # group exceedance thresholds by target name
        exceedance_map = {}
        for item in exceedance_cfg:
            name = item.get("qoi")
            thr = float(item.get("threshold"))
            metric = item.get("metric", "component")
            exceedance_map.setdefault((name, metric), []).append(thr)

        for spec in qoi_specs:
            if not spec["is_vector"]:
                qty = spec["columns"][0]
                vals = models[qty]["informed"]["f_on_unique_knots"]
                row = _statistics_row(
                    {"quantity": qty}, vals, weights, quantiles,
                    exceedance_map.get((qty, "component"), []),
                )
                pd.DataFrame([row]).to_csv(
                    os.path.join(stats_dir, f"stats_{qty}.csv"), index=False
                )
            else:
                name = spec["name"]
                cols = spec["columns"]
                # Nodal values: one row per target node, one column per component.
                component_values = []
                for c in cols:
                    component_values.append(models[c]["informed"]["f_on_unique_knots"])
                vals = np.vstack(component_values).T

                # per-component stats
                rows = []
                comp_indices = []
                comp_means = []
                comp_stds = []
                for j, col in enumerate(cols):
                    row = _statistics_row(
                        {"index": j, "label": _infer_vector_label(col), "component": col},
                        vals[:, j], weights, quantiles,
                        exceedance_map.get((name, "component"), []),
                    )
                    rows.append(row)
                    comp_indices.append(j)
                    comp_means.append(row["mean"])
                    comp_stds.append(row["std"])

                pd.DataFrame(rows).to_csv(
                    os.path.join(stats_dir, f"stats_{name}_components.csv"), index=False
                )

                # Mean ± std plot along the vector index
                if comp_indices and plot_vector_mean_std_enabled:
                    plot_vector_mean_std(
                        component_indices=np.array(comp_indices),
                        mean_values=np.array(comp_means, dtype=float),
                        std_values=np.array(comp_stds, dtype=float),
                        qty_name=name,
                        save_path=os.path.join(stats_dir, f"stats_{name}_mean_std.png"),
                        xlabel=_infer_vector_label(cols[0]) or "Vector index"
                    )

                # aggregate vector metrics
                agg_rows = []
                metrics = {}
                if "l2" in vector_metrics:
                    metrics["l2"] = np.linalg.norm(vals, axis=1)
                if "mean" in vector_metrics:
                    metrics["mean"] = np.mean(vals, axis=1)
                if "rms" in vector_metrics:
                    metrics["rms"] = np.sqrt(np.mean(vals ** 2, axis=1))
                if "max" in vector_metrics:
                    metrics["max"] = np.max(vals, axis=1)

                for metric_name, metric_vals in metrics.items():
                    row = _statistics_row(
                        {"metric": metric_name}, metric_vals, weights, quantiles,
                        exceedance_map.get((name, metric_name), []),
                    )
                    agg_rows.append(row)

                pd.DataFrame(agg_rows).to_csv(
                    os.path.join(stats_dir, f"stats_{name}_aggregates.csv"), index=False
                )

    print(f"\nPhase 3 complete. Results saved to '{output_dir}'.")

