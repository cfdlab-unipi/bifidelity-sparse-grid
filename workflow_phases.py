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
import io
import re
import numpy as np
import pandas as pd

# --- Import from your custom toolboxes ---
from sparse_grid_toolbox import (
    PdfSpec,
    create_sparse_grid_structure, reduce_sparse_grid_knots,
    evaluate_sparse_grid_interpolant, compute_discrepancy_indicator,
    lev2knots_doubling, lev2knots_lin, lev2knots_2step,
    build_1d_knots_and_domains_from_pdfspecs,
    pdfspec_to_domain_and_rho,
    compute_sparse_grid_quadrature_weights
)
from plotting_tools import (
    plot_error_comparison_barchart, plot_combined_response_surfaces,
    plot_quadrature_points_slice, plot_discrepancy_by_point,
    plot_error_distribution, plot_vector_mean_std
)

# =============================================================================
# Global settings
# =============================================================================

ROUND_DECIMALS = 6  # for deterministic point keys (set operations, dedupe)
DEFAULT_TOL = 1e-6   # matching tolerance in physical space (snap-to-grid)

# =============================================================================
# QoI helpers (scalar + vector support)
# =============================================================================

def _expand_qoi_specs(p_def: dict) -> list:
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

def _qoi_columns_from_specs(qoi_specs: list) -> list:
    return [c for spec in qoi_specs for c in spec["columns"]]

def _describe_qois_for_msg(qoi_specs: list) -> str:
    parts = []
    for spec in qoi_specs:
        if spec["is_vector"]:
            parts.append(f"{spec['name']} (vector, {spec['size']} components)")
        else:
            parts.append(spec["name"])
    return ", ".join(parts)

def _resolve_plot_columns(plot_cfg: dict, qoi_specs: list) -> list:
    """
    Decide which QoI columns to plot.
    Default: all scalar quantities + vector quantities limited to plot_indices (if provided).
    If plot_cfg["plot_quantities"] is provided, only those are used.
    """
    name_to_spec = {s["name"]: s for s in qoi_specs}
    requested = plot_cfg.get("plot_quantities")
    cols = []

    def _add_from_spec(spec):
        if not spec["is_vector"]:
            cols.extend(spec["columns"])
        else:
            if spec["plot_indices"]:
                for i in spec["plot_indices"]:
                    if 0 <= int(i) < spec["size"]:
                        cols.append(spec["columns"][int(i)])

    if requested:
        for name in requested:
            spec = name_to_spec.get(name)
            if spec is None:
                if name in _qoi_columns_from_specs(qoi_specs):
                    cols.append(name)
                else:
                    raise ValueError(f"plot_quantities contains unknown QoI '{name}'.")
            else:
                _add_from_spec(spec)
    else:
        for spec in qoi_specs:
            _add_from_spec(spec)

    seen = set()
    ordered = []
    for c in cols:
        if c not in seen:
            ordered.append(c)
            seen.add(c)
    return ordered

def _refinement_values_from_qoi(df: pd.DataFrame, ref_name: str, qoi_specs: list) -> np.ndarray:
    """
    Return a 1D array for the refinement criterion.
    - If ref_name is a scalar QoI column -> use it directly.
    - If ref_name is a vector QoI name -> use the L2 norm across components.
    """
    if ref_name in df.columns:
        return df[ref_name].values.astype(float)

    name_to_spec = {s["name"]: s for s in qoi_specs}
    spec = name_to_spec.get(ref_name)
    if spec and spec["is_vector"]:
        vals = df[spec["columns"]].values.astype(float)
        return np.linalg.norm(vals, axis=1)

    raise ValueError(
        f"Unknown refinement_criterion '{ref_name}'. "
        f"Expected a scalar QoI column or a vector QoI name."
    )

def _parse_pdf_specs(p_def: dict) -> list:
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
        if pdf_type in ("uniform", "beta", "triangular"):
            params.setdefault("a", a)
            params.setdefault("b", b)
        else:
            # For unbounded PDFs, use config domains as explicit truncation when provided.
            params.setdefault("a", a)
            params.setdefault("b", b)

        # Basic parameter validation
        if pdf_type == "normal":
            sigma = float(params.get("sigma", 1.0))
            if sigma <= 0.0:
                raise ValueError(f"Normal PDF requires sigma > 0 (dim '{dim_names[i]}').")
        elif pdf_type == "exponential":
            lam = float(params.get("lambda", params.get("lam", 1.0)))
            if lam <= 0.0:
                raise ValueError(f"Exponential PDF requires lambda > 0 (dim '{dim_names[i]}').")
        elif pdf_type == "gamma":
            alpha = float(params.get("alpha", 0.0))
            beta = float(params.get("beta", 1.0))
            if alpha <= -1.0 or beta <= 0.0:
                raise ValueError(f"Gamma PDF requires alpha > -1 and beta > 0 (dim '{dim_names[i]}').")
        elif pdf_type == "beta":
            alpha = float(params.get("alpha", 0.0))
            beta = float(params.get("beta", 0.0))
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

def _choose_level_to_knots(pdf_specs: list) -> callable:
    """
    Select a global level->knots function.
    - If any dimension is uniform (Clenshaw-Curtis), use doubling for nestedness.
    - Else, if any symmetric Leja is requested, use 2-step (odd counts).
    - Otherwise, linear is fine for Leja.
    """
    if any(s.name.lower() == "uniform" for s in pdf_specs):
        return lev2knots_doubling
    if any(bool(s.symmetric) for s in pdf_specs):
        return lev2knots_2step
    return lev2knots_lin

def _build_pdf_functions(pdf_specs: list):
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

def _weighted_quantile(values, weights, quantiles):
    """Compute weighted quantiles for 1D values."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.size == 0:
        return np.array([])
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

def _infer_vector_label(component_name: str) -> str:
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
    f_map = {tuple(np.round(k, 12)): float(v) for k, v in zip(unique_knots.T, f_values)}

    rows = []
    for grid in S:
        grid_coeff = float(grid.get("coeff", 0.0))
        idx_vec = grid.get("idx")
        idx_str = ",".join(str(int(x)) for x in idx_vec) if idx_vec is not None else ""

        for j in range(int(grid["size"])):
            node = grid["knots"][:, j]
            node_key = tuple(np.round(node, 12))
            f_val = f_map.get(node_key, np.nan)
            combined = grid_coeff * f_val if not np.isnan(f_val) else np.nan

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

def _normalize_header_tokens(line: str) -> str:
    """Normalize a header line by collapsing whitespace and joining with tabs."""
    tokens = re.split(r"\s+", line.strip())
    return "\t".join(tokens)

def _read_to_fill_file(path: str, expected_columns: list, sep_regex: str = r"\s+") -> pd.DataFrame:
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

    missing = [c for c in expected_columns if c not in header_tokens]
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

def _filter_fully_filled_rows(df: pd.DataFrame, dim_names: list, qoi_columns: list) -> pd.DataFrame:
    """Keep only rows where all dimensions and all QoI columns are present (not NaN)."""
    required = dim_names + qoi_columns
    mask = ~df[required].isnull().any(axis=1)
    return df.loc[mask].copy()

def _deduplicate_by_point(df: pd.DataFrame, dim_names: list, keep: str = "first") -> pd.DataFrame:
    """
    Deduplicate rows by a rounded coordinate key (ROUND_DECIMALS).
    Default keep='first' makes baseline win when df = concat([base, ref]).
    """
    df2 = df.copy()
    df2["_pt_key"] = df2[dim_names].apply(
        lambda r: tuple(np.round(r.values.astype(float), ROUND_DECIMALS)),
        axis=1
    )
    df2 = df2.drop_duplicates(subset=["_pt_key"], keep=keep).drop(columns=["_pt_key"])
    return df2

def _point_key_array(knots_T: np.ndarray) -> set:
    """knots_T shape: (npts, ndim) -> set of rounded tuple keys."""
    return set(map(tuple, np.round(knots_T.astype(float), ROUND_DECIMALS)))

def _require_filled_file(path_filled: str, path_template: str) -> str:
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

# =============================================================================
# Workflow helpers
# =============================================================================

def _find_elbow_point(sorted_values: np.ndarray) -> int:
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
    base_knots: np.ndarray,
    base_values: np.ndarray,
    all_target_knots: np.ndarray,
    all_known_data_df: pd.DataFrame,
    quantity_name: str,
    dim_names: list,
    base_grid_structure,
    round_decimals: int = ROUND_DECIMALS,
    nn_tolerance: float = DEFAULT_TOL,
    warn: bool = True,
    tolerance: float = None,   # <-- backward-compatible alias
) -> np.ndarray:
    """
    Build nodal values on the TARGET grid:
      1) exact match by rounded key (stable, no tolerance dependence)
      2) fallback: nearest-neighbor snap within nn_tolerance
      3) remaining nodes are interpolated from the BASE model

    Backward compatibility:
      - if caller passes `tolerance=...`, it overrides nn_tolerance.
    """

    if tolerance is not None:
        nn_tolerance = float(tolerance)

    canonical = all_target_knots.T.astype(float)  # (M, N)
    canon_keys = [tuple(np.round(p, round_decimals)) for p in canonical]
    canon_index = {k: i for i, k in enumerate(canon_keys)}

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
        k = tuple(np.round(p, round_decimals))
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
                if warn:
                    print(
                        f"[WARN] '{quantity_name}': point {p} not on target grid "
                        f"(min_dist={np.sqrt(dist2[j]):.3e} > tol={nn_tolerance:.3e}). Ignored."
                    )

        if warn:
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

        for idx, p, v in zip(missing, pts.T, est):
            coords = ", ".join(
                f"{name}={val:.6g}" for name, val in zip(dim_names, p)
            )
            print(f"  - ({coords})  ->  {quantity_name} = {v:.6e}")


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
    pdf_specs = _parse_pdf_specs(p_def)
    knots_funcs, domains = build_1d_knots_and_domains_from_pdfspecs(pdf_specs)
    level_to_knots = _choose_level_to_knots(pdf_specs)

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

    knots_base = df_base[dim_names].values.T
    ref_criterion_name = ref_settings["refinement_criterion"]
    values_base_criterion = _refinement_values_from_qoi(df_base, ref_criterion_name, qoi_specs)

    # Candidate points = target grid minus base grid
    N = len(dim_names)
    w_start, w_target = sg_settings["base_level"], sg_settings["target_level"]
    pdf_specs = _parse_pdf_specs(p_def)
    knots_funcs, domains = build_1d_knots_and_domains_from_pdfspecs(pdf_specs)
    level_to_knots = _choose_level_to_knots(pdf_specs)

    S_target = create_sparse_grid_structure(N, w_target, level_to_knots, knots_funcs, domains)
    knots_target_total = reduce_sparse_grid_knots(S_target)

    base_set = _point_key_array(knots_base.T)
    target_set = _point_key_array(knots_target_total.T)

    candidate_tuples = sorted(list(target_set - base_set))
    candidate_knots = np.array(candidate_tuples, dtype=float).T if candidate_tuples else np.empty((N, 0), dtype=float)

    if candidate_knots.shape[1] == 0:
        print("No new candidate points found for refinement. The grid may already be at the target level.")
        return

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
        level_to_knots_func=level_to_knots
    )

    # Determine budget
    final_budget = 0
    if ref_settings.get("use_auto_budget", False):
        max_budget = int(ref_settings.get("max_budget", len(indicators)))
        method = ref_settings.get("auto_budget_method", "elbow")
        if method == "elbow":
            print("  -> Calculating budget with 'Elbow' method...")
            sorted_ind = np.sort(indicators)[::-1]
            optimal = _find_elbow_point(sorted_ind)
            final_budget = min(optimal, max_budget)
            print(f"     - Elbow suggests: {optimal}. Final budget (cap {max_budget}): {final_budget}")
        elif method == "threshold":
            threshold_perc = float(ref_settings.get("auto_budget_threshold", 0.1))
            print(f"  -> Calculating budget with 'Threshold' method ({threshold_perc*100:.0f}%)...")
            if np.max(indicators) > 1e-12:
                thr = np.max(indicators) * threshold_perc
                num = int(np.sum(indicators >= thr))
                final_budget = min(num, max_budget)
                print(f"     - Threshold value: {thr:.4e}. Points above: {num}. Final budget: {final_budget}")
            else:
                print("     - Max indicator is zero. Budget set to 0.")
                final_budget = 0
        else:
            raise ValueError(f"Unknown auto_budget_method='{method}'")
    else:
        final_budget = min(int(ref_settings.get("max_budget", 0)), len(indicators))
        print(f"  -> Using fixed budget: {final_budget}")

    best_local_indices = np.argsort(indicators)[-final_budget:] if final_budget > 0 else np.array([], dtype=int)
    selected_knots_arr = candidate_knots[:, best_local_indices] if best_local_indices.size else np.empty((N, 0), dtype=float)

    print(f"\nSelected {selected_knots_arr.shape[1]} new points for refinement.")

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

    selected_set = _point_key_array(selected_knots_arr.T) if selected_knots_arr.size else set()

    # OPTIONAL points: everything in target that is not in base and not in selected
    remaining_tuples = sorted(list(target_set - base_set - selected_set))
    remaining_knots = np.array(remaining_tuples, dtype=float)
    df_remaining = pd.DataFrame(remaining_knots, columns=dim_names) if remaining_knots.size else pd.DataFrame(columns=dim_names)
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

    # Merge data; baseline wins on duplicates
    df_all = pd.concat([df_base, df_ref], ignore_index=True)
    df_all = _deduplicate_by_point(df_all, dim_names, keep="first")

    print(f"Loaded {len(df_base)} baseline points and {len(df_ref)} refinement points (fully filled).")

    # Grid structures
    N = len(dim_names)
    w_start, w_target = sg_settings["base_level"], sg_settings["target_level"]
    pdf_specs = _parse_pdf_specs(p_def)
    knots_funcs, domains = build_1d_knots_and_domains_from_pdfspecs(pdf_specs)
    level_to_knots = _choose_level_to_knots(pdf_specs)

    S_start  = create_sparse_grid_structure(N, w_start,  level_to_knots, knots_funcs, domains)
    S_target = create_sparse_grid_structure(N, w_target, level_to_knots, knots_funcs, domains)

    knots_base         = df_base[dim_names].values.T
    knots_target_total = reduce_sparse_grid_knots(S_target)

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
            tolerance=DEFAULT_TOL
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
            print(f"ERROR: Could not read validation file '{validation_file}'. Details: {e}")
            return

        test_points = df_test[dim_names].values.T.astype(float)

        plot_columns = _resolve_plot_columns(config.get("plotting", {}), qoi_specs)
        for qty in plot_columns:
            if qty not in df_test.columns:
                print(f"  - WARNING: Quantity '{qty}' not found in validation file. Skipping error plots for it.")
                continue

            true_values = df_test[qty].values.astype(float)

            pred_base = evaluate_sparse_grid_interpolant(test_points, **models[qty]["base"])
            pred_inf  = evaluate_sparse_grid_interpolant(test_points, **models[qty]["informed"])

            error_base = 100.0 * np.abs(pred_base - true_values) / (np.abs(true_values) + 1e-12)
            error_inf  = 100.0 * np.abs(pred_inf  - true_values) / (np.abs(true_values) + 1e-12)

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
    selected_knots_arr = df_ref[dim_names].values.T.astype(float) if len(df_ref) else np.empty((N, 0), dtype=float)

    plot_columns = _resolve_plot_columns(config.get("plotting", {}), qoi_specs)
    for slice_info in config["plotting"]["slices_2d"]:
        slice_id = "_".join([f"{k}{str(v).replace('.', 'p')}" for k, v in slice_info["fixed"].items()])

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

        plot_quadrature_points_slice(
            base_knots=knots_base,
            target_knots=knots_target_total,
            selected_knots=selected_knots_arr,
            dim_names=dim_names,
            varying_dims=slice_info["varying"],
            fixed_dims=slice_info["fixed"],
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
                stats = _weighted_stats(vals, weights, quantiles)

                row = {
                    "quantity": qty,
                    "mean": stats.get("mean"),
                    "var": stats.get("var"),
                    "std": stats.get("std"),
                    "skew": stats.get("skew"),
                    "kurtosis_excess": stats.get("kurtosis_excess"),
                }
                for qv, qn in zip(stats.get("quantiles", []), quantiles):
                    row[f"q{qn:.3f}"] = qv

                for thr in exceedance_map.get((qty, "component"), []):
                    row[f"P_gt_{thr}"] = np.sum(weights[vals > thr])

                pd.DataFrame([row]).to_csv(
                    os.path.join(stats_dir, f"stats_{qty}.csv"), index=False
                )
            else:
                name = spec["name"]
                cols = spec["columns"]
                vals = np.vstack([models[c]["informed"]["f_on_unique_knots"] for c in cols]).T

                # per-component stats
                rows = []
                comp_indices = []
                comp_means = []
                comp_stds = []
                for j, col in enumerate(cols):
                    stats = _weighted_stats(vals[:, j], weights, quantiles)
                    row = {
                        "index": j,
                        "label": _infer_vector_label(col),
                        "component": col,
                        "mean": stats.get("mean"),
                        "var": stats.get("var"),
                        "std": stats.get("std"),
                        "skew": stats.get("skew"),
                        "kurtosis_excess": stats.get("kurtosis_excess"),
                    }
                    for qv, qn in zip(stats.get("quantiles", []), quantiles):
                        row[f"q{qn:.3f}"] = qv
                    for thr in exceedance_map.get((name, "component"), []):
                        row[f"P_gt_{thr}"] = np.sum(weights[vals[:, j] > thr])
                    rows.append(row)
                    comp_indices.append(j)
                    comp_means.append(stats.get("mean"))
                    comp_stds.append(stats.get("std"))

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
                    stats = _weighted_stats(metric_vals, weights, quantiles)
                    row = {
                        "metric": metric_name,
                        "mean": stats.get("mean"),
                        "var": stats.get("var"),
                        "std": stats.get("std"),
                        "skew": stats.get("skew"),
                        "kurtosis_excess": stats.get("kurtosis_excess"),
                    }
                    for qv, qn in zip(stats.get("quantiles", []), quantiles):
                        row[f"q{qn:.3f}"] = qv
                    for thr in exceedance_map.get((name, metric_name), []):
                        row[f"P_gt_{thr}"] = np.sum(weights[metric_vals > thr])
                    agg_rows.append(row)

                pd.DataFrame(agg_rows).to_csv(
                    os.path.join(stats_dir, f"stats_{name}_aggregates.csv"), index=False
                )

    print(f"\nPhase 3 complete. Results saved to '{output_dir}'.")

