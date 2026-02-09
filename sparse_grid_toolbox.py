"""
Sparse-grid interpolation toolbox.

Key features:
- Total-degree sparse grid construction.
- Clenshaw-Curtis 1D nodes (nested with suitable level-to-knots).
- Weighted Leja 1D nodes matched to user-provided pdfs (nested by construction).
- Robust discrepancy indicator for adaptive refinement.

The weighted Leja sequence is built by recursively maximizing:
    sqrt(rho(t)) * Π_{k=1}^{j-1} |t - t_k|
i.e. score(t) = 0.5*log(rho(t)) + Σ log(|t - t_k|).
"""

from __future__ import annotations

import numpy as np
import itertools
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
from scipy.special import comb


# =============================================================================
# Level-to-knots functions (nestedness-aware)
# =============================================================================

def lev2knots_lin(level: int) -> int:
    """m(i_n)=i_n (not necessarily nested for all knot families, but ok for Leja)."""
    if level <= 0:
        return 0
    return int(level)

def lev2knots_2step(level: int) -> int:
    """m(i_n)=2(i_n-1)+1 => 1,3,5,7,... (good for symmetric Leja)."""
    if level <= 0:
        return 0
    return int(2 * (level - 1) + 1)

def lev2knots_doubling(level: int) -> int:
    """m(1)=1, m(i_n)=2^(i_n-1)+1 => 1,3,5,9,17,... (nested CC, nested trap)."""
    if level <= 0:
        return 0
    if level == 1:
        return 1
    return int(2 ** (level - 1) + 1)

def lev2knots_tripling(level: int) -> int:
    """m(i_n)=3^(i_n-1) (nested midpoint)."""
    if level <= 0:
        return 0
    return int(3 ** (level - 1))


# =============================================================================
# 1D knots: Clenshaw-Curtis (uniform) + Weighted Leja (general pdf)
# =============================================================================

def knots_cc_1d(nn: int, x_a: float, x_b: float) -> np.ndarray:
    """Generate nn Clenshaw-Curtis nodes in [x_a, x_b]."""
    nn = int(nn)
    if nn <= 0:
        return np.array([], dtype=float)
    if nn == 1:
        return np.array([(x_a + x_b) / 2.0], dtype=float)

    n = nn - 1
    x_canon = np.cos(np.arange(0, n + 1) * np.pi / n)
    x_canon[np.abs(x_canon) < 1e-15] = 0.0
    return (x_b - x_a) / 2.0 * x_canon + (x_a + x_b) / 2.0


def _weighted_leja_1d(
    nn: int,
    x_a: float,
    x_b: float,
    rho: Callable[[np.ndarray], np.ndarray],
    grid_size: int = 2001,
    symmetric: bool = False,
) -> np.ndarray:
    """
    Weighted Leja sequence on [x_a, x_b] using a dense candidate grid (brute force).

    Recurrence:
        t^(j) = argmax_{t in [x_a,x_b]}  sqrt(rho(t)) * Π_{k<j} |t - t^(k)|

    Nested by construction (return first nn nodes).
    If symmetric=True, enforce symmetry about mid-point by selecting one side and mirroring.
    (Useful for symmetric pdfs like normal and beta with alpha=beta).
    """
    nn = int(nn)
    if nn <= 0:
        return np.array([], dtype=float)

    grid = np.linspace(x_a, x_b, int(grid_size))
    rho_vals = np.asarray(rho(grid), dtype=float)
    rho_vals = np.maximum(rho_vals, 1e-300)  # avoid log(0)

    mid = 0.5 * (x_a + x_b)

    def score_with_nodes(nodes: List[float]) -> np.ndarray:
        # score(t) = 0.5*log(rho(t)) + sum_k log(|t-x_k|)
        d = np.zeros_like(grid)
        if nodes:
            # accumulate log distances in a stable way
            d = np.sum(np.log(np.abs(grid[:, None] - np.array(nodes)[None, :]) + 1e-300), axis=1)
        return 0.5 * np.log(rho_vals) + d

    # --- initialization: start from mode (max rho) ---
    nodes: List[float] = []
    idx0 = int(np.argmax(rho_vals))
    x0 = float(grid[idx0])
    nodes.append(x0)

    if not symmetric:
        while len(nodes) < nn:
            sc = score_with_nodes(nodes)
            nodes.append(float(grid[int(np.argmax(sc))]))
        return np.array(nodes, dtype=float)

    # symmetric construction:
    # keep nodes symmetric around mid. Start from the mode; then add pairs (x, 2mid-x)
    # If the mode is not exactly mid, the first node will be mirrored only when we add the first pair.
    sym_nodes: List[float] = [x0]

    # if the first node isn't at mid, ensure its mirror appears early
    if abs(x0 - mid) > 1e-14 and len(sym_nodes) < nn:
        sym_nodes.append(float(2.0 * mid - x0))

    while len(sym_nodes) < nn:
        sc = score_with_nodes(sym_nodes)
        x_new = float(grid[int(np.argmax(sc))])
        x_mir = float(2.0 * mid - x_new)

        # add x_new
        if len(sym_nodes) < nn:
            sym_nodes.append(x_new)
        # add mirror
        if len(sym_nodes) < nn:
            sym_nodes.append(x_mir)

    return np.array(sym_nodes[:nn], dtype=float)


# =============================================================================
# PDFs (consistent with the usual parametrizations; normalization not required)
# =============================================================================

def _pdf_uniform(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return np.where((x >= a) & (x <= b), 1.0 / max(b - a, 1e-300), 0.0)

def _pdf_normal(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-300)
    return (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def _pdf_exponential(x: np.ndarray, lam: float) -> np.ndarray:
    lam = max(float(lam), 1e-300)
    return np.where(x >= 0.0, lam * np.exp(-lam * x), 0.0)

def _pdf_gamma(x: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    # Here we use the "rate" beta > 0; shape parameter as in the SGMK table.
    beta = max(float(beta), 1e-300)
    # Normalization constant is irrelevant for Leja, but keep a consistent shape
    return np.where(x >= 0.0, (x ** alpha) * np.exp(-beta * x), 0.0)

def _pdf_beta_on_interval(x: np.ndarray, a: float, b: float, alpha: float, beta: float) -> np.ndarray:
    # Defined on [a,b], shape parameters alpha,beta > -1
    L = max(b - a, 1e-300)
    z = (x - a) / L
    return np.where((z >= 0.0) & (z <= 1.0), (z ** alpha) * ((1.0 - z) ** beta), 0.0)

def _pdf_triangular_decreasing(x: np.ndarray, a: float, b: float) -> np.ndarray:
    # decreasing triangular on [a,b] as in SGMK table: rho(x)=2(b-x)/(b-a)^2
    L = max(b - a, 1e-300)
    return np.where((x >= a) & (x <= b), 2.0 * (b - x) / (L ** 2), 0.0)


# =============================================================================
# PDF specification layer (you pass one spec per dimension)
# =============================================================================

@dataclass(frozen=True)
class PdfSpec:
    name: str  # 'uniform'|'normal'|'exponential'|'gamma'|'beta'|'triangular'
    params: Dict[str, float]
    # optional truncation for unbounded domains (normal, exponential, gamma)
    # Examples:
    #   normal: {'n_sigma': 8}
    #   exponential/gamma: {'q': 12} meaning upper ~= q/beta or q/lam
    trunc: Optional[Dict[str, float]] = None
    symmetric: bool = False  # request symmetric Leja when meaningful


def pdfspec_to_domain_and_rho(spec: PdfSpec) -> Tuple[Tuple[float, float], Callable[[np.ndarray], np.ndarray]]:
    """
    Return (domain=[a,b], rho(x)) for the provided PdfSpec.
    For unbounded supports, use a robust truncation (override via spec.trunc).
    """
    name = spec.name.lower().strip()
    p = {k: float(v) for k, v in (spec.params or {}).items()}
    t = spec.trunc or {}

    # Allow explicit domain override for any PDF via params a,b
    if "a" in p and "b" in p:
        a, b = p["a"], p["b"]
        if name == "uniform":
            return (a, b), (lambda x: _pdf_uniform(x, a, b))
        if name == "triangular":
            return (a, b), (lambda x: _pdf_triangular_decreasing(x, a, b))
        if name == "beta":
            alpha, beta = p["alpha"], p["beta"]
            return (a, b), (lambda x: _pdf_beta_on_interval(x, a, b, alpha, beta))
        if name == "normal":
            mu, sigma = p.get("mu", 0.0), p.get("sigma", 1.0)
            return (a, b), (lambda x: _pdf_normal(x, mu, sigma))
        if name == "exponential":
            lam = p.get("lam", 1.0)
            return (a, b), (lambda x: _pdf_exponential(x, lam))
        if name == "gamma":
            alpha = p.get("alpha", 0.0)
            beta = p.get("beta", 1.0)
            return (a, b), (lambda x: _pdf_gamma(x, alpha, beta))
        raise ValueError(f"Unsupported PdfSpec.name='{spec.name}'")

    if name == "uniform":
        a, b = p["a"], p["b"]
        return (a, b), (lambda x: _pdf_uniform(x, a, b))

    if name == "triangular":
        a, b = p["a"], p["b"]
        return (a, b), (lambda x: _pdf_triangular_decreasing(x, a, b))

    if name == "beta":
        a, b = p["a"], p["b"]
        alpha, beta = p["alpha"], p["beta"]
        return (a, b), (lambda x: _pdf_beta_on_interval(x, a, b, alpha, beta))

    if name == "normal":
        mu, sigma = p.get("mu", 0.0), p.get("sigma", 1.0)
        n_sigma = float(t.get("n_sigma", 8.0))
        a = mu - n_sigma * sigma
        b = mu + n_sigma * sigma
        return (a, b), (lambda x: _pdf_normal(x, mu, sigma))

    if name == "exponential":
        lam = p.get("lam", 1.0)
        # choose upper by quantile-like truncation: x_max ~ q/lam (q default 12 -> exp(-12) ~ 6e-6)
        q = float(t.get("q", 12.0))
        a, b = 0.0, q / max(lam, 1e-300)
        return (a, b), (lambda x: _pdf_exponential(x, lam))

    if name == "gamma":
        alpha = p.get("alpha", 0.0)
        beta = p.get("beta", 1.0)  # rate
        q = float(t.get("q", 20.0))
        a, b = 0.0, q / max(beta, 1e-300)
        return (a, b), (lambda x: _pdf_gamma(x, alpha, beta))

    raise ValueError(f"Unsupported PdfSpec.name='{spec.name}'")


def suggest_lev2knots_for_nestedness(
    knots_family: str,
    spec: Optional[PdfSpec] = None,
) -> Callable[[int], int]:
    """
    Heuristic consistent with the SGMK guidance:
    - Clenshaw–Curtis: doubling
    - midpoint: tripling
    - (weighted) Leja: linear is OK; for symmetric Leja prefer 2-step or doubling
    """
    kf = knots_family.lower().strip()

    if kf in {"clenshaw-curtis", "cc"}:
        return lev2knots_doubling
    if kf in {"midpoint"}:
        return lev2knots_tripling
    if kf in {"leja", "weighted-leja", "wleja"}:
        if spec is not None and bool(spec.symmetric):
            # add knots in pairs to preserve symmetry
            return lev2knots_2step
        return lev2knots_lin

    # default conservative choice
    return lev2knots_lin


def make_knots_func_from_pdfspec(
    spec: PdfSpec,
    grid_size: int = 2001,
) -> Tuple[Callable[[int, float, float], np.ndarray], Tuple[float, float]]:
    """
    Build a 1D knots function consistent with the provided pdf_spec.
    Returns (knots_func_1d(nn, a, b), domain=(a,b)).
    """
    domain, rho = pdfspec_to_domain_and_rho(spec)
    a, b = domain

    if spec.name.lower().strip() == "uniform":
        # Use nested Clenshaw-Curtis for uniform PDFs.
        return knots_cc_1d, domain

    def knots_func(nn: int, x_a: float, x_b: float) -> np.ndarray:
        # ignore x_a, x_b passed by caller; enforce the spec domain consistently
        return _weighted_leja_1d(
            nn=int(nn),
            x_a=a,
            x_b=b,
            rho=rho,
            grid_size=int(grid_size),
            symmetric=bool(spec.symmetric),
        )

    return knots_func, domain


# =============================================================================
# Sparse-grid structure (combination technique, total degree)
# =============================================================================

def tensor_grid_knots(
    N: int,
    m: np.ndarray,
    knots_func_1d: Union[Callable[[int, float, float], np.ndarray], Sequence[Callable[[int, float, float], np.ndarray]]],
    domains: Sequence[Tuple[float, float]],
) -> Dict[str, object]:
    """Build tensor-product grid nodes given per-dimension resolutions."""
    if isinstance(knots_func_1d, (list, tuple)):
        if len(knots_func_1d) != N:
            raise ValueError("knots_func_1d list length must match N.")
        knots_per_dim = [
            knots_func_1d[i](int(m[i]), float(domains[i][0]), float(domains[i][1])) for i in range(N)
        ]
    else:
        knots_per_dim = [
            knots_func_1d(int(m[i]), float(domains[i][0]), float(domains[i][1])) for i in range(N)
        ]

    grids = np.meshgrid(*knots_per_dim, indexing="ij")
    points = np.vstack([grid.ravel() for grid in grids])
    return {"knots": points, "size": points.shape[1], "knots_per_dim": knots_per_dim, "m": np.array(m, dtype=int)}


def generate_multi_indices_total_degree(N: int, level: int) -> np.ndarray:
    """Generate total-degree multi-indices for a sparse grid."""
    valid_indices = []
    for i_tuple in itertools.product(range(1, level + N + 1), repeat=N):
        if sum(i_tuple) <= level + N:
            valid_indices.append(i_tuple)
    return np.array(sorted(valid_indices, key=lambda x: (sum(x), x)), dtype=int)


def create_sparse_grid_structure(
    N: int,
    level: int,
    level_to_knots_func: Callable[[int], int],
    knots_func_1d: Union[Callable[[int, float, float], np.ndarray], Sequence[Callable[[int, float, float], np.ndarray]]],
    domains: Sequence[Tuple[float, float]],
) -> List[Dict[str, object]]:
    """Create a sparse-grid structure (tensor grids plus combination coefficients)."""
    multi_indices = generate_multi_indices_total_degree(N, level)
    S: List[Dict[str, object]] = []

    for idx_vec in multi_indices:
        q = int(np.sum(idx_vec))
        coeff = ((-1) ** (level + N - q)) * comb(N - 1, level + N - q, exact=True)
        if abs(coeff) < 1e-12:
            continue

        m = np.array([int(level_to_knots_func(int(val))) for val in idx_vec], dtype=int)
        grid = tensor_grid_knots(N, m, knots_func_1d, domains)
        grid["coeff"] = float(coeff)
        grid["idx"] = np.array(idx_vec, dtype=int)
        S.append(grid)

    return S


def reduce_sparse_grid_knots(S: List[Dict[str, object]], tol: float = 1e-14) -> np.ndarray:
    """Extract and uniquify nodes from all tensor grids in S."""
    if not S:
        return np.array([[]], dtype=float)

    all_knots = np.hstack([grid["knots"] for grid in S]).T
    if all_knots.shape[0] == 0:
        return np.array([[]], dtype=float)

    sorted_indices = np.lexsort(np.flipud(all_knots.T))
    sorted_knots = all_knots[sorted_indices, :]
    diffs = np.diff(sorted_knots, axis=0)
    is_unique_mask = np.concatenate(([True], np.max(np.abs(diffs), axis=1) > tol))
    return sorted_knots[is_unique_mask, :].T


# =============================================================================
# Interpolant evaluation (tensor Lagrange, global)
# =============================================================================

def _evaluate_lagrange_poly_1d(x_eval: float, knots_1d: np.ndarray, j_index: int) -> float:
    """Evaluate the j-th 1D Lagrange basis polynomial."""
    num, den = 1.0, 1.0
    knot_j = float(knots_1d[j_index])
    for k, knot_k in enumerate(knots_1d):
        if k != j_index:
            num *= (x_eval - float(knot_k))
            den *= (knot_j - float(knot_k))
    return float(num / den) if abs(den) > 1e-16 else 1.0


def evaluate_sparse_grid_interpolant(
    y_eval_points: np.ndarray,
    S: List[Dict[str, object]],
    unique_knots: np.ndarray,
    f_on_unique_knots: np.ndarray,
) -> np.ndarray:
    """Evaluate the sparse-grid interpolant on evaluation points."""
    if unique_knots.size == 0:
        return np.array([], dtype=float)

    f_map = {tuple(np.round(k, 12)): float(v) for k, v in zip(unique_knots.T, f_on_unique_knots)}
    N = int(unique_knots.shape[0])
    num_eval_points = int(y_eval_points.shape[1])
    f_approx_values = np.zeros(num_eval_points, dtype=float)

    for p in range(num_eval_points):
        y_eval = y_eval_points[:, p]
        total_value = 0.0

        for grid in S:
            tensor_interpolant = 0.0
            knots_nd = grid["knots"]
            knots_per_dim = grid["knots_per_dim"]

            for j_node_idx in range(int(grid["size"])):
                node_y_j = knots_nd[:, j_node_idx]
                f_val = f_map.get(tuple(np.round(node_y_j, 12)))
                if f_val is None:
                    continue

                lagrange_basis = 1.0
                for n in range(N):
                    knots_1d_n = np.asarray(knots_per_dim[n], dtype=float)
                    # locate 1D index (safe because nodes were built from these arrays)
                    knot_idx_1d = int(np.where(np.isclose(knots_1d_n, node_y_j[n]))[0][0])
                    lagrange_basis *= _evaluate_lagrange_poly_1d(float(y_eval[n]), knots_1d_n, knot_idx_1d)

                tensor_interpolant += f_val * lagrange_basis

            total_value += float(grid["coeff"]) * tensor_interpolant

        f_approx_values[p] = total_value

    return f_approx_values


# =============================================================================
# Discrepancy indicator (robust hybrid relative/absolute)
# =============================================================================

def compute_discrepancy_indicator(
    candidate_knots: np.ndarray,
    w_start: int,
    N: int,
    domains: Sequence[Tuple[float, float]],
    all_known_knots: np.ndarray,
    all_known_values: np.ndarray,
    knots_func_1d: Union[Callable[[int, float, float], np.ndarray], Sequence[Callable[[int, float, float], np.ndarray]]] = knots_cc_1d,
    level_to_knots_func: Callable[[int], int] = lev2knots_doubling,
) -> np.ndarray:
    """Compute a robust discrepancy indicator (hierarchical error)."""
    if candidate_knots.shape[1] == 0:
        return np.array([], dtype=float)

    S_start = create_sparse_grid_structure(N, w_start, level_to_knots_func, knots_func_1d, domains)
    S_prev = create_sparse_grid_structure(N, w_start - 1, level_to_knots_func, knots_func_1d, domains)

    preds_start = evaluate_sparse_grid_interpolant(candidate_knots, S_start, all_known_knots, all_known_values)
    preds_prev = evaluate_sparse_grid_interpolant(candidate_knots, S_prev, all_known_knots, all_known_values)

    abs_diff = np.abs(preds_start - preds_prev)
    denominator = np.abs(preds_start)
    indicator = np.zeros_like(denominator)

    non_zero_mask = denominator > 1e-9
    indicator[non_zero_mask] = abs_diff[non_zero_mask] / denominator[non_zero_mask]
    indicator[~non_zero_mask] = abs_diff[~non_zero_mask]

    return indicator


# =============================================================================
# Convenience: build per-dimension Leja knots + domains + suggested lev2knots
# =============================================================================

def build_1d_knots_and_domains_from_pdfspecs(
    pdf_specs: Sequence[PdfSpec],
    grid_size: int = 2001,
) -> Tuple[List[Callable[[int, float, float], np.ndarray]], List[Tuple[float, float]]]:
    """
    Given one PdfSpec per dimension, return:
      - knots_funcs: list of weighted-Leja generators, one per dimension
      - domains: list of [a,b] per dimension (after truncation if needed)
    """
    knots_funcs: List[Callable[[int, float, float], np.ndarray]] = []
    domains: List[Tuple[float, float]] = []

    for spec in pdf_specs:
        kf, dom = make_knots_func_from_pdfspec(spec, grid_size=grid_size)
        knots_funcs.append(kf)
        domains.append(dom)

    return knots_funcs, domains


def suggest_global_lev2knots_for_pdfspecs(
    knots_family: str,
    pdf_specs: Sequence[PdfSpec],
) -> Callable[[int], int]:
    """
    Return a single lev2knots function to be used globally.
    If any dimension requests symmetric Leja, prefer 2-step for Leja family.
    """
    if knots_family.lower().strip() in {"leja", "weighted-leja", "wleja"}:
        if any(bool(s.symmetric) for s in pdf_specs):
            return lev2knots_2step
        return lev2knots_lin

    return suggest_lev2knots_for_nestedness(knots_family, None)


# =============================================================================
# Sparse-grid quadrature weights (weighted integrals)
# =============================================================================

def _lagrange_quadrature_weights_1d(
    knots: np.ndarray,
    rho: Callable[[np.ndarray], np.ndarray],
    a: float,
    b: float,
    quad_order: int = 200,
) -> np.ndarray:
    """
    Compute 1D quadrature weights by integrating Lagrange basis polynomials
    against the PDF rho on [a, b] using Gauss-Legendre quadrature.
    """
    from numpy.polynomial.legendre import leggauss

    knots = np.asarray(knots, dtype=float)
    if knots.size == 0:
        return np.array([], dtype=float)

    xq, wq = leggauss(int(quad_order))
    x = 0.5 * (b - a) * xq + 0.5 * (b + a)
    w = 0.5 * (b - a) * wq

    rho_vals = rho(x)
    w_rho = w * rho_vals

    diff = x[:, None] - knots[None, :]
    diff[np.abs(diff) < 1e-15] = 1e-15
    prod_all = np.prod(diff, axis=1)

    denom = np.ones_like(knots)
    for j in range(knots.size):
        denom[j] = np.prod(knots[j] - np.delete(knots, j))

    L = (prod_all[:, None] / diff) / denom[None, :]
    return np.sum(w_rho[:, None] * L, axis=0)


def compute_sparse_grid_quadrature_weights(
    S: List[Dict[str, object]],
    unique_knots: np.ndarray,
    pdf_funcs: Sequence[Callable[[np.ndarray], np.ndarray]],
    domains: Sequence[Tuple[float, float]],
    quad_order: int = 200,
    tol: float = 1e-12,
    pdf_keys: Optional[Sequence[str]] = None,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute sparse-grid quadrature weights on the union of knots.

    For each tensor grid in S, build tensor-product weights from 1D Lagrange
    quadrature weights, then combine them with sparse-grid coefficients.
    """
    if unique_knots.size == 0:
        return np.array([], dtype=float)

    if len(pdf_funcs) != len(domains):
        raise ValueError("pdf_funcs length must match domains length.")

    if pdf_keys is None:
        pdf_keys = [str(i) for i in range(len(pdf_funcs))]

    unique_keys = [tuple(np.round(k, 12)) for k in unique_knots.T]
    index_map = {k: i for i, k in enumerate(unique_keys)}
    weights = np.zeros(unique_knots.shape[1], dtype=float)

    cache: Dict[Tuple[object, ...], np.ndarray] = {}
    for grid in S:
        weights_1d: List[np.ndarray] = []
        for dim, knots_1d in enumerate(grid["knots_per_dim"]):
            key = (
                dim,
                tuple(np.round(np.asarray(knots_1d, dtype=float), 12)),
                pdf_keys[dim],
                float(domains[dim][0]),
                float(domains[dim][1]),
                int(quad_order),
            )
            if key not in cache:
                cache[key] = _lagrange_quadrature_weights_1d(
                    np.asarray(knots_1d, dtype=float),
                    pdf_funcs[dim],
                    float(domains[dim][0]),
                    float(domains[dim][1]),
                    quad_order=int(quad_order),
                )
            weights_1d.append(cache[key])

        mesh = np.meshgrid(*weights_1d, indexing="ij")
        w_tensor = np.prod(np.array(mesh), axis=0).ravel()

        for j in range(int(grid["size"])):
            node = grid["knots"][:, j]
            idx = index_map.get(tuple(np.round(node, 12)))
            if idx is None:
                continue
            weights[idx] += float(grid["coeff"]) * w_tensor[j]

    if normalize:
        total = float(np.sum(weights))
        if abs(total) > tol:
            weights /= total

    return weights
