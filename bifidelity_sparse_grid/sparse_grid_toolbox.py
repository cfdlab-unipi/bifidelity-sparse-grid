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
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
from scipy.special import comb
# Re-export the established node functions so existing numerical calls keep
# their names while the one-dimensional rules remain in a small module.
from .nodes import (
    knots_cc_1d, knots_leja_line_1d, knots_leja_sym_line_1d,
    leja_nodes_and_weights_1d, _weighted_leja_1d,
)


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


def generate_multi_indices_total_degree(N, level):
    """Generate positive multi-indices with sum at most N + level."""
    if N < 1 or int(N) != N:
        raise ValueError("N must be a positive integer.")
    if level < 0 or int(level) != level:
        raise ValueError("level must be a non-negative integer.")

    # Start with zero-based degrees and extend one dimension at a time.
    # Discard inadmissible degrees immediately, so high dimensions remain usable.
    N = int(N)
    level = int(level)
    degrees = [()]
    for dimension in range(N):
        next_degrees = []
        for prefix in degrees:
            remaining_degree = level - sum(prefix)
            for degree in range(remaining_degree + 1):
                next_degrees.append(prefix + (degree,))
        degrees = next_degrees

    # Add one because the one-dimensional grid levels start at one.
    degrees.sort(key=lambda values: (sum(values), values))
    return np.asarray(degrees, dtype=int) + 1


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

def _clenshaw_curtis_barycentric_weights(num_knots):
    """Return closed-form barycentric weights for ordered CC nodes."""
    if num_knots < 1:
        raise ValueError("num_knots must be at least 1")
    if num_knots == 1:
        return np.ones(1)

    weights = (-1.0) ** np.arange(num_knots)
    weights[0] *= 0.5
    weights[-1] *= 0.5
    weights.setflags(write=False)
    return weights


def _generic_barycentric_weights(knots_tuple):
    # Leja nodes do not have the closed-form CC weights. This direct formula is
    # stable for the supported Leja prefixes, which contain at most 31 nodes.
    knots = np.asarray(knots_tuple, dtype=float)
    num_knots = knots.size
    weights = np.ones(num_knots, dtype=float)

    for node_index in range(num_knots):
        denominator = 1.0
        for other_index in range(num_knots):
            if node_index != other_index:
                denominator *= knots[node_index] - knots[other_index]
        weights[node_index] = 1.0 / denominator

    weights /= np.max(np.abs(weights))
    weights.setflags(write=False)
    return weights


def _are_clenshaw_curtis_knots(knots_1d):
    # Keep the original exact CC weights whenever the given nodes are a CC grid.
    # This prevents the Leja extension from changing the established CC pathway.
    knots = np.asarray(knots_1d, dtype=float)
    if knots.size <= 1:
        return True

    expected_knots = knots_cc_1d(knots.size, np.min(knots), np.max(knots))
    tolerance = 100.0 * np.finfo(float).eps * max(1.0, np.max(np.abs(knots)))
    return np.allclose(knots, expected_knots, rtol=0.0, atol=tolerance)


def _barycentric_weights_for_knots(knots_1d):
    # Select the fast closed form for CC and the general formula for Leja.
    knots = np.asarray(knots_1d, dtype=float)
    if _are_clenshaw_curtis_knots(knots):
        return _clenshaw_curtis_barycentric_weights(knots.size)
    return _generic_barycentric_weights(tuple(knots))


def _evaluate_lagrange_poly_1d(x_eval, knots_1d, j_index):
    """Evaluate one cardinal basis function with the barycentric formula.

    CC uses closed-form weights; Leja uses weights from its actual nodes.
    At a node, return the cardinal value directly to avoid a removable 0/0.
    """
    knots_1d = np.asarray(knots_1d, dtype=float)
    num_knots = knots_1d.size
    if not 0 <= j_index < num_knots:
        raise IndexError("j_index is outside the knot array")
    if num_knots == 1:
        return 1.0

    differences = float(x_eval) - knots_1d
    scale = max(1.0, float(np.max(np.abs(knots_1d))))
    matches = np.flatnonzero(np.abs(differences) <= 1e-14 * scale)
    if matches.size:
        return 1.0 if j_index == int(matches[0]) else 0.0

    weights = _barycentric_weights_for_knots(knots_1d)
    terms = weights / differences
    return float(terms[j_index] / np.sum(terms))

def _evaluate_lagrange_basis_matrix_1d(points_1d, knots_1d):
    """Evaluate all one-dimensional cardinal functions at many points.

    Each row of the returned matrix belongs to one evaluation point. Each
    column belongs to one knot. The barycentric formula is used because it is
    stable for the CC and Leja knots used by this toolbox.
    """
    points_1d = np.asarray(points_1d, dtype=float)
    knots_1d = np.asarray(knots_1d, dtype=float)
    num_knots = knots_1d.size

    if num_knots == 1:
        return np.ones((points_1d.size, 1), dtype=float)

    weights = _barycentric_weights_for_knots(knots_1d)
    differences = points_1d[:, None] - knots_1d[None, :]
    scale = max(1.0, float(np.max(np.abs(knots_1d))))

    # At a knot the barycentric formula is formally 0/0. The cardinal
    # property gives the answer exactly: one at that knot and zero elsewhere.
    matches = np.abs(differences) <= 1.0e-14 * scale
    safe_differences = differences.copy()
    safe_differences[matches] = 1.0
    terms = weights[None, :] / safe_differences
    row_sums = np.sum(terms, axis=1, keepdims=True)
    row_sums[np.any(matches, axis=1)] = 1.0
    basis = terms / row_sums

    matching_rows = np.flatnonzero(np.any(matches, axis=1))
    if matching_rows.size:
        matching_columns = np.argmax(matches[matching_rows], axis=1)
        basis[matching_rows] = 0.0
        basis[matching_rows, matching_columns] = 1.0

    return basis


def evaluate_smolyak_from_nodal_values(
    evaluation_points,
    S,
    unique_knots,
    nodal_values,
    batch_size=1024,
    show_progress=False,
):
    """Evaluate one or more Smolyak interpolants from their nodal values.

    Parameters
    ----------
    evaluation_points : numpy.ndarray
        Points with shape ``(N, number_of_points)``.
    S : list of dict
        Sparse-grid structure returned by ``create_sparse_grid_structure``.
    unique_knots : numpy.ndarray
        Deduplicated nodes of ``S``, with shape ``(N, number_of_knots)``.
    nodal_values : numpy.ndarray
        Values at ``unique_knots``. It can be one vector with shape
        ``(number_of_knots,)`` or several channels with shape
        ``(number_of_channels, number_of_knots)``.
    batch_size : int, optional
        Number of evaluation points processed together. Batching avoids a
        prohibitively large temporary tensor in high dimension.
    show_progress : bool, optional
        Print progress every 250 Smolyak tensor terms. It is off by default
        so this standard evaluator is quiet inside tests and plotting tools.

    Returns
    -------
    numpy.ndarray
        A one-dimensional prediction vector when one value channel is passed,
        otherwise an array with one row per value channel.

    Notes
    -----
    This is the standard high-dimensional evaluator. It reuses the
    one-dimensional basis values for every function channel and every tensor
    term with the same knots. It is numerically equivalent to evaluating the
    Smolyak combination formula point by point, but it is much faster when a
    validation set or a plot contains many points.
    """
    evaluation_points = np.asarray(evaluation_points, dtype=float)
    unique_knots = np.asarray(unique_knots, dtype=float)
    nodal_values = np.asarray(nodal_values, dtype=float)

    if evaluation_points.ndim != 2:
        raise ValueError("evaluation_points must have shape (N, number_of_points)")
    if unique_knots.ndim != 2:
        raise ValueError("unique_knots must have shape (N, number_of_knots)")
    if evaluation_points.shape[0] != unique_knots.shape[0]:
        raise ValueError("evaluation_points and unique_knots have different dimensions")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    input_was_one_channel = nodal_values.ndim == 1
    if input_was_one_channel:
        nodal_values = nodal_values[None, :]
    if nodal_values.ndim != 2:
        raise ValueError("nodal_values must be one vector or a two-dimensional array")
    if nodal_values.shape[1] != unique_knots.shape[1]:
        raise ValueError("nodal_values and unique_knots have incompatible sizes")

    num_points = evaluation_points.shape[1]
    num_channels = nodal_values.shape[0]
    predictions = np.zeros((num_channels, num_points), dtype=float)

    # Map each local tensor-grid node back to its column in the shared,
    # deduplicated nodal-value array. Rounding agrees with reduce_sparse_grid_knots.
    unique_lookup = {}
    for index in range(unique_knots.shape[1]):
        point_key = tuple(np.round(unique_knots[:, index], 12))
        unique_lookup[point_key] = index

    # Collect the actual one-dimensional knots stored in S. The key includes
    # the dimension: different dimensions may have different physical domains.
    knots_by_dimension = {}
    for grid in S:
        for dimension, knots_1d in enumerate(grid["knots_per_dim"]):
            knots_by_dimension[(dimension, tuple(knots_1d))] = np.asarray(
                knots_1d, dtype=float
            )

    # These basis matrices are the expensive geometric part of the calculation.
    # They are independent of function values and are reused by all channels.
    basis_cache = {}
    for key, knots_1d in knots_by_dimension.items():
        dimension = key[0]
        basis_cache[key] = _evaluate_lagrange_basis_matrix_1d(
            evaluation_points[dimension], knots_1d
        )

    num_terms = len(S)
    for term_number, grid in enumerate(S, start=1):
        counts = []
        for knots_1d in grid["knots_per_dim"]:
            counts.append(len(knots_1d))
        counts = tuple(counts)

        tensor_indices = []
        for point in grid["knots"].T:
            point_key = tuple(np.round(point, 12))
            tensor_indices.append(unique_lookup[point_key])
        tensor_indices = np.asarray(tensor_indices, dtype=int)

        tensor_values = nodal_values[:, tensor_indices].reshape(
            (num_channels,) + counts
        )

        active_dimensions = []
        inactive_axes = []
        for dimension, count in enumerate(counts):
            if count > 1:
                active_dimensions.append(dimension)
            else:
                # Axis zero stores the value channel, hence +1 here.
                inactive_axes.append(dimension + 1)
        if inactive_axes:
            tensor_values = np.squeeze(tensor_values, axis=tuple(inactive_axes))

        coefficient = float(grid["coeff"])
        if not active_dimensions:
            predictions += coefficient * tensor_values.reshape(num_channels, 1)
            continue

        for start in range(0, num_points, batch_size):
            stop = min(start + batch_size, num_points)

            # Contract the last active dimension first. Each following
            # contraction removes one more tensor-product dimension.
            last_dimension = active_dimensions[-1]
            last_basis = basis_cache[
                (last_dimension, tuple(grid["knots_per_dim"][last_dimension]))
            ][start:stop]
            temporary = np.tensordot(
                last_basis, tensor_values, axes=(1, len(active_dimensions))
            )

            for dimension in reversed(active_dimensions[:-1]):
                basis = basis_cache[(dimension, tuple(grid["knots_per_dim"][dimension]))][start:stop]
                temporary = np.einsum(
                    "bf...j,bj->bf...", temporary, basis, optimize=True
                )

            predictions[:, start:stop] += coefficient * temporary.T

        if show_progress and (term_number % 250 == 0 or term_number == num_terms):
            print(f"      Smolyak term {term_number}/{num_terms}", flush=True)

    if input_was_one_channel:
        return predictions[0]
    return predictions


def evaluate_sparse_grid_interpolant(y_eval_points, S, unique_knots, f_on_unique_knots):
    """Evaluate one sparse-grid interpolant using the standard Smolyak evaluator.

    This compatibility wrapper preserves the historical public API. New code
    may call ``evaluate_smolyak_from_nodal_values`` directly when it needs to
    evaluate several functions or several model channels at the same points.
    """
    return evaluate_smolyak_from_nodal_values(
        y_eval_points,
        S,
        unique_knots,
        f_on_unique_knots,
    )


# =============================================================================
# Discrepancy indicator (robust hybrid relative/absolute)
# =============================================================================

def compute_discrepancy_indicator(
    candidate_knots, w_start, N, domains, all_known_knots, all_known_values,
    knots_func_1d=knots_cc_1d, level_to_knots_func=lev2knots_doubling,
    indicator_type="legacy", relative_epsilon=1e-5, auxiliary_level=None,
):
    """Compare baseline and auxiliary interpolants at candidate points.

    Values have shape (number_of_knots,) for a scalar, or
    (number_of_components, number_of_knots) for a vector quantity.
    Vector discrepancies are norms of interpolated component differences.
    The default preserves the historical scalar indicator for old configs.
    """
    if indicator_type not in ("absolute", "relative", "symmetric_relative", "legacy"):
        raise ValueError("indicator_type must be absolute, symmetric_relative, relative, or legacy.")
    if not np.isfinite(relative_epsilon) or relative_epsilon <= 0:
        raise ValueError("relative_epsilon must be finite and positive.")
    if auxiliary_level is None:
        auxiliary_level = w_start - 1
    if auxiliary_level < 0 or auxiliary_level >= w_start:
        raise ValueError("auxiliary_level must be non-negative and below base_level.")
    if candidate_knots.shape[1] == 0:
        return np.array([], dtype=float)

    S_start = create_sparse_grid_structure(N, w_start, level_to_knots_func, knots_func_1d, domains)
    S_prev = create_sparse_grid_structure(N, auxiliary_level, level_to_knots_func, knots_func_1d, domains)
    preds_start = evaluate_smolyak_from_nodal_values(candidate_knots, S_start, all_known_knots, all_known_values)
    preds_prev = evaluate_smolyak_from_nodal_values(candidate_knots, S_prev, all_known_knots, all_known_values)

    # Interpolate every component before computing the vector norm. Taking a
    # norm at the input nodes would lose component signs and cancellations.
    if preds_start.ndim == 2:
        difference = np.linalg.norm(preds_start - preds_prev, axis=0)
        magnitude_start = np.linalg.norm(preds_start, axis=0)
        magnitude_prev = np.linalg.norm(preds_prev, axis=0)
    else:
        difference = np.abs(preds_start - preds_prev)
        magnitude_start = np.abs(preds_start)
        magnitude_prev = np.abs(preds_prev)

    if indicator_type == "absolute":
        return difference
    if indicator_type == "legacy":
        indicator = difference.copy()
        non_zero = magnitude_start > 1e-9
        indicator[non_zero] /= magnitude_start[non_zero]
        return indicator

    # Use one scale S over all candidates. The stabilizer prevents spikes
    # when both interpolants approach zero; a zero function has zero indicator.
    scale = max(float(np.max(magnitude_start)), float(np.max(magnitude_prev)))
    if scale == 0.0:
        return np.zeros_like(difference)
    denominator = magnitude_start / scale + magnitude_prev / scale + relative_epsilon
    return 2.0 * (difference / scale) / denominator


def select_refinement_points(indicators, nominal_budget, include_ties=True, tie_tolerance=1e-3, indicator_cutoff=None):
    """Return selected indices in descending order, including cutoff ties."""
    indicators = np.asarray(indicators, dtype=float)
    if indicators.ndim != 1 or not np.all(np.isfinite(indicators)) or np.any(indicators < 0):
        raise ValueError("Indicators must be a finite, non-negative vector.")
    if not np.isfinite(nominal_budget) or nominal_budget < 0 or int(nominal_budget) != nominal_budget:
        raise ValueError("nominal_budget must be a non-negative integer.")
    if not np.isfinite(tie_tolerance) or not 0 <= tie_tolerance < 1:
        raise ValueError("tie_tolerance must lie between zero (included) and one (excluded).")
    if indicator_cutoff is not None and (not np.isfinite(indicator_cutoff) or indicator_cutoff < 0):
        raise ValueError("indicator_cutoff must be finite and non-negative.")
    nominal_budget = min(int(nominal_budget), indicators.size)
    if nominal_budget == 0 or indicators.size == 0:
        return np.array([], dtype=int)

    # Stable sorting makes strict-budget selections repeatable even for ties.
    ranking = np.argsort(-indicators, kind="stable")
    if indicator_cutoff is None:
        if not include_ties:
            return ranking[:nominal_budget]
        cutoff = indicators[ranking[nominal_budget - 1]]
    else:
        # A threshold cutoff is tau*max(eta), not the last retained value.
        cutoff = float(indicator_cutoff)
    tolerance = tie_tolerance * abs(cutoff) if include_ties else 0.0
    return ranking[indicators[ranking] >= cutoff - tolerance]


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

    # Integrate the same stable barycentric basis used by interpolation.
    basis = _evaluate_lagrange_basis_matrix_1d(x, knots)
    return np.sum(w_rho[:, None] * basis, axis=0)


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
