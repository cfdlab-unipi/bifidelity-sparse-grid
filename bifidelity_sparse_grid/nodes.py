"""One-dimensional Clenshaw-Curtis, verified Leja, and PDF-weighted nodes.

The CSV stores each verified Leja node once. Prefix quadrature weights are
retained for compatibility; weighted statistics use the configured PDFs.
"""

import csv
from importlib.resources import files
from typing import Callable, List

import numpy as np

# Load the small packaged dataset once. CSV decimal strings preserve exactly
# the same floating-point nodes and weights as the previous Python tables.
with files("bifidelity_sparse_grid").joinpath("data/leja.csv").open("r") as file:
    LEJA_ROWS = list(csv.DictReader(file))
CLASSICAL_LEJA_NODES = np.array([float(row["classical_node"]) for row in LEJA_ROWS])
SYMMETRIC_LEJA_NODES = np.array([float(row["symmetric_node"]) for row in LEJA_ROWS])
MAX_LEJA_NODES = len(LEJA_ROWS)


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


def _leja_nodes(num_knots, x_a, x_b, sequence):
    """Read a nested prefix and map it to the requested physical interval."""
    if int(num_knots) != num_knots or not 1 <= num_knots <= MAX_LEJA_NODES:
        raise ValueError(f"Leja rules provide between 1 and {MAX_LEJA_NODES} nodes.")
    if not np.isfinite(x_a) or not np.isfinite(x_b) or x_b <= x_a:
        raise ValueError("Leja domain bounds must be finite with max > min.")
    count = int(num_knots)
    if sequence == "line":
        canonical = CLASSICAL_LEJA_NODES[:count]
    elif sequence == "symmetric":
        canonical = SYMMETRIC_LEJA_NODES[:count]
    else:
        raise ValueError("sequence must be line or symmetric.")
    return 0.5 * (x_a + x_b) + 0.5 * (x_b - x_a) * canonical


def leja_nodes_and_weights_1d(num_knots, x_a, x_b, sequence):
    """Return the original nodes and uniform-probability quadrature weights."""
    nodes = _leja_nodes(num_knots, x_a, x_b, sequence)
    if sequence == "line":
        column = "classical_weights"
    else:
        column = "symmetric_weights"
    weights = np.fromstring(LEJA_ROWS[int(num_knots) - 1][column], sep=" ")
    return nodes, weights


def knots_leja_line_1d(num_knots, x_a, x_b):
    """Classical real Leja nodes, beginning with 1, -1, and 0."""
    return _leja_nodes(num_knots, x_a, x_b, "line")


def knots_leja_sym_line_1d(num_knots, x_a, x_b):
    """Symmetric real Leja nodes, beginning with 0, 1, and -1."""
    return _leja_nodes(num_knots, x_a, x_b, "symmetric")


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

    if nn > int(grid_size):
        raise ValueError(f"Weighted Leja requires at most {grid_size} nodes on its candidate grid.")
    grid = np.linspace(x_a, x_b, int(grid_size))
    rho_vals = np.asarray(rho(grid), dtype=float)
    if not np.all(np.isfinite(rho_vals)):
        raise ValueError("Weighted Leja needs a finite density at the domain endpoints; use an explicit CC or tabulated Leja grid for singular densities.")
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
