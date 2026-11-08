"""One-dimensional Clenshaw-Curtis, verified Leja, and PDF-weighted nodes.

The CSV stores each verified Leja node once. Prefix quadrature weights are
retained for compatibility; weighted statistics use the configured PDFs.
"""

import csv
from importlib.resources import files

import numpy as np

# Load the small packaged dataset once. CSV decimal strings preserve exactly
# the same floating-point nodes and weights as the previous Python tables.
with files("bifidelity_sparse_grid").joinpath("data/leja.csv").open("r") as file:
    LEJA_ROWS = list(csv.DictReader(file))
CLASSICAL_LEJA_NODES = np.array([float(row["classical_node"]) for row in LEJA_ROWS])
SYMMETRIC_LEJA_NODES = np.array([float(row["symmetric_node"]) for row in LEJA_ROWS])
MAX_LEJA_NODES = len(LEJA_ROWS)


def knots_cc_1d(nn, x_a, x_b):
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


def _leja_score(grid, rho_vals, nodes):
    """Weighted Leja score on the candidate grid, for the nodes chosen so far.

    score(t) = 0.5*log(rho(t)) + sum_k log(|t - x_k|)

    The tiny 1e-300 avoids log(0) at nodes already chosen: their score becomes
    about -690, so they are never selected again.
    """
    log_distances = np.zeros_like(grid)
    if nodes:
        distances = np.abs(grid[:, None] - np.array(nodes)[None, :])
        log_distances = np.sum(np.log(distances + 1e-300), axis=1)
    return 0.5 * np.log(rho_vals) + log_distances


def _weighted_leja_1d(nn, x_a, x_b, rho, grid_size=2001, symmetric=False):
    """
    Weighted Leja sequence on [x_a, x_b] using a dense candidate grid (brute force).

    Recurrence:
        t^(j) = argmax_{t in [x_a,x_b]}  sqrt(rho(t)) * Π_{k<j} |t - t^(k)|

    Nested by construction (return first nn nodes).
    If symmetric=True, start from the mid-point and add mirrored pairs.
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

    if not symmetric:
        # Start from the mode (largest density on the grid), then add one
        # node at a time where the score is largest.
        nodes = [float(grid[int(np.argmax(rho_vals))])]
        while len(nodes) < nn:
            score = _leja_score(grid, rho_vals, nodes)
            nodes.append(float(grid[int(np.argmax(score))]))
        return np.array(nodes, dtype=float)

    # symmetric construction:
    # keep nodes symmetric around mid. Start exactly from mid, then add pairs
    # (x, 2mid-x). The midpoint is the mode of every supported symmetric PDF
    # (normal with mu at the centre, beta with alpha == beta > 0). Starting from
    # the grid argmax instead can give a first node a few ulps away from mid;
    # its mirror is then a near-duplicate node. For a flat density (beta with
    # alpha == beta == 0) the argmax is the left endpoint, and the midpoint was
    # later added twice. Both cases made the interpolation singular.
    sym_nodes = [float(mid)]

    while len(sym_nodes) < nn:
        score = _leja_score(grid, rho_vals, sym_nodes)
        x_new = float(grid[int(np.argmax(score))])
        x_mir = float(2.0 * mid - x_new)

        # add x_new
        sym_nodes.append(x_new)
        # add its mirror, unless nn nodes are already present
        if len(sym_nodes) < nn:
            sym_nodes.append(x_mir)

    return np.array(sym_nodes, dtype=float)
