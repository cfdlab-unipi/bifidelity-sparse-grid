# sparse_grid_toolbox.py
# =============================================================================
#
# TOOLBOX PER L'INTERPOLAZIONE SPARSE-GRID (V5 - ROBUSTO)
# - Utilizza la regola "total degree" per gli indici.
# - Include un indicatore di discrepanza numericamente robusto per
#   il raffinamento adattivo.
#
# =============================================================================

import numpy as np
import itertools
from scipy.special import comb

# --- Funzioni per la generazione dei nodi 1D ---

def knots_cc_1d(nn, x_a, x_b):
    """Genera nn nodi di Clenshaw-Curtis nel dominio [x_a, x_b]."""
    if nn == 1:
        return np.array([(x_a + x_b) / 2.0])
    n = nn - 1
    x_canon = np.cos(np.arange(0, n + 1) * np.pi / n)
    x_canon[np.abs(x_canon) < 1e-15] = 0.0
    return (x_b - x_a) / 2.0 * x_canon + (x_a + x_b) / 2.0

def lev2knots_doubling(level):
    """Converte un livello in un numero di nodi (regola di Clenshaw-Curtis)."""
    if level <= 0: return 0
    if level == 1: return 1
    return 2**(level - 1) + 1

# --- Funzioni per la costruzione della Struttura Sparse Grid ---

def tensor_grid_knots(N, m, knots_func_1d, domains):
    """Crea i nodi di una griglia tensoriale date le risoluzioni per dimensione."""
    knots_per_dim = [knots_func_1d(int(m[i]), domains[i][0], domains[i][1]) for i in range(N)]
    grids = np.meshgrid(*knots_per_dim, indexing='ij')
    points = np.vstack([grid.ravel() for grid in grids])
    return {'knots': points, 'size': points.shape[1], 'knots_per_dim': knots_per_dim}

def generate_multi_indices_total_degree(N, level):
    """Genera i multi-indici per una sparse grid basata sulla regola "total degree"."""
    valid_indices = []
    for i_tuple in itertools.product(range(1, level + N + 1), repeat=N):
        if sum(i_tuple) <= level + N:
            valid_indices.append(i_tuple)
    return np.array(sorted(valid_indices, key=lambda x: (sum(x), x)))

def create_sparse_grid_structure(N, level, level_to_knots_func, knots_func_1d, domains):
    """Crea la struttura della sparse grid (collezione di griglie tensoriali con coefficienti)."""
    multi_indices = generate_multi_indices_total_degree(N, level)
    S = []
    for idx_vec in multi_indices:
        q = sum(idx_vec)
        coeff = ((-1)**(level + N - q)) * comb(N - 1, level + N - q, exact=True)
        if abs(coeff) < 1e-12:
            continue
        m = np.array([level_to_knots_func(val) for val in idx_vec], dtype=int)
        grid = tensor_grid_knots(N, m, knots_func_1d, domains)
        grid['coeff'] = coeff
        S.append(grid)
    return S

def reduce_sparse_grid_knots(S, tol=1e-14):
    """Estrae e rende unici i nodi da tutte le griglie tensoriali nella struttura S."""
    if not S: return np.array([[]])
    all_knots = np.hstack([grid['knots'] for grid in S]).T
    if all_knots.shape[0] == 0: return np.array([[]])
    sorted_indices = np.lexsort(np.flipud(all_knots.T))
    sorted_knots = all_knots[sorted_indices, :]
    diffs = np.diff(sorted_knots, axis=0)
    is_unique_mask = np.concatenate(([True], np.max(np.abs(diffs), axis=1) > tol))
    return sorted_knots[is_unique_mask, :].T

# --- Funzioni per la Valutazione dell'Interpolante ---

def _evaluate_lagrange_poly_1d(x_eval, knots_1d, j_index):
    """Valuta il j-esimo polinomio di base di Lagrange 1D."""
    num, den = 1.0, 1.0
    knot_j = knots_1d[j_index]
    for k, knot_k in enumerate(knots_1d):
        if k != j_index:
            num *= (x_eval - knot_k)
            den *= (knot_j - knot_k)
    return num / den if abs(den) > 1e-16 else 1.0

def evaluate_sparse_grid_interpolant(y_eval_points, S, unique_knots, f_on_unique_knots):
    """Valuta l'interpolante sparse grid su un set di punti di valutazione."""
    f_map = {tuple(np.round(k, 12)): v for k, v in zip(unique_knots.T, f_on_unique_knots)}
    N = unique_knots.shape[0]
    num_eval_points = y_eval_points.shape[1]
    f_approx_values = np.zeros(num_eval_points)
    for p in range(num_eval_points):
        y_eval = y_eval_points[:, p]
        total_value = 0.0
        for grid in S:
            tensor_interpolant = 0.0
            for j_node_idx in range(grid['size']):
                node_y_j = grid['knots'][:, j_node_idx]
                f_val = f_map.get(tuple(np.round(node_y_j, 12)))
                if f_val is None:
                    continue
                lagrange_basis = 1.0
                for n in range(N):
                    knots_1d_n = grid['knots_per_dim'][n]
                    knot_idx_1d = np.where(np.isclose(knots_1d_n, node_y_j[n]))[0][0]
                    lagrange_basis *= _evaluate_lagrange_poly_1d(y_eval[n], knots_1d_n, knot_idx_1d)
                tensor_interpolant += f_val * lagrange_basis
            total_value += grid['coeff'] * tensor_interpolant
        f_approx_values[p] = total_value
    return f_approx_values

def compute_discrepancy_indicator(candidate_knots, w_start, N, domains, all_known_knots, all_known_values):
    """Calcola l'indicatore di discrepanza (errore gerarchico) in modo robusto."""
    if candidate_knots.shape[1] == 0:
        return np.array([])
        
    S_start = create_sparse_grid_structure(N, w_start, lev2knots_doubling, knots_cc_1d, domains)
    S_prev = create_sparse_grid_structure(N, w_start - 1, lev2knots_doubling, knots_cc_1d, domains)
    
    preds_start = evaluate_sparse_grid_interpolant(candidate_knots, S_start, all_known_knots, all_known_values)
    preds_prev = evaluate_sparse_grid_interpolant(candidate_knots, S_prev, all_known_knots, all_known_values)
    
    # --- LOGICA DI CALCOLO IBRIDA E ROBUSTA ---
    abs_diff = np.abs(preds_start - preds_prev)
    denominator = np.abs(preds_start)
    indicator = np.zeros_like(denominator)
    
    # Maschera per identificare dove il denominatore è abbastanza grande per una divisione stabile
    non_zero_mask = denominator > 1e-9
    
    # CASO 1: Il valore predetto è lontano da zero -> usa l'errore RELATIVO
    indicator[non_zero_mask] = abs_diff[non_zero_mask] / denominator[non_zero_mask]
    
    # CASO 2: Il valore predetto è vicino a zero -> usa l'errore ASSOLUTO per evitare instabilità
    indicator[~non_zero_mask] = abs_diff[~non_zero_mask]
    
    return indicator