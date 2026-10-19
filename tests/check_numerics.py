"""Run numerical regression checks with Python and the toolkit dependencies."""

import itertools
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

# Use the source files from this checkout, with no external benchmark directory.
REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
from bifidelity_sparse_grid.sparse_grid_toolbox import (
    PdfSpec, build_1d_knots_and_domains_from_pdfspecs,
    compute_discrepancy_indicator, compute_sparse_grid_quadrature_weights,
    create_sparse_grid_structure, evaluate_smolyak_from_nodal_values,
    generate_multi_indices_total_degree, knots_cc_1d, knots_leja_line_1d,
    knots_leja_sym_line_1d, lev2knots_doubling, lev2knots_lin,
    reduce_sparse_grid_knots, select_refinement_points, leja_nodes_and_weights_1d,
)
from bifidelity_sparse_grid.workflow_phases import _build_informed_model_values, _parse_pdf_specs
from bifidelity_sparse_grid.sparse_grid_toolbox import pdfspec_to_domain_and_rho
from bifidelity_sparse_grid.nodes import _weighted_leja_1d

# Compare index generation against exhaustive enumeration on small problems.
# The 11-dimensional count also checks the case that was formerly impractical.
for dimension in range(1, 5):
    for level in range(4):
        expected = []
        for indices in itertools.product(range(1, level + 2), repeat=dimension):
            if sum(indices) <= dimension + level:
                expected.append(indices)
        expected.sort(key=lambda values: (sum(values), values))
        np.testing.assert_array_equal(generate_multi_indices_total_degree(dimension, level), expected)
assert len(generate_multi_indices_total_degree(11, 3)) == math.comb(14, 3)
print("PASS: admissible multi-indices and high-dimensional node generation")

# Check cardinal interpolation, polynomial reproduction, several value channels,
# and batching for each supported grid family, including unequal domains.
rng = np.random.default_rng(42)
domains = [[-2.0, 3.0], [0.0, 2.0]]
points = rng.uniform(size=(2, 31))
points[0] = -2.0 + 5.0 * points[0]
points[1] *= 2.0
rules = [(knots_cc_1d, lev2knots_doubling), (knots_leja_line_1d, lev2knots_lin), (knots_leja_sym_line_1d, lev2knots_lin)]
for rule, growth in rules:
    grid = create_sparse_grid_structure(2, 2, growth, rule, domains)
    knots = reduce_sparse_grid_knots(grid)
    values = 2.0 + knots[0] ** 2 - knots[1] + knots[0] * knots[1]
    exact = 2.0 + points[0] ** 2 - points[1] + points[0] * points[1]
    np.testing.assert_allclose(evaluate_smolyak_from_nodal_values(knots, grid, knots, values), values, atol=2e-11)
    channels = np.vstack([values, -3.0 * values, np.ones_like(values)])
    expected = np.vstack([exact, -3.0 * exact, np.ones_like(exact)])
    for batch_size in [1, 7, 100]:
        actual = evaluate_smolyak_from_nodal_values(points, grid, knots, channels, batch_size=batch_size)
        np.testing.assert_allclose(actual, expected, atol=2e-11)

    # This independent product-form Lagrange sum checks arbitrary nodal values,
    # beyond the polynomial examples above, against the combination formula.
    arbitrary = rng.normal(size=knots.shape[1])
    reference = np.zeros(3)
    for point_index, point in enumerate(points[:, :3].T):
        for tensor in grid:
            for node in tensor["knots"].T:
                node_index = np.flatnonzero(np.all(np.isclose(knots.T, node, rtol=0, atol=1e-13), axis=1))[0]
                basis = 1.0
                for dimension, knots_1d in enumerate(tensor["knots_per_dim"]):
                    for other in knots_1d:
                        if abs(other - node[dimension]) > 1e-13:
                            basis *= (point[dimension] - other) / (node[dimension] - other)
                reference[point_index] += tensor["coeff"] * arbitrary[node_index] * basis
    actual = evaluate_smolyak_from_nodal_values(points[:, :3], grid, knots, arbitrary)
    np.testing.assert_allclose(actual, reference, atol=2e-11)

    # With no extra evaluations the reconstructed target model must reproduce
    # the baseline; with all target values it must reproduce the target model.
    target = create_sparse_grid_structure(2, 3, growth, rule, domains)
    target_knots = reduce_sparse_grid_knots(target)
    known = pd.DataFrame(knots.T, columns=["x", "y"])
    known["response"] = arbitrary
    inherited = _build_informed_model_values(knots, arbitrary, target_knots, known, "response", ["x", "y"], grid)
    prediction = evaluate_smolyak_from_nodal_values(points, target, target_knots, inherited)
    np.testing.assert_allclose(prediction, evaluate_smolyak_from_nodal_values(points, grid, knots, arbitrary), atol=2e-10)
    known = pd.DataFrame(target_knots.T, columns=["x", "y"])
    known["response"] = rng.normal(size=target_knots.shape[1])
    complete = _build_informed_model_values(knots, arbitrary, target_knots, known, "response", ["x", "y"], grid)
    np.testing.assert_allclose(complete, known["response"].values, atol=1e-13)
print("PASS: CC and Leja interpolation, batching, vector channels, baseline and target limits")

# Prefixes must stay nested, since previous simulations are reused at new levels.
for rule in [knots_leja_line_1d, knots_leja_sym_line_1d]:
    previous = rule(1, -1, 1)
    for count in range(2, 32):
        current = rule(count, -1, 1)
        np.testing.assert_allclose(current[:count - 1], previous, rtol=0, atol=1e-13)
        previous = current
for sequence in ["line", "symmetric"]:
    for count in range(1, 32):
        nodes, weights = leja_nodes_and_weights_1d(count, -1, 1, sequence)
        assert len(nodes) == count and len(weights) == count
        np.testing.assert_allclose(weights.sum(), 1.0, atol=5e-12)
print("PASS: all 31 tabulated Leja prefixes remain nested")

# Weighted nonuniform rules must retain interpolation and probability weights.
specs = [PdfSpec("normal", {"mu": 0.0, "sigma": 0.7, "a": -2.0, "b": 2.0}, symmetric=True)]
node_rules, weighted_domains = build_1d_knots_and_domains_from_pdfspecs(specs)
grid = create_sparse_grid_structure(1, 2, lev2knots_lin, node_rules, weighted_domains)
knots = reduce_sparse_grid_knots(grid)
values = knots[0] ** 2
np.testing.assert_allclose(evaluate_smolyak_from_nodal_values(knots, grid, knots, values), values, atol=1e-12)
rho = lambda x: np.exp(-0.5 * (x / 0.7) ** 2)
weights = compute_sparse_grid_quadrature_weights(grid, knots, [rho], weighted_domains, quad_order=80)
np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-12)
np.testing.assert_allclose(weights @ knots[0], 0.0, atol=1e-12)
print("PASS: weighted-PDF interpolation and quadrature")

# Compare the stabilized indicator with its defining equation. Vector components
# change sign here, so interpolating their nodal norms would give a wrong answer.
grid = create_sparse_grid_structure(1, 1, lev2knots_doubling, knots_cc_1d, [[-1, 1]])
knots = reduce_sparse_grid_knots(grid)
points = np.array([[-0.75, 0.0, 0.25, 0.8]])
values = np.vstack([knots[0], -2.0 * knots[0]])
absolute = np.sqrt(5.0) * np.abs(points[0])
actual = compute_discrepancy_indicator(points, 1, 1, [[-1, 1]], knots, values, indicator_type="absolute")
np.testing.assert_allclose(actual, absolute, atol=1e-12)
expected = 2.0 * absolute / (absolute + 1e-5 * absolute.max())
actual = compute_discrepancy_indicator(points, 1, 1, [[-1, 1]], knots, values, indicator_type="symmetric_relative")
np.testing.assert_allclose(actual, expected, atol=1e-12)
zero = compute_discrepancy_indicator(points, 1, 1, [[-1, 1]], knots, np.zeros(knots.shape[1]), indicator_type="relative")
np.testing.assert_array_equal(zero, np.zeros(points.shape[1]))
print("PASS: scalar/vector indicators and zero-function stabilization")

# Verify nominal count, actual count, strict limits, and near-equal cutoffs.
indicators = np.array([3.0, 2.0, 1.999, 0.1])
np.testing.assert_array_equal(select_refinement_points(indicators, 2), [0, 1, 2])
np.testing.assert_array_equal(select_refinement_points(indicators, 2, include_ties=False), [0, 1])
assert len(select_refinement_points(indicators, 0)) == 0
np.testing.assert_array_equal(select_refinement_points(np.zeros(4), 4), np.arange(4))
for invalid in [np.array([np.nan]), np.array([-1.0]), np.array([np.inf])]:
    try:
        select_refinement_points(invalid, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid indicators must be rejected.")
print("PASS: deterministic selection, cutoff ties, zero budgets, and invalid inputs")
# Apply equivalence tolerance to the threshold, not the last retained rank.
indicators = np.array([10.0, 5.1, 4.997, 0.0])
np.testing.assert_array_equal(select_refinement_points(indicators, 2, indicator_cutoff=5.0), [0, 1, 2])
np.testing.assert_array_equal(select_refinement_points(indicators, 2, include_ties=False, indicator_cutoff=5.0), [0, 1])
print("PASS: threshold cutoff and equivalence band")

# Check reference candidate and selected counts for both analytical functions.
for case in ["oscillatory", "ishigami"]:
    if case == "oscillatory":
        dimension, domain = 4, [0.0, 1.0]
        expected_candidates, expected_selected = 96, 47
    else:
        dimension, domain = 3, [-np.pi, np.pi]
        expected_candidates, expected_selected = 44, 24
    domains = [domain] * dimension
    baseline_grid = create_sparse_grid_structure(dimension, 2, lev2knots_doubling, knots_cc_1d, domains)
    target_grid = create_sparse_grid_structure(dimension, 3, lev2knots_doubling, knots_cc_1d, domains)
    baseline_knots = reduce_sparse_grid_knots(baseline_grid)
    target_knots = reduce_sparse_grid_knots(target_grid)
    baseline_keys = set(map(tuple, np.round(baseline_knots.T, 12)))
    candidate_indices = []
    for index, point in enumerate(target_knots.T):
        if tuple(np.round(point, 12)) not in baseline_keys:
            candidate_indices.append(index)
    candidates = target_knots[:, candidate_indices]
    if case == "oscillatory":
        frequencies = np.pi * np.array([1.5, 3.0, 0.5, 4.5])
        values = np.sum(np.cos(frequencies[:, None] * baseline_knots), axis=0)
    else:
        values = np.sin(baseline_knots[0]) + 7 * np.sin(baseline_knots[1]) ** 2 + 0.1 * baseline_knots[2] ** 4 * np.sin(baseline_knots[0])
    indicators = compute_discrepancy_indicator(candidates, 2, dimension, domains, baseline_knots, values, indicator_type="symmetric_relative")
    cutoff = 0.05 * np.max(indicators)
    count_above = int(np.sum(indicators >= cutoff))
    selected = select_refinement_points(indicators, count_above, indicator_cutoff=cutoff)
    assert len(candidate_indices) == expected_candidates and len(selected) == expected_selected
    print(f"PASS: {case} threshold selection: {len(selected)}/{len(candidate_indices)}")

# Check documented distribution forms independently of their input parsing.
problem = {"dim_names": ["x"], "domains": [[0.0, 1.0]], "pdfs": [{"type": "exponential", "params": {"lambda": 3.0}}]}
spec = _parse_pdf_specs(problem)[0]
_, density = pdfspec_to_domain_and_rho(spec)
points_1d = np.array([0.0, 0.25, 0.75])
np.testing.assert_allclose(density(points_1d), 3.0 * np.exp(-3.0 * points_1d))
problem["pdfs"] = ["beta"]
spec = _parse_pdf_specs(problem)[0]
_, density = pdfspec_to_domain_and_rho(spec)
np.testing.assert_allclose(density(points_1d), np.ones(3))

# A finite candidate search cannot supply more distinct nodes than candidates.
# Infinite endpoint densities must also fail before duplicate nodes are produced.
for count, density, size in [(4, np.ones_like, 3), (2, lambda x: np.full_like(x, np.inf), 5)]:
    try:
        _weighted_leja_1d(count, 0.0, 1.0, density, grid_size=size)
    except ValueError:
        pass
    else:
        raise AssertionError("Unsupported weighted-Leja requests must fail.")
print("PASS: exponential alias, beta defaults, and weighted-Leja limits")
print("All numerical checks passed.")
