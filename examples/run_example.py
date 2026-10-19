"""Run all three toolkit phases on an analytical function.

Change the command-line parameters to compare the two functions and grids.
All generated data and figures are written inside the chosen output folder.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from scipy.stats import qmc

# Import the toolkit from the repository containing this example.
REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
from bifidelity_sparse_grid.workflow_phases import run_phase_1, run_phase_2, run_phase_3

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--case", choices=["ishigami", "oscillatory"], default="ishigami")
parser.add_argument("--grid", choices=["CC", "leja_line", "leja_sym"], default="CC")
parser.add_argument("--budget-percent", type=float, default=20.0)
parser.add_argument("--output-dir", type=Path)
args = parser.parse_args()

# The same function supplies baseline, selected, and independent validation data.
# Replace this function with calls to your own model for another demonstration.
def evaluate_example(points):
    if args.case == "ishigami":
        return np.sin(points[0]) + 7.0 * np.sin(points[1]) ** 2 + 0.1 * points[2] ** 4 * np.sin(points[0])
    frequencies = np.pi * np.array([1.5, 3.0, 0.5, 4.5])
    return np.sum(np.cos(frequencies[:, None] * points), axis=0)

if args.case == "ishigami":
    dimension = 3
    domain = [-np.pi, np.pi]
else:
    dimension = 4
    domain = [0.0, 1.0]

dim_names = []
for index in range(dimension):
    dim_names.append(f"x{index + 1}")

folder = args.output_dir
if folder is None:
    folder = REPOSITORY / "examples" / "output" / f"{args.case}_{args.grid}"
folder = folder.resolve()
data_dir = folder / "data"
output_dir = folder / "analysis"
data_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)

# Fix the remaining coordinates to the domain midpoint for the response plots.
fixed_coordinates = {}
for name in dim_names[2:]:
    fixed_coordinates[name] = 0.5 * (domain[0] + domain[1])

# Leja grows by one node per one-dimensional level, while CC doubles.
# These small defaults give broadly similar node counts, without claiming a
# matched-cost comparison. The actual counts are saved in the selection JSON.
if args.grid == "CC":
    auxiliary_level = 1
    base_level = 2
    target_level = 3
else:
    auxiliary_level = 2
    base_level = 3
    target_level = 5

config = {
    "project_name": args.case,
    "data_dir": str(data_dir),
    "output_dir": str(output_dir),
    "problem_definition": {
        "dim_names": dim_names,
        "domains": [domain] * dimension,
        "output_quantities": ["response"],
    },
    "sparse_grid_settings": {
        "grid_type": args.grid,
        "auxiliary_level": auxiliary_level,
        "base_level": base_level,
        "target_level": target_level,
    },
    "refinement_settings": {
        "refinement_criterion": "response",
        "indicator_type": "symmetric_relative",
        "relative_epsilon": 1e-5,
        "budget_percent": args.budget_percent,
        "include_ties": True,
        "tie_tolerance": 1e-3,
    },
    "validation": {"validation_data_file": str(data_dir / "validation.csv")},
    "plotting": {
        "plot_resolution": 20,
        "slices_2d": [{"varying": dim_names[:2], "fixed": fixed_coordinates}],
    },
    "statistics": {"enabled": True, "quad_order": 40, "export_lagrange_coeffs": False},
}
with open(folder / "config.json", "w") as file:
    json.dump(config, file, indent=2)

# Phase 1 writes the baseline template. Evaluate every row with the exact model.
run_phase_1(config)
baseline = pd.read_csv(data_dir / f"{args.case}_phase1_to_fill.dat", sep=r"\s+")
baseline["response"] = evaluate_example(baseline[dim_names].values.T)
baseline.to_csv(data_dir / f"{args.case}_phase1.dat", sep="\t", index=False)

# Phase 2 ranks candidate nodes. Evaluate only the selected ones; the toolkit
# inherits values at all remaining target nodes from the baseline interpolant.
run_phase_2(config)
selection = pd.read_csv(output_dir / f"{args.case}_selection.csv")
refinement = selection.loc[selection["selected"], dim_names].copy()
refinement["response"] = evaluate_example(refinement[dim_names].values.T)
refinement.to_csv(data_dir / f"{args.case}_phase2.dat", sep="\t", index=False)

# Independent scrambled-Sobol points give the same validation set for all grids.
sampler = qmc.Sobol(dimension, scramble=True, seed=42)
validation_points = qmc.scale(sampler.random_base2(10), [domain[0]] * dimension, [domain[1]] * dimension)
validation = pd.DataFrame(validation_points, columns=dim_names)
validation["response"] = evaluate_example(validation_points.T)
validation.to_csv(data_dir / "validation.csv", index=False)

# Phase 3 writes response surfaces, error plots, scalar metrics, and statistics.
run_phase_3(config)
print(f"\nResults: {output_dir}")
print(pd.read_csv(output_dir / f"{args.case}_validation_metrics.csv").to_string(index=False))
