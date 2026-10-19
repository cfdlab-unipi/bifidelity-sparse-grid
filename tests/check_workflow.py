"""Check file I/O and all three phases with vector simulation outputs."""

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
from bifidelity_sparse_grid.workflow_phases import run_phase_1, run_phase_2, run_phase_3

# Exercise an empty selection, a partial selection, and the full target. Use
# weighted-PDF automatic rules in the partial case to check the old default.
for budget_percent, budget_mode in [(0, "percent"), (1, "percent"), (20, "percent"), (100, "percent"), (20, "fixed"), (20, "threshold"), (20, "elbow")]:
    with tempfile.TemporaryDirectory(prefix="sparse_grid_workflow_") as temporary:
        folder = Path(temporary)
        config = {
            "project_name": "vector_test",
            "data_dir": str(folder),
            "output_dir": str(folder),
            "problem_definition": {
                "dim_names": ["x", "y"],
                "domains": [[-1, 1], [-1, 1]],
                "output_quantities": [{"name": "signal", "size": 2}],
            },
            "sparse_grid_settings": {"base_level": 1, "target_level": 2},
            "refinement_settings": {
                "refinement_criterion": "signal",
                "indicator_type": "symmetric_relative",
                "budget_percent": budget_percent,
            },
            "validation": {"validation_data_file": str(folder / "validation.csv")},
            "plotting": {"slices_2d": [], "plot_resolution": 10},
            "statistics": {"enabled": False, "export_lagrange_coeffs": False},
        }
        # Verify the historical fixed-count and automatic budget settings too.
        if budget_mode != "percent":
            del config["refinement_settings"]["budget_percent"]
            config["refinement_settings"]["include_ties"] = False
            if budget_mode == "fixed":
                config["refinement_settings"]["max_budget"] = 2
            else:
                config["refinement_settings"]["use_auto_budget"] = True
                config["refinement_settings"]["auto_budget_method"] = budget_mode
                config["refinement_settings"]["max_budget"] = 3

        if budget_percent == 20:
            config["problem_definition"]["pdfs"] = ["uniform", {"type": "normal", "params": {"mu": 0, "sigma": 0.6}}]
        elif budget_percent == 100:
            # The old configuration omits indicator_type. Its legacy mode
            # must still execute through the same input/output workflow.
            del config["refinement_settings"]["indicator_type"]

        # Capture routine phase messages so a failed assertion is easy to see.
        with contextlib.redirect_stdout(io.StringIO()):
            run_phase_1(config)
            baseline = pd.read_csv(folder / "vector_test_phase1_to_fill.dat", sep=r"\s+")
            baseline["signal_t0000"] = baseline["x"] + 2 * baseline["y"]
            baseline["signal_t0001"] = baseline["x"] ** 2 * baseline["y"]
            baseline.to_csv(folder / "vector_test_phase1.dat", sep="\t", index=False)
            run_phase_2(config)

            selection = pd.read_csv(folder / "vector_test_selection.csv")
            refinement = selection.loc[selection["selected"], ["x", "y"]].copy()
            refinement["signal_t0000"] = refinement["x"] + 2 * refinement["y"]
            refinement["signal_t0001"] = refinement["x"] ** 2 * refinement["y"]
            refinement_path = folder / "vector_test_phase2.dat"
            refinement.to_csv(refinement_path, sep="\t", index=False)

            # A separate validation file checks metric export for components
            # even when no vector components have been requested for plotting.
            validation = pd.DataFrame({"x": [-0.7, 0.2, 0.9], "y": [0.4, -0.5, 0.3]})
            validation["signal_t0000"] = validation["x"] + 2 * validation["y"]
            validation["signal_t0001"] = validation["x"] ** 2 * validation["y"]
            validation.to_csv(folder / "validation.csv", index=False)
            run_phase_3(config)

            metrics = pd.read_csv(folder / "vector_test_validation_metrics.csv")
            assert len(metrics) == 4
            assert np.all(np.isfinite(metrics["rmse"]))
            with open(folder / "vector_test_selection.json") as file:
                summary = json.load(file)
            assert summary["selected_count"] == int(selection["selected"].sum())
            assert summary["total_evaluations"] == len(baseline) + len(refinement)
            if budget_mode == "percent":
                expected_nominal = int(np.floor(budget_percent * len(selection) / 100.0))
                assert summary["nominal_budget"] == expected_nominal
                if expected_nominal == 0:
                    assert len(refinement) == 0
            if budget_mode == "fixed":
                assert len(refinement) <= 2
            elif budget_mode in ["threshold", "elbow"]:
                assert len(refinement) <= 3
            if budget_percent == 0:
                assert len(refinement) == 0
            if budget_percent == 100:
                assert len(refinement) == len(selection)
                assert metrics.loc[metrics["model"] == "informed", "rmse"].max() < 1e-11

            # An incomplete required simulation must fail explicitly. Optional
            # unfilled points are allowed by the file reader and remain ignored.
            if len(refinement):
                refinement.iloc[1:].to_csv(refinement_path, sep="\t", index=False)
                try:
                    run_phase_3(config)
                except ValueError as error:
                    assert "selected refinement points" in str(error)
                else:
                    raise AssertionError("Missing selected evaluations must be rejected.")
        print(f"PASS: vector workflow, selection {budget_mode} ({budget_percent}), file I/O, metrics, and required-point checks")
print("All workflow checks passed.")
