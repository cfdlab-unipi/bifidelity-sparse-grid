"""Check an installed toolkit from outside the source checkout.

Run with the Python interpreter of an environment containing the installed
package. All generated projects and simulation files are temporary.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

# Keep the interpreter's original path: resolving a virtual-environment Python
# symlink would accidentally select the base environment instead.
python = sys.executable
with tempfile.TemporaryDirectory(prefix="sparse_grid_cli_") as temporary:
    workspace = Path(temporary)
    project = workspace / "my-study"
    command = [python, "-m", "bifidelity_sparse_grid"]
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    environment["OPENBLAS_NUM_THREADS"] = "1"

    # Initialization creates a standalone project without requiring a checkout.
    result = subprocess.run(command + ["init", str(project)], cwd=workspace, env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    config_path = project / "config.json"
    original_config = config_path.read_bytes()
    assert "data/my-study_phase1_to_fill.dat" in (project / "README.md").read_text()
    result = subprocess.run(command + ["init", str(project)], cwd=workspace, env=environment, capture_output=True, text=True)
    assert result.returncode == 1
    assert config_path.read_bytes() == original_config
    print("PASS: installed project initialization and protection against overwriting")

    # Use installed Leja data, relative paths, and an independent validation file.
    config = json.loads(config_path.read_text())
    config["sparse_grid_settings"]["grid_type"] = "leja_sym"
    config["validation"]["validation_data_file"] = "validation.csv"
    config_path.write_text(json.dumps(config, indent=4))
    files_before = set(project.rglob("*"))
    result = subprocess.run(command + ["check", "--config", str(config_path)], cwd=workspace, env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert set(project.rglob("*")) == files_before
    print("PASS: configuration checks write no files")

    # Execute all three phases from a different working directory. The analytic
    # response substitutes for an external simulation in this integration check.
    result = subprocess.run(command + ["--phase", "1", "--config", str(config_path)], cwd=workspace, env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    baseline = pd.read_csv(project / "data/my-study_phase1_to_fill.dat", sep=r"\s+")
    baseline["response"] = baseline["x1"] ** 2 + np.sin(3 * baseline["x2"]) + baseline["x1"] * baseline["x2"]
    baseline.to_csv(project / "data/my-study_phase1.dat", sep="\t", index=False)
    result = subprocess.run(command + ["--phase", "2", "--config", str(config_path)], cwd=workspace, env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    selection = pd.read_csv(project / "output/my-study_selection.csv")
    refinement = selection.loc[selection["selected"], ["x1", "x2"]].copy()
    refinement["response"] = refinement["x1"] ** 2 + np.sin(3 * refinement["x2"]) + refinement["x1"] * refinement["x2"]
    refinement.to_csv(project / "data/my-study_phase2.dat", sep="\t", index=False)
    validation = pd.DataFrame({"x1": [0.15, 0.4, 0.8], "x2": [0.7, 0.3, 0.5]})
    validation["response"] = validation["x1"] ** 2 + np.sin(3 * validation["x2"]) + validation["x1"] * validation["x2"]
    validation.to_csv(project / "validation.csv", index=False)
    result = subprocess.run(command + ["--phase", "3", "--config", str(config_path)], cwd=workspace, env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    metrics = pd.read_csv(project / "output/my-study_validation_metrics.csv")
    assert len(metrics) == 2 and np.all(np.isfinite(metrics["rmse"]))
    assert not (workspace / "data").exists()
    assert not (workspace / "output").exists()
    print("PASS: installed three-phase workflow, packaged Leja data, and config-relative paths")

    # Constant references have no range for percentage normalization.
    validation["response"] = 1.0
    validation.to_csv(project / "validation.csv", index=False)
    result = subprocess.run(command + ["--phase", "3", "--config", str(config_path)], cwd=workspace, env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "Constant reference" in result.stdout
    print("PASS: constant validation references retain absolute metrics")

    (project / "validation.csv").write_text("")
    result = subprocess.run(command + ["--phase", "3", "--config", str(config_path)], cwd=workspace, env=environment, capture_output=True, text=True)
    assert result.returncode == 1
    assert "Could not read validation file" in result.stderr
    assert "Phase completed successfully" not in result.stdout
    print("PASS: unreadable validation data returns an error")

    # Invalid settings must fail before files are generated, with a concise
    # message instead of a traceback intended for a Python developer.
    config["refinement_settings"]["budget_percent"] = -1
    config_path.write_text(json.dumps(config, indent=4))
    result = subprocess.run(command + ["check", "--config", str(config_path)], cwd=workspace, env=environment, capture_output=True, text=True)
    assert result.returncode == 1
    assert "budget_percent" in result.stderr and "Traceback" not in result.stderr
    print("PASS: invalid configuration produces a clear failure")

    # The complete vector starter is also distributed inside the wheel.
    vector_project = workspace / "vector-study"
    result = subprocess.run(command + ["init", str(vector_project), "--template", "vector"], cwd=workspace, env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "hifi-data/vector-study_phase1_to_fill.dat" in (vector_project / "README.md").read_text()
    result = subprocess.run(command + ["check", "--config", str(vector_project / "config.json")], cwd=workspace, env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    print("PASS: installed vector project template")
print("All installed CLI checks passed.")
