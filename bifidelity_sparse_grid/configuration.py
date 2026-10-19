"""Read project settings and check common mistakes before generating files."""

import json
import math
from pathlib import Path


def load_configuration(config_path):
    """Resolve project files relative to config.json, independently of the shell."""
    config_path = Path(config_path).expanduser().resolve()
    with open(config_path, "r") as file:
        config = json.load(file)
    if not isinstance(config, dict):
        raise ValueError("The configuration must be a JSON object.")

    # Keep simulation inputs and analysis outputs beside the configuration.
    # Absolute paths are also allowed and retain their original meaning.
    for key, default in [("data_dir", "data"), ("output_dir", "output")]:
        path = Path(config.get(key, default)).expanduser()
        if not path.is_absolute():
            path = config_path.parent / path
        config[key] = str(path.resolve())
    validation_file = config.get("validation", {}).get("validation_data_file")
    if validation_file:
        path = Path(validation_file).expanduser()
        if not path.is_absolute():
            path = config_path.parent / path
        config["validation"]["validation_data_file"] = str(path.resolve())

    # Plotting is optional: a one-dimensional problem can omit slices entirely.
    config.setdefault("plotting", {})
    config["plotting"].setdefault("slices_2d", [])
    config["plotting"].setdefault("plot_resolution", 40)
    for slice_info in config["plotting"]["slices_2d"]:
        slice_info.setdefault("fixed", {})
    return config


def check_configuration(config):
    """Check problem, grid, refinement, and slice settings without writing files."""
    # Reuse the numerical workflow's own settings, rather than maintaining a
    # second implementation of PDF parsing or grid-family selection.
    from .workflow_phases import _expand_qoi_specs, _grid_settings, _resolve_plot_columns

    project_name = config["project_name"]
    if not isinstance(project_name, str) or not project_name.strip():
        raise ValueError("project_name must be a non-empty string.")
    if "/" in project_name or "\\" in project_name or project_name in [".", ".."]:
        raise ValueError("project_name must be a name, without directory separators.")
    problem = config["problem_definition"]
    dimensions = problem["dim_names"]
    domains = problem["domains"]
    if not dimensions or not all(isinstance(name, str) and name.strip() for name in dimensions):
        raise ValueError("dim_names must contain non-empty input names.")
    if len(set(dimensions)) != len(dimensions):
        raise ValueError("Input names in dim_names must be unique.")
    if len(domains) != len(dimensions):
        raise ValueError("Provide one domain per input dimension.")
    for domain in domains:
        if len(domain) != 2 or not all(math.isfinite(value) for value in domain) or domain[1] <= domain[0]:
            raise ValueError("Each domain must contain two finite bounds with max > min.")

    quantities = _expand_qoi_specs(problem)
    if not quantities:
        raise ValueError("Define at least one output quantity.")
    quantity_names = []
    columns = []
    for quantity in quantities:
        quantity_names.append(quantity["name"])
        columns.extend(quantity["columns"])
    if len(set(quantity_names)) != len(quantity_names) or len(set(columns)) != len(columns):
        raise ValueError("Output quantity names and expanded columns must be unique.")
    # Simulation templates use whitespace-separated columns, so names must
    # not contain spaces or tabs. Catch this before asking for model evaluations.
    for name in dimensions + columns:
        if not isinstance(name, str) or not name or any(character.isspace() for character in name):
            raise ValueError("Input and output column names must be non-empty and contain no whitespace.")
    if set(columns).intersection(dimensions):
        raise ValueError("Input and output column names must be different.")
    _grid_settings(problem, config["sparse_grid_settings"])

    refinement = config["refinement_settings"]
    if refinement["refinement_criterion"] not in quantity_names + columns:
        raise ValueError("refinement_criterion must name an output quantity or vector component.")
    if refinement.get("indicator_type", "legacy") not in ["legacy", "absolute", "relative", "symmetric_relative"]:
        raise ValueError("Unknown indicator_type; use absolute or symmetric_relative.")
    epsilon = refinement.get("relative_epsilon", 1e-5)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("relative_epsilon must be finite and positive.")
    tolerance = refinement.get("tie_tolerance", 1e-3)
    if not math.isfinite(tolerance) or not 0 <= tolerance < 1:
        raise ValueError("tie_tolerance must lie between zero (included) and one (excluded).")
    for key in ["include_ties", "use_auto_budget"]:
        if key in refinement and not isinstance(refinement[key], bool):
            raise ValueError(f"{key} must be true or false, without quotation marks.")
    if "max_budget" in refinement:
        count = refinement["max_budget"]
        if not math.isfinite(count) or count < 0 or int(count) != count:
            raise ValueError("max_budget must be a non-negative integer.")
    if "budget_percent" in refinement:
        percent = refinement["budget_percent"]
        if not math.isfinite(percent) or not 0 <= percent <= 100:
            raise ValueError("budget_percent must lie between zero and 100.")
    if refinement.get("use_auto_budget", False):
        if refinement.get("auto_budget_method", "elbow") not in ["elbow", "threshold"]:
            raise ValueError("auto_budget_method must be elbow or threshold.")
        threshold = refinement.get("auto_budget_threshold", 0.1)
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("auto_budget_threshold must lie between zero and one.")

    # A slice varies exactly two inputs. Any fixed input must be a different,
    # known dimension with a coordinate inside its physical domain.
    plotting = config.get("plotting", {})
    _resolve_plot_columns(plotting, quantities)
    resolution = plotting.get("plot_resolution", 40)
    if int(resolution) != resolution or resolution < 2:
        raise ValueError("plot_resolution must be an integer of at least two.")
    for slice_info in plotting.get("slices_2d", []):
        varying = slice_info["varying"]
        if len(varying) != 2 or len(set(varying)) != 2 or not set(varying).issubset(dimensions):
            raise ValueError("Each slice must vary two different known inputs.")
        for name, value in slice_info.get("fixed", {}).items():
            if name not in dimensions or name in varying:
                raise ValueError("Fixed slice inputs must be known and different from the varying inputs.")
            bounds = domains[dimensions.index(name)]
            if not math.isfinite(value) or not bounds[0] <= value <= bounds[1]:
                raise ValueError(f"Fixed slice coordinate for '{name}' is outside its domain.")
