"""Create a project, check its settings, or execute one simulation phase."""

import argparse
from importlib.resources import files
import json
from pathlib import Path
import sys
import traceback

from .configuration import load_configuration, check_configuration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=["init", "check", "run"], default="run")
    parser.add_argument("project_directory", nargs="?", help="New project folder, used with init.")
    parser.add_argument("--template", choices=["scalar", "vector"], default="scalar", help="Starting configuration for init.")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], help="1: baseline points; 2: select points; 3: analyze.")
    parser.add_argument("--config", default="config.json", help="Project configuration; data paths are relative to this file.")
    parser.add_argument("--debug", action="store_true", help="Show a Python traceback when a command fails.")
    args = parser.parse_args()

    try:
        if args.command == "init":
            if args.phase is not None:
                parser.error("init does not accept --phase.")
            directory = Path(args.project_directory or ".").expanduser().resolve()
            config_path = directory / "config.json"
            if config_path.exists():
                raise ValueError(f"Configuration already exists: {config_path}. Choose a new project folder.")
            with files("bifidelity_sparse_grid").joinpath(f"templates/{args.template}.json").open("r") as file:
                config = json.load(file)
            config["project_name"] = directory.name
            directory.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w") as file:
                json.dump(config, file, indent=4)
                file.write("\n")
            for key in ["data_dir", "output_dir"]:
                (directory / config[key]).mkdir(parents=True, exist_ok=True)

            # Leave an existing README alone if the user initialized a folder
            # containing other material. Project initialization never overwrites it.
            readme = directory / "README.md"
            if not readme.exists():
                template = files("bifidelity_sparse_grid").joinpath("templates/PROJECT_README.md").read_text()
                readme.write_text(template.format(
                    project_name=config["project_name"],
                    data_dir=config["data_dir"],
                    output_dir=config["output_dir"],
                ))
            print(f"Project created: {directory}")
            print(f"Edit {config_path}, then run sparse-grid check --config \"{config_path}\".")
            return 0

        if args.project_directory is not None:
            parser.error("The project directory argument is only used with init; use --config for existing projects.")
        if args.command == "run" and args.phase is None:
            parser.error("Choose --phase 1, 2, or 3, or use init/check.")
        if args.command == "check" and args.phase is not None:
            parser.error("check does not accept --phase.")

        config = load_configuration(args.config)
        check_configuration(config)
        if args.command == "check":
            print(f"Configuration valid: {Path(args.config).resolve()}")
            print(f"Project: {config['project_name']}")
            print(f"Input dimensions: {len(config['problem_definition']['dim_names'])}")
            print(f"Data directory: {config['data_dir']}")
            print(f"Output directory: {config['output_dir']}")
            return 0

        # Load the numerical workflow only when it is needed. Help and project
        # initialization require no plotting backend or simulation data.
        from .workflow_phases import run_phase_1, run_phase_2, run_phase_3
        Path(config["data_dir"]).mkdir(parents=True, exist_ok=True)
        Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
        if args.phase == 1:
            run_phase_1(config)
        elif args.phase == 2:
            run_phase_2(config)
        else:
            run_phase_3(config)
        print("\nPhase completed successfully.")
        return 0
    except KeyError as error:
        print(f"ERROR: Missing configuration field or required grid point: {error}", file=sys.stderr)
        if args.debug:
            traceback.print_exc()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        if args.debug:
            traceback.print_exc()
    return 1
