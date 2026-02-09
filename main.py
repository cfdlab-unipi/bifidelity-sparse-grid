"""
Command-line entry point for the adaptive sparse-grid workflow.

Phases:
  1) Generate baseline grid points for user evaluation.
  2) Select informative refinement points.
  3) Build models, validate (optional), and generate plots.
"""

import argparse
import json
import os
import traceback
from workflow_phases import run_phase_1, run_phase_2, run_phase_3

def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive refinement workflow for sparse grids.")

    parser.add_argument(
        '--phase',
        type=int,
        required=True,
        choices=[1, 2, 3],
        help="The workflow phase to execute: 1 (Generate baseline points), 2 (Select refinement points), 3 (Final analysis)."
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.json',
        help="Path to the JSON configuration file."
    )

    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            config = json.load(f)

        # Ensure output and data directories exist.
        os.makedirs(config.get('output_dir', 'output'), exist_ok=True)
        os.makedirs(config.get('data_dir', 'data_input'), exist_ok=True)

        if args.phase == 1:
            print(">>> EXECUTING PHASE 1: Generating Baseline Grid Points <<<")
            run_phase_1(config)
        elif args.phase == 2:
            print(">>> EXECUTING PHASE 2: Selecting Refinement Points <<<")
            run_phase_2(config)
        elif args.phase == 3:
            print(">>> EXECUTING PHASE 3: Final Analysis and Plotting <<<")
            run_phase_3(config)

        print("\n>>> Phase completed successfully! <<<")

    except FileNotFoundError as e:
        print(f"\nCRITICAL ERROR: File not found - {e}")
        print("Please ensure the configuration file and necessary data files exist.")
    except Exception as e:
        print(f"\nUNEXPECTED ERROR DURING EXECUTION:\n")
        traceback.print_exc()

if __name__ == '__main__':
    main()
