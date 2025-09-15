# main.py
# =============================================================================
#
# MAIN CONTROL SCRIPT
# - This script acts as the entry point for the adaptive sparse grid workflow.
# - It uses command-line arguments to select and execute different phases:
#   1. Generate baseline points for the user to evaluate.
#   2. Select new, informative points for refinement.
#   3. Perform the final analysis and generate all plots.
#
# =============================================================================

import json
import argparse
import os
import traceback
from workflow_phases import run_phase_1, run_phase_2, run_phase_3

if __name__ == '__main__':
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
        # Load the configuration from the JSON file
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        # Create output directories if they don't exist
        # Note: Changed 'dati_da_riempire' to a more standard 'data_input'
        os.makedirs(config.get('output_dir', 'output'), exist_ok=True)
        os.makedirs(config.get('data_dir', 'data_input'), exist_ok=True)

        # Execute the chosen phase
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