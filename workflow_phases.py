# workflow_phases.py
# =============================================================================
#
# CORE ANALYSIS WORKFLOW LOGIC (V1.0)
# - Implements the three-phase adaptive refinement workflow.
# - Phase 1: Generates initial grid points for evaluation.
# - Phase 2: Selects new points for refinement based on evaluated data.
# - Phase 3: Builds final models, validates them, and generates all plots.
#
# =============================================================================

import pandas as pd
import numpy as np
import os

# --- Import from your custom toolboxes ---
from sparse_grid_toolbox import (
    create_sparse_grid_structure, reduce_sparse_grid_knots,
    evaluate_sparse_grid_interpolant, compute_discrepancy_indicator,
    lev2knots_doubling, knots_cc_1d
)
from plotting_tools import (
    plot_error_comparison_barchart, plot_combined_response_surfaces,
    plot_quadrature_points_slice, plot_discrepancy_by_point,
    plot_error_distribution
)

# =============================================================================
# HELPER FUNCTIONS (Adapted from original code)
# =============================================================================

def _find_elbow_point(sorted_values):
    """Finds the elbow point in a sorted array of values."""
    if len(sorted_values) < 3:
        return len(sorted_values)
    n_points = len(sorted_values)
    all_coords = np.vstack((range(n_points), sorted_values)).T
    first_point, last_point = all_coords[0], all_coords[-1]
    line_vec = last_point - first_point
    line_vec_norm = line_vec / (np.sqrt(np.sum(line_vec**2)) + 1e-9)
    vec_from_first = all_coords - first_point
    scalar_product = np.sum(vec_from_first * np.tile(line_vec_norm, (n_points, 1)), axis=1)
    vec_from_first_parallel = np.outer(scalar_product, line_vec_norm)
    distances = np.sqrt(np.sum((vec_from_first - vec_from_first_parallel)**2, axis=1))
    return np.argmax(distances) + 1

def _build_informed_model_values(base_knots, base_values, all_target_knots,
                                 all_known_data_df, quantity_name, dim_names,
                                 base_grid_structure, tolerance=1e-9):
    """
    Constructs the final value vector for the informed (refined) model.

    This function robustly combines known data points (baseline and refined)
    with interpolated values for any remaining points in the target grid structure.
    It uses a "snap-to-grid" approach to handle potential floating-point
    inaccuracies between programmatically generated grid points and those read
    from data files.

    Args:
        base_knots (np.ndarray): Coordinates of the baseline grid points.
        base_values (np.ndarray): Function values on the baseline grid.
        all_target_knots (np.ndarray): Coordinates of all points in the final,
                                       refined grid structure.
        all_known_data_df (pd.DataFrame): DataFrame containing all available
                                          simulation/experimental data (base + refined).
        quantity_name (str): The name of the output column in the DataFrame.
        dim_names (list): List of column names for the input dimensions.
        base_grid_structure (dict): The sparse grid structure object for the base model.
        tolerance (float): The maximum distance to consider two points a match.

    Returns:
        np.ndarray: A vector of values ordered correctly for `all_target_knots`.
    """
    # 1. Prepare coordinate arrays for matching
    canonical_points = all_target_knots.T  # Shape: (num_canonical, dims)
    data_points = all_known_data_df[dim_names].values # Shape: (num_data, dims)
    data_values = all_known_data_df[quantity_name].values

    # 2. Initialize the final values vector. Using NaN helps identify missing points.
    final_values = np.full(canonical_points.shape[0], np.nan)

    # 3. For each data point from files, find its nearest neighbor in the canonical grid
    #    and "snap" its value into place if the distance is within tolerance.
    for i, data_point in enumerate(data_points):
        # Calculate squared Euclidean distances from the current data point to all canonical points
        distances_sq = np.sum((data_point - canonical_points)**2, axis=1)
        
        # Find the index of the closest canonical point
        best_match_idx = np.argmin(distances_sq)
        
        # If the closest point is within the tolerance, consider it a match
        if distances_sq[best_match_idx] < tolerance**2:
            # Assign the known value to the corresponding position in the final vector
            final_values[best_match_idx] = data_values[i]

    # 4. Identify any points in the final vector that are still NaN. These are the
    #    points that truly need to be estimated via interpolation.
    missing_indices = np.where(np.isnan(final_values))[0]

    # 5. If there are missing points, interpolate their values using the base model
    if len(missing_indices) > 0:
        points_to_interpolate = all_target_knots[:, missing_indices]
        
        estimated_values = evaluate_sparse_grid_interpolant(
            points_to_interpolate,
            base_grid_structure,
            base_knots,
            base_values
        )
        # Fill in the missing values with the interpolated ones
        final_values[missing_indices] = estimated_values

    # Final sanity check to ensure the vector is complete
    if np.isnan(final_values).any():
        raise RuntimeError(f"FATAL: Final values vector for '{quantity_name}' still contains NaNs after interpolation.")

    return final_values


# =============================================================================
# PHASE 1: GENERATE BASELINE GRID
# =============================================================================

def run_phase_1(config):
    """
    Generates the baseline sparse grid points and saves them to a CSV file
    for the user to fill in with simulation/experimental results.
    """
    proj_name = config['project_name']
    p_def = config['problem_definition']
    sg_settings = config['sparse_grid_settings']
    
    N = len(p_def['dim_names'])
    w_start = sg_settings['base_level']
    domains = p_def['domains']

    # 1. Create the base grid structure and find unique knots
    S_start = create_sparse_grid_structure(N, w_start, lev2knots_doubling, knots_cc_1d, domains)
    knots_base = reduce_sparse_grid_knots(S_start)
    
    print(f"Generated {knots_base.shape[1]} unique points for the baseline grid (level w={w_start}).")
    
    # 2. Prepare a DataFrame for the user
    df_out = pd.DataFrame(knots_base.T, columns=p_def['dim_names'])
    for qty in p_def['output_quantities']:
        df_out[qty] = np.nan  # Add empty columns to be filled
    
    # 3. Save the file
    output_path = os.path.join(config['data_dir'], f"{proj_name}_phase1_to_fill.dat")
    # Usiamo to_csv con sep='\t' per creare un file separato da tab
    df_out.to_csv(output_path, sep='\t', index=False)
    
    print(f"\nACTION REQUIRED:")
    print(f"  -> Please open the file '{output_path}' with a text editor or spreadsheet program.")
    print(f"  -> Fill in the values for the output columns: {', '.join(p_def['output_quantities'])}")
    print(f"  -> Then, run the script again with --phase 2.")


# =============================================================================
# PHASE 2: SELECT REFINEMENT POINTS
# =============================================================================

def run_phase_2(config):
    """
    Reads the filled baseline data, selects the best points for refinement
    using a discrepancy indicator, and saves ONLY the new points to a new CSV file.
    """
    proj_name = config['project_name']
    p_def = config['problem_definition']
    sg_settings = config['sparse_grid_settings']
    ref_settings = config['refinement_settings']
    output_dir = config['output_dir']

    # 1. Load the data filled in by the user from Phase 1
    input_path = os.path.join(config['data_dir'], f"{proj_name}_phase1_to_fill.dat")
    try:
        df_base = pd.read_csv(input_path, sep='\t')
    except FileNotFoundError:
        print(f"ERROR: The file '{input_path}' was not found. Please run phase 1 first.")
        return
    
    if df_base[p_def['output_quantities']].isnull().values.any():
        raise ValueError(f"The file '{input_path}' contains missing values. Please fill all data before proceeding.")

    knots_base = df_base[p_def['dim_names']].values.T
    ref_criterion_name = ref_settings['refinement_criterion']
    values_base_criterion = df_base[ref_criterion_name].values
    
    # 2. Identify candidate points for refinement
    N = len(p_def['dim_names'])
    w_start, w_target = sg_settings['base_level'], sg_settings['target_level']
    domains = p_def['domains']
    
    S_target = create_sparse_grid_structure(N, w_target, lev2knots_doubling, knots_cc_1d, domains)
    knots_target_total = reduce_sparse_grid_knots(S_target)
    
    set_base = set(map(tuple, np.round(knots_base.T, 12)))
    candidate_tuples = sorted(list(set(map(tuple, np.round(knots_target_total.T, 12))) - set_base))
    candidate_knots = np.array(candidate_tuples).T if candidate_tuples else np.empty((N, 0))

    if candidate_knots.shape[1] == 0:
        print("No new candidate points found for refinement. The grid may already be at the target level.")
        return

    # 3. Compute discrepancy indicator and select the best points
    print(f"Analyzing {candidate_knots.shape[1]} candidate points for refinement based on '{ref_criterion_name}'...")
    indicators = compute_discrepancy_indicator(candidate_knots, w_start, N, domains, knots_base, values_base_criterion)

    # 4. Determine the budget for new points
    final_budget = 0
    if ref_settings.get('use_auto_budget', False):
        max_budget = ref_settings.get('max_budget', len(indicators))
        method = ref_settings.get('auto_budget_method', 'elbow')
        if method == 'elbow':
            print("  -> Calculating budget with 'Elbow' method...")
            sorted_indicators = np.sort(indicators)[::-1]
            optimal_count = _find_elbow_point(sorted_indicators)
            final_budget = min(optimal_count, max_budget)
            print(f"     - 'Elbow' suggests: {optimal_count}. Final budget (capped by max_budget={max_budget}): {final_budget}")
        elif method == 'threshold':
            threshold_perc = ref_settings.get('auto_budget_threshold', 0.1)
            print(f"  -> Calculating budget with 'Threshold' method ({threshold_perc*100:.0f}%)...")
            if np.max(indicators) > 1e-9:
                threshold_value = np.max(indicators) * threshold_perc
                num_selected = np.sum(indicators >= threshold_value)
                final_budget = min(num_selected, max_budget)
                print(f"     - Threshold value: {threshold_value:.4e}. Points above: {num_selected}. Final budget: {final_budget}")
            else:
                print("     - Max indicator is zero. Budget set to 0.")
    else: 
        final_budget = min(ref_settings.get('max_budget', 0), len(indicators))
        print(f"  -> Using fixed budget: {final_budget}")

    best_local_indices = np.argsort(indicators)[-final_budget:] if final_budget > 0 else []
    selected_knots_arr = candidate_knots[:, best_local_indices]
    
    print(f"\nSelected {selected_knots_arr.shape[1]} new points for refinement.")
    
    # 5. Generate and save the discrepancy plot
    print("\nGenerating discrepancy plot to visualize the selection...")
    plot_discrepancy_by_point(
        candidate_tuples=candidate_tuples,
        indicators=indicators,
        budget=final_budget,
        refinement_criterion=ref_criterion_name,
        save_path=os.path.join(output_dir, f"{proj_name}_phase2_discrepancy_plot.png")
    )
    
    # 6. Save ONLY the new points to a new CSV file
    df_out = pd.DataFrame(selected_knots_arr.T, columns=p_def['dim_names'])
    for qty in p_def['output_quantities']:
        df_out[qty] = np.nan
        
    output_path = os.path.join(config['data_dir'], f"{proj_name}_phase2_to_fill.dat")
    df_out.to_csv(output_path, sep='\t', index=False)

    print(f"\nACTION REQUIRED:")
    print(f"  -> Please open the new file '{output_path}'")
    print(f"  -> Fill in the values for these new points.")
    print(f"  -> Then, run the script again with --phase 3 for the final analysis.")


# =============================================================================
# PHASE 3: FINAL ANALYSIS AND PLOTTING
# =============================================================================

def run_phase_3(config):
    """
    Loads all data (baseline + refinement), builds the final surrogate models,
    performs validation if test data is provided, and generates all plots.
    """
    print("\n--- Starting Final Analysis and Plotting ---")
    proj_name = config['project_name']
    p_def = config['problem_definition']
    sg_settings = config['sparse_grid_settings']
    output_dir = config['output_dir']

    # --- 1. Load all available data ---
    path_base = os.path.join(config['data_dir'], f"{proj_name}_phase1_to_fill.dat")
    path_refinement = os.path.join(config['data_dir'], f"{proj_name}_phase2_to_fill.dat")
    
    try:
        df_base = pd.read_csv(path_base, sep='\t')
        df_refinement = pd.read_csv(path_refinement, sep='\t')
    except FileNotFoundError as e:
        print(f"ERROR: Could not find required data file: {e.filename}")
        print("Please ensure you have completed phases 1 and 2.")
        return
        
    df_all_data = pd.concat([df_base, df_refinement], ignore_index=True)
    
    print(f"Loaded {len(df_base)} baseline points and {len(df_refinement)} refinement points.")

    # --- 2. Prepare grid structures and models ---
    N = len(p_def['dim_names'])
    w_start, w_target = sg_settings['base_level'], sg_settings['target_level']
    domains = p_def['domains']
    
    S_start = create_sparse_grid_structure(N, w_start, lev2knots_doubling, knots_cc_1d, domains)
    S_target = create_sparse_grid_structure(N, w_target, lev2knots_doubling, knots_cc_1d, domains)
    knots_base = df_base[p_def['dim_names']].values.T
    knots_target_total = reduce_sparse_grid_knots(S_target)

    # Prepare a map of all known refinement data for the builder function
    df_refinement['tuple'] = df_refinement[p_def['dim_names']].apply(lambda row: tuple(np.round(row.values, 12)), axis=1)
    
    # --- 3. Build models for each output quantity ---
    models = {}
    for qty in p_def['output_quantities']:
        print(f"Building models for quantity: '{qty}'")

        # Baseline model data
        values_base = df_base[qty].values

        # Informed model data - USIAMO LA NUOVA FUNZIONE ROBUSTA
        values_informed = _build_informed_model_values(
            base_knots=knots_base,
            base_values=values_base,
            all_target_knots=knots_target_total,
            all_known_data_df=df_all_data, # Passiamo l'intero DataFrame
            quantity_name=qty,
            dim_names=p_def['dim_names'],
            base_grid_structure=S_start
        )

        models[qty] = {
            'base': {'S': S_start, 'unique_knots': knots_base, 'f_on_unique_knots': values_base},
            'informed': {'S': S_target, 'unique_knots': knots_target_total, 'f_on_unique_knots': values_informed}
        }

    # --- 4. Perform validation if validation file is provided ---
    validation_file = config.get('validation', {}).get('validation_data_file')
    if validation_file and os.path.exists(validation_file):
        print(f"\nValidation file found: '{validation_file}'. Calculating model errors...")
        validation_file = config.get('validation', {}).get('validation_data_file')
        if validation_file and os.path.exists(validation_file):
            print(f"\nValidation file found: '{validation_file}'. Calculating model errors...")
            try:
                if validation_file.endswith('.dat') or validation_file.endswith('.tsv'):
                    df_test = pd.read_csv(validation_file, sep='\t')
                else: # Default to comma-separated
                    df_test = pd.read_csv(validation_file)
            except Exception as e:
                print(f"ERROR: Could not read validation file '{validation_file}'. Details: {e}")
                return
            
        test_points = df_test[p_def['dim_names']].values.T
        
        for qty in p_def['output_quantities']:
            if qty not in df_test.columns:
                print(f"  - WARNING: Quantity '{qty}' not found in validation file. Skipping error plots for it.")
                continue

            true_values = df_test[qty].values
            
            # Predict with both models
            pred_base = evaluate_sparse_grid_interpolant(test_points, **models[qty]['base'])
            pred_informed = evaluate_sparse_grid_interpolant(test_points, **models[qty]['informed'])
            
            # Calculate percentage error
            error_base = 100 * np.abs(pred_base - true_values) / (np.abs(true_values) + 1e-9)
            error_informed = 100 * np.abs(pred_informed - true_values) / (np.abs(true_values) + 1e-9)

            print(f"  - Average Error for '{qty}': Baseline={error_base.mean():.2f}%, Informed={error_informed.mean():.2f}%")

            # Plot error comparisons
            df_errors = pd.DataFrame({
                'TestID': df_test.index,
                'Baseline Model Error [%]': error_base,
                'Informed Model Error [%]': error_informed
            })
            
            plot_error_comparison_barchart(df_errors, qty, save_path=os.path.join(output_dir, f"error_barchart_{qty}.png"))
            plot_error_distribution(error_base, error_informed, qty, save_path=os.path.join(output_dir, f"error_distribution_{qty}.png"))

    else:
        print("\nNo validation file specified or found. Skipping error analysis plots.")

    # --- 5. Generate Response Surface and Quadrature Point Plots ---
    print("\nGenerating final plots...")
    for slice_info in config['plotting']['slices_2d']:
        slice_id = "_".join([f"{k}{str(v).replace('.', 'p')}" for k, v in slice_info['fixed'].items()])
        
        for qty in p_def['output_quantities']:
            plot_combined_response_surfaces(
                slice_info, domains, p_def['dim_names'], config['plotting']['plot_resolution'],
                base_model=models[qty]['base'],
                informed_model=models[qty]['informed'],
                val_name=qty,
                save_path=os.path.join(output_dir, f"response_surface_{qty}_{slice_id}.png")
            )
        
        selected_knots_arr = df_refinement[p_def['dim_names']].values.T
        plot_quadrature_points_slice(
            base_knots=knots_base,
            target_knots=knots_target_total,
            selected_knots=selected_knots_arr,
            dim_names=p_def['dim_names'],
            varying_dims=slice_info['varying'],
            fixed_dims=slice_info['fixed'],
            save_path=os.path.join(output_dir, f"quadrature_points_{slice_id}.png")
        )
    
    print(f"\nAnalysis complete. All outputs are in the '{output_dir}' directory.")