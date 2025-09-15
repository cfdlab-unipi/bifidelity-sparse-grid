# plotting_tools.py
# =============================================================================
#
# PLOTTING TOOLBOX (V2.0 - English, Two-Model Comparison)
# - Generates plots for the adaptive sparse grid workflow.
# - Supports comparison between a 'Baseline' and an 'Informed' model.
# - Includes response surfaces, error charts, and diagnostic plots.
#
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

from sparse_grid_toolbox import evaluate_sparse_grid_interpolant

def plot_error_comparison_barchart(df_results, val_name, save_path=None):
    """
    Generates a bar chart comparing the prediction error on test points
    for the baseline and informed models.
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    x = np.arange(len(df_results['TestID']))
    width = 0.35
    
    ax.bar(x - width/2, df_results['Baseline Model Error [%]'], width, label='Baseline Model', color='C0', alpha=0.7)
    ax.bar(x + width/2, df_results['Informed Model Error [%]'], width, label='Informed Model', color='C1')
    
    ax.set_title(f'Prediction Error Comparison for {val_name}', fontsize=20)
    ax.set_xlabel('Validation Point ID', fontsize=18)
    ax.set_ylabel('Normalized Percentage Error (%)', fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(df_results['TestID'])
    ax.legend(fontsize=18)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  -> Saved error bar chart to: {save_path}")
        plt.close(fig)
    else:
        plt.show()

def plot_combined_response_surfaces(slice_info, domains, dim_names, grid_resolution, base_model, informed_model, val_name, save_path=None):
    """
    Generates a 2-panel comparison of response surfaces for the baseline and informed models.
    """
    varying_dims = slice_info['varying']
    fixed_dims = slice_info['fixed']
    print(f"  -> Generating {grid_resolution}x{grid_resolution} response surface for {val_name} on {varying_dims[0]}-{varying_dims[1]} slice...")

    idx_vary_1, idx_vary_2 = dim_names.index(varying_dims[0]), dim_names.index(varying_dims[1])
    range_1 = np.linspace(domains[idx_vary_1][0], domains[idx_vary_1][1], grid_resolution)
    range_2 = np.linspace(domains[idx_vary_2][0], domains[idx_vary_2][1], grid_resolution)
    Grid_1, Grid_2 = np.meshgrid(range_1, range_2)
    
    eval_points = np.zeros((len(dim_names), Grid_1.size))
    for i, dim in enumerate(dim_names):
        if dim == varying_dims[0]: eval_points[i, :] = Grid_1.ravel()
        elif dim == varying_dims[1]: eval_points[i, :] = Grid_2.ravel()
        else: eval_points[i, :] = fixed_dims[dim]
        
    F_base = evaluate_sparse_grid_interpolant(eval_points, **base_model).reshape(Grid_1.shape)
    F_informed = evaluate_sparse_grid_interpolant(eval_points, **informed_model).reshape(Grid_1.shape)

    global_min = min(F_base.min(), F_informed.min())
    global_max = max(F_base.max(), F_informed.max())
    levels = np.linspace(global_min, global_max, 21)
    cmap = 'viridis' if val_name.lower().startswith('v') else 'inferno'
    
    fig = plt.figure(figsize=(16, 6))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.08], wspace=0.2) 
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1], sharey=ax0)
    cbar_ax = fig.add_subplot(gs[0, 2])
    plt.setp(ax1.get_yticklabels(), visible=False)
    
    models_to_plot = [
        (ax0, F_base, 'Baseline Model'),
        (ax1, F_informed, 'Informed Model')
    ]
    
    contour_mappable = None
    for i, (ax, F_grid, title) in enumerate(models_to_plot):
        contour_mappable = ax.contourf(Grid_1, Grid_2, F_grid, levels=levels, cmap=cmap, vmin=global_min, vmax=global_max)
        contours = ax.contour(Grid_1, Grid_2, F_grid, levels=8, colors='black', linestyles='dashed', linewidths=0.8)
        ax.clabel(contours, inline=True, fontsize=10, fmt='%.2f')
        ax.set_title(title, fontsize=18)
        ax.set_xlabel(f'{varying_dims[0]}', fontsize=16)
        if i == 0: ax.set_ylabel(f'{varying_dims[1]}', fontsize=16)
        ax.grid(True, linestyle='--', alpha=0.5)

    cbar = fig.colorbar(contour_mappable, cax=cbar_ax)
    cbar.set_label(val_name, fontsize=16)
    
    slice_str = ", ".join([f'{k}={v:.2f}' for k, v in fixed_dims.items()])
    fig.suptitle(f'Response Surface for {val_name} (Slice: {slice_str})', fontsize=20)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  -> Saved response surface to: {save_path}")
        plt.close(fig)
    else:
        plt.show()

def plot_discrepancy_by_point(candidate_tuples, indicators, budget, refinement_criterion, save_path=None):
    """
    Generates a bar plot showing the discrepancy indicator value for each candidate point,
    highlighting those that were selected based on the budget.
    """
    print(f"  -> Generating discrepancy plot (Criterion: {refinement_criterion})...")
    if len(indicators) == 0:
        print("     - WARNING: No candidate points to plot.")
        return
        
    sorted_data = sorted(zip(indicators, candidate_tuples), key=lambda item: item[0], reverse=True)
    sorted_indicators = [item[0] for item in sorted_data]
    
    num_candidates = len(sorted_indicators)
    x_positions = np.arange(num_candidates)
    colors = ['C0'] * budget + ['C1'] * (num_candidates - budget)
    
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.bar(x_positions, sorted_indicators, color=colors, alpha=0.9, width=1.0)
    
    ax.set_title(f'Discrepancy Indicator Ranking (Criterion: {refinement_criterion})', fontsize=20)
    ax.set_xlabel('Candidate Point Rank (Sorted by Discrepancy)', fontsize=18)
    ax.set_ylabel('Discrepancy Indicator Value', fontsize=18)
    
    legend_elements = [
        Patch(facecolor='C0', alpha=0.9, label=f'Selected Points (Budget = {budget})'),
        Patch(facecolor='C1', alpha=0.9, label=f'Discarded Points ({num_candidates - budget})')
    ]
    ax.legend(handles=legend_elements, fontsize=18)
    
    # Use log scale if the values span several orders of magnitude
    max_val, min_val = np.max(sorted_indicators), np.min(sorted_indicators)
    if max_val > 0 and (max_val / (min_val + 1e-16)) > 100:
        ax.set_yscale('log')
    
    ax.grid(True, which="both", linestyle='--', alpha=0.6, axis='y')
    ax.set_xlim(-0.5, num_candidates - 0.5)
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  -> Saved discrepancy plot to: {save_path}")
        plt.close(fig)
    else:
        plt.show()

def plot_quadrature_points_slice(base_knots, target_knots, selected_knots, dim_names, varying_dims, fixed_dims, save_path=None):
    """
    Shows the quadrature points on a 2D slice, highlighting the baseline points,
    all potential target points, and the newly selected points for refinement.
    """
    print(f"  -> Generating quadrature points plot for slice {varying_dims[0]}-{varying_dims[1]}...")
    
    def filter_knots(knots, fixed_indices, fixed_values, tol=1e-9):
        if knots.shape[1] == 0: return np.empty((knots.shape[0], 0))
        mask = np.all(np.isclose(knots[fixed_indices, :], np.array(fixed_values)[:, np.newaxis], atol=tol), axis=0)
        return knots[:, mask]

    idx_vary_1, idx_vary_2 = dim_names.index(varying_dims[0]), dim_names.index(varying_dims[1])
    fixed_indices = [dim_names.index(d) for d in fixed_dims.keys()]
    fixed_values = list(fixed_dims.values())
    
    base_slice_knots = filter_knots(base_knots, fixed_indices, fixed_values)
    target_slice_knots = filter_knots(target_knots, fixed_indices, fixed_values)
    selected_slice_knots = filter_knots(selected_knots, fixed_indices, fixed_values)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(target_slice_knots[idx_vary_1, :], target_slice_knots[idx_vary_2, :], marker='s', color='lightgray', s=80, label=f'Target Grid Candidates ({target_slice_knots.shape[1]} points)')
    ax.scatter(base_slice_knots[idx_vary_1, :], base_slice_knots[idx_vary_2, :], marker='o', color='blue', s=60, alpha=0.9, label=f'Baseline Grid Points ({base_slice_knots.shape[1]} points)')
    if selected_slice_knots.shape[1] > 0:
        ax.scatter(selected_slice_knots[idx_vary_1, :], selected_slice_knots[idx_vary_2, :], marker='*', color='red', s=300, edgecolors='black', label=f'Selected Refinement Points ({selected_slice_knots.shape[1]} points)', zorder=10)

    slice_str = ", ".join([f'{k}={v}' for k, v in fixed_dims.items()])
    ax.set_title(f'Quadrature Points on Slice ({slice_str})', fontsize=18)
    ax.set_xlabel(f'{varying_dims[0]}', fontsize=16)
    ax.set_ylabel(f'{varying_dims[1]}', fontsize=16)
    ax.legend(fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  -> Saved quadrature points plot to: {save_path}")
        plt.close(fig)
    else:
        plt.show()

def plot_error_distribution(errors_base, errors_informed, val_name, save_path=None, bins=15):
    """
    Generates a histogram to compare the error distribution of the baseline
    and informed models.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.hist(errors_base, bins=bins, density=True, color='C0', alpha=0.7, label='Baseline Model')
    ax.hist(errors_informed, bins=bins, density=True, color='C1', alpha=0.7, label='Informed Model')

    ax.set_title(f'Distribution of Prediction Errors for {val_name}', fontsize=18)
    ax.set_xlabel('Normalized Percentage Error (%)', fontsize=16)
    ax.set_ylabel('Density', fontsize=16)
    ax.legend(fontsize=14)
    ax.grid(linestyle='--', alpha=0.7, axis='y')
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  -> Saved error distribution plot to: {save_path}")
        plt.close(fig)
    else:
        plt.show()