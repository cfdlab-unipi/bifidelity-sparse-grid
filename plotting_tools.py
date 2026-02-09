# plotting_tools.py
# =============================================================================
# PLOTTING UTILITIES TOOLBOX
#
# Utilities for plotting and comparing models in the adaptive sparse-grid
# workflow (baseline vs informed).
#
# Conventions:
#   - Global colors are defined at the top of the file and can be overridden
#     per-function via keyword arguments.
#   - If save_path is provided, the figure is saved; in all cases the plot is
#     shown and the figure is closed.
# =============================================================================

"""Plotting utilities for the adaptive sparse-grid workflow."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

from sparse_grid_toolbox import evaluate_sparse_grid_interpolant

# -----------------------------------------------------------------------------
# Global colors (override from caller if needed)
# -----------------------------------------------------------------------------

BASELINE_COLOR = "#7CC9FF"
INFORMED_COLOR = "#F47C27"
CANDIDATE_SELECTED_COLOR = "red"
CANDIDATE_UNSELECTED_COLOR = "gray"
MEAN_LINE_COLOR = "black"
STD_BAND_COLOR = "gray"


def _finalize_plot(fig, save_path=None, save_dpi=300, save_msg=None):
    """
    Save/show/close helper.

    If save_path is provided, the figure is saved. The plot is always shown and
    then the figure is closed to release resources.
    """
    if save_path:
        fig.savefig(save_path, dpi=save_dpi, bbox_inches="tight")
        if save_msg:
            print(save_msg)
    plt.show()
    plt.close(fig)


def _extract_slice_info(slice_info: dict, dim_names: list[str], domains: list[tuple[float, float]]):
    """
    Return (varying_dims, fixed_dims) from either key style and fill missing fixed dims
    with midpoints to avoid KeyError.
    """
    if "varying" in slice_info:
        varying_dims = slice_info.get("varying", [])
        fixed_dims = dict(slice_info.get("fixed", {}))
    else:
        varying_dims = slice_info.get("varying_dims", [])
        fixed_dims = dict(slice_info.get("fixed_dims", {}))

    for i, dim in enumerate(dim_names):
        if dim not in varying_dims and dim not in fixed_dims:
            fixed_dims[dim] = 0.5 * (domains[i][0] + domains[i][1])

    return varying_dims, fixed_dims


# -----------------------------------------------------------------------------
# Error plots
# -----------------------------------------------------------------------------

def plot_error_comparison_barchart(
    df_results,
    val_name,
    save_path=None,
    figsize=(16, 8),
    bar_width=0.35,
    bar_alpha=0.7,
    xrotation=45,
    label_fontsize=12,
    title_fontsize=16,
    tick_fontsize=10,
    legend_fontsize=12,
    baseline_color=None,
    informed_color=None,
    save_dpi=300,
):
    """
    Bar chart comparing prediction errors on validation points.
    Accepts either:
      - columns: 'Baseline Model Error [%]' and 'Informed Model Error [%]'
      - legacy columns: 'Errore Base {val_name} [%]' and 'Errore Hybrid {val_name} [%]'
    """
    if baseline_color is None:
        baseline_color = BASELINE_COLOR
    if informed_color is None:
        informed_color = INFORMED_COLOR

    fig, ax = plt.subplots(figsize=figsize)

    if "Baseline Model Error [%]" in df_results.columns:
        err_base = df_results["Baseline Model Error [%]"]
        err_inf = df_results["Informed Model Error [%]"]
        x_labels = df_results.get("TestID", df_results.index)
        label_base = "Baseline model"
        label_inf = "Informed model"
    else:
        err_base = df_results[f"Errore Base {val_name} [%]"]
        err_inf = df_results[f"Errore Hybrid {val_name} [%]"]
        x_labels = df_results.get("Test", df_results.index)
        label_base = "Baseline model"
        label_inf = "Hierarchical model"

    x = np.arange(len(err_base))

    ax.bar(x - bar_width / 2, err_base, bar_width, label=label_base, color=baseline_color, alpha=bar_alpha)
    ax.bar(x + bar_width / 2, err_inf, bar_width, label=label_inf, color=informed_color, alpha=bar_alpha)

    ax.set_title(f"Prediction error comparison for {val_name}", fontsize=title_fontsize)
    ax.set_xlabel("Validation point ID", fontsize=label_fontsize)
    ax.set_ylabel("Normalized percentage error (%)", fontsize=label_fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=xrotation, ha="right")
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
    ax.legend(fontsize=legend_fontsize)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    fig.tight_layout()

    _finalize_plot(
        fig,
        save_path=save_path,
        save_dpi=save_dpi,
        save_msg=f"  -> Saved error bar chart to: {save_path}",
    )


def plot_error_distribution(
    errors_base,
    errors_informed,
    val_name,
    save_path=None,
    bins=15,
    add_kde=False,
    figsize=(12, 7),
    hist_alpha=0.7,
    kde_linewidth=1.5,
    label_fontsize=12,
    title_fontsize=16,
    tick_fontsize=10,
    legend_fontsize=12,
    baseline_color=None,
    informed_color=None,
    save_dpi=300,
):
    """Histogram comparing baseline vs informed error distributions."""
    if baseline_color is None:
        baseline_color = BASELINE_COLOR
    if informed_color is None:
        informed_color = INFORMED_COLOR

    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(errors_base, bins=bins, density=True, color=baseline_color, alpha=hist_alpha, label="Baseline model")
    ax.hist(errors_informed, bins=bins, density=True, color=informed_color, alpha=hist_alpha, label="Informed model")

    if add_kde:
        try:
            from scipy.stats import gaussian_kde

            xmin = min(np.min(errors_base), np.min(errors_informed))
            xmax = max(np.max(errors_base), np.max(errors_informed))
            x_range = np.linspace(xmin, xmax, 200)
            kde_base = gaussian_kde(np.asarray(errors_base))
            kde_inf = gaussian_kde(np.asarray(errors_informed))
            ax.plot(x_range, kde_base(x_range), color=baseline_color, linewidth=kde_linewidth, label="Baseline KDE")
            ax.plot(x_range, kde_inf(x_range), color=informed_color, linewidth=kde_linewidth, label="Informed KDE")
        except Exception as exc:
            print(f"  -> WARNING: KDE failed ({exc}). Continuing without KDE.")

    ax.set_title(f"Distribution of prediction errors for {val_name}", fontsize=title_fontsize)
    ax.set_xlabel("Normalized percentage error (%)", fontsize=label_fontsize)
    ax.set_ylabel("Density", fontsize=label_fontsize)
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
    ax.legend(fontsize=legend_fontsize)
    ax.grid(linestyle="--", alpha=0.7, axis="y")
    fig.tight_layout()

    _finalize_plot(
        fig,
        save_path=save_path,
        save_dpi=save_dpi,
        save_msg=f"  -> Saved error distribution plot to: {save_path}",
    )


# -----------------------------------------------------------------------------
# Response surfaces
# -----------------------------------------------------------------------------

def plot_combined_response_surfaces(
    slice_info,
    domains,
    dim_names,
    grid_resolution,
    base_model,
    informed_model,
    val_name,
    save_path=None,
    figsize=(16, 6),
    n_levels=21,
    n_contours=8,
    contour_linewidth=0.8,
    label_fontsize=12,
    title_fontsize=14,
    tick_fontsize=10,
    cbar_label_fontsize=12,
    cmap=None,
    save_dpi=300,
):
    """Two-panel comparison of response surfaces (baseline vs informed)."""
    varying_dims, fixed_dims = _extract_slice_info(slice_info, dim_names, domains)
    print(
        f"  -> Generating {grid_resolution}x{grid_resolution} response surface for {val_name} "
        f"on {varying_dims[0]}-{varying_dims[1]} slice..."
    )

    idx_vary_1, idx_vary_2 = dim_names.index(varying_dims[0]), dim_names.index(varying_dims[1])
    range_1 = np.linspace(domains[idx_vary_1][0], domains[idx_vary_1][1], grid_resolution)
    range_2 = np.linspace(domains[idx_vary_2][0], domains[idx_vary_2][1], grid_resolution)
    Grid_1, Grid_2 = np.meshgrid(range_1, range_2)

    eval_points = np.zeros((len(dim_names), Grid_1.size))
    for i, dim in enumerate(dim_names):
        if dim == varying_dims[0]:
            eval_points[i, :] = Grid_1.ravel()
        elif dim == varying_dims[1]:
            eval_points[i, :] = Grid_2.ravel()
        else:
            eval_points[i, :] = fixed_dims[dim]

    F_base = evaluate_sparse_grid_interpolant(eval_points, **base_model).reshape(Grid_1.shape)
    F_inf = evaluate_sparse_grid_interpolant(eval_points, **informed_model).reshape(Grid_1.shape)

    global_min = min(F_base.min(), F_inf.min())
    global_max = max(F_base.max(), F_inf.max())
    levels = np.linspace(global_min, global_max, n_levels)
    if cmap is None:
        cmap = "viridis" if val_name.lower().startswith("v") else "inferno"

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.08], wspace=0.2)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1], sharey=ax0)
    cbar_ax = fig.add_subplot(gs[0, 2])
    plt.setp(ax1.get_yticklabels(), visible=False)

    contour_mappable = None
    for i_ax, (ax, F_grid, title) in enumerate(
        [(ax0, F_base, "Baseline model"), (ax1, F_inf, "Informed model")]
    ):
        contour_mappable = ax.contourf(
            Grid_1, Grid_2, F_grid, levels=levels, cmap=cmap, vmin=global_min, vmax=global_max
        )
        contours = ax.contour(
            Grid_1,
            Grid_2,
            F_grid,
            levels=n_contours,
            colors="black",
            linestyles="dashed",
            linewidths=contour_linewidth,
        )
        ax.clabel(contours, inline=True, fontsize=tick_fontsize, fmt="%.2f")
        ax.set_title(title, fontsize=title_fontsize)
        ax.set_xlabel(f"{varying_dims[0]}", fontsize=label_fontsize)
        if i_ax == 0:
            ax.set_ylabel(f"{varying_dims[1]}", fontsize=label_fontsize)
        ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
        ax.grid(True, linestyle="--", alpha=0.5)

    cbar = fig.colorbar(contour_mappable, cax=cbar_ax)
    cbar.set_label(val_name, fontsize=cbar_label_fontsize)
    cbar.ax.tick_params(axis="y", which="major", labelsize=tick_fontsize)

    if fixed_dims:
        slice_str = ", ".join([f"{k}={v:.2f}" for k, v in fixed_dims.items()])
        title = f"Response surface for {val_name} (slice: {slice_str})"
    else:
        title = f"Response surface for {val_name} (full domain)"
    fig.suptitle(title, fontsize=title_fontsize + 2)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    _finalize_plot(
        fig,
        save_path=save_path,
        save_dpi=save_dpi,
        save_msg=f"  -> Saved response surface to: {save_path}",
    )


def plot_response_surface(
    slice_info,
    domains,
    dim_names,
    grid_resolution,
    model,
    val_name,
    title,
    save_path=None,
    figsize=(8, 6),
    n_levels=21,
    n_contours=8,
    contour_linewidth=0.8,
    label_fontsize=12,
    title_fontsize=14,
    tick_fontsize=10,
    cbar_label_fontsize=12,
    cmap=None,
    save_dpi=300,
):
    """Single-panel response surface plot for one model."""
    varying_dims, fixed_dims = _extract_slice_info(slice_info, dim_names, domains)

    idx_vary_1, idx_vary_2 = dim_names.index(varying_dims[0]), dim_names.index(varying_dims[1])
    range_1 = np.linspace(domains[idx_vary_1][0], domains[idx_vary_1][1], grid_resolution)
    range_2 = np.linspace(domains[idx_vary_2][0], domains[idx_vary_2][1], grid_resolution)
    Grid_1, Grid_2 = np.meshgrid(range_1, range_2)

    eval_points = np.zeros((len(dim_names), Grid_1.size))
    for i, dim in enumerate(dim_names):
        if dim == varying_dims[0]:
            eval_points[i, :] = Grid_1.ravel()
        elif dim == varying_dims[1]:
            eval_points[i, :] = Grid_2.ravel()
        else:
            eval_points[i, :] = fixed_dims[dim]

    F = evaluate_sparse_grid_interpolant(eval_points, **model).reshape(Grid_1.shape)

    levels = np.linspace(F.min(), F.max(), n_levels)
    if cmap is None:
        cmap = "viridis" if val_name.lower().startswith("v") else "inferno"

    fig, ax = plt.subplots(figsize=figsize)
    contour_mappable = ax.contourf(Grid_1, Grid_2, F, levels=levels, cmap=cmap)
    contours = ax.contour(
        Grid_1,
        Grid_2,
        F,
        levels=n_contours,
        colors="black",
        linestyles="dashed",
        linewidths=contour_linewidth,
    )
    ax.clabel(contours, inline=True, fontsize=tick_fontsize, fmt="%.2f")
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(f"{varying_dims[0]}", fontsize=label_fontsize)
    ax.set_ylabel(f"{varying_dims[1]}", fontsize=label_fontsize)
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
    ax.grid(True, linestyle="--", alpha=0.5)

    cbar = fig.colorbar(contour_mappable, ax=ax)
    cbar.set_label(val_name, fontsize=cbar_label_fontsize)
    cbar.ax.tick_params(axis="y", which="major", labelsize=tick_fontsize)

    fig.tight_layout()
    _finalize_plot(
        fig,
        save_path=save_path,
        save_dpi=save_dpi,
        save_msg=f"  -> Saved response surface to: {save_path}",
    )


# -----------------------------------------------------------------------------
# Discrepancy and quadrature points
# -----------------------------------------------------------------------------

def plot_discrepancy_by_point(
    candidate_tuples,
    indicators,
    budget,
    refinement_criterion,
    save_path=None,
    figsize=(18, 8),
    bar_alpha=0.9,
    bar_width=1.0,
    label_fontsize=12,
    title_fontsize=16,
    tick_fontsize=10,
    legend_fontsize=12,
    selected_color="C0",
    discarded_color="C1",
    save_dpi=300,
):
    """Bar plot of discrepancy indicator values, highlighting selected points."""
    print(f"  -> Generating discrepancy plot (criterion: {refinement_criterion})...")
    if len(indicators) == 0:
        print("     - WARNING: No candidate points to plot.")
        return

    sorted_data = sorted(zip(indicators, candidate_tuples), key=lambda item: item[0], reverse=True)
    sorted_indicators = [item[0] for item in sorted_data]
    num_candidates = len(sorted_indicators)
    x_positions = np.arange(num_candidates)
    colors = [selected_color] * budget + [discarded_color] * (num_candidates - budget)

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x_positions, sorted_indicators, color=colors, alpha=bar_alpha, width=bar_width)

    ax.set_title(f"Discrepancy indicator ranking (criterion: {refinement_criterion})", fontsize=title_fontsize)
    ax.set_xlabel("Candidate point rank (sorted by discrepancy)", fontsize=label_fontsize)
    ax.set_ylabel("Discrepancy indicator value", fontsize=label_fontsize)

    legend_elements = [
        Patch(facecolor=selected_color, alpha=bar_alpha, label=f"Selected points (budget = {budget})"),
        Patch(facecolor=discarded_color, alpha=bar_alpha, label=f"Discarded points ({num_candidates - budget})"),
    ]
    ax.legend(handles=legend_elements, fontsize=legend_fontsize)
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)

    max_val, min_val = np.max(sorted_indicators), np.min(sorted_indicators)
    if max_val > 0 and (max_val / (min_val + 1e-16)) > 100:
        ax.set_yscale("log")

    ax.grid(True, which="both", linestyle="--", alpha=0.6, axis="y")
    ax.set_xlim(-0.5, num_candidates - 0.5)
    fig.tight_layout()

    _finalize_plot(
        fig,
        save_path=save_path,
        save_dpi=save_dpi,
        save_msg=f"  -> Saved discrepancy plot to: {save_path}",
    )


def plot_quadrature_points_slice(
    base_knots,
    target_knots,
    selected_knots,
    dim_names,
    varying_dims,
    fixed_dims,
    save_path=None,
    figsize=(10, 8),
    target_size=80,
    base_size=60,
    selected_size=300,
    label_fontsize=12,
    title_fontsize=16,
    tick_fontsize=10,
    legend_fontsize=12,
    baseline_color=None,
    target_color="lightgray",
    selected_color=None,
    save_dpi=300,
):
    """2D scatter plot of baseline/candidate/selected points on a slice."""
    if baseline_color is None:
        baseline_color = BASELINE_COLOR
    if selected_color is None:
        selected_color = CANDIDATE_SELECTED_COLOR
    print(f"  -> Generating quadrature points plot for slice {varying_dims[0]}-{varying_dims[1]}...")

    def filter_knots(knots, fixed_indices, fixed_values, tol=1e-9):
        if knots.shape[1] == 0:
            return np.empty((knots.shape[0], 0))
        mask = np.all(
            np.isclose(knots[fixed_indices, :], np.array(fixed_values)[:, np.newaxis], atol=tol),
            axis=0,
        )
        return knots[:, mask]

    idx_vary_1, idx_vary_2 = dim_names.index(varying_dims[0]), dim_names.index(varying_dims[1])
    fixed_indices = [dim_names.index(d) for d in fixed_dims.keys()]
    fixed_values = list(fixed_dims.values())

    base_slice_knots = filter_knots(base_knots, fixed_indices, fixed_values)
    target_slice_knots = filter_knots(target_knots, fixed_indices, fixed_values)
    selected_slice_knots = filter_knots(selected_knots, fixed_indices, fixed_values)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(
        target_slice_knots[idx_vary_1, :],
        target_slice_knots[idx_vary_2, :],
        marker="s",
        color=target_color,
        s=target_size,
        label=f"Target grid candidates ({target_slice_knots.shape[1]} points)",
    )
    ax.scatter(
        base_slice_knots[idx_vary_1, :],
        base_slice_knots[idx_vary_2, :],
        marker="o",
        color=baseline_color,
        s=base_size,
        alpha=0.9,
        label=f"Baseline grid points ({base_slice_knots.shape[1]} points)",
    )
    if selected_slice_knots.shape[1] > 0:
        ax.scatter(
            selected_slice_knots[idx_vary_1, :],
            selected_slice_knots[idx_vary_2, :],
            marker="*",
            color=selected_color,
            s=selected_size,
            edgecolors="black",
            label=f"Selected refinement points ({selected_slice_knots.shape[1]} points)",
            zorder=10,
        )

    if fixed_dims:
        slice_str = ", ".join([f"{k}={v:.2f}" for k, v in fixed_dims.items()])
        title = f"Quadrature points on slice ({slice_str})"
    else:
        title = "Quadrature points on full domain"
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(f"{varying_dims[0]}", fontsize=label_fontsize)
    ax.set_ylabel(f"{varying_dims[1]}", fontsize=label_fontsize)
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
    ax.legend(fontsize=legend_fontsize)
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()

    _finalize_plot(
        fig,
        save_path=save_path,
        save_dpi=save_dpi,
        save_msg=f"  -> Saved quadrature points plot to: {save_path}",
    )


# -----------------------------------------------------------------------------
# Vector mean/std plot
# -----------------------------------------------------------------------------

def plot_vector_mean_std(
    component_indices,
    mean_values,
    std_values,
    qty_name,
    save_path=None,
    xlabel="Component index",
    figsize=(12, 6),
    band_alpha=0.3,
    line_width=2.0,
    label_fontsize=12,
    title_fontsize=16,
    tick_fontsize=10,
    legend_fontsize=11,
    mean_color=None,
    std_color=None,
    save_dpi=300,
):
    """Plot mean along a vector index with ±1 std band."""
    component_indices = np.asarray(component_indices)
    mean_values = np.asarray(mean_values)
    std_values = np.asarray(std_values)

    if mean_color is None:
        mean_color = MEAN_LINE_COLOR
    if std_color is None:
        std_color = STD_BAND_COLOR

    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(
        component_indices,
        mean_values - std_values,
        mean_values + std_values,
        color=std_color,
        alpha=band_alpha,
        label="±1 std",
    )
    ax.plot(component_indices, mean_values, color=mean_color, linewidth=line_width, label="Mean")
    ax.set_title(f"{qty_name} mean ± std", fontsize=title_fontsize)
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(qty_name, fontsize=label_fontsize)
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=legend_fontsize)
    fig.tight_layout()

    _finalize_plot(
        fig,
        save_path=save_path,
        save_dpi=save_dpi,
        save_msg=f"  -> Saved mean/std plot to: {save_path}",
    )
