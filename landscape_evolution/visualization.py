"""
Visualization Module

This module provides plotting functions for landscape evolution results.

Key visualizations:
- Initial vs final topography
- Erosion and deposition maps
- River networks (from flow accumulation)
- Layer exposure maps
- Cross-sections showing stratigraphy
- Time series of landscape change
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from typing import Optional, Tuple, List
from .world_state import WorldState
from .simulator import SimulationHistory


def plot_initial_vs_final(
    history: SimulationHistory,
    pixel_scale_m: float,
    figsize: Tuple[float, float] = (14, 6),
    cmap: str = 'terrain',
    save_path: Optional[str] = None
):
    """
    Plot initial vs final topography side-by-side.
    
    Parameters
    ----------
    history : SimulationHistory
        Simulation history
    pixel_scale_m : float
        Grid spacing (m)
    figsize : tuple
        Figure size
    cmap : str
        Colormap name
    save_path : str, optional
        If provided, save figure to this path
    """
    if len(history.surface_snapshots) < 2:
        print("Need at least 2 snapshots to compare")
        return
    
    initial = history.surface_snapshots[0]
    final = history.surface_snapshots[-1]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Shared elevation limits
    vmin = min(initial.min(), final.min())
    vmax = max(initial.max(), final.max())
    
    # Convert to km for display
    extent_km = [0, initial.shape[1] * pixel_scale_m / 1000,
                 0, initial.shape[0] * pixel_scale_m / 1000]
    
    # Initial
    im0 = axes[0].imshow(initial, cmap=cmap, vmin=vmin, vmax=vmax,
                         extent=extent_km, origin='lower')
    axes[0].set_title(f'Initial Surface (t={history.times[0]:.1f} yr)')
    axes[0].set_xlabel('X (km)')
    axes[0].set_ylabel('Y (km)')
    plt.colorbar(im0, ax=axes[0], label='Elevation (m)')
    
    # Final
    im1 = axes[1].imshow(final, cmap=cmap, vmin=vmin, vmax=vmax,
                         extent=extent_km, origin='lower')
    axes[1].set_title(f'Final Surface (t={history.times[-1]:.1f} yr)')
    axes[1].set_xlabel('X (km)')
    axes[1].set_ylabel('Y (km)')
    plt.colorbar(im1, ax=axes[1], label='Elevation (m)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def plot_erosion_deposition_maps(
    history: SimulationHistory,
    pixel_scale_m: float,
    figsize: Tuple[float, float] = (16, 5),
    save_path: Optional[str] = None
):
    """
    Plot erosion and deposition maps, and net change.
    
    Parameters
    ----------
    history : SimulationHistory
        Simulation history
    pixel_scale_m : float
        Grid spacing (m)
    figsize : tuple
        Figure size
    save_path : str, optional
        If provided, save figure to this path
    """
    if len(history.erosion_maps) == 0:
        print("No erosion/deposition data available")
        return
    
    erosion = history.get_total_erosion()
    deposition = history.get_total_deposition()
    net_change = history.get_net_change()
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    extent_km = [0, erosion.shape[1] * pixel_scale_m / 1000,
                 0, erosion.shape[0] * pixel_scale_m / 1000]
    
    # Erosion (positive values = material removed)
    im0 = axes[0].imshow(erosion, cmap='Reds', extent=extent_km, origin='lower')
    axes[0].set_title('Cumulative Erosion')
    axes[0].set_xlabel('X (km)')
    axes[0].set_ylabel('Y (km)')
    plt.colorbar(im0, ax=axes[0], label='Erosion (m)')
    
    # Deposition (positive values = material added)
    im1 = axes[1].imshow(deposition, cmap='Blues', extent=extent_km, origin='lower')
    axes[1].set_title('Cumulative Deposition')
    axes[1].set_xlabel('X (km)')
    plt.colorbar(im1, ax=axes[1], label='Deposition (m)')
    
    # Net change
    vmax = max(abs(net_change.min()), abs(net_change.max()))
    im2 = axes[2].imshow(net_change, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                         extent=extent_km, origin='lower')
    axes[2].set_title('Net Elevation Change')
    axes[2].set_xlabel('X (km)')
    plt.colorbar(im2, ax=axes[2], label='Change (m)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def plot_river_network(
    surface_elev: np.ndarray,
    flow_accum: np.ndarray,
    pixel_scale_m: float,
    threshold_cells: Optional[int] = None,
    figsize: Tuple[float, float] = (10, 10),
    save_path: Optional[str] = None
):
    """
    Plot river network overlaid on topography.
    
    Parameters
    ----------
    surface_elev : np.ndarray
        Surface elevation (m)
    flow_accum : np.ndarray
        Flow accumulation (number of cells)
    pixel_scale_m : float
        Grid spacing (m)
    threshold_cells : int, optional
        Minimum flow accumulation to show as river
    figsize : tuple
        Figure size
    save_path : str, optional
        If provided, save figure to this path
    """
    if threshold_cells is None:
        # Default: top 5% of flow accumulation
        threshold_cells = np.percentile(flow_accum, 95)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    extent_km = [0, surface_elev.shape[1] * pixel_scale_m / 1000,
                 0, surface_elev.shape[0] * pixel_scale_m / 1000]
    
    # Plot topography
    im = ax.imshow(surface_elev, cmap='terrain', extent=extent_km, origin='lower')
    
    # Overlay rivers
    rivers = flow_accum >= threshold_cells
    
    # Create a masked array for rivers (blue)
    river_overlay = np.ma.masked_where(~rivers, flow_accum)
    ax.imshow(river_overlay, cmap='Blues', alpha=0.7, extent=extent_km, origin='lower')
    
    ax.set_title('River Network')
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    plt.colorbar(im, ax=ax, label='Elevation (m)')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def plot_layer_exposure(
    world: WorldState,
    figsize: Tuple[float, float] = (10, 10),
    save_path: Optional[str] = None
):
    """
    Plot which layer is exposed at the surface.
    
    Parameters
    ----------
    world : WorldState
        Current world state
    figsize : tuple
        Figure size
    save_path : str, optional
        If provided, save figure to this path
    """
    top_layer_idx = world.get_top_layer_map()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    extent_km = [0, world.nx * world.pixel_scale_m / 1000,
                 0, world.ny * world.pixel_scale_m / 1000]
    
    # Create colormap for layers
    n_layers = len(world.layer_names)
    colors = plt.cm.tab20(np.linspace(0, 1, n_layers))
    cmap = mcolors.ListedColormap(colors)
    
    im = ax.imshow(top_layer_idx, cmap=cmap, vmin=0, vmax=n_layers-1,
                   extent=extent_km, origin='lower')
    
    ax.set_title('Surface Layer Exposure')
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    
    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[i], label=world.layer_names[i])
        for i in range(n_layers)
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def plot_cross_section(
    world: WorldState,
    row: Optional[int] = None,
    figsize: Tuple[float, float] = (14, 6),
    vertical_exaggeration: float = 1.0,
    min_draw_thickness: float = 0.1,
    save_path: Optional[str] = None
):
    """
    Plot a cross-section through the stratigraphy.
    
    Parameters
    ----------
    world : WorldState
        Current world state
    row : int, optional
        Row index for cross-section. If None, uses middle row.
    figsize : tuple
        Figure size
    vertical_exaggeration : float
        Vertical exaggeration factor
    min_draw_thickness : float
        Minimum thickness (m) to draw a layer
    save_path : str, optional
        If provided, save figure to this path
    """
    if row is None:
        row = world.ny // 2
    
    # Extract cross-section
    x_coords = np.arange(world.nx) * world.pixel_scale_m / 1000  # km
    surface = world.surface_elev[row, :]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot surface
    ax.plot(x_coords, surface * vertical_exaggeration, 'k-', linewidth=2, label='Surface')
    
    # Plot layer interfaces
    colors = plt.cm.tab20(np.linspace(0, 1, len(world.layer_names)))
    
    for i, layer_name in enumerate(world.layer_names):
        interface = world.layer_interfaces[layer_name][row, :]
        thickness = world.layer_thickness[layer_name][row, :]
        
        # Only plot if layer is present somewhere
        if thickness.max() > min_draw_thickness:
            ax.plot(x_coords, interface * vertical_exaggeration,
                   color=colors[i], linestyle='--', alpha=0.7)
            ax.fill_between(
                x_coords,
                interface * vertical_exaggeration,
                surface * vertical_exaggeration,
                where=(thickness > min_draw_thickness),
                color=colors[i],
                alpha=0.3,
                label=layer_name
            )
            
            # Update surface for next layer
            surface = interface
    
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel(f'Elevation (m, VE={vertical_exaggeration}x)')
    ax.set_title(f'Cross-Section at Row {row}')
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def plot_evolution_summary(
    history: SimulationHistory,
    world: WorldState,
    flow_accum: Optional[np.ndarray] = None,
    figsize: Tuple[float, float] = (18, 12),
    save_path: Optional[str] = None
):
    """
    Create a comprehensive summary plot of landscape evolution.
    
    Shows:
    - Initial and final topography
    - Erosion and deposition maps
    - River network (if flow_accum provided)
    - Layer exposure
    
    Parameters
    ----------
    history : SimulationHistory
        Simulation history
    world : WorldState
        Final world state
    flow_accum : np.ndarray, optional
        Flow accumulation for river network
    figsize : tuple
        Figure size
    save_path : str, optional
        If provided, save figure to this path
    """
    # Determine layout
    if flow_accum is not None:
        nrows, ncols = 3, 2
    else:
        nrows, ncols = 2, 2
    
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(nrows, ncols, figure=fig)
    
    pixel_scale_m = world.pixel_scale_m
    extent_km = [0, world.nx * pixel_scale_m / 1000,
                 0, world.ny * pixel_scale_m / 1000]
    
    # 1. Initial topography
    ax1 = fig.add_subplot(gs[0, 0])
    initial = history.surface_snapshots[0]
    im1 = ax1.imshow(initial, cmap='terrain', extent=extent_km, origin='lower')
    ax1.set_title(f'Initial (t={history.times[0]:.1f} yr)')
    ax1.set_xlabel('X (km)')
    ax1.set_ylabel('Y (km)')
    plt.colorbar(im1, ax=ax1, label='Elevation (m)')
    
    # 2. Final topography
    ax2 = fig.add_subplot(gs[0, 1])
    final = history.surface_snapshots[-1]
    im2 = ax2.imshow(final, cmap='terrain', extent=extent_km, origin='lower')
    ax2.set_title(f'Final (t={history.times[-1]:.1f} yr)')
    ax2.set_xlabel('X (km)')
    plt.colorbar(im2, ax=ax2, label='Elevation (m)')
    
    # 3. Erosion
    ax3 = fig.add_subplot(gs[1, 0])
    erosion = history.get_total_erosion()
    im3 = ax3.imshow(erosion, cmap='Reds', extent=extent_km, origin='lower')
    ax3.set_title('Cumulative Erosion')
    ax3.set_xlabel('X (km)')
    ax3.set_ylabel('Y (km)')
    plt.colorbar(im3, ax=ax3, label='Erosion (m)')
    
    # 4. Deposition
    ax4 = fig.add_subplot(gs[1, 1])
    deposition = history.get_total_deposition()
    im4 = ax4.imshow(deposition, cmap='Blues', extent=extent_km, origin='lower')
    ax4.set_title('Cumulative Deposition')
    ax4.set_xlabel('X (km)')
    plt.colorbar(im4, ax=ax4, label='Deposition (m)')
    
    # 5. River network (if provided)
    if flow_accum is not None:
        ax5 = fig.add_subplot(gs[2, 0])
        im5 = ax5.imshow(final, cmap='terrain', extent=extent_km, origin='lower')
        threshold = np.percentile(flow_accum, 95)
        rivers = flow_accum >= threshold
        river_overlay = np.ma.masked_where(~rivers, flow_accum)
        ax5.imshow(river_overlay, cmap='Blues', alpha=0.7, extent=extent_km, origin='lower')
        ax5.set_title('River Network')
        ax5.set_xlabel('X (km)')
        ax5.set_ylabel('Y (km)')
        
        # 6. Layer exposure
        ax6 = fig.add_subplot(gs[2, 1])
        top_layer_idx = world.get_top_layer_map()
        n_layers = len(world.layer_names)
        colors = plt.cm.tab20(np.linspace(0, 1, n_layers))
        cmap = mcolors.ListedColormap(colors)
        im6 = ax6.imshow(top_layer_idx, cmap=cmap, vmin=0, vmax=n_layers-1,
                        extent=extent_km, origin='lower')
        ax6.set_title('Surface Layer Exposure')
        ax6.set_xlabel('X (km)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def plot_time_series(
    history: SimulationHistory,
    figsize: Tuple[float, float] = (12, 8),
    save_path: Optional[str] = None
):
    """
    Plot time series of landscape metrics.
    
    Parameters
    ----------
    history : SimulationHistory
        Simulation history
    figsize : tuple
        Figure size
    save_path : str, optional
        If provided, save figure to this path
    """
    if len(history.times) < 2:
        print("Need at least 2 snapshots for time series")
        return
    
    times = np.array(history.times)
    
    # Compute metrics at each snapshot
    mean_elev = [surf.mean() for surf in history.surface_snapshots]
    std_elev = [surf.std() for surf in history.surface_snapshots]
    total_erosion = [ero.sum() for ero in history.erosion_maps]
    total_deposition = [dep.sum() for dep in history.deposition_maps]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Mean elevation
    axes[0, 0].plot(times, mean_elev, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Time (yr)')
    axes[0, 0].set_ylabel('Mean Elevation (m)')
    axes[0, 0].set_title('Mean Elevation Over Time')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Elevation relief
    axes[0, 1].plot(times, std_elev, 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Time (yr)')
    axes[0, 1].set_ylabel('Elevation Std Dev (m)')
    axes[0, 1].set_title('Relief Over Time')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Cumulative erosion
    axes[1, 0].plot(times, total_erosion, 'r-', linewidth=2)
    axes[1, 0].set_xlabel('Time (yr)')
    axes[1, 0].set_ylabel('Total Erosion (m)')
    axes[1, 0].set_title('Cumulative Erosion')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Cumulative deposition
    axes[1, 1].plot(times, total_deposition, 'b-', linewidth=2)
    axes[1, 1].set_xlabel('Time (yr)')
    axes[1, 1].set_ylabel('Total Deposition (m)')
    axes[1, 1].set_title('Cumulative Deposition')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def plot_erosion_analysis(
    erosion: np.ndarray,
    surface_elev: np.ndarray,
    pixel_scale_m: float,
    row_for_profile: Optional[int] = None,
    figsize: Tuple[float, float] = (16, 10),
    cmap: str = 'Reds',
    save_path: Optional[str] = None
):
    """
    Comprehensive erosion analysis plot.
    
    Shows:
    - Erosion map
    - Erosion histogram
    - Erosion profile (cross-section)
    - Erosion vs elevation scatter
    
    Parameters
    ----------
    erosion : np.ndarray
        Cumulative erosion depth (m), positive values = material removed
    surface_elev : np.ndarray
        Current surface elevation (m)
    pixel_scale_m : float
        Grid spacing (m)
    row_for_profile : int, optional
        Row index for profile. If None, uses middle row.
    figsize : tuple
        Figure size
    cmap : str
        Colormap for erosion map
    save_path : str, optional
        If provided, save figure to this path
    """
    if row_for_profile is None:
        row_for_profile = erosion.shape[0] // 2
    
    # Create figure with custom layout
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Convert to km for display
    extent_km = [0, erosion.shape[1] * pixel_scale_m / 1000,
                 0, erosion.shape[0] * pixel_scale_m / 1000]
    
    # 1. Erosion map (large, top)
    ax1 = fig.add_subplot(gs[0:2, 0])
    im1 = ax1.imshow(erosion, cmap=cmap, extent=extent_km, origin='lower')
    ax1.set_title('Cumulative Erosion Map', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X (km)')
    ax1.set_ylabel('Y (km)')
    
    # Mark profile location
    profile_y_km = row_for_profile * pixel_scale_m / 1000
    ax1.axhline(profile_y_km, color='cyan', linestyle='--', linewidth=2, 
                label=f'Profile at y={profile_y_km:.1f} km')
    ax1.legend(loc='upper right')
    
    cbar1 = plt.colorbar(im1, ax=ax1, label='Erosion Depth (m)')
    
    # Add statistics text
    stats_text = (
        f"Total eroded: {erosion.sum():.1f} m\n"
        f"Mean: {erosion.mean():.2f} m\n"
        f"Max: {erosion.max():.2f} m\n"
        f"Std: {erosion.std():.2f} m"
    )
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. Erosion histogram
    ax2 = fig.add_subplot(gs[0, 1])
    erosion_values = erosion[erosion > 0]  # Only where erosion occurred
    if len(erosion_values) > 0:
        ax2.hist(erosion_values, bins=50, color='darkred', alpha=0.7, edgecolor='black')
        ax2.axvline(erosion_values.mean(), color='blue', linestyle='--', 
                   linewidth=2, label=f'Mean: {erosion_values.mean():.2f} m')
        ax2.axvline(np.median(erosion_values), color='green', linestyle='--',
                   linewidth=2, label=f'Median: {np.median(erosion_values):.2f} m')
        ax2.set_xlabel('Erosion Depth (m)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Erosion Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No erosion data', ha='center', va='center',
                transform=ax2.transAxes, fontsize=12)
    
    # 3. Erosion profile (cross-section)
    ax3 = fig.add_subplot(gs[1, 1])
    x_coords = np.arange(erosion.shape[1]) * pixel_scale_m / 1000
    erosion_profile = erosion[row_for_profile, :]
    
    ax3.fill_between(x_coords, 0, erosion_profile, color='darkred', alpha=0.5)
    ax3.plot(x_coords, erosion_profile, 'r-', linewidth=2)
    ax3.set_xlabel('Distance (km)')
    ax3.set_ylabel('Erosion Depth (m)')
    ax3.set_title(f'Erosion Profile (Row {row_for_profile})')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(bottom=0)
    
    # 4. Erosion vs elevation scatter
    ax4 = fig.add_subplot(gs[2, :])
    
    # Sample points for scatter (avoid too many points)
    ny, nx = erosion.shape
    sample_step = max(1, ny // 50)  # Sample ~2500 points max
    
    erosion_sample = erosion[::sample_step, ::sample_step].flatten()
    elev_sample = surface_elev[::sample_step, ::sample_step].flatten()
    
    # Only plot where erosion occurred
    mask = erosion_sample > 0
    if mask.sum() > 0:
        scatter = ax4.scatter(elev_sample[mask], erosion_sample[mask],
                            c=erosion_sample[mask], cmap=cmap,
                            alpha=0.5, s=10, edgecolors='none')
        ax4.set_xlabel('Current Elevation (m)')
        ax4.set_ylabel('Erosion Depth (m)')
        ax4.set_title('Erosion vs Elevation')
        ax4.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax4, label='Erosion (m)')
        
        # Add trend line
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(
            elev_sample[mask], erosion_sample[mask]
        )
        trend_x = np.array([elev_sample[mask].min(), elev_sample[mask].max()])
        trend_y = slope * trend_x + intercept
        ax4.plot(trend_x, trend_y, 'b--', linewidth=2, 
                label=f'Trend (R²={r_value**2:.3f})')
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'No erosion data', ha='center', va='center',
                transform=ax4.transAxes, fontsize=12)
    
    plt.suptitle('Erosion Analysis', fontsize=16, fontweight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def plot_erosion_rate_map(
    erosion_rate: np.ndarray,
    pixel_scale_m: float,
    flow_accum: Optional[np.ndarray] = None,
    figsize: Tuple[float, float] = (14, 6),
    cmap: str = 'hot_r',
    save_path: Optional[str] = None
):
    """
    Plot erosion rate map, optionally with river network overlay.
    
    Parameters
    ----------
    erosion_rate : np.ndarray
        Erosion rate (m/yr)
    pixel_scale_m : float
        Grid spacing (m)
    flow_accum : np.ndarray, optional
        Flow accumulation for river overlay
    figsize : tuple
        Figure size
    cmap : str
        Colormap
    save_path : str, optional
        If provided, save figure to this path
    """
    extent_km = [0, erosion_rate.shape[1] * pixel_scale_m / 1000,
                 0, erosion_rate.shape[0] * pixel_scale_m / 1000]
    
    if flow_accum is not None:
        # Two panels: erosion rate and erosion + rivers
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Left: erosion rate only
        im1 = axes[0].imshow(erosion_rate, cmap=cmap, extent=extent_km, origin='lower')
        axes[0].set_title('Erosion Rate')
        axes[0].set_xlabel('X (km)')
        axes[0].set_ylabel('Y (km)')
        plt.colorbar(im1, ax=axes[0], label='Erosion Rate (m/yr)')
        
        # Right: erosion rate with rivers
        im2 = axes[1].imshow(erosion_rate, cmap=cmap, extent=extent_km, origin='lower')
        
        # Overlay rivers
        threshold = np.percentile(flow_accum, 95)
        rivers = flow_accum >= threshold
        river_overlay = np.ma.masked_where(~rivers, flow_accum)
        axes[1].imshow(river_overlay, cmap='Blues', alpha=0.5, 
                      extent=extent_km, origin='lower')
        
        axes[1].set_title('Erosion Rate + River Network')
        axes[1].set_xlabel('X (km)')
        plt.colorbar(im2, ax=axes[1], label='Erosion Rate (m/yr)')
        
    else:
        # Single panel
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(erosion_rate, cmap=cmap, extent=extent_km, origin='lower')
        ax.set_title('Erosion Rate')
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        plt.colorbar(im, ax=ax, label='Erosion Rate (m/yr)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
