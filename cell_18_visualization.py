"""
EROSION SYSTEM - VISUALIZATION AND ANALYSIS

Creates comprehensive plots showing:
1. Initial vs Final topography (side by side)
2. Elevation change map (erosion/deposition)
3. Rivers and lakes overlay on final topography
4. Discharge map showing drainage network
5. Cross-sections showing erosion depth
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import matplotlib.patches as mpatches
from typing import Dict, Tuple

def plot_erosion_results(
    results: Dict,
    pixel_scale_m: float,
    figsize: Tuple[int, int] = (20, 12),
    cmap_terrain: str = "terrain",
    river_discharge_threshold: float = 5000.0,
    lake_discharge_threshold: float = 1000.0
):
    """
    Create comprehensive visualization of erosion simulation results.
    
    Parameters
    ----------
    results : dict
        Output from run_erosion_simulation.
    pixel_scale_m : float
        Grid cell size [m].
    figsize : tuple
        Figure size (width, height).
    cmap_terrain : str
        Colormap for terrain elevation.
    river_discharge_threshold : float
        Discharge threshold for rivers [m³/yr].
    lake_discharge_threshold : float
        Discharge threshold for lakes [m³/yr].
    """
    # Extract data
    elev_initial = results["elevation_initial"]
    elev_final = results["elevation_final"]
    diagnostics = results["diagnostics_history"][-1]  # Last timestep
    
    ny, nx = elev_initial.shape
    extent_km = [0, nx * pixel_scale_m / 1000, 0, ny * pixel_scale_m / 1000]
    
    # Compute changes
    dz = elev_final - elev_initial
    
    # Detect rivers and lakes
    Q = diagnostics["Q"]
    flow_dir = diagnostics["flow_dir"]
    receivers = diagnostics["receivers"]
    
    rivers = detect_rivers(Q, flow_dir, receivers, discharge_threshold=river_discharge_threshold)
    lakes, lake_labels = detect_lakes(elev_final, flow_dir, Q, 
                                      min_discharge_threshold=lake_discharge_threshold,
                                      min_area_cells=4)
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # --- ROW 1: Initial and Final Topography ---
    
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.imshow(elev_initial, extent=extent_km, origin="lower", 
                     cmap=cmap_terrain, aspect="auto")
    ax1.set_title("Initial Topography", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Distance (km)")
    ax1.set_ylabel("Distance (km)")
    plt.colorbar(im1, ax=ax1, label="Elevation (m)")
    
    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.imshow(elev_final, extent=extent_km, origin="lower",
                     cmap=cmap_terrain, aspect="auto")
    ax2.set_title("Final Topography", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Distance (km)")
    ax2.set_ylabel("Distance (km)")
    plt.colorbar(im2, ax=ax2, label="Elevation (m)")
    
    ax3 = plt.subplot(2, 3, 3)
    dz_lim = max(abs(dz.min()), abs(dz.max()))
    im3 = ax3.imshow(dz, extent=extent_km, origin="lower",
                     cmap="RdBu_r", vmin=-dz_lim, vmax=dz_lim, aspect="auto")
    ax3.set_title("Elevation Change\n(Red=Erosion, Blue=Deposition)", 
                  fontsize=14, fontweight="bold")
    ax3.set_xlabel("Distance (km)")
    ax3.set_ylabel("Distance (km)")
    plt.colorbar(im3, ax=ax3, label="Change (m)")
    
    # --- ROW 2: Rivers/Lakes and Discharge ---
    
    ax4 = plt.subplot(2, 3, 4)
    # Show final topography with rivers and lakes
    im4 = ax4.imshow(elev_final, extent=extent_km, origin="lower",
                     cmap="gray", aspect="auto", alpha=0.6)
    # Overlay rivers in blue
    river_overlay = np.ma.masked_where(~rivers, elev_final)
    ax4.imshow(river_overlay, extent=extent_km, origin="lower",
               cmap="Blues", aspect="auto", alpha=0.8, vmin=elev_final.min(), 
               vmax=elev_final.max())
    # Overlay lakes in cyan
    lake_overlay = np.ma.masked_where(~lakes, elev_final)
    ax4.imshow(lake_overlay, extent=extent_km, origin="lower",
               cmap="cool", aspect="auto", alpha=0.9)
    ax4.set_title("Rivers and Lakes", fontsize=14, fontweight="bold")
    ax4.set_xlabel("Distance (km)")
    ax4.set_ylabel("Distance (km)")
    
    # Create legend
    river_patch = mpatches.Patch(color='blue', label=f'Rivers ({np.sum(rivers)} cells)')
    lake_patch = mpatches.Patch(color='cyan', label=f'Lakes ({np.sum(lakes)} cells)')
    ax4.legend(handles=[river_patch, lake_patch], loc='upper right')
    
    ax5 = plt.subplot(2, 3, 5)
    # Discharge map (log scale for better visualization)
    Q_log = np.log10(Q + 1)
    im5 = ax5.imshow(Q_log, extent=extent_km, origin="lower",
                     cmap="viridis", aspect="auto")
    ax5.set_title("Discharge (log₁₀)", fontsize=14, fontweight="bold")
    ax5.set_xlabel("Distance (km)")
    ax5.set_ylabel("Distance (km)")
    plt.colorbar(im5, ax=ax5, label="log₁₀(Q) [log m³/yr]")
    
    ax6 = plt.subplot(2, 3, 6)
    # Cross-section comparison
    mid_row = ny // 2
    x_km = np.arange(nx) * pixel_scale_m / 1000
    ax6.plot(x_km, elev_initial[mid_row, :], 'k-', linewidth=2, label="Initial")
    ax6.plot(x_km, elev_final[mid_row, :], 'r-', linewidth=2, label="Final")
    ax6.fill_between(x_km, elev_initial[mid_row, :], elev_final[mid_row, :],
                     where=(elev_final[mid_row, :] < elev_initial[mid_row, :]),
                     color='red', alpha=0.3, label="Erosion")
    ax6.fill_between(x_km, elev_initial[mid_row, :], elev_final[mid_row, :],
                     where=(elev_final[mid_row, :] >= elev_initial[mid_row, :]),
                     color='blue', alpha=0.3, label="Deposition")
    ax6.set_title(f"Cross-Section (row {mid_row})", fontsize=14, fontweight="bold")
    ax6.set_xlabel("Distance (km)")
    ax6.set_ylabel("Elevation (m)")
    ax6.legend(loc='best')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Print statistics
    print("\n" + "="*80)
    print("EROSION STATISTICS")
    print("="*80)
    print(f"Initial elevation: {elev_initial.min():.1f} - {elev_initial.max():.1f} m")
    print(f"Final elevation: {elev_final.min():.1f} - {elev_final.max():.1f} m")
    print(f"Mean elevation change: {dz.mean():.2f} m")
    print(f"Max erosion: {-dz.min():.2f} m")
    print(f"Max deposition: {dz.max():.2f} m")
    print(f"Total volume eroded: {-dz[dz<0].sum() * pixel_scale_m**2 / 1e9:.3f} km³")
    print(f"Total volume deposited: {dz[dz>0].sum() * pixel_scale_m**2 / 1e9:.3f} km³")
    print(f"Net volume change: {dz.sum() * pixel_scale_m**2 / 1e9:.3f} km³")
    print(f"\nRiver cells: {np.sum(rivers)} ({np.sum(rivers)/rivers.size*100:.1f}%)")
    print(f"Lake cells: {np.sum(lakes)} ({np.sum(lakes)/lakes.size*100:.1f}%)")
    print(f"Number of lakes: {lake_labels.max()}")
    print(f"Max discharge: {Q.max():.1f} m³/yr")
    print(f"Mean discharge: {Q.mean():.1f} m³/yr")
    print("="*80 + "\n")
    
    return fig


def plot_elevation_history(
    results: Dict,
    pixel_scale_m: float,
    figsize: Tuple[int, int] = (18, 4)
):
    """
    Plot elevation evolution over time (snapshots).
    
    Parameters
    ----------
    results : dict
        Output from run_erosion_simulation.
    pixel_scale_m : float
        Grid cell size [m].
    figsize : tuple
        Figure size.
    """
    elev_history = results["elevation_history"]
    time_points = results["time_points"]
    
    # Select up to 5 snapshots to show
    n_snapshots = min(5, len(elev_history))
    indices = np.linspace(0, len(elev_history)-1, n_snapshots, dtype=int)
    
    fig, axes = plt.subplots(1, n_snapshots, figsize=figsize)
    
    if n_snapshots == 1:
        axes = [axes]
    
    ny, nx = elev_history[0].shape
    extent_km = [0, nx * pixel_scale_m / 1000, 0, ny * pixel_scale_m / 1000]
    
    vmin = min(e.min() for e in elev_history)
    vmax = max(e.max() for e in elev_history)
    
    for i, idx in enumerate(indices):
        ax = axes[i]
        elev = elev_history[idx]
        t = time_points[idx]
        t_real = t * TIME_ACCELERATION
        
        im = ax.imshow(elev, extent=extent_km, origin="lower",
                      cmap="terrain", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(f"t = {t:.0f} yr\n({t_real:.0f} real yr)")
        ax.set_xlabel("Distance (km)")
        if i == 0:
            ax.set_ylabel("Distance (km)")
    
    plt.tight_layout()
    return fig

print("\n✅ Visualization module loaded!")
