"""
DIAGNOSTIC WATER SNAPSHOT - Final Rivers & Lakes Visualization

This cell provides a function to freeze erosion and run a diagnostic
water-only pass to visualize the final drainage network.

Purpose:
- Freeze terrain (no erosion)
- Apply strong rain event
- Compute water flow and ponding
- Generate "screenshot" showing rivers and lakes on final terrain
"""

import numpy as np
from typing import Dict, Tuple

def compute_water_snapshot(
    elevation: np.ndarray,
    pixel_scale_m: float,
    rain_intensity: float = 0.1,
    water_depth_k: float = 0.001,
    max_water_depth: float = 5.0,
    water_min_depth: float = 0.01,
    slope_lake_threshold: float = 0.01,
    infiltration_fraction: float = 0.3
) -> Dict:
    """
    Compute diagnostic water distribution WITHOUT erosion.
    
    This is a "freeze frame" water pass:
    1. Apply rain
    2. Compute runoff
    3. Compute flow directions
    4. Compute discharge Q
    5. Convert Q to water depth
    6. Classify rivers vs lakes
    
    NO EROSION OR SEDIMENT TRANSPORT - just water flow!
    
    Parameters:
    -----------
    elevation : ndarray
        Final terrain elevation (frozen, no changes)
    pixel_scale_m : float
        Grid spacing
    rain_intensity : float
        Uniform rain to apply (m)
    water_depth_k : float
        Scaling factor: water_depth = k * Q
    max_water_depth : float
        Clamp maximum water depth
    water_min_depth : float
        Minimum depth to consider as water
    slope_lake_threshold : float
        Slope below which water is "lake", above is "river"
    infiltration_fraction : float
        Fraction of rain that infiltrates
    
    Returns:
    --------
    dict with:
        - water_depth: Water depth at each cell
        - Q: Discharge (water flux)
        - river_mask: Boolean mask of river cells
        - lake_mask: Boolean mask of lake cells
        - flow_dir: Flow direction indices
        - slope: Slope along flow direction
    """
    ny, nx = elevation.shape
    
    print(f"\n{'='*60}")
    print("DIAGNOSTIC WATER SNAPSHOT (No Erosion)")
    print(f"{'='*60}")
    print(f"  Terrain: {nx}×{ny}")
    print(f"  Rain: {rain_intensity} m")
    print(f"  Infiltration: {infiltration_fraction*100:.0f}%")
    
    # ========================================
    # STEP 1: Apply Rain → Runoff
    # ========================================
    
    rain = np.ones((ny, nx), dtype=np.float32) * rain_intensity
    infiltration = rain * infiltration_fraction
    runoff = np.maximum(0.0, rain - infiltration)
    
    print(f"\n  Runoff: {runoff.mean():.4f} m (mean)")
    
    # ========================================
    # STEP 2: Compute Flow Directions (D8)
    # ========================================
    
    flow_dir = -np.ones((ny, nx), dtype=np.int32)  # -1 = NONE (pit)
    receivers = np.zeros((ny, nx, 2), dtype=np.int32)
    
    # 8 neighbors: N, NE, E, SE, S, SW, W, NW
    di = [-1, -1,  0,  1, 1, 1, 0, -1]
    dj = [ 0,  1,  1,  1, 0, -1, -1, -1]
    distances = [pixel_scale_m, pixel_scale_m*np.sqrt(2), pixel_scale_m, pixel_scale_m*np.sqrt(2),
                 pixel_scale_m, pixel_scale_m*np.sqrt(2), pixel_scale_m, pixel_scale_m*np.sqrt(2)]
    
    for i in range(ny):
        for j in range(nx):
            z_c = elevation[i, j]
            best_slope = 0.0
            best_dir = -1
            
            for k in range(8):
                ni = i + di[k]
                nj = j + dj[k]
                
                if 0 <= ni < ny and 0 <= nj < nx:
                    z_n = elevation[ni, nj]
                    if z_n < z_c:
                        slope = (z_c - z_n) / distances[k]
                        if slope > best_slope:
                            best_slope = slope
                            best_dir = k
                            receivers[i, j] = [ni, nj]
            
            flow_dir[i, j] = best_dir
    
    pits = np.sum(flow_dir == -1)
    print(f"  Flow direction: {pits} pit cells (potential lakes)")
    
    # ========================================
    # STEP 3: Compute Discharge Q
    # ========================================
    
    Q = np.zeros((ny, nx), dtype=np.float32)
    
    # Sort cells by elevation (high to low)
    flat_indices = np.argsort(elevation.ravel())[::-1]
    coords_sorted = [(idx // nx, idx % nx) for idx in flat_indices]
    
    for (i, j) in coords_sorted:
        Q[i, j] += runoff[i, j]
        
        if flow_dir[i, j] >= 0:
            ni, nj = receivers[i, j]
            Q[ni, nj] += Q[i, j]
    
    print(f"  Discharge Q: {Q.min():.6f} - {Q.max():.6f} (range)")
    print(f"  High Q cells (>0.1): {np.sum(Q > 0.1)}")
    
    # ========================================
    # STEP 4: Compute Slope Along Flow
    # ========================================
    
    slope = np.zeros((ny, nx), dtype=np.float32)
    
    for i in range(ny):
        for j in range(nx):
            if flow_dir[i, j] >= 0:
                ni, nj = receivers[i, j]
                k = flow_dir[i, j]
                dist = distances[k]
                slope[i, j] = (elevation[i, j] - elevation[ni, nj]) / dist
            else:
                slope[i, j] = 0.0
    
    slope = np.maximum(slope, 0.0)
    
    print(f"  Slope: {slope.mean():.6f} (mean)")
    
    # ========================================
    # STEP 5: Convert Q to Water Depth
    # ========================================
    
    water_depth = water_depth_k * Q
    water_depth = np.minimum(water_depth, max_water_depth)
    
    print(f"  Water depth: {water_depth.min():.4f} - {water_depth.max():.4f} m")
    
    # ========================================
    # STEP 6: Classify Rivers vs Lakes
    # ========================================
    
    # Lake: Water present, low slope (flat)
    # River: Water present, higher slope (flowing)
    
    water_present = water_depth > water_min_depth
    
    lake_mask = water_present & (slope < slope_lake_threshold)
    river_mask = water_present & (slope >= slope_lake_threshold)
    
    n_lake_cells = np.sum(lake_mask)
    n_river_cells = np.sum(river_mask)
    
    print(f"\n  Classification:")
    print(f"    Lake cells: {n_lake_cells} ({100*n_lake_cells/(ny*nx):.2f}%)")
    print(f"    River cells: {n_river_cells} ({100*n_river_cells/(ny*nx):.2f}%)")
    
    print(f"{'='*60}")
    print("✓ Water snapshot computed")
    print(f"{'='*60}\n")
    
    return {
        'water_depth': water_depth,
        'Q': Q,
        'river_mask': river_mask,
        'lake_mask': lake_mask,
        'flow_dir': flow_dir,
        'slope': slope,
        'runoff': runoff
    }


def visualize_water_on_terrain(elevation: np.ndarray, 
                                water_data: Dict,
                                title: str = "Final Rivers & Lakes"):
    """
    Create beautiful visualization of water overlay on terrain.
    
    Shows:
    - Base elevation map
    - Rivers (blue lines)
    - Lakes (blue patches)
    - Transparent water overlay
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    water_depth = water_data['water_depth']
    Q = water_data['Q']
    river_mask = water_data['river_mask']
    lake_mask = water_data['lake_mask']
    slope = water_data['slope']
    
    # ========================================
    # Plot 1: Elevation with Water Overlay
    # ========================================
    ax = axes[0, 0]
    
    # Shaded relief
    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(elevation, cmap=plt.cm.terrain, vert_exag=0.1, blend_mode='soft')
    ax.imshow(rgb, origin='lower')
    
    # Overlay water
    water_overlay = np.ma.masked_where(water_depth < 0.01, water_depth)
    im = ax.imshow(water_overlay, cmap='Blues', alpha=0.6, origin='lower', vmin=0, vmax=water_depth.max())
    
    ax.set_title("Terrain + Water Overlay")
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='Water Depth (m)', fraction=0.046)
    
    # ========================================
    # Plot 2: Rivers and Lakes (Classified)
    # ========================================
    ax = axes[0, 1]
    
    # Base terrain
    ax.imshow(elevation, cmap='terrain', origin='lower', alpha=0.5)
    
    # Rivers: bright blue
    river_display = np.zeros_like(elevation)
    river_display[river_mask] = 1.0
    ax.imshow(np.ma.masked_where(river_display == 0, river_display), 
              cmap='Blues', alpha=0.9, origin='lower', vmin=0, vmax=1)
    
    # Lakes: cyan
    lake_display = np.zeros_like(elevation)
    lake_display[lake_mask] = 1.0
    ax.imshow(np.ma.masked_where(lake_display == 0, lake_display), 
              cmap='winter', alpha=0.7, origin='lower', vmin=0, vmax=1)
    
    ax.set_title(f"Rivers (Blue) + Lakes (Cyan)\n{np.sum(river_mask)} river cells, {np.sum(lake_mask)} lake cells")
    ax.axis('off')
    
    # ========================================
    # Plot 3: Discharge (Q) - Water Flux
    # ========================================
    ax = axes[0, 2]
    
    Q_log = np.log10(Q + 1e-6)
    im = ax.imshow(Q_log, cmap='YlGnBu', origin='lower')
    ax.set_title("Discharge (log₁₀ Q)\nDrainage Network")
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='log₁₀(Q)', fraction=0.046)
    
    # ========================================
    # Plot 4: Water Depth
    # ========================================
    ax = axes[1, 0]
    
    im = ax.imshow(water_depth, cmap='Blues', origin='lower', vmin=0)
    ax.set_title("Water Depth")
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='Depth (m)', fraction=0.046)
    
    # ========================================
    # Plot 5: Slope
    # ========================================
    ax = axes[1, 1]
    
    slope_display = np.log10(slope + 1e-6)
    im = ax.imshow(slope_display, cmap='plasma', origin='lower')
    ax.set_title("Slope (log₁₀)\nSteep = Rivers, Flat = Lakes")
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='log₁₀(slope)', fraction=0.046)
    
    # ========================================
    # Plot 6: Binary Water Mask
    # ========================================
    ax = axes[1, 2]
    
    water_mask = (water_depth > 0.01).astype(float)
    im = ax.imshow(water_mask, cmap='binary_r', origin='lower')
    ax.set_title(f"Water Present/Absent\n{np.sum(water_mask)} wet cells")
    ax.axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


print("✓ Water snapshot module loaded")
print("  Functions: compute_water_snapshot(), visualize_water_on_terrain()")
