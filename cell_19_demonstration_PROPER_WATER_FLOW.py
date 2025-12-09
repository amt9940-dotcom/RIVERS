"""
EROSION DEMONSTRATION WITH PROPER WATER FLOW SIMULATION

Features:
- Uses realistic layers from cells 0-9
- Non-uniform rain from wind physics
- Shows erosion progress after each epoch
- PROPER WATER FLOW: Simulates water flowing downhill, tracks actual water location
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import time

print("\n" + "="*80)
print("EROSION SIMULATION WITH PROPER WATER FLOW")
print("="*80)

# ==============================================================================
# VALIDATE INPUT DATA
# ==============================================================================

print("\nValidating input data...")

# Check for global variables
if 'GLOBAL_STRATA' not in globals():
    raise RuntimeError("GLOBAL_STRATA not found! Run cells 0-9 first.")
if 'GLOBAL_WEATHER_DATA' not in globals():
    raise RuntimeError("GLOBAL_WEATHER_DATA not found! Run cells 0-9 first.")
if 'GLOBAL_RAIN_TIMESERIES' not in globals():
    raise RuntimeError("GLOBAL_RAIN_TIMESERIES not found! Run cells 0-9 first.")

print("✓ All global variables found")

# Extract data
elevation_initial = GLOBAL_STRATA['surface_elev'].copy()
thickness_initial = {k: v.copy() for k, v in GLOBAL_STRATA['thickness'].items()}
layer_order = GLOBAL_STRATA['layer_order'].copy()
pixel_scale_m = GLOBAL_STRATA['pixel_scale_m']
rain_timeseries = GLOBAL_RAIN_TIMESERIES.copy()

print(f"✓ Using terrain from cells 0-9: {elevation_initial.shape}")
print(f"✓ Using {len(layer_order)} layers: {layer_order}")
print(f"✓ Using {len(rain_timeseries)} years of rain data")

# ==============================================================================
# RUN EROSION WITH EPOCH SNAPSHOTS
# ==============================================================================

print("\n" + "="*80)
print("RUNNING EROSION SIMULATION")
print("="*80)

# Simulation parameters
num_epochs = 5
years_per_epoch = 20
total_years = num_epochs * years_per_epoch

print(f"\nSimulation plan:")
print(f"  Total years: {total_years}")
print(f"  Epochs: {num_epochs}")
print(f"  Years per epoch: {years_per_epoch}")
print(f"  Real-world equivalent: {total_years * TIME_ACCELERATION:.0f} years")

# Storage for epoch snapshots
epoch_elevations = []
epoch_layers = []
epoch_years = []

# Current state
elevation = elevation_initial.copy()
thickness = {k: v.copy() for k, v in thickness_initial.items()}

# Initial snapshot
epoch_elevations.append(elevation.copy())
top_idx, top_name = compute_top_layer_map(thickness, layer_order)
epoch_layers.append(top_name.copy())
epoch_years.append(0)

print(f"\nEpoch 0: Initial state")
print(f"  Elevation: {elevation.min():.1f} - {elevation.max():.1f} m")

# Run simulation epoch by epoch
for epoch in range(1, num_epochs + 1):
    print(f"\n{'='*60}")
    print(f"Epoch {epoch}/{num_epochs}: Years {(epoch-1)*years_per_epoch} → {epoch*years_per_epoch}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Run erosion for this epoch
    results = run_erosion_simulation(
        elevation_initial=elevation,
        thickness_initial=thickness,
        layer_order=layer_order,
        rain_timeseries=rain_timeseries[(epoch-1)*years_per_epoch:epoch*years_per_epoch],
        pixel_scale_m=pixel_scale_m,
        dt=1.0,
        num_timesteps=years_per_epoch,
        save_interval=years_per_epoch,
        apply_diffusion=True,
        verbose=True
    )
    
    elapsed = time.time() - start_time
    
    # Update state for next epoch
    elevation = results['elevation_final']
    thickness = results['thickness_final']
    
    # Save snapshot
    epoch_elevations.append(elevation.copy())
    top_idx, top_name = compute_top_layer_map(thickness, layer_order)
    epoch_layers.append(top_name.copy())
    epoch_years.append(epoch * years_per_epoch)
    
    # Stats
    elev_change = elevation - elevation_initial
    print(f"\n✓ Epoch {epoch} complete in {elapsed:.1f} s")
    print(f"  Elevation: {elevation.min():.1f} - {elevation.max():.1f} m")
    print(f"  Total erosion: {-elev_change.min():.1f} m (max)")
    print(f"  Mean change: {elev_change.mean():.2f} m")

print("\n" + "="*80)
print("✅ EROSION SIMULATION COMPLETE")
print("="*80)

# ==============================================================================
# PROPER WATER FLOW SIMULATION
# ==============================================================================

print("\n" + "="*80)
print("SIMULATING WATER FLOW (Proper Physics)")
print("="*80)

# 8-connected neighbors
NEIGHBOR_OFFSETS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

def simulate_water_flow(height, rain_amount=1.0, steps=200, flow_factor=0.5):
    """
    Simulate water flowing downhill on fixed terrain.
    
    Parameters
    ----------
    height : np.ndarray (ny, nx)
        Terrain elevation [m]
    rain_amount : float
        Initial water added to every cell [m]
    steps : int
        Number of flow iterations
    flow_factor : float
        How aggressively water flows (0-1)
    
    Returns
    -------
    water : np.ndarray (ny, nx)
        Water depth at each cell after flow [m]
    """
    height = height.astype(np.float64)
    rows, cols = height.shape
    
    # Initialize water depth (uniform rainfall)
    water = np.full_like(height, rain_amount, dtype=np.float64)
    
    print(f"  Starting water flow simulation...")
    print(f"    Initial rain: {rain_amount:.3f} m per cell")
    print(f"    Flow iterations: {steps}")
    print(f"    Flow factor: {flow_factor}")
    
    for step in range(steps):
        surface = height + water  # Water surface elevation
        dwater = np.zeros_like(water)
        
        # For each cell, flow water to lowest neighbor
        for i in range(rows):
            for j in range(cols):
                current_surface = surface[i, j]
                lowest_surface = current_surface
                lowest_pos = None
                
                # Find lowest neighboring surface
                for di, dj in NEIGHBOR_OFFSETS:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        ns = surface[ni, nj]
                        if ns < lowest_surface:
                            lowest_surface = ns
                            lowest_pos = (ni, nj)
                
                # No lower neighbor → local minimum (potential lake)
                if lowest_pos is None:
                    continue
                
                drop = current_surface - lowest_surface
                if drop <= 0:
                    continue
                
                # Amount of water that flows out
                outflow = min(water[i, j], drop * flow_factor)
                if outflow > 0:
                    dwater[i, j] -= outflow
                    dwater[lowest_pos] += outflow
        
        water += dwater
        
        # Progress update
        if (step + 1) % 50 == 0:
            print(f"    Iteration {step+1}/{steps}: max water depth = {water.max():.3f} m")
    
    print(f"  ✓ Water flow complete")
    print(f"    Final max depth: {water.max():.3f} m")
    print(f"    Final mean depth: {water.mean():.3f} m")
    
    return water


def classify_water_cells(height, water, min_depth=0.01):
    """
    Classify water cells into lakes vs rivers.
    
    Parameters
    ----------
    height : np.ndarray (ny, nx)
        Terrain elevation [m]
    water : np.ndarray (ny, nx)
        Water depth [m]
    min_depth : float
        Minimum depth to consider as water [m]
    
    Returns
    -------
    lakes : np.ndarray (ny, nx), bool
        True where lakes exist (water with no downhill outlet)
    rivers : np.ndarray (ny, nx), bool
        True where rivers exist (water with downhill outlet)
    """
    height = height.astype(np.float64)
    surface = height + water
    rows, cols = surface.shape
    
    lakes = np.zeros_like(surface, dtype=bool)
    rivers = np.zeros_like(surface, dtype=bool)
    
    for i in range(rows):
        for j in range(cols):
            if water[i, j] < min_depth:
                continue
            
            current_surface = surface[i, j]
            has_downhill = False
            
            # Check if any neighbor is lower
            for di, dj in NEIGHBOR_OFFSETS:
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    if surface[ni, nj] < current_surface - 1e-9:
                        has_downhill = True
                        break
            
            if has_downhill:
                rivers[i, j] = True
            else:
                lakes[i, j] = True
    
    return lakes, rivers


# Run water flow simulation on final terrain
print("\nFreezing terrain at final elevation...")
elevation_final = epoch_elevations[-1].copy()

print("Applying rainfall and simulating flow...")
water_depth = simulate_water_flow(
    height=elevation_final,
    rain_amount=1.0,      # 1 meter of rain
    steps=300,            # 300 flow iterations
    flow_factor=0.4       # Moderate flow speed
)

print("\nClassifying water features...")
lakes_mask, rivers_mask = classify_water_cells(
    height=elevation_final,
    water=water_depth,
    min_depth=0.02  # Only show water > 2cm deep
)

num_river_cells = np.sum(rivers_mask)
num_lake_cells = np.sum(lakes_mask)
ny, nx = elevation_final.shape

print(f"✓ Water classification complete:")
print(f"  River cells: {num_river_cells} ({100*num_river_cells/(ny*nx):.2f}%)")
print(f"  Lake cells: {num_lake_cells} ({100*num_lake_cells/(ny*nx):.2f}%)")
print(f"  Total water cells: {num_river_cells + num_lake_cells} ({100*(num_river_cells + num_lake_cells)/(ny*nx):.2f}%)")

print("\n" + "="*80)
print("✅ WATER FLOW SIMULATION COMPLETE")
print("="*80)

# ==============================================================================
# EPOCH-BY-EPOCH VISUALIZATION
# ==============================================================================

print("\nGenerating epoch-by-epoch plots...")

fig, axes = plt.subplots(3, num_epochs + 1, figsize=(4*(num_epochs+1), 12))

# Compute elevation range for consistent colorscale
all_elevs = np.concatenate([e.flatten() for e in epoch_elevations])
vmin, vmax = np.percentile(all_elevs, [1, 99])

# Compute total change
initial_elev = epoch_elevations[0]
max_erosion = 0

# Row 1: Elevation maps
for i, (elev, year) in enumerate(zip(epoch_elevations, epoch_years)):
    ax = axes[0, i]
    im = ax.imshow(elev, cmap='terrain', vmin=vmin, vmax=vmax, origin='lower')
    ax.set_title(f"Year {year}\n({year * TIME_ACCELERATION:.0f} real years)")
    ax.axis('off')
    if i == num_epochs:
        plt.colorbar(im, ax=ax, label='Elevation (m)', fraction=0.046)

axes[0, 0].set_ylabel("ELEVATION", fontsize=12, fontweight='bold')

# Row 2: Surface material
layer_colors_map = {
    'Topsoil': 0, 'Subsoil': 1, 'Colluvium': 2,
    'Saprolite': 3, 'WeatheredBR': 4, 'Basement': 5
}

for i, (layer_map, year) in enumerate(zip(epoch_layers, epoch_years)):
    ax = axes[1, i]
    
    # Convert layer names to colors
    ny_plot, nx_plot = layer_map.shape
    color_map = np.zeros((ny_plot, nx_plot))
    for ii in range(ny_plot):
        for jj in range(nx_plot):
            color_map[ii, jj] = layer_colors_map.get(layer_map[ii, jj], 5)
    
    im = ax.imshow(color_map, cmap='tab10', vmin=0, vmax=5, origin='lower')
    ax.set_title(f"Year {year}")
    ax.axis('off')
    
    if i == num_epochs:
        cbar = plt.colorbar(im, ax=ax, ticks=[0,1,2,3,4,5], fraction=0.046)
        cbar.set_ticklabels(['Topsoil', 'Subsoil', 'Colluvium', 'Saprolite', 'W.BR', 'Basement'])

axes[1, 0].set_ylabel("SURFACE MATERIAL", fontsize=12, fontweight='bold')

# Row 3: Erosion depth (cumulative)
for i, (elev, year) in enumerate(zip(epoch_elevations, epoch_years)):
    ax = axes[2, i]
    
    erosion_depth = initial_elev - elev  # Positive = erosion
    max_erosion = max(max_erosion, erosion_depth.max())
    
    im = ax.imshow(erosion_depth, cmap='hot_r', vmin=0, vmax=None, origin='lower')
    ax.set_title(f"Year {year}")
    ax.axis('off')
    
    if i == num_epochs:
        plt.colorbar(im, ax=ax, label='Erosion (m)', fraction=0.046)

axes[2, 0].set_ylabel("EROSION DEPTH", fontsize=12, fontweight='bold')

plt.suptitle(f"Erosion Evolution Over {total_years} Years ({total_years * TIME_ACCELERATION:.0f} Real Years)", 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()

print("✓ Epoch plots generated")

# ==============================================================================
# WATER FLOW VISUALIZATION (PROPER - Only Where Water Is)
# ==============================================================================

print("\nGenerating water flow visualization...")

fig = plt.figure(figsize=(20, 12))

# Create 2x3 grid
ax1 = plt.subplot(2, 3, 1)
ax2 = plt.subplot(2, 3, 2)
ax3 = plt.subplot(2, 3, 3)
ax4 = plt.subplot(2, 3, 4)
ax5 = plt.subplot(2, 3, 5)
ax6 = plt.subplot(2, 3, 6)

# Plot 1: Final terrain elevation
im1 = ax1.imshow(elevation_final, cmap='terrain', origin='lower')
ax1.set_title("Final Terrain Elevation", fontsize=14, fontweight='bold')
ax1.axis('off')
plt.colorbar(im1, ax=ax1, label='Elevation (m)', fraction=0.046)

# Plot 2: Water depth (all water)
water_to_show = np.where(water_depth > 0.02, water_depth, np.nan)
im2 = ax2.imshow(water_to_show, cmap='Blues', origin='lower')
ax2.set_title("Water Depth (All Water)", fontsize=14, fontweight='bold')
ax2.axis('off')
plt.colorbar(im2, ax=ax2, label='Depth (m)', fraction=0.046)

# Plot 3: Rivers only
rivers_depth = np.where(rivers_mask, water_depth, np.nan)
im3 = ax3.imshow(rivers_depth, cmap='Blues', origin='lower')
ax3.set_title("Rivers Only", fontsize=14, fontweight='bold')
ax3.axis('off')
plt.colorbar(im3, ax=ax3, label='Depth (m)', fraction=0.046)

# Plot 4: Lakes only
lakes_depth = np.where(lakes_mask, water_depth, np.nan)
im4 = ax4.imshow(lakes_depth, cmap='Blues', origin='lower')
ax4.set_title("Lakes Only", fontsize=14, fontweight='bold')
ax4.axis('off')
plt.colorbar(im4, ax=ax4, label='Depth (m)', fraction=0.046)

# Plot 5: **MAIN VISUALIZATION** - Terrain with water overlay (NO BLUE HUE!)
terrain_rgb = plt.cm.terrain((elevation_final - elevation_final.min()) / 
                             (elevation_final.max() - elevation_final.min() + 1e-9))[:, :, :3]

# Create water overlay - ONLY where water exists
water_overlay = np.zeros((ny, nx, 4))  # RGBA
water_overlay[:, :, 3] = 0.0  # Transparent everywhere by default

# Rivers: bright blue, semi-transparent
water_overlay[rivers_mask, 0] = 0.0  # R
water_overlay[rivers_mask, 1] = 0.4  # G
water_overlay[rivers_mask, 2] = 1.0  # B (bright blue)
water_overlay[rivers_mask, 3] = 0.7  # Alpha

# Lakes: cyan, more opaque
water_overlay[lakes_mask, 0] = 0.0  # R
water_overlay[lakes_mask, 1] = 0.7  # G
water_overlay[lakes_mask, 2] = 1.0  # B
water_overlay[lakes_mask, 3] = 0.85  # Alpha

# Composite - terrain stays as-is, only water cells get blue
ax5.imshow(terrain_rgb, origin='lower')
ax5.imshow(water_overlay, origin='lower')
ax5.set_title("🌊 TERRAIN + RIVERS + LAKES 🌊\n(Proper Water Flow - No Blue Hue!)", 
              fontsize=14, fontweight='bold', color='blue')
ax5.axis('off')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='blue', alpha=0.7, label=f'Rivers ({num_river_cells} cells)'),
    Patch(facecolor='cyan', alpha=0.85, label=f'Lakes ({num_lake_cells} cells)')
]
ax5.legend(handles=legend_elements, loc='upper right', fontsize=10)

# Plot 6: Erosion with water overlay
erosion_final = initial_elev - elevation_final
erosion_rgb = plt.cm.hot_r(erosion_final / (erosion_final.max() + 1e-9))[:, :, :3]

ax6.imshow(erosion_rgb, origin='lower')
ax6.imshow(water_overlay, origin='lower')
ax6.set_title("Erosion Depth + Water", fontsize=14, fontweight='bold')
ax6.axis('off')

plt.suptitle("PROPER WATER FLOW SIMULATION\n(Water tracked cell-by-cell, flows downhill to lowest neighbor)", 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

print("✓ Water flow visualization complete")

# ==============================================================================
# CROSS-SECTION WITH WATER
# ==============================================================================

print("\nGenerating cross-section with water...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

# Cross-section at middle row
mid_row = ny // 2
x_coords = np.arange(nx) * pixel_scale_m

# Plot 1: Elevation profile with water
ax1.fill_between(x_coords, 0, elevation_final[mid_row, :], 
                 color='saddlebrown', alpha=0.7, label='Terrain')
water_surface = elevation_final[mid_row, :] + water_depth[mid_row, :]
water_exists = water_depth[mid_row, :] > 0.02
ax1.fill_between(x_coords, elevation_final[mid_row, :], water_surface, 
                 where=water_exists,
                 color='cyan', alpha=0.6, label='Water')
ax1.plot(x_coords, elevation_final[mid_row, :], 'k-', linewidth=1.5, label='Ground Surface')
ax1.set_ylabel("Elevation (m)", fontsize=12)
ax1.set_title(f"Cross-Section at Row {mid_row} (with Water)", fontsize=14, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Plot 2: Water depth profile
ax2.plot(x_coords, water_depth[mid_row, :], 'b-', linewidth=2, label='Water Depth')
ax2.set_xlabel("Distance (m)", fontsize=12)
ax2.set_ylabel("Water Depth (m)", fontsize=12)
ax2.set_title("Water Depth Profile", fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Highlight rivers and lakes
river_locs = np.where(rivers_mask[mid_row, :])[0]
lake_locs = np.where(lakes_mask[mid_row, :])[0]
if len(river_locs) > 0:
    ax2.scatter(river_locs * pixel_scale_m, water_depth[mid_row, river_locs],
                color='blue', s=30, zorder=10, label='Rivers', alpha=0.7)
if len(lake_locs) > 0:
    ax2.scatter(lake_locs * pixel_scale_m, water_depth[mid_row, lake_locs],
                color='cyan', s=50, zorder=10, label='Lakes', alpha=0.7)
ax2.legend()

plt.tight_layout()
plt.show()

print("✓ Cross-section generated")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\nSimulation:")
print(f"  Duration: {total_years} sim years = {total_years * TIME_ACCELERATION:.0f} real years")
print(f"  Epochs: {num_epochs}")

print(f"\nErosion:")
final_erosion = initial_elev - epoch_elevations[-1]
print(f"  Mean: {final_erosion.mean():.2f} m")
print(f"  Max: {final_erosion.max():.2f} m")
print(f"  Std dev: {final_erosion.std():.2f} m")

print(f"\nWater Flow (Proper Simulation):")
print(f"  Initial rainfall: 1.0 m uniform")
print(f"  Flow iterations: 300")
print(f"  Final water depth:")
print(f"    Max: {water_depth.max():.3f} m")
print(f"    Mean: {water_depth.mean():.3f} m")
print(f"  River cells: {num_river_cells} ({100*num_river_cells/(ny*nx):.2f}%)")
print(f"  Lake cells: {num_lake_cells} ({100*num_lake_cells/(ny*nx):.2f}%)")

print("\n" + "="*80)
print("✅ COMPLETE: EROSION + PROPER WATER FLOW")
print("="*80)
print("\nKey improvements:")
print("  ✓ Water tracked cell-by-cell (not discharge Q)")
print("  ✓ Water flows to lowest neighbor iteratively")
print("  ✓ Lakes = water with no downhill outlet")
print("  ✓ Rivers = water with downhill neighbor")
print("  ✓ NO BLUE HUE - only water cells are colored")
print("  ✓ Terrain stays normal colors where there's no water")
print("="*80 + "\n")
