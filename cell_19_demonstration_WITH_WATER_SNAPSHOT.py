"""
EROSION DEMONSTRATION WITH EPOCH VISUALIZATION + FINAL WATER SNAPSHOT

Features:
- Uses realistic layers from cells 0-9
- Non-uniform rain from wind physics
- Shows erosion progress after each epoch
- FINAL WATER SNAPSHOT: Shows rivers and lakes on final terrain
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import time

print("\n" + "="*80)
print("EROSION SIMULATION WITH EPOCH VISUALIZATION + WATER SNAPSHOT")
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
num_epochs = 5  # Number of epochs to show
years_per_epoch = 20  # Years between snapshots
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
        constants={
            'TIME_ACCELERATION': TIME_ACCELERATION,
            'RAIN_BOOST': RAIN_BOOST,
            'BASE_K': BASE_K,
            'MAX_ERODE_PER_STEP': MAX_ERODE_PER_STEP,
            'FLAT_K': FLAT_K,
            'SLOPE_THRESHOLD': SLOPE_THRESHOLD,
            'M_DISCHARGE': M_DISCHARGE,
            'N_SLOPE': N_SLOPE,
            'HALF_LOSS_FRACTION': HALF_LOSS_FRACTION,
            'CAPACITY_K': CAPACITY_K,
            'CAPACITY_M': CAPACITY_M,
            'CAPACITY_N': CAPACITY_N,
            'INFILTRATION_FRACTION': INFILTRATION_FRACTION,
            'DIFFUSION_K': DIFFUSION_K,
            'ERODIBILITY_MAP': ERODIBILITY_MAP
        },
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
# FINAL WATER SNAPSHOT (Diagnostic Water-Only Pass)
# ==============================================================================

print("\n" + "="*80)
print("COMPUTING FINAL WATER SNAPSHOT (Rivers + Lakes)")
print("="*80)

print("\nFreezing terrain at final elevation...")
print("Applying diagnostic rain event (water-only, no erosion)...")

# Use final elevation
elevation_final = epoch_elevations[-1].copy()
ny, nx = elevation_final.shape

# Apply a strong uniform rain event for water visualization
SNAPSHOT_RAIN_INTENSITY = 0.01  # m/hour
SNAPSHOT_RAIN_DURATION = 24  # hours
SNAPSHOT_RAIN_BOOST = 50.0  # Boost factor to fill channels

rain_snapshot = np.ones((ny, nx), dtype=np.float32) * SNAPSHOT_RAIN_INTENSITY * SNAPSHOT_RAIN_DURATION * SNAPSHOT_RAIN_BOOST

print(f"  Rain intensity: {SNAPSHOT_RAIN_INTENSITY * SNAPSHOT_RAIN_DURATION * SNAPSHOT_RAIN_BOOST:.2f} m")

# Compute runoff (simple)
infiltration_frac = INFILTRATION_FRACTION
runoff_snapshot = rain_snapshot * (1 - infiltration_frac)

# Compute flow direction from final elevation
flow_dir_snapshot, receivers_snapshot, distances_snapshot = compute_flow_direction_d8(elevation_final, pixel_scale_m)

# Compute discharge Q (water accumulation)
discharge_snapshot = compute_discharge(elevation_final, flow_dir_snapshot, receivers_snapshot, runoff_snapshot, pixel_scale_m)

print(f"✓ Flow and discharge computed")
print(f"  Max discharge: {discharge_snapshot.max():.2e} m³/s")
print(f"  Mean discharge: {discharge_snapshot.mean():.2e} m³/s")

# Convert discharge to water depth (simple model)
WATER_DEPTH_K = 0.01  # Tunable coefficient
MAX_WATER_DEPTH = 5.0  # Maximum water depth (m)

water_depth = WATER_DEPTH_K * np.sqrt(discharge_snapshot)  # sqrt to avoid extreme values
water_depth = np.clip(water_depth, 0, MAX_WATER_DEPTH)

print(f"✓ Water depth computed")
print(f"  Max depth: {water_depth.max():.2f} m")
print(f"  Mean depth: {water_depth.mean():.3f} m")

# Detect rivers and lakes
WATER_MIN_DEPTH = 0.05  # Minimum depth to consider (m)
SLOPE_LAKE_THRESHOLD = 0.01  # Slope threshold for lake vs river

# Compute slopes along flow direction
grad_y, grad_x = np.gradient(elevation_final, pixel_scale_m)
slope_mag = np.sqrt(grad_x**2 + grad_y**2)

# Classify water features
river_mask = (water_depth > WATER_MIN_DEPTH) & (slope_mag > SLOPE_LAKE_THRESHOLD)
lake_mask = (water_depth > WATER_MIN_DEPTH) & (slope_mag <= SLOPE_LAKE_THRESHOLD)

num_river_cells = np.sum(river_mask)
num_lake_cells = np.sum(lake_mask)

print(f"\n✓ Water features detected:")
print(f"  River cells: {num_river_cells} ({100*num_river_cells/(ny*nx):.2f}%)")
print(f"  Lake cells: {num_lake_cells} ({100*num_lake_cells/(ny*nx):.2f}%)")

# Also use the existing river/lake detection from cell 16
try:
    river_mask_advanced = detect_rivers(discharge_snapshot, pixel_scale_m, 
                                        discharge_threshold=np.percentile(discharge_snapshot, 95))
    lake_mask_advanced, lake_labels = detect_lakes(elevation_final, discharge_snapshot, 
                                                     pixel_scale_m, min_lake_area_m2=100)
    
    num_lakes = len(np.unique(lake_labels)) - 1  # Exclude 0 (no lake)
    print(f"\n✓ Advanced detection:")
    print(f"  River cells (advanced): {np.sum(river_mask_advanced)} ({100*np.sum(river_mask_advanced)/(ny*nx):.2f}%)")
    print(f"  Number of lakes: {num_lakes}")
    print(f"  Lake cells (advanced): {np.sum(lake_mask_advanced)} ({100*np.sum(lake_mask_advanced)/(ny*nx):.2f}%)")
    
    # Use advanced masks
    river_mask = river_mask_advanced
    lake_mask = lake_mask_advanced
except:
    print("\n⚠ Advanced detection failed, using simple classification")

print("\n" + "="*80)
print("✅ WATER SNAPSHOT COMPLETE")
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
# FINAL WATER SNAPSHOT VISUALIZATION (Rivers + Lakes Overlay)
# ==============================================================================

print("\nGenerating final water snapshot visualization...")

fig = plt.figure(figsize=(20, 12))

# Create 2x3 grid
ax1 = plt.subplot(2, 3, 1)
ax2 = plt.subplot(2, 3, 2)
ax3 = plt.subplot(2, 3, 3)
ax4 = plt.subplot(2, 3, 4)
ax5 = plt.subplot(2, 3, 5)
ax6 = plt.subplot(2, 3, 6)

# Plot 1: Final elevation (terrain)
im1 = ax1.imshow(elevation_final, cmap='terrain', origin='lower')
ax1.set_title("Final Terrain Elevation", fontsize=14, fontweight='bold')
ax1.axis('off')
plt.colorbar(im1, ax=ax1, label='Elevation (m)', fraction=0.046)

# Plot 2: Discharge (Q) - shows water flux
im2 = ax2.imshow(np.log10(discharge_snapshot + 1e-6), cmap='Blues', origin='lower')
ax2.set_title("Discharge (log₁₀ Q)", fontsize=14, fontweight='bold')
ax2.axis('off')
plt.colorbar(im2, ax=ax2, label='log₁₀(m³/s)', fraction=0.046)

# Plot 3: Water depth
im3 = ax3.imshow(water_depth, cmap='Blues', vmin=0, vmax=water_depth.max(), origin='lower')
ax3.set_title("Water Depth", fontsize=14, fontweight='bold')
ax3.axis('off')
plt.colorbar(im3, ax=ax3, label='Depth (m)', fraction=0.046)

# Plot 4: Rivers and Lakes (Binary masks)
water_vis = np.zeros((ny, nx, 3))
water_vis[:, :, 0] = 0.8  # Red channel (terrain base)
water_vis[:, :, 1] = 0.7  # Green channel
water_vis[:, :, 2] = 0.6  # Blue channel

# Rivers = blue lines
water_vis[river_mask, 0] = 0.0
water_vis[river_mask, 1] = 0.5
water_vis[river_mask, 2] = 1.0

# Lakes = cyan filled
water_vis[lake_mask, 0] = 0.0
water_vis[lake_mask, 1] = 0.8
water_vis[lake_mask, 2] = 0.9

ax4.imshow(water_vis, origin='lower')
ax4.set_title("Rivers (blue) + Lakes (cyan)", fontsize=14, fontweight='bold')
ax4.axis('off')

# Plot 5: **MAIN VISUALIZATION** - Terrain with rivers/lakes overlay
# This is the "screenshot" requested
terrain_rgb = plt.cm.terrain((elevation_final - elevation_final.min()) / (elevation_final.max() - elevation_final.min() + 1e-9))[:, :, :3]

# Create water overlay
water_overlay = np.zeros((ny, nx, 4))  # RGBA
water_overlay[:, :, 3] = 0.0  # Transparent by default

# Rivers: bright blue, semi-transparent
water_overlay[river_mask, 0] = 0.0  # R
water_overlay[river_mask, 1] = 0.4  # G
water_overlay[river_mask, 2] = 1.0  # B (bright blue)
water_overlay[river_mask, 3] = 0.8  # Alpha (semi-transparent)

# Lakes: cyan, more opaque
water_overlay[lake_mask, 0] = 0.0  # R
water_overlay[lake_mask, 1] = 0.7  # G
water_overlay[lake_mask, 2] = 1.0  # B
water_overlay[lake_mask, 3] = 0.9  # Alpha (more opaque)

# Composite
ax5.imshow(terrain_rgb, origin='lower')
ax5.imshow(water_overlay, origin='lower')
ax5.set_title("🌊 FINAL WATER SNAPSHOT 🌊\nTerrain + Rivers + Lakes", 
              fontsize=14, fontweight='bold', color='blue')
ax5.axis('off')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='blue', alpha=0.8, label=f'Rivers ({num_river_cells} cells)'),
    Patch(facecolor='cyan', alpha=0.9, label=f'Lakes ({num_lake_cells} cells)')
]
ax5.legend(handles=legend_elements, loc='upper right', fontsize=10)

# Plot 6: Erosion depth with water overlay
erosion_final = initial_elev - elevation_final
erosion_rgb = plt.cm.hot_r(erosion_final / (erosion_final.max() + 1e-9))[:, :, :3]

ax6.imshow(erosion_rgb, origin='lower')
ax6.imshow(water_overlay, origin='lower')
ax6.set_title("Erosion Depth + Water Features", fontsize=14, fontweight='bold')
ax6.axis('off')

plt.suptitle("FINAL WATER SNAPSHOT - Diagnostic Water-Only Pass\n(No erosion, just water flow and ponding)", 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

print("✓ Water snapshot visualization complete")

# ==============================================================================
# CROSS-SECTION WITH WATER
# ==============================================================================

print("\nGenerating cross-section with water table...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

# Cross-section at middle row
mid_row = ny // 2
x_coords = np.arange(nx) * pixel_scale_m

# Plot 1: Elevation profile with water
ax1.fill_between(x_coords, 0, elevation_final[mid_row, :], color='saddlebrown', alpha=0.7, label='Terrain')
water_surface = elevation_final[mid_row, :] + water_depth[mid_row, :]
ax1.fill_between(x_coords, elevation_final[mid_row, :], water_surface, 
                 where=(water_depth[mid_row, :] > WATER_MIN_DEPTH),
                 color='cyan', alpha=0.6, label='Water')
ax1.plot(x_coords, elevation_final[mid_row, :], 'k-', linewidth=1.5, label='Ground Surface')
ax1.set_ylabel("Elevation (m)", fontsize=12)
ax1.set_title(f"Cross-Section at Row {mid_row} (with Water)", fontsize=14, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Plot 2: Discharge profile
ax2.semilogy(x_coords, discharge_snapshot[mid_row, :], 'b-', linewidth=2, label='Discharge Q')
ax2.set_xlabel("Distance (m)", fontsize=12)
ax2.set_ylabel("Discharge (m³/s, log scale)", fontsize=12)
ax2.set_title("Discharge Profile (Shows River Locations)", fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, which='both')
ax2.legend()

# Highlight river locations
river_locs = np.where(river_mask[mid_row, :])[0]
if len(river_locs) > 0:
    ax2.scatter(river_locs * pixel_scale_m, discharge_snapshot[mid_row, river_locs],
                color='red', s=50, zorder=10, label='River cells')
    ax2.legend()

plt.tight_layout()
plt.show()

print("✓ Cross-section with water generated")

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

print(f"\nWater Features (Final Snapshot):")
print(f"  Snapshot rain: {SNAPSHOT_RAIN_INTENSITY * SNAPSHOT_RAIN_DURATION * SNAPSHOT_RAIN_BOOST:.2f} m")
print(f"  Max discharge: {discharge_snapshot.max():.2e} m³/s")
print(f"  River cells: {num_river_cells} ({100*num_river_cells/(ny*nx):.2f}%)")
print(f"  Lake cells: {num_lake_cells} ({100*num_lake_cells/(ny*nx):.2f}%)")
print(f"  Max water depth: {water_depth.max():.2f} m")

print("\n" + "="*80)
print("✅ COMPLETE: EROSION + EPOCH ANALYSIS + WATER SNAPSHOT")
print("="*80)
print("\nKey visualizations generated:")
print("  1. Epoch-by-epoch evolution (elevation, materials, erosion)")
print("  2. Final water snapshot (terrain + rivers + lakes overlay)")
print("  3. Cross-section with water table")
print("\n✓ Water snapshot shows:")
print("  • Rivers (blue lines) - high discharge, sloped channels")
print("  • Lakes (cyan areas) - water ponding in flat basins")
print("  • Final terrain with realistic drainage network")
print("="*80 + "\n")
