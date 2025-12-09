"""
EROSION DEMONSTRATION WITH EPOCH-BY-EPOCH VISUALIZATION

Features:
- Uses realistic layers from cells 0-9
- Non-uniform rain from wind physics
- Shows erosion progress after each epoch
- Tracks how different materials erode at different rates
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import time

print("\n" + "="*80)
print("EROSION SIMULATION WITH EPOCH VISUALIZATION")
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
print("✅ SIMULATION COMPLETE")
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
    ny, nx = layer_map.shape
    color_map = np.zeros((ny, nx))
    for ii in range(ny):
        for jj in range(nx):
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
# EROSION RATE ANALYSIS
# ==============================================================================

print("\nGenerating erosion rate analysis...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Erosion vs time (mean)
ax = axes[0]
mean_erosion = [initial_elev.mean() - e.mean() for e in epoch_elevations]
ax.plot(epoch_years, mean_erosion, 'o-', linewidth=2, markersize=8)
ax.set_xlabel("Simulation Year")
ax.set_ylabel("Mean Erosion (m)")
ax.set_title("Average Erosion Over Time")
ax.grid(True, alpha=0.3)

# Plot 2: Max erosion vs time
ax = axes[1]
max_erosion_per_epoch = [np.max(initial_elev - e) for e in epoch_elevations]
ax.plot(epoch_years, max_erosion_per_epoch, 's-', linewidth=2, markersize=8, color='red')
ax.set_xlabel("Simulation Year")
ax.set_ylabel("Maximum Erosion (m)")
ax.set_title("Maximum Erosion Over Time")
ax.grid(True, alpha=0.3)

# Plot 3: Erosion histogram (final)
ax = axes[2]
final_erosion = initial_elev - epoch_elevations[-1]
ax.hist(final_erosion.flatten(), bins=50, edgecolor='black', alpha=0.7)
ax.set_xlabel("Erosion Depth (m)")
ax.set_ylabel("Number of Cells")
ax.set_title(f"Erosion Distribution (Year {total_years})")
ax.axvline(final_erosion.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {final_erosion.mean():.2f} m')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.suptitle("Erosion Statistics", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("✓ Erosion rate analysis complete")

# ==============================================================================
# MATERIAL EXPOSURE TRACKING
# ==============================================================================

print("\nGenerating material exposure tracking...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Count exposed materials at each epoch
material_counts = {layer: [] for layer in layer_order}

for layer_map in epoch_layers:
    counts = {layer: 0 for layer in layer_order}
    unique, counts_arr = np.unique(layer_map, return_counts=True)
    for mat, cnt in zip(unique, counts_arr):
        if mat in counts:
            counts[mat] = cnt
    
    for layer in layer_order:
        material_counts[layer].append(counts[layer])

# Plot 1: Stacked area chart
ax = axes[0]
bottom = np.zeros(len(epoch_years))
colors = plt.cm.tab10(np.linspace(0, 1, len(layer_order)))

for i, layer in enumerate(layer_order):
    counts = np.array(material_counts[layer])
    ax.fill_between(epoch_years, bottom, bottom + counts, label=layer, alpha=0.7, color=colors[i])
    bottom += counts

ax.set_xlabel("Simulation Year")
ax.set_ylabel("Number of Exposed Cells")
ax.set_title("Surface Material Exposure Over Time")
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3, axis='y')

# Plot 2: Percentage change
ax = axes[1]
total_cells = epoch_elevations[0].size

for i, layer in enumerate(layer_order):
    counts = np.array(material_counts[layer])
    percentages = 100 * counts / total_cells
    ax.plot(epoch_years, percentages, 'o-', label=layer, linewidth=2, markersize=6, color=colors[i])

ax.set_xlabel("Simulation Year")
ax.set_ylabel("Surface Coverage (%)")
ax.set_title("Material Exposure Percentage Over Time")
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

plt.suptitle("Surface Material Evolution", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("✓ Material exposure tracking complete")

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

print(f"\nInitial vs Final Elevation:")
print(f"  Initial: {initial_elev.mean():.1f} ± {initial_elev.std():.1f} m")
print(f"  Final: {epoch_elevations[-1].mean():.1f} ± {epoch_elevations[-1].std():.1f} m")

print(f"\nMaterial Exposure (Final):")
for layer in layer_order:
    pct = 100 * material_counts[layer][-1] / total_cells
    print(f"  {layer:15s}: {pct:5.1f}%")

print("\n" + "="*80)
print("✅ EPOCH-BY-EPOCH EROSION ANALYSIS COMPLETE")
print("="*80)
print("\nKey observations:")
print("  • Different materials erode at different rates")
print("  • Topsoil erodes fastest → exposes deeper layers")
print("  • Valleys deepen where rain concentrates (wind channels)")
print("  • Ridges resist erosion (less rain, harder rock)")
print("="*80 + "\n")
