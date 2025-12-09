#!/usr/bin/env python3
"""
Complete Erosion Simulator - Python Script Version

This is a plain Python script with all code in the correct order.
Just run: python3 run_erosion_simulator.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors

print("="*70)
print("LANDSCAPE EVOLUTION SIMULATOR - EROSION ANALYSIS")
print("="*70)

# =============================================================================
# 1. IMPORT LANDSCAPE EVOLUTION PACKAGE
# =============================================================================
print("\n[1/10] Importing landscape evolution package...")

from landscape_evolution import (
    WorldState,
    TectonicUplift,
    WeatherGenerator,
    FlowRouter,
    LandscapeEvolutionSimulator,
    plot_erosion_analysis,
    plot_erosion_rate_map
)

from landscape_evolution.terrain_generation import (
    quantum_seeded_topography,
    denormalize_elevation
)

from landscape_evolution.initial_stratigraphy import (
    create_slope_dependent_stratigraphy
)

from landscape_evolution.visualization import (
    plot_initial_vs_final,
    plot_erosion_deposition_maps,
    plot_river_network,
    plot_cross_section
)

print("✓ Package imported successfully")

# =============================================================================
# 2. PARAMETERS
# =============================================================================
print("\n[2/10] Setting parameters...")

N = 256  # Grid size (256x256)
pixel_scale_m = 100.0  # 100 meters per pixel
elev_range_m = (0.0, 1000.0)  # Elevation range: 0 to 1000 m

print(f"  Grid: {N}×{N}")
print(f"  Pixel scale: {pixel_scale_m} m")
print(f"  Elevation range: {elev_range_m[0]}-{elev_range_m[1]} m")

# =============================================================================
# 3. GENERATE TERRAIN
# =============================================================================
print("\n[3/10] Generating initial terrain...")
print("  This may take a moment...")

z_norm, rng = quantum_seeded_topography(
    N=N,
    beta=3.1,           # Smoothness parameter
    warp_amp=0.12,      # Domain warping
    ridged_alpha=0.18,  # Ridge sharpening
    random_seed=42      # For reproducibility
)

surface_elev = denormalize_elevation(z_norm, elev_range_m)

print(f"✓ Terrain generated")
print(f"  Elevation range: [{surface_elev.min():.1f}, {surface_elev.max():.1f}] m")
print(f"  Mean elevation: {surface_elev.mean():.1f} m")

# Visualize
plt.figure(figsize=(10, 8))
plt.imshow(surface_elev, cmap='terrain', origin='lower')
plt.colorbar(label='Elevation (m)', shrink=0.8)
plt.title('Initial Terrain', fontsize=14, fontweight='bold')
plt.xlabel('X (pixels)')
plt.ylabel('Y (pixels)')
plt.tight_layout()
plt.savefig('01_initial_terrain.png', dpi=150, bbox_inches='tight')
print("  Saved: 01_initial_terrain.png")
plt.close()

# =============================================================================
# 4. INITIALIZE WORLD STATE
# =============================================================================
print("\n[4/10] Initializing world state with stratigraphy...")

layer_names = [
    "Topsoil",
    "Colluvium",
    "Saprolite",
    "WeatheredBR",
    "Sandstone",
    "Shale",
    "Basement"
]

print(f"  Layers: {len(layer_names)}")

world = WorldState(
    nx=N,
    ny=N,
    pixel_scale_m=pixel_scale_m,
    layer_names=layer_names
)

create_slope_dependent_stratigraphy(
    world,
    surface_elev=surface_elev,
    pixel_scale_m=pixel_scale_m,
    base_regolith_m=2.0,
    base_saprolite_m=5.0,
    bedrock_thickness_m=100.0
)

print("✓ World state initialized")

# =============================================================================
# 5. SET UP FORCING
# =============================================================================
print("\n[5/10] Setting up external forcing...")

# Tectonic uplift
tectonics = TectonicUplift(N, N, pixel_scale_m)
tectonics.set_uniform_uplift(1e-3)  # 1 mm/yr

print(f"  Tectonics: {tectonics}")

# Weather/climate
weather = WeatherGenerator(
    N, N, pixel_scale_m,
    mean_annual_precip_m=1.0,
    wind_direction_deg=270.0,
    orographic_factor=0.5
)

print(f"  Weather: {weather}")

# =============================================================================
# 6. CREATE SIMULATOR
# =============================================================================
print("\n[6/10] Creating simulator...")

simulator = LandscapeEvolutionSimulator(
    world=world,
    tectonics=tectonics,
    weather=weather,
    snapshot_interval=50,
    verbose=True
)

print("✓ Simulator created")

# =============================================================================
# 7. RUN SIMULATION
# =============================================================================
print("\n[7/10] Running simulation...")

total_time = 5000.0  # 5,000 years
dt = 10.0            # 10-year time steps

print(f"  Duration: {total_time:.0f} years")
print(f"  Time step: {dt} years")
print(f"  Total steps: {int(total_time/dt)}")
print("\n" + "="*70)

history = simulator.run(total_time=total_time, dt=dt)

print("="*70)
print(f"✓ SIMULATION COMPLETE")
print(f"  Final time: {world.time:.1f} years")
print(f"  Snapshots: {len(history.times)}")

# =============================================================================
# 8. COMPUTE WATER ROUTING
# =============================================================================
print("\n[8/10] Computing water routing...")

flow_router = FlowRouter(pixel_scale_m)
flow_dir, slope, flow_accum = flow_router.compute_flow(
    world.surface_elev,
    fill_depressions=False
)

print(f"✓ Flow routing complete")
print(f"  Max flow accumulation: {flow_accum.max():.0f} cells")
print(f"  Mean slope: {slope.mean():.4f} m/m")

# =============================================================================
# 9. EROSION ANALYSIS
# =============================================================================
print("\n[9/10] Creating erosion visualizations...")

# Get erosion data
cumulative_erosion = history.get_total_erosion()
erosion_rate = cumulative_erosion / total_time

print(f"  Total erosion: {cumulative_erosion.sum():.1f} m")
print(f"  Mean erosion: {cumulative_erosion.mean():.2f} m")
print(f"  Max erosion: {cumulative_erosion.max():.2f} m")
print(f"  Mean rate: {erosion_rate.mean():.2e} m/yr")

# Plot 1: Comprehensive erosion analysis
print("\n  Creating erosion analysis plot...")
plot_erosion_analysis(
    erosion=cumulative_erosion,
    surface_elev=world.surface_elev,
    pixel_scale_m=pixel_scale_m,
    row_for_profile=N//2,
    save_path='02_erosion_analysis.png'
)
plt.close('all')

# Plot 2: Erosion rate with rivers
print("  Creating erosion rate + rivers plot...")
plot_erosion_rate_map(
    erosion_rate=erosion_rate,
    pixel_scale_m=pixel_scale_m,
    flow_accum=flow_accum,
    save_path='03_erosion_rate_rivers.png'
)
plt.close('all')

# =============================================================================
# 10. ADDITIONAL VISUALIZATIONS
# =============================================================================
print("\n[10/10] Creating additional visualizations...")

# Initial vs final
print("  - Initial vs final topography...")
plot_initial_vs_final(history, pixel_scale_m, save_path='04_initial_vs_final.png')
plt.close('all')

# Erosion and deposition
print("  - Erosion and deposition maps...")
plot_erosion_deposition_maps(history, pixel_scale_m, save_path='05_erosion_deposition.png')
plt.close('all')

# River network
print("  - River network...")
plot_river_network(
    world.surface_elev,
    flow_accum,
    pixel_scale_m,
    threshold_cells=0.01 * N * N,
    save_path='06_river_network.png'
)
plt.close('all')

# Cross-section
print("  - Cross-section...")
plot_cross_section(
    world,
    row=N//2,
    vertical_exaggeration=2.0,
    save_path='07_cross_section.png'
)
plt.close('all')

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================
print("\n" + "="*70)
print("SIMULATION SUMMARY")
print("="*70)

initial_surface = history.surface_snapshots[0]
final_surface = history.surface_snapshots[-1]
net_change = final_surface - initial_surface
total_erosion = history.get_total_erosion()
total_deposition = history.get_total_deposition()

print(f"\nSimulation Time: {total_time:.0f} years")
print(f"Time Steps: {int(total_time/dt)}")
print(f"Grid Size: {N}×{N} ({N*N:,} cells)")
print(f"Pixel Scale: {pixel_scale_m} m")

print("\nELEVATION CHANGES:")
print(f"  Initial mean: {initial_surface.mean():.1f} m")
print(f"  Final mean: {final_surface.mean():.1f} m")
print(f"  Change: {(final_surface.mean() - initial_surface.mean()):.1f} m")

print("\nEROSION:")
print(f"  Total: {total_erosion.sum():.1f} m (cumulative)")
print(f"  Mean: {total_erosion.mean():.2f} m")
print(f"  Max: {total_erosion.max():.2f} m")
print(f"  Mean rate: {(total_erosion.mean()/total_time):.2e} m/yr")

print("\nDEPOSITION:")
print(f"  Total: {total_deposition.sum():.1f} m (cumulative)")
print(f"  Mean: {total_deposition.mean():.2f} m")
print(f"  Max: {total_deposition.max():.2f} m")

print("\nNET CHANGE:")
print(f"  Total lowering: {net_change[net_change < 0].sum():.1f} m")
print(f"  Total raising: {net_change[net_change > 0].sum():.1f} m")
print(f"  Net: {net_change.sum():.1f} m")

print("\nUPLIFT:")
uplift_total = tectonics.uplift_rate.mean() * total_time
print(f"  Rate: {tectonics.uplift_rate.mean():.2e} m/yr")
print(f"  Total: {uplift_total:.2f} m")

# =============================================================================
# DONE
# =============================================================================
print("\n" + "="*70)
print("✓ ALL COMPLETE!")
print("="*70)
print("\nOutput files created:")
print("  01_initial_terrain.png")
print("  02_erosion_analysis.png")
print("  03_erosion_rate_rivers.png")
print("  04_initial_vs_final.png")
print("  05_erosion_deposition.png")
print("  06_river_network.png")
print("  07_cross_section.png")
print("\n" + "="*70)
