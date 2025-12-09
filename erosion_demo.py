#!/usr/bin/env python3
"""
Erosion Model Demo - Standalone Example

This script demonstrates how to use the erosion model with your terrain generator.
Run this file directly: python erosion_demo.py
"""

# Import the terrain generator (your existing code)
from terrain_generator import quantum_seeded_topography, generate_stratigraphy

# Import the erosion model (new code)
from erosion_model import (
    run_erosion_simulation,
    plot_erosion_evolution,
    plot_cross_section_evolution
)

import numpy as np
import matplotlib.pyplot as plt
import copy

print("=" * 80)
print("EROSION MODEL DEMO")
print("=" * 80)

# -----------------------------------------------------------------------------
# STEP 1: Generate initial terrain and stratigraphy
# -----------------------------------------------------------------------------
print("\n1. Generating quantum-seeded terrain...")

# Parameters
N = 256  # Grid size (use 128 for faster, 512 for higher quality)
elev_range_m = 2000.0
pixel_scale_m = 100.0
seed = 42

# Generate terrain
z_norm, rng = quantum_seeded_topography(N=N, beta=3.0, random_seed=seed)
print(f"   ✓ Terrain generated: {N}×{N}")

# Generate stratigraphy
print("2. Generating stratigraphy...")
strata = generate_stratigraphy(
    z_norm=z_norm,
    elev_range_m=elev_range_m,
    pixel_scale_m=pixel_scale_m,
    rng=rng,
    dip_deg=5.0,
    dip_dir_deg=45.0,
)

print(f"   ✓ Surface elevation: {strata['surface_elev'].min():.1f} - {strata['surface_elev'].max():.1f} m")

# Save initial state for comparison
strata_initial = copy.deepcopy(strata)

# -----------------------------------------------------------------------------
# STEP 2: Set up erosion parameters
# -----------------------------------------------------------------------------
print("\n3. Setting up erosion parameters...")

num_epochs = 50
dt = 1000.0  # years per epoch

# Erosion parameters (tuned for stability)
K_channel = 1e-6      # Channel erosion coefficient
D_hillslope = 0.005   # Hillslope diffusivity (m²/year)
uplift_rate = 0.0001  # Tectonic uplift (m/year = 0.1 mm/year)

print(f"   Epochs: {num_epochs}")
print(f"   Time step: {dt} years")
print(f"   Total time: {num_epochs * dt / 1000:.1f} kyr")
print(f"   K_channel: {K_channel:.2e}")
print(f"   Uplift: {uplift_rate * 1000:.2f} mm/year")

# Optional: Create spatially variable uplift (growing dome)
ny, nx = strata["surface_elev"].shape
uplift_field = np.zeros((ny, nx))
center_i, center_j = ny // 2, nx // 2

for i in range(ny):
    for j in range(nx):
        dist = np.sqrt((i - center_i)**2 + (j - center_j)**2)
        # Gaussian dome of uplift
        uplift_field[i, j] = uplift_rate * np.exp(-(dist / (N/4))**2)

# -----------------------------------------------------------------------------
# STEP 3: Run erosion simulation
# -----------------------------------------------------------------------------
print("\n4. Running erosion simulation...")
print("   (This may take a few minutes...)")

history = run_erosion_simulation(
    strata=strata,
    pixel_scale_m=pixel_scale_m,
    num_epochs=num_epochs,
    dt=dt,
    rainfall_func=None,  # Use uniform rainfall (or create custom function)
    uplift_rate=uplift_field,  # Spatially variable uplift
    K_channel=K_channel,
    D_hillslope=D_hillslope,
    verbose=True
)

print(f"✓ Simulation complete!")

# -----------------------------------------------------------------------------
# STEP 4: Compute statistics
# -----------------------------------------------------------------------------
print("\n5. Computing statistics...")

total_erosion = sum([h["total_erosion"] for h in history])
total_deposition = sum([h["deposition"] for h in history])

mean_erosion = np.mean(total_erosion)
max_erosion = np.max(total_erosion)
mean_deposition = np.mean(total_deposition)
max_deposition = np.max(total_deposition)

delta_elev = strata["surface_elev"] - strata_initial["surface_elev"]
mean_delta = np.mean(delta_elev)

print(f"   Mean erosion: {mean_erosion:.2f} m")
print(f"   Max erosion: {max_erosion:.2f} m")
print(f"   Mean deposition: {mean_deposition:.2f} m")
print(f"   Max deposition: {max_deposition:.2f} m")
print(f"   Mean elevation change: {mean_delta:.2f} m")

# -----------------------------------------------------------------------------
# STEP 5: Visualize results
# -----------------------------------------------------------------------------
print("\n6. Creating visualizations...")

# Before/after comparison
fig1 = plot_erosion_evolution(
    strata_initial, 
    strata, 
    history[-1],
    pixel_scale_m
)
plt.suptitle(f"Erosion Model Results (t = {num_epochs * dt / 1000:.1f} kyr)", 
             fontsize=14, y=1.00)
plt.savefig("erosion_results.png", dpi=150, bbox_inches='tight')
print("   ✓ Saved: erosion_results.png")

# Cross-section
row_idx = N // 2
fig2 = plot_cross_section_evolution(
    strata_initial,
    strata,
    row_idx,
    pixel_scale_m
)
plt.suptitle(f"Cross-Section Evolution (row {row_idx})", 
             fontsize=14, y=0.995)
plt.savefig("erosion_cross_section.png", dpi=150, bbox_inches='tight')
print("   ✓ Saved: erosion_cross_section.png")

plt.show()

print("\n" + "=" * 80)
print("DEMO COMPLETE!")
print("=" * 80)
print("\nCheck the output:")
print("  - erosion_results.png")
print("  - erosion_cross_section.png")
print("\nTo customize, edit the parameters in this script and run again!")
