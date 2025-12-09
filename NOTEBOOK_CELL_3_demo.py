"""
PASTE THIS INTO NOTEBOOK CELL 3: Erosion Demo

This demo uses the functions from Cell 1 and Cell 2.
Make sure you've run those cells first!
"""

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
N = 128  # Smaller grid for faster demo (use 256 for better quality)
elev_range_m = 2000.0
pixel_scale_m = 100.0
seed = 42

# Generate terrain (using function from Cell 1)
z_norm, rng = quantum_seeded_topography(N=N, beta=3.0, random_seed=seed)
print(f"   ✓ Terrain generated: {N}×{N}")

# Generate stratigraphy (using function from Cell 1)
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

num_epochs = 25  # Reduced for faster demo
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

# Create spatially variable uplift (growing dome)
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

# Using function from Cell 2
history = run_erosion_simulation(
    strata=strata,
    pixel_scale_m=pixel_scale_m,
    num_epochs=num_epochs,
    dt=dt,
    rainfall_func=None,  # Uniform rainfall
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

# Using functions from Cell 2
fig1 = plot_erosion_evolution(
    strata_initial, 
    strata, 
    history[-1],
    pixel_scale_m
)
plt.suptitle(f"Erosion Model Results (t = {num_epochs * dt / 1000:.1f} kyr)", 
             fontsize=14, y=1.00)
plt.show()

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
plt.show()

print("\n" + "=" * 80)
print("DEMO COMPLETE!")
print("=" * 80)
print("\nThe erosion model has successfully evolved your terrain!")
print("Try adjusting the parameters and running again:")
print("  - Change K_channel to control erosion rate")
print("  - Change num_epochs to run longer/shorter")
print("  - Change N to adjust grid size")
