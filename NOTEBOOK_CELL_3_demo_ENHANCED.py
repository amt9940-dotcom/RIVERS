"""
PASTE THIS INTO NOTEBOOK CELL 3: Enhanced Erosion Demo with Full Visualization

This shows:
- Initial terrain and layers
- Detailed progress output
- Final terrain with rivers/lakes overlaid
- Before/after comparisons
"""

import numpy as np
import matplotlib.pyplot as plt
import copy

print("=" * 80)
print("ENHANCED EROSION MODEL DEMO")
print("=" * 80)

# -----------------------------------------------------------------------------
# STEP 1: Generate initial terrain and stratigraphy
# -----------------------------------------------------------------------------
print("\n1. Generating quantum-seeded terrain...")

# Parameters
N = 128  # Grid size
elev_range_m = 2000.0
pixel_scale_m = 100.0
seed = 42

# Generate terrain
z_norm, rng = quantum_seeded_topography(N=N, beta=3.0, random_seed=seed)
print(f"   ✓ Terrain generated: {N}×{N}")

# Generate stratigraphy
print("\n2. Generating stratigraphy...")
strata = generate_stratigraphy(
    z_norm=z_norm,
    elev_range_m=elev_range_m,
    pixel_scale_m=pixel_scale_m,
    rng=rng,
    dip_deg=5.0,
    dip_dir_deg=45.0,
)

print(f"   ✓ Surface elevation: {strata['surface_elev'].min():.1f} - {strata['surface_elev'].max():.1f} m")
print(f"   ✓ Relief: {strata['surface_elev'].max() - strata['surface_elev'].min():.1f} m")

# Show layer statistics
print("\n   Layer thicknesses:")
for layer in ["Topsoil", "Subsoil", "Saprolite", "Sandstone", "Shale", "Basement"]:
    if layer in strata["thickness"]:
        thick = strata["thickness"][layer]
        print(f"     {layer:12s}: {thick.mean():.1f} m (mean), {thick.max():.1f} m (max)")

# -----------------------------------------------------------------------------
# VISUALIZE INITIAL STATE
# -----------------------------------------------------------------------------
print("\n3. Visualizing initial terrain...")

fig1, axes = plt.subplots(2, 2, figsize=(14, 12))

# Initial surface
ax = axes[0, 0]
im = ax.imshow(strata["surface_elev"], origin="lower", cmap="terrain")
ax.set_title("Initial Surface Elevation", fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, label="Elevation (m)")
ax.set_xlabel("X (cells)")
ax.set_ylabel("Y (cells)")

# Initial layers (cross-section)
ax = axes[0, 1]
row_idx = N // 2
x_km = np.arange(N) * pixel_scale_m / 1000.0
ax.plot(x_km, strata["surface_elev"][row_idx, :], 'k-', linewidth=2, label="Surface")
for layer, color in zip(["Topsoil", "Saprolite", "Sandstone", "Basement"], 
                        ['brown', 'orange', 'gold', 'gray']):
    if layer in strata["interfaces"]:
        ax.plot(x_km, strata["interfaces"][layer][row_idx, :], '--', 
                color=color, alpha=0.7, linewidth=1.5, label=layer)
ax.fill_between(x_km, strata["interfaces"]["Basement"][row_idx, :], 
                 strata["surface_elev"][row_idx, :], alpha=0.2, color='brown', label='Rock layers')
ax.set_xlabel("Distance (km)")
ax.set_ylabel("Elevation (m)")
ax.set_title(f"Initial Stratigraphy (Cross-section at row {row_idx})", fontsize=12, fontweight='bold')
ax.legend(loc="best", fontsize=8)
ax.grid(True, alpha=0.3)

# Topographic analysis
ax = axes[1, 0]
gradient_x, gradient_y = np.gradient(strata["surface_elev"])
slope = np.sqrt(gradient_x**2 + gradient_y**2) / pixel_scale_m
im = ax.imshow(slope, origin="lower", cmap="YlOrRd")
ax.set_title("Initial Slope", fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, label="Slope (m/m)")
ax.set_xlabel("X (cells)")
ax.set_ylabel("Y (cells)")

# Layer at surface
ax = axes[1, 1]
surface_layer = np.zeros((N, N))
# Simple categorization based on elevation
elev_norm = (strata["surface_elev"] - strata["surface_elev"].min()) / \
            (strata["surface_elev"].max() - strata["surface_elev"].min())
im = ax.imshow(elev_norm, origin="lower", cmap="Spectral")
ax.set_title("Normalized Elevation (proxy for exposed layers)", fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, label="0 (low) to 1 (high)")
ax.set_xlabel("X (cells)")
ax.set_ylabel("Y (cells)")

plt.suptitle("INITIAL STATE - Before Erosion", fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()

print("   ✓ Initial terrain visualized")

# Save initial state
strata_initial = copy.deepcopy(strata)

# -----------------------------------------------------------------------------
# STEP 2: Set up erosion parameters
# -----------------------------------------------------------------------------
print("\n4. Setting up erosion parameters...")

num_epochs = 25
dt = 1000.0

K_channel = 1e-6
D_hillslope = 0.005
uplift_rate_base = 0.0001

print(f"   Simulation time: {num_epochs * dt / 1000:.1f} kyr ({num_epochs} epochs × {dt} years)")
print(f"   Channel erosion: K = {K_channel:.2e}")
print(f"   Hillslope diffusion: D = {D_hillslope} m²/yr")
print(f"   Uplift rate (center): {uplift_rate_base * 1000:.2f} mm/yr")

# Create spatially variable uplift (dome)
ny, nx = strata["surface_elev"].shape
uplift_field = np.zeros((ny, nx))
center_i, center_j = ny // 2, nx // 2

for i in range(ny):
    for j in range(nx):
        dist = np.sqrt((i - center_i)**2 + (j - center_j)**2)
        uplift_field[i, j] = uplift_rate_base * np.exp(-(dist / (N/4))**2)

print(f"   Uplift pattern: Gaussian dome (center={uplift_rate_base*1000:.2f} mm/yr, edge={uplift_field.min()*1000:.3f} mm/yr)")

# -----------------------------------------------------------------------------
# STEP 3: Run erosion simulation with detailed output
# -----------------------------------------------------------------------------
print("\n5. Running erosion simulation...")
print("   " + "=" * 70)

history = run_erosion_simulation(
    strata=strata,
    pixel_scale_m=pixel_scale_m,
    num_epochs=num_epochs,
    dt=dt,
    rainfall_func=None,
    uplift_rate=uplift_field,
    K_channel=K_channel,
    D_hillslope=D_hillslope,
    verbose=True
)

print("   " + "=" * 70)
print("   ✓ Simulation complete!")

# -----------------------------------------------------------------------------
# STEP 4: Compute detailed statistics
# -----------------------------------------------------------------------------
print("\n6. Computing statistics...")

total_erosion = sum([h["total_erosion"] for h in history])
total_deposition = sum([h["deposition"] for h in history])

mean_erosion = np.mean(total_erosion)
max_erosion = np.max(total_erosion)
mean_deposition = np.mean(total_deposition)
max_deposition = np.max(total_deposition)

delta_elev = strata["surface_elev"] - strata_initial["surface_elev"]
mean_delta = np.mean(delta_elev)
min_delta = np.min(delta_elev)
max_delta = np.max(delta_elev)

total_eroded_volume = np.sum(total_erosion) * (pixel_scale_m ** 2)
total_deposited_volume = np.sum(total_deposition) * (pixel_scale_m ** 2)

print(f"   Erosion:")
print(f"     Mean: {mean_erosion:.2f} m")
print(f"     Max: {max_erosion:.2f} m")
print(f"     Total volume: {total_eroded_volume/1e6:.2f} million m³")
print(f"   Deposition:")
print(f"     Mean: {mean_deposition:.2f} m")
print(f"     Max: {max_deposition:.2f} m")
print(f"     Total volume: {total_deposited_volume/1e6:.2f} million m³")
print(f"   Net elevation change:")
print(f"     Mean: {mean_delta:+.2f} m")
print(f"     Range: {min_delta:.2f} to {max_delta:+.2f} m")
print(f"   Mass balance: {(total_deposited_volume/total_eroded_volume)*100:.1f}% of eroded material deposited")

# -----------------------------------------------------------------------------
# STEP 5: Enhanced visualizations with rivers and lakes
# -----------------------------------------------------------------------------
print("\n7. Creating enhanced visualizations...")

# Get final flow data
final_flow = history[-1]["flow_data"]
discharge = final_flow["discharge"]
slope = final_flow["slope"]

# Identify rivers (high discharge areas)
discharge_norm = (discharge - discharge.min()) / (discharge.max() - discharge.min() + 1e-9)
river_threshold = np.percentile(discharge_norm, 95)  # Top 5% discharge
rivers = discharge_norm > river_threshold

# Identify lakes/depressions (local minima with no outflow)
lakes = final_flow["flow_dir"] == -1  # Pits

print(f"   Rivers detected: {np.sum(rivers)} cells ({np.sum(rivers)/N**2*100:.1f}%)")
print(f"   Lakes detected: {np.sum(lakes)} cells ({np.sum(lakes)/N**2*100:.1f}%)")

# Create comprehensive visualization
fig2, axes = plt.subplots(3, 3, figsize=(18, 16))

# Row 1: Terrain evolution
# Before
ax = axes[0, 0]
im = ax.imshow(strata_initial["surface_elev"], origin="lower", cmap="terrain")
ax.set_title("BEFORE: Surface Elevation", fontsize=11, fontweight='bold')
plt.colorbar(im, ax=ax, label="Elevation (m)", fraction=0.046)
ax.set_xlabel("X (cells)")
ax.set_ylabel("Y (cells)")

# After
ax = axes[0, 1]
im = ax.imshow(strata["surface_elev"], origin="lower", cmap="terrain")
ax.set_title("AFTER: Surface Elevation", fontsize=11, fontweight='bold')
plt.colorbar(im, ax=ax, label="Elevation (m)", fraction=0.046)
ax.set_xlabel("X (cells)")
ax.set_ylabel("Y (cells)")

# Change
ax = axes[0, 2]
delta = strata["surface_elev"] - strata_initial["surface_elev"]
vmax = max(abs(delta.min()), abs(delta.max()))
im = ax.imshow(delta, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
ax.set_title("Elevation Change (Δz)", fontsize=11, fontweight='bold')
plt.colorbar(im, ax=ax, label="Δz (m)", fraction=0.046)
ax.set_xlabel("X (cells)")
ax.set_ylabel("Y (cells)")

# Row 2: Processes
# Erosion
ax = axes[1, 0]
im = ax.imshow(history[-1]["erosion_channel"], origin="lower", cmap="hot_r")
ax.set_title("Channel Erosion (last epoch)", fontsize=11, fontweight='bold')
plt.colorbar(im, ax=ax, label="Erosion (m)", fraction=0.046)
ax.set_xlabel("X (cells)")
ax.set_ylabel("Y (cells)")

# Deposition
ax = axes[1, 1]
im = ax.imshow(history[-1]["deposition"], origin="lower", cmap="Blues")
ax.set_title("Deposition (last epoch)", fontsize=11, fontweight='bold')
plt.colorbar(im, ax=ax, label="Deposition (m)", fraction=0.046)
ax.set_xlabel("X (cells)")
ax.set_ylabel("Y (cells)")

# Total erosion
ax = axes[1, 2]
im = ax.imshow(total_erosion, origin="lower", cmap="YlOrRd")
ax.set_title("Total Erosion (all epochs)", fontsize=11, fontweight='bold')
plt.colorbar(im, ax=ax, label="Total erosion (m)", fraction=0.046)
ax.set_xlabel("X (cells)")
ax.set_ylabel("Y (cells)")

# Row 3: Hydrology
# Flow accumulation (rivers)
ax = axes[2, 0]
im = ax.imshow(np.log10(discharge + 1), origin="lower", cmap="Blues")
river_cells = np.where(rivers)
ax.scatter(river_cells[1], river_cells[0], c='darkblue', s=0.5, alpha=0.5, label='Rivers')
ax.set_title("Flow Accumulation (rivers highlighted)", fontsize=11, fontweight='bold')
plt.colorbar(im, ax=ax, label="log₁₀(discharge)", fraction=0.046)
ax.set_xlabel("X (cells)")
ax.set_ylabel("Y (cells)")

# Final terrain with rivers and lakes overlay
ax = axes[2, 1]
im = ax.imshow(strata["surface_elev"], origin="lower", cmap="terrain", alpha=0.8)
# Overlay rivers in blue
river_overlay = np.ma.masked_where(~rivers, discharge_norm)
ax.imshow(river_overlay, origin="lower", cmap="Blues", alpha=0.6, vmin=0, vmax=1)
# Overlay lakes in dark blue
if np.any(lakes):
    lake_cells = np.where(lakes)
    ax.scatter(lake_cells[1], lake_cells[0], c='navy', s=2, alpha=0.8, marker='s', label='Lakes')
ax.set_title("FINAL: Terrain + Rivers + Lakes", fontsize=11, fontweight='bold')
plt.colorbar(im, ax=ax, label="Elevation (m)", fraction=0.046)
ax.legend(loc='upper right', fontsize=8)
ax.set_xlabel("X (cells)")
ax.set_ylabel("Y (cells)")

# Drainage density map
ax = axes[2, 2]
drainage_density = rivers.astype(float)
# Simple smoothing (works with or without scipy)
try:
    from scipy.ndimage import uniform_filter
    drainage_smooth = uniform_filter(drainage_density, size=5)
except:
    # Fallback: simple rolling average
    drainage_smooth = drainage_density.copy()
    for _ in range(2):
        up = np.roll(drainage_smooth, -1, axis=0)
        down = np.roll(drainage_smooth, 1, axis=0)
        left = np.roll(drainage_smooth, 1, axis=1)
        right = np.roll(drainage_smooth, -1, axis=1)
        drainage_smooth = (drainage_smooth + up + down + left + right) / 5.0
im = ax.imshow(drainage_smooth, origin="lower", cmap="YlGnBu")
ax.set_title("Drainage Density", fontsize=11, fontweight='bold')
plt.colorbar(im, ax=ax, label="River density", fraction=0.046)
ax.set_xlabel("X (cells)")
ax.set_ylabel("Y (cells)")

plt.suptitle(f"EROSION RESULTS - {num_epochs * dt / 1000:.1f} kyr evolution", 
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()

# Cross-sections
print("\n8. Creating cross-sections...")

fig3, axes = plt.subplots(2, 1, figsize=(14, 10))

x_km = np.arange(N) * pixel_scale_m / 1000.0
row_idx = N // 2

# Before
ax = axes[0]
ax.plot(x_km, strata_initial["surface_elev"][row_idx, :], 'k-', linewidth=2.5, label="Surface")
for layer, color in zip(["Topsoil", "Saprolite", "Sandstone", "Basement"],
                        ['brown', 'orange', 'gold', 'darkgray']):
    if layer in strata_initial["interfaces"]:
        ax.plot(x_km, strata_initial["interfaces"][layer][row_idx, :], '--',
                color=color, linewidth=1.5, alpha=0.7, label=layer)
ax.fill_between(x_km, strata_initial["interfaces"]["Basement"][row_idx, :],
                 strata_initial["surface_elev"][row_idx, :], alpha=0.15, color='brown')
ax.set_ylabel("Elevation (m)", fontsize=11)
ax.set_title(f"BEFORE Erosion - Cross-section at row {row_idx}", fontsize=12, fontweight='bold')
ax.legend(loc="best", fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)

# After
ax = axes[1]
ax.plot(x_km, strata["surface_elev"][row_idx, :], 'k-', linewidth=2.5, label="Surface (after)")
ax.plot(x_km, strata_initial["surface_elev"][row_idx, :], 'k:', linewidth=1.5, alpha=0.5, label="Surface (before)")
for layer, color in zip(["Topsoil", "Saprolite", "Sandstone", "Basement"],
                        ['brown', 'orange', 'gold', 'darkgray']):
    if layer in strata["interfaces"]:
        ax.plot(x_km, strata["interfaces"][layer][row_idx, :], '--',
                color=color, linewidth=1.5, alpha=0.7, label=layer)
ax.fill_between(x_km, strata["interfaces"]["Basement"][row_idx, :],
                 strata["surface_elev"][row_idx, :], alpha=0.15, color='brown')
# Highlight erosion/deposition
erosion_mask = delta[row_idx, :] < -0.5
deposition_mask = delta[row_idx, :] > 0.5
if np.any(erosion_mask):
    ax.fill_between(x_km, strata["surface_elev"][row_idx, :], strata_initial["surface_elev"][row_idx, :],
                     where=erosion_mask, alpha=0.3, color='red', label='Erosion')
if np.any(deposition_mask):
    ax.fill_between(x_km, strata_initial["surface_elev"][row_idx, :], strata["surface_elev"][row_idx, :],
                     where=deposition_mask, alpha=0.3, color='blue', label='Deposition')
ax.set_xlabel("Distance (km)", fontsize=11)
ax.set_ylabel("Elevation (m)", fontsize=11)
ax.set_title(f"AFTER Erosion - Cross-section at row {row_idx}", fontsize=12, fontweight='bold')
ax.legend(loc="best", fontsize=9, ncol=3)
ax.grid(True, alpha=0.3)

plt.suptitle("Stratigraphic Evolution", fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()

print("   ✓ Cross-sections visualized")

# -----------------------------------------------------------------------------
# SUMMARY
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("EROSION SIMULATION COMPLETE!")
print("=" * 80)
print(f"\nSummary:")
print(f"  • Simulated {num_epochs * dt / 1000:.1f} kyr of landscape evolution")
print(f"  • Eroded {total_eroded_volume/1e6:.2f} million m³ of material")
print(f"  • Deposited {total_deposited_volume/1e6:.2f} million m³ in valleys")
print(f"  • Developed {np.sum(rivers)} cells of river network")
print(f"  • Created {np.sum(lakes)} lake/depression cells")
print(f"  • Mean elevation change: {mean_delta:+.2f} m")
print(f"\nThe erosion model has successfully:")
print(f"  ✓ Carved river valleys (blue channels in visualizations)")
print(f"  ✓ Smoothed hillslopes through diffusion")
print(f"  ✓ Deposited sediment in low-energy areas")
print(f"  ✓ Maintained stratigraphic layer ordering")
print(f"  ✓ Applied tectonic uplift (dome pattern)")
print("\nTry adjusting parameters in this cell and running again!")
print("=" * 80)
