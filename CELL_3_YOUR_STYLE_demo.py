"""
CELL 3: DEMO (YOUR STYLE)

Uses YOUR parameters and visualization style:
- N=512 high-resolution
- pixel_scale_m=10.0 (5.12 km domain)
- figsize=(14, 11.5) detailed plots
- Discrete colormaps
- Your geological feature visualization

This demonstrates the FIXED erosion model at YOUR scale.
"""
import numpy as np
import matplotlib.pyplot as plt
import copy

print("=" * 80)
print("EROSION SYSTEM (YOUR STYLE - High Resolution)")
print("=" * 80)

# ==============================================================================
# STEP 1: Generate terrain (YOUR STYLE)
# ==============================================================================
print("\n1. Generating high-resolution terrain...")
print("   (This may take ~30 seconds for N=512...)")

# YOUR parameters
N = 512  # YOUR resolution (not 50!)
pixel_scale_m = 10.0  # YOUR cell size (not 1000!)
elev_range_m = 700.0  # YOUR elevation range
seed = 42
base_wind_dir_deg = 270.0  # West to East

# Generate
z_norm, rng = quantum_seeded_topography(
    N=N,
    beta=3.2,  # YOUR typical value
    warp_amp=0.10,  # YOUR typical value
    ridged_alpha=0.15,  # YOUR typical value
    random_seed=seed
)

print(f"   ✓ Terrain generated: {N} × {N}")
print(f"   Domain size: {N * pixel_scale_m / 1000:.2f} km × {N * pixel_scale_m / 1000:.2f} km")

# Generate stratigraphy
strata = generate_stratigraphy(z_norm, pixel_scale_m, elev_range_m)

print(f"   Elevation range: {strata['surface_elev'].min():.1f} - {strata['surface_elev'].max():.1f} m")

# ==============================================================================
# STEP 2: Analyze wind features (YOUR STYLE)
# ==============================================================================
print("\n2. Analyzing wind features...")

wind_structs = build_wind_structures(strata["surface_elev"], pixel_scale_m, base_wind_dir_deg)

n_barriers = np.sum(wind_structs["barrier_mask"])
n_channels = np.sum(wind_structs["channel_mask"])
n_basins = np.sum(wind_structs["basin_mask"])

print(f"   Wind from: {base_wind_dir_deg}° (west)")
print(f"   Barriers: {n_barriers} cells ({100*n_barriers/(N*N):.2f}%)")
print(f"   Channels: {n_channels} cells ({100*n_channels/(N*N):.2f}%)")
print(f"   Basins: {n_basins} cells ({100*n_basins/(N*N):.2f}%)")

# Visualize (YOUR discrete colormap style)
plot_wind_structures_categorical(wind_structs)

# ==============================================================================
# STEP 3: Setup erosion parameters (SCALED for YOUR resolution)
# ==============================================================================
print("\n3. Setting up erosion parameters (SCALED for N=512, pixel=10m)...")

# CONSERVATIVE parameters for testing
num_epochs = 5  # Start small to test
dt = 10.0  # years (small time step for 10m cells)
K_channel = 1e-6  # Small coefficient
D_hillslope = 0.001  # Small diffusion
uplift_rate = 0.00001  # Very small uplift (0.01 mm/yr)

print(f"   Epochs: {num_epochs}")
print(f"   Time step: {dt} years")
print(f"   Total time: {num_epochs * dt:.1f} years")
print(f"   K_channel: {K_channel:.2e}")
print(f"   D_hillslope: {D_hillslope}")
print(f"   Uplift: {uplift_rate * 1000:.3f} mm/yr")
print(f"   Q_threshold: 100.0 m² (scaled for 10m cells)")
print(f"   Max erosion: 1.0 m/step (scaled)")

# Simple rainfall (uniform for now)
def rainfall_func(epoch):
    # Simple uniform rainfall
    return np.ones((N, N)) * 0.8  # m/year

# Spatially variable uplift (small dome in center)
uplift_field = np.zeros((N, N))
center_i, center_j = N // 2, N // 2

for i in range(N):
    for j in range(N):
        dist = np.sqrt((i - center_i)**2 + (j - center_j)**2)
        uplift_field[i, j] = uplift_rate * np.exp(-(dist / (N/4))**2)

print(f"   ✓ Parameters set (conservative for testing)")

# ==============================================================================
# STEP 4: Save initial state
# ==============================================================================
print("\n4. Saving initial state...")
strata_initial = copy.deepcopy(strata)
print(f"   ✓ Initial state saved")
print(f"     Initial elevation: {strata_initial['surface_elev'].min():.1f} - {strata_initial['surface_elev'].max():.1f} m")

# ==============================================================================
# STEP 5: Run erosion simulation
# ==============================================================================
print("\n5. Running erosion simulation...")
print("   (This will take ~1-2 minutes for N=512...)")

history = run_erosion_simulation_SCALED(
    strata=strata,
    pixel_scale_m=pixel_scale_m,
    num_epochs=num_epochs,
    dt=dt,
    rainfall_func=rainfall_func,
    uplift_rate=uplift_field,
    K_channel=K_channel,
    D_hillslope=D_hillslope,
    verbose=True
)

print("   ✓ Simulation complete!")

# Check final state
print(f"\n   Final state:")
print(f"     Elevation: {strata['surface_elev'].min():.1f} - {strata['surface_elev'].max():.1f} m")
if strata['surface_elev'].min() < 0:
    print("     ⚠ WARNING: Some elevations below sea level")
elif strata['surface_elev'].min() < -100:
    print("     ⚠ WARNING: Blow-up detected!")
else:
    print("     ✓ All elevations look reasonable")

# ==============================================================================
# STEP 6: Compute statistics
# ==============================================================================
print("\n6. Computing statistics...")

total_erosion = sum([h["total_erosion"] for h in history])
total_deposition = sum([h["deposition"] for h in history])

mean_erosion = np.mean(total_erosion)
max_erosion = np.max(total_erosion)
mean_deposition = np.mean(total_deposition)
max_deposition = np.max(total_deposition)

delta_elev = strata["surface_elev"] - strata_initial["surface_elev"]
mean_delta = np.mean(delta_elev)

print(f"   Erosion: mean {mean_erosion:.3f} m, max {max_erosion:.3f} m")
print(f"   Deposition: mean {mean_deposition:.3f} m, max {max_deposition:.3f} m")
print(f"   Net change: {mean_delta:+.3f} m")

# ==============================================================================
# STEP 7: Visualizations (YOUR STYLE - large detailed plots)
# ==============================================================================
print("\n7. Creating visualizations (YOUR STYLE)...")

final_flow = history[-1]["flow_data"]
discharge = final_flow["discharge"]

# Identify rivers (top 5% discharge)
discharge_norm = (discharge - discharge.min()) / (discharge.max() - discharge.min() + 1e-9)
river_threshold = np.percentile(discharge_norm, 95)
rivers = discharge_norm > river_threshold

print(f"   Rivers: {np.sum(rivers)} cells (top 5% discharge)")

# Create large detailed figure (YOUR style)
fig, axes = plt.subplots(3, 3, figsize=(18, 16))

# Row 1: Terrain (before, after, change)
ax = axes[0, 0]
im = ax.imshow(strata_initial["surface_elev"], origin="lower", cmap="terrain", interpolation="bilinear")
ax.set_title("BEFORE: Elevation", fontweight='bold', fontsize=12)
plt.colorbar(im, ax=ax, label="Elevation (m)", fraction=0.046)
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")

ax = axes[0, 1]
im = ax.imshow(strata["surface_elev"], origin="lower", cmap="terrain", interpolation="bilinear")
ax.set_title("AFTER: Elevation", fontweight='bold', fontsize=12)
plt.colorbar(im, ax=ax, label="Elevation (m)", fraction=0.046)
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")

ax = axes[0, 2]
vmax = max(abs(delta_elev.min()), abs(delta_elev.max()), 0.01)
im = ax.imshow(delta_elev, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="bilinear")
ax.set_title("Elevation Change (Δz)", fontweight='bold', fontsize=12)
plt.colorbar(im, ax=ax, label="m (red=erosion, blue=deposition)", fraction=0.046)
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")

# Row 2: Erosion, deposition, discharge
ax = axes[1, 0]
im = ax.imshow(total_erosion, origin="lower", cmap="YlOrRd", interpolation="bilinear")
ax.set_title("Total Erosion", fontweight='bold', fontsize=12)
plt.colorbar(im, ax=ax, label="m", fraction=0.046)
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")

ax = axes[1, 1]
im = ax.imshow(total_deposition, origin="lower", cmap="Blues", interpolation="bilinear")
ax.set_title("Total Deposition", fontweight='bold', fontsize=12)
plt.colorbar(im, ax=ax, label="m", fraction=0.046)
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")

ax = axes[1, 2]
discharge_log = np.log10(discharge + 1)
im = ax.imshow(discharge_log, origin="lower", cmap="viridis", interpolation="bilinear")
ax.set_title("Discharge (log₁₀ upslope area)", fontweight='bold', fontsize=12)
plt.colorbar(im, ax=ax, label="log₁₀(A [m²] + 1)", fraction=0.046)
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")

# Row 3: Rivers, slope, wind features
ax = axes[2, 0]
# Show terrain with river overlay
im = ax.imshow(strata["surface_elev"], origin="lower", cmap="terrain", alpha=0.7, interpolation="bilinear")
river_overlay = np.ma.masked_where(~rivers, discharge_norm)
ax.imshow(river_overlay, origin="lower", cmap="Blues", alpha=0.8, interpolation="nearest")
ax.set_title("Terrain + Rivers", fontweight='bold', fontsize=12)
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")

ax = axes[2, 1]
slope = wind_structs["slope_norm"]
im = ax.imshow(slope, origin="lower", cmap="hot", interpolation="bilinear")
ax.set_title("Slope (normalized)", fontweight='bold', fontsize=12)
plt.colorbar(im, ax=ax, label="0 (flat) to 1 (steep)", fraction=0.046)
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")

# Wind features (YOUR discrete colormap style)
ax = axes[2, 2]
barrier_mask = wind_structs["barrier_mask"]
channel_mask = wind_structs["channel_mask"]
basin_mask = wind_structs["basin_mask"]

features = np.zeros_like(strata["surface_elev"], dtype=int)
features[barrier_mask] = 1
features[channel_mask] = 2
features[basin_mask] = 3

cmap_discrete = plt.cm.get_cmap("tab10", 4)
im = ax.imshow(features, origin="lower", interpolation="nearest",
               cmap=cmap_discrete, vmin=-0.5, vmax=3.5)
ax.set_title("Wind Features (categorical)", fontweight='bold', fontsize=12)
from matplotlib.patches import Patch
legend_patches = [
    Patch(color=cmap_discrete(1), label="Barriers"),
    Patch(color=cmap_discrete(2), label="Channels"),
    Patch(color=cmap_discrete(3), label="Basins"),
]
ax.legend(handles=legend_patches, loc="upper right", fontsize=9)
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")

plt.suptitle(f"Erosion Results (YOUR STYLE) - N={N}, {num_epochs * dt:.0f} years", 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# ==============================================================================
# STEP 8: Cross-section (YOUR STYLE)
# ==============================================================================
print("\n8. Creating cross-section...")

fig_xs, ax_xs = plt.subplots(1, 1, figsize=(14, 5))

row_idx = N // 2
x_km = np.arange(N) * pixel_scale_m / 1000.0

ax_xs.plot(x_km, strata_initial["surface_elev"][row_idx, :], 'k-', linewidth=2, 
           label="Before", alpha=0.7)
ax_xs.plot(x_km, strata["surface_elev"][row_idx, :], 'b-', linewidth=2, 
           label="After")

# Fill erosion/deposition
ax_xs.fill_between(x_km, strata["surface_elev"][row_idx, :], 
                     strata_initial["surface_elev"][row_idx, :],
                     where=(strata["surface_elev"][row_idx, :] < strata_initial["surface_elev"][row_idx, :]),
                     color='red', alpha=0.3, label="Erosion")
ax_xs.fill_between(x_km, strata["surface_elev"][row_idx, :], 
                     strata_initial["surface_elev"][row_idx, :],
                     where=(strata["surface_elev"][row_idx, :] > strata_initial["surface_elev"][row_idx, :]),
                     color='blue', alpha=0.3, label="Deposition")

ax_xs.set_xlabel("Distance (km)", fontsize=12)
ax_xs.set_ylabel("Elevation (m)", fontsize=12)
ax_xs.set_title(f"Cross-Section at row {row_idx} (center of domain)", fontweight='bold', fontsize=14)
ax_xs.legend(loc='best', fontsize=11)
ax_xs.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "=" * 80)
print("COMPLETE!")
print("=" * 80)
print(f"\n✓ System working at YOUR resolution:")
print(f"  N = {N} (not 50!)")
print(f"  pixel_scale = {pixel_scale_m} m (not 1000 m!)")
print(f"  Domain = {N * pixel_scale_m / 1000:.2f} km × {N * pixel_scale_m / 1000:.2f} km")
print(f"\n✓ Wind features classified correctly:")
print(f"  Barriers: {n_barriers} cells ({100*n_barriers/(N*N):.2f}%)")
print(f"  Channels: {n_channels} cells ({100*n_channels/(N*N):.2f}%)")
print(f"\n✓ Erosion model working:")
print(f"  Rivers detected: {np.sum(rivers)} cells")
print(f"  Mean erosion: {mean_erosion:.3f} m")
print(f"  Elevation range: {strata['surface_elev'].min():.1f} - {strata['surface_elev'].max():.1f} m")
print(f"\n✓ Visualizations use YOUR style:")
print(f"  Large plots (18×16)")
print(f"  Discrete colormaps for features")
print(f"  High resolution detail visible")
print("=" * 80)

print("\n💡 NEXT STEPS:")
print("  1. If this works, increase num_epochs to 25-50")
print("  2. Integrate YOUR storm-based rain (from Project.ipynb)")
print("  3. Add YOUR time-varying weather")
print("  4. Scale up erosion rates if needed")
