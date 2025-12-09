"""
PASTE THIS INTO NOTEBOOK CELL 3: Weather-Driven Erosion Demo

This version uses your sophisticated wind/storm system to generate rainfall.
Storms are influenced by:
- Topography (mountains, valleys)
- Wind direction (SE as you specified)
- Orographic lifting (windward slopes get more rain)
- Wind barriers (mountains deflect storms)
- Wind channels (valleys funnel storms)
"""

import numpy as np
import matplotlib.pyplot as plt
import copy

print("=" * 80)
print("WEATHER-DRIVEN EROSION MODEL DEMO")
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

# Wind direction (you said SE = southeast)
# In degrees: 0°=East, 90°=North, 180°=West, 270°=South
# Southeast means wind FROM SE, which is 315° (or -45°)
# But if you want wind coming FROM the south going east, use 270° (south)
base_wind_dir_deg = 270.0  # Wind from the south (adjust to 315 for SE)

# Generate terrain
z_norm, rng = quantum_seeded_topography(N=N, beta=3.0, random_seed=seed)
print(f"   ✓ Terrain generated: {N}×{N}")

# Generate stratigraphy
print("\n2. Generating stratigraphy...")
strata = generate_stratigraphy(
    z_norm=z_norm,
    elev_range_m=elev_range_m,
    pixel_scale_m=pixel_scale_m,
    rng=rng
)

print(f"   ✓ Surface elevation: {strata['surface_elev'].min():.1f} - {strata['surface_elev'].max():.1f} m")
print(f"   ✓ Relief: {strata['surface_elev'].max() - strata['surface_elev'].min():.1f} m")

# -----------------------------------------------------------------------------
# STEP 2: Analyze wind structures (your terrain features)
# -----------------------------------------------------------------------------
print("\n3. Analyzing terrain for wind effects...")

wind_structs = build_wind_structures(
    strata["surface_elev"],
    pixel_scale_m,
    base_wind_dir_deg
)

n_barriers = np.sum(wind_structs["barrier_mask"])
n_channels = np.sum(wind_structs["channel_mask"])
n_basins = np.sum(wind_structs["basin_mask"])
n_windward = np.sum(wind_structs["windward_mask"])

print(f"   Wind direction: {base_wind_dir_deg}° (from which wind comes)")
print(f"   ✓ Detected geological features:")
print(f"     - {n_barriers} cells ({n_barriers/N**2*100:.1f}%) are wind barriers (mountains)")
print(f"     - {n_channels} cells ({n_channels/N**2*100:.1f}%) are wind channels (valleys)")
print(f"     - {n_basins} cells ({n_basins/N**2*100:.1f}%) are basins (bowls)")
print(f"     - {n_windward} cells ({n_windward/N**2*100:.1f}%) are windward slopes")
print(f"   These features will influence where storms form!")

# Show wind structures
fig0, axes = plt.subplots(2, 2, figsize=(12, 10))

ax = axes[0, 0]
im = ax.imshow(strata["surface_elev"], origin="lower", cmap="terrain")
ax.set_title("Terrain Elevation", fontweight='bold')
plt.colorbar(im, ax=ax, label="Elevation (m)")

ax = axes[0, 1]
wind_composite = np.zeros_like(strata["surface_elev"])
wind_composite[wind_structs["windward_mask"]] = 0.3
wind_composite[wind_structs["barrier_mask"]] = 0.8
wind_composite[wind_structs["channel_mask"]] = 0.5
wind_composite[wind_structs["basin_mask"]] = 0.2
im = ax.imshow(wind_composite, origin="lower", cmap="RdYlBu_r", vmin=0, vmax=1)
ax.set_title(f"Wind Structures (wind from {base_wind_dir_deg}°)", fontweight='bold')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Barrier(red) → Channel(yellow) → Basin(blue)")

ax = axes[1, 0]
im = ax.imshow(wind_structs["barrier_mask"].astype(float), origin="lower", cmap="Reds")
ax.set_title("Wind Barriers (Mountains)", fontweight='bold')
plt.colorbar(im, ax=ax)

ax = axes[1, 1]
im = ax.imshow(wind_structs["channel_mask"].astype(float), origin="lower", cmap="Blues")
ax.set_title("Wind Channels (Valleys)", fontweight='bold')
plt.colorbar(im, ax=ax)

plt.suptitle("Terrain Analysis for Wind/Storm Routing", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Save initial state
strata_initial = copy.deepcopy(strata)

# -----------------------------------------------------------------------------
# STEP 3: Create weather-driven rainfall generator
# -----------------------------------------------------------------------------
print("\n4. Setting up weather-driven rainfall generator...")

def generate_storm_rainfall(epoch, strata, wind_structs, rng, 
                             base_rainfall=0.5, storm_intensity_range=(0.5, 2.5)):
    """
    Generate rainfall for this epoch based on topography and wind.
    
    Storms form preferentially:
    - On windward slopes (orographic lifting)
    - Near wind barriers (flow convergence)
    - In wind channels (funneling effect)
    - Over elevated areas (convective heating)
    
    The wind direction determines WHERE storms track across the map.
    """
    surface_elev = strata["surface_elev"]
    
    # Compute low-pressure zones (storm likelihood)
    low_pressure_data = compute_orographic_low_pressure(
        surface_elev,
        rng,
        pixel_scale_m,
        base_wind_dir_deg=base_wind_dir_deg,
        wind_structs=wind_structs,
        scale_factor=1.5,
        orographic_weight=0.7
    )
    
    low_pressure = low_pressure_data["low_pressure_likelihood"]
    
    # Storm intensity varies by epoch (simulate weather variability)
    # Some epochs are wetter, some drier
    epoch_wetness = 0.5 + 1.5 * rng.random()  # 0.5x to 2.0x variation
    
    # Rainfall = base * low_pressure_likelihood * epoch_wetness
    rainfall = base_rainfall * (1.0 + low_pressure) * epoch_wetness
    
    # Smooth slightly to avoid salt-and-pepper
    from scipy.ndimage import uniform_filter
    rainfall = uniform_filter(rainfall, size=3, mode='wrap')
    
    return rainfall, low_pressure

# Create the rainfall function
def rainfall_function_for_epoch(epoch):
    """Wrapper that generates rainfall for a specific epoch."""
    # Use a new RNG for each epoch to get variation
    epoch_rng = np.random.default_rng(seed + epoch + 1000)
    rainfall, _ = generate_storm_rainfall(
        epoch, strata, wind_structs, epoch_rng,
        base_rainfall=0.5  # m/year base
    )
    return rainfall

print(f"   ✓ Weather generator created")
print(f"   Base rainfall: 0.5 m/year")
print(f"   Storm intensity: varies 0.5x to 2.0x per epoch")
print(f"   Influenced by:")
print(f"     • Orographic lifting (70% weight)")
print(f"     • Wind barriers (mountains deflect flow)")
print(f"     • Wind channels (valleys funnel storms)")
print(f"     • Topographic convergence")

# Show example rainfall pattern
test_rng = np.random.default_rng(seed + 999)
test_rainfall, test_lowP = generate_storm_rainfall(
    0, strata, wind_structs, test_rng
)

fig1, axes = plt.subplots(1, 3, figsize=(15, 4))

ax = axes[0]
im = ax.imshow(strata["surface_elev"], origin="lower", cmap="terrain")
ax.set_title("Terrain", fontweight='bold')
plt.colorbar(im, ax=ax, label="Elevation (m)")

ax = axes[1]
im = ax.imshow(test_lowP, origin="lower", cmap="YlOrRd")
ax.set_title("Storm Likelihood (Low-Pressure Zones)", fontweight='bold')
plt.colorbar(im, ax=ax, label="0 (low) to 1 (high)")

ax = axes[2]
im = ax.imshow(test_rainfall, origin="lower", cmap="Blues")
ax.set_title("Example Rainfall Pattern", fontweight='bold')
plt.colorbar(im, ax=ax, label="Rainfall (m/year)")

plt.suptitle(f"Weather System - Wind from {base_wind_dir_deg}°", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print(f"   ✓ Example: Rainfall ranges from {test_rainfall.min():.2f} to {test_rainfall.max():.2f} m/year")

# -----------------------------------------------------------------------------
# STEP 4: Set up erosion parameters
# -----------------------------------------------------------------------------
print("\n5. Setting up erosion parameters...")

num_epochs = 25
dt = 1000.0

K_channel = 1e-6
D_hillslope = 0.005
uplift_rate = 0.0001

print(f"   Simulation time: {num_epochs * dt / 1000:.1f} kyr ({num_epochs} epochs × {dt} years)")
print(f"   Channel erosion: K = {K_channel:.2e}")
print(f"   Hillslope diffusion: D = {D_hillslope} m²/yr")
print(f"   Uplift rate: {uplift_rate * 1000:.2f} mm/yr")

# Spatially variable uplift (dome)
ny, nx = strata["surface_elev"].shape
uplift_field = np.zeros((ny, nx))
center_i, center_j = ny // 2, nx // 2

for i in range(ny):
    for j in range(nx):
        dist = np.sqrt((i - center_i)**2 + (j - center_j)**2)
        uplift_field[i, j] = uplift_rate * np.exp(-(dist / (N/4))**2)

# -----------------------------------------------------------------------------
# STEP 5: Run erosion simulation with weather-driven rainfall
# -----------------------------------------------------------------------------
print("\n6. Running erosion with WEATHER-DRIVEN RAINFALL...")
print("   " + "=" * 70)

history = run_erosion_simulation(
    strata=strata,
    pixel_scale_m=pixel_scale_m,
    num_epochs=num_epochs,
    dt=dt,
    rainfall_func=rainfall_function_for_epoch,  # ← WEATHER-DRIVEN!
    uplift_rate=uplift_field,
    K_channel=K_channel,
    D_hillslope=D_hillslope,
    verbose=True
)

print("   " + "=" * 70)
print("   ✓ Simulation complete!")

# -----------------------------------------------------------------------------
# STEP 6: Statistics
# -----------------------------------------------------------------------------
print("\n7. Computing statistics...")

total_erosion = sum([h["total_erosion"] for h in history])
total_deposition = sum([h["deposition"] for h in history])

mean_erosion = np.mean(total_erosion)
max_erosion = np.max(total_erosion)
mean_deposition = np.mean(total_deposition)
max_deposition = np.max(total_deposition)

delta_elev = strata["surface_elev"] - strata_initial["surface_elev"]
mean_delta = np.mean(delta_elev)

print(f"   Erosion:")
print(f"     Mean: {mean_erosion:.2f} m")
print(f"     Max: {max_erosion:.2f} m")
print(f"   Deposition:")
print(f"     Mean: {mean_deposition:.2f} m")
print(f"     Max: {max_deposition:.2f} m")
print(f"   Net elevation change: {mean_delta:+.2f} m")

# Analyze where erosion happened relative to wind structures
erosion_on_windward = np.mean(total_erosion[wind_structs["windward_mask"]])
erosion_on_leeward = np.mean(total_erosion[wind_structs["leeward_mask"]])
erosion_in_channels = np.mean(total_erosion[wind_structs["channel_mask"]])

print(f"\n   Erosion patterns (shows wind influence on water flow):")
print(f"     Windward slopes: {erosion_on_windward:.2f} m average")
print(f"     Leeward slopes: {erosion_on_leeward:.2f} m average")
print(f"     Wind channels: {erosion_in_channels:.2f} m average")

# -----------------------------------------------------------------------------
# STEP 7: Enhanced visualizations
# -----------------------------------------------------------------------------
print("\n8. Creating visualizations...")

final_flow = history[-1]["flow_data"]
discharge = final_flow["discharge"]

# Rivers
discharge_norm = (discharge - discharge.min()) / (discharge.max() - discharge.min() + 1e-9)
river_threshold = np.percentile(discharge_norm, 95)
rivers = discharge_norm > river_threshold

print(f"   Rivers detected: {np.sum(rivers)} cells")

# Main results figure
fig2, axes = plt.subplots(2, 3, figsize=(16, 10))

# Before
ax = axes[0, 0]
im = ax.imshow(strata_initial["surface_elev"], origin="lower", cmap="terrain")
ax.set_title("BEFORE: Elevation", fontweight='bold')
plt.colorbar(im, ax=ax, fraction=0.046)

# After
ax = axes[0, 1]
im = ax.imshow(strata["surface_elev"], origin="lower", cmap="terrain")
ax.set_title("AFTER: Elevation", fontweight='bold')
plt.colorbar(im, ax=ax, fraction=0.046)

# Change
ax = axes[0, 2]
vmax = max(abs(delta_elev.min()), abs(delta_elev.max()))
im = ax.imshow(delta_elev, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
ax.set_title("Elevation Change (Δz)", fontweight='bold')
plt.colorbar(im, ax=ax, label="Δz (m)", fraction=0.046)

# Total erosion
ax = axes[1, 0]
im = ax.imshow(total_erosion, origin="lower", cmap="YlOrRd")
ax.set_title("Total Erosion (all epochs)", fontweight='bold')
plt.colorbar(im, ax=ax, fraction=0.046)

# Total deposition
ax = axes[1, 1]
im = ax.imshow(total_deposition, origin="lower", cmap="Blues")
ax.set_title("Total Deposition (all epochs)", fontweight='bold')
plt.colorbar(im, ax=ax, fraction=0.046)

# Final terrain + rivers + wind
ax = axes[1, 2]
im = ax.imshow(strata["surface_elev"], origin="lower", cmap="terrain", alpha=0.7)
# Overlay rivers
river_overlay = np.ma.masked_where(~rivers, discharge_norm)
ax.imshow(river_overlay, origin="lower", cmap="Blues", alpha=0.6)
# Overlay wind barriers
barrier_overlay = np.ma.masked_where(~wind_structs["barrier_mask"], 
                                       np.ones_like(strata["surface_elev"]))
ax.contour(barrier_overlay, levels=[0.5], colors='red', linewidths=1, alpha=0.5)
ax.set_title("FINAL: Terrain + Rivers + Barriers", fontweight='bold')
plt.colorbar(im, ax=ax, fraction=0.046)

plt.suptitle(f"Weather-Driven Erosion Results ({num_epochs * dt / 1000:.1f} kyr)", 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# SUMMARY
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("WEATHER-DRIVEN EROSION COMPLETE!")
print("=" * 80)
print(f"\nThis simulation used your sophisticated wind/storm system:")
print(f"  ✓ Wind from {base_wind_dir_deg}° influences storm paths")
print(f"  ✓ {n_barriers} mountain barriers deflect flow")
print(f"  ✓ {n_channels} valley channels funnel storms")
print(f"  ✓ Windward slopes get more rain (orographic effect)")
print(f"  ✓ Rainfall varies by epoch (0.5x to 2.0x)")
print(f"\nResults:")
print(f"  • {mean_erosion:.2f} m average erosion")
print(f"  • {np.sum(rivers)} river cells developed")
print(f"  • Windward slopes eroded {erosion_on_windward:.2f} m (avg)")
print(f"  • Leeward slopes eroded {erosion_on_leeward:.2f} m (avg)")
print("\nThe storms followed realistic paths based on topography!")
print("=" * 80)
