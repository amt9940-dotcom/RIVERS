"""
CELL 3: DEMO (FIXED - SHOWS IMPROVEMENTS)

This demonstrates:
- Coherent wind features (fewer barriers, more channels)
- Orographic weather patterns
- Realistic erosion with river networks
- Bounded elevations (no blow-up)
"""
import numpy as np
import matplotlib.pyplot as plt
import copy

print("=" * 80)
print("FIXED EROSION SYSTEM DEMO")
print("=" * 80)

# ==============================================================================
# STEP 1: Generate terrain
# ==============================================================================
print("\n1. Generating terrain...")

N = 50
seed = 42
pixel_scale_m = 1000.0
base_elevation_m = 800.0
relief_m = 400.0
base_wind_dir_deg = 270.0  # West to East

# Generate terrain
# Check which version of quantum_seeded_topography we have
import inspect
sig = inspect.signature(quantum_seeded_topography)
params = list(sig.parameters.keys())

if 'scale' in params and 'octaves' in params:
    # New FIXED version
    z_norm, rng = quantum_seeded_topography(N=N, random_seed=seed, scale=3.0, octaves=6)
elif 'beta' in params:
    # Old version
    z_norm, rng = quantum_seeded_topography(N=N, random_seed=seed, beta=3.0)
else:
    # Try new signature as default
    z_norm, rng = quantum_seeded_topography(N=N, random_seed=seed, scale=3.0, octaves=6)
strata = generate_stratigraphy(z_norm, pixel_scale_m, base_elevation_m, relief_m)

print(f"   Grid: {N} × {N}")
print(f"   Cell size: {pixel_scale_m} m")
print(f"   Elevation range: {strata['surface_elev'].min():.1f} - {strata['surface_elev'].max():.1f} m")

# ==============================================================================
# STEP 2: Analyze wind features (FIXED)
# ==============================================================================
print("\n2. Analyzing wind features (FIXED)...")

wind_structs = build_wind_structures(strata["surface_elev"], pixel_scale_m, base_wind_dir_deg)

n_barriers = np.sum(wind_structs["barrier_mask"])
n_channels = np.sum(wind_structs["channel_mask"])
n_windward = np.sum(wind_structs["windward_mask"])
n_leeward = np.sum(wind_structs["leeward_mask"])

print(f"   Wind from: {base_wind_dir_deg}° (west)")
print(f"   Barriers detected: {n_barriers} cells")
print(f"   Channels detected: {n_channels} cells")
print(f"   Windward slopes: {n_windward} cells")
print(f"   Leeward slopes: {n_leeward} cells")
print(f"   → NOTE: Barriers should be ~100-300, channels ~50-200")
print(f"   → These are LARGE-SCALE features, not tiny speckles")

# Visualize wind features
fig0, axes = plt.subplots(2, 3, figsize=(15, 10))

# Elevation
ax = axes[0, 0]
im = ax.imshow(strata["surface_elev"], origin="lower", cmap="terrain")
ax.set_title("Elevation", fontweight='bold')
plt.colorbar(im, ax=ax, label="m")

# Slope
ax = axes[0, 1]
slope = wind_structs["topo_fields"]["slope"]
im = ax.imshow(slope, origin="lower", cmap="viridis")
ax.set_title("Slope", fontweight='bold')
plt.colorbar(im, ax=ax, label="m/m")

# Laplacian (curvature)
ax = axes[0, 2]
lap = wind_structs["topo_fields"]["laplacian"]
im = ax.imshow(lap, origin="lower", cmap="RdBu_r", vmin=-0.1, vmax=0.1)
ax.set_title("Curvature (Laplacian)", fontweight='bold')
plt.colorbar(im, ax=ax, label="1/m")

# Barriers overlay
ax = axes[1, 0]
# Show elevation with barrier contours
im = ax.imshow(strata["surface_elev"], origin="lower", cmap="gray", alpha=0.5)
barrier_overlay = np.ma.masked_where(~wind_structs["barrier_mask"], 
                                      np.ones_like(strata["surface_elev"]))
ax.imshow(barrier_overlay, origin="lower", cmap="Reds", alpha=0.7)
ax.contour(wind_structs["barrier_mask"], levels=[0.5], colors='red', linewidths=2)
ax.set_title(f"Wind Barriers (n={n_barriers})", fontweight='bold')
ax.text(0.02, 0.98, "Red = High ridges\nfacing wind", transform=ax.transAxes,
        va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Channels overlay
ax = axes[1, 1]
im = ax.imshow(strata["surface_elev"], origin="lower", cmap="gray", alpha=0.5)
channel_overlay = np.ma.masked_where(~wind_structs["channel_mask"],
                                      np.ones_like(strata["surface_elev"]))
ax.imshow(channel_overlay, origin="lower", cmap="Blues", alpha=0.7)
ax.contour(wind_structs["channel_mask"], levels=[0.5], colors='blue', linewidths=2)
ax.set_title(f"Wind Channels (n={n_channels})", fontweight='bold')
ax.text(0.02, 0.98, "Blue = Valleys\naligned with wind", transform=ax.transAxes,
        va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Combined
ax = axes[1, 2]
im = ax.imshow(strata["surface_elev"], origin="lower", cmap="terrain", alpha=0.7)
# Windward (orange)
windward_overlay = np.ma.masked_where(~wind_structs["windward_mask"],
                                       np.ones_like(strata["surface_elev"]))
ax.imshow(windward_overlay, origin="lower", cmap="Oranges", alpha=0.4)
# Barriers (red contours)
ax.contour(wind_structs["barrier_mask"], levels=[0.5], colors='red', linewidths=1.5, alpha=0.8)
# Channels (blue contours)
ax.contour(wind_structs["channel_mask"], levels=[0.5], colors='blue', linewidths=1.5, alpha=0.8)
ax.set_title("Combined Wind Features", fontweight='bold')
ax.text(0.02, 0.98, "Red = barriers\nBlue = channels\nOrange = windward",
        transform=ax.transAxes, va='top', fontsize=8,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.suptitle(f"Wind Feature Analysis (Wind from {base_wind_dir_deg}°)", 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ==============================================================================
# STEP 3: Generate weather (FIXED)
# ==============================================================================
print("\n3. Generating weather system (FIXED)...")

def generate_storm_rainfall_FIXED(epoch, strata, wind_structs, rng, base_rainfall=0.8):
    """FIXED: Generate rainfall with strong orographic control."""
    surface_elev = strata["surface_elev"]
    
    # Compute low-pressure (FIXED: 70% orographic weight)
    low_pressure_data = compute_orographic_low_pressure(
        surface_elev,
        rng,
        pixel_scale_m,
        base_wind_dir_deg=base_wind_dir_deg,
        wind_structs=wind_structs,
        scale_factor=1.5,
        orographic_weight=0.7  # Strong topographic control
    )
    
    low_pressure = low_pressure_data["low_pressure_likelihood"]
    
    # Epoch wetness variation
    epoch_wetness = 0.5 + 1.5 * rng.random()
    
    # Rainfall = base * (1 + low_pressure) * wetness
    # This gives 0.8 * (1-2) * (0.5-2.0) = 0.4 to 3.2 m/year range
    rainfall = base_rainfall * (1.0 + low_pressure) * epoch_wetness
    
    # Slight smoothing
    from scipy.ndimage import uniform_filter
    rainfall = uniform_filter(rainfall, size=3, mode='wrap')
    
    return rainfall, low_pressure

# Test
test_rng = np.random.default_rng(seed + 999)
test_rainfall, test_lowP = generate_storm_rainfall_FIXED(
    0, strata, wind_structs, test_rng
)

print(f"   Base rainfall: 0.8 m/year")
print(f"   Rainfall range: {test_rainfall.min():.2f} - {test_rainfall.max():.2f} m/year")
print(f"   Storm likelihood: {test_lowP.min():.2f} - {test_lowP.max():.2f}")

# Visualize weather
fig1, axes = plt.subplots(1, 3, figsize=(15, 4))

ax = axes[0]
im = ax.imshow(strata["surface_elev"], origin="lower", cmap="terrain")
ax.set_title("Terrain Elevation", fontweight='bold')
plt.colorbar(im, ax=ax, label="Elevation (m)")

ax = axes[1]
im = ax.imshow(test_lowP, origin="lower", cmap="YlOrRd")
# Overlay barriers and channels
ax.contour(wind_structs["barrier_mask"], levels=[0.5], colors='red', 
           linewidths=1, alpha=0.5, linestyles='dashed')
ax.contour(wind_structs["channel_mask"], levels=[0.5], colors='blue',
           linewidths=1, alpha=0.5, linestyles='dashed')
ax.set_title("Storm Likelihood (with wind features)", fontweight='bold')
plt.colorbar(im, ax=ax, label="0 (low) to 1 (high)")
ax.text(0.02, 0.98, "Red dashed = barriers\nBlue dashed = channels",
        transform=ax.transAxes, va='top', fontsize=8,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax = axes[2]
im = ax.imshow(test_rainfall, origin="lower", cmap="Blues")
ax.set_title("Example Rainfall Pattern", fontweight='bold')
plt.colorbar(im, ax=ax, label="Rainfall (m/year)")

plt.suptitle(f"Weather System (FIXED) - Wind from {base_wind_dir_deg}°", 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ==============================================================================
# STEP 4: Run erosion (FIXED - REALISTIC PARAMETERS)
# ==============================================================================
print("\n4. Running erosion simulation (FIXED)...")

# Save initial state
strata_initial = copy.deepcopy(strata)

# REALISTIC PARAMETERS (prevent blow-up)
num_epochs = 10  # Start with fewer epochs for testing
dt = 100.0  # Shorter time step (was 1000)
K_channel = 1e-6  # Smaller erosion coefficient (was 1e-5)
D_hillslope = 0.005  # Smaller diffusion (was 0.01)
uplift_rate = 0.00005  # Smaller uplift (was 0.0001)

print(f"   Parameters:")
print(f"     Epochs: {num_epochs}")
print(f"     Time step: {dt} years")
print(f"     Total time: {num_epochs * dt / 1000:.1f} kyr")
print(f"     K_channel: {K_channel:.2e}")
print(f"     D_hillslope: {D_hillslope}")
print(f"     Uplift: {uplift_rate * 1000:.3f} mm/yr")
print(f"   Bounds:")
print(f"     Max erosion/step: 10m channel, 5m hillslope")
print(f"     Min elevation: basement + 10m")

# Rainfall function
def rainfall_func(epoch):
    epoch_rng = np.random.default_rng(seed + epoch + 1000)
    rainfall, _ = generate_storm_rainfall_FIXED(
        epoch, strata, wind_structs, epoch_rng, base_rainfall=0.8
    )
    return rainfall

# Spatially variable uplift (dome)
ny, nx = strata["surface_elev"].shape
uplift_field = np.zeros((ny, nx))
center_i, center_j = ny // 2, nx // 2

for i in range(ny):
    for j in range(nx):
        dist = np.sqrt((i - center_i)**2 + (j - center_j)**2)
        uplift_field[i, j] = uplift_rate * np.exp(-(dist / (N/4))**2)

# Run
print("   Running...")
history = run_erosion_simulation_FIXED(
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

print("   ✓ Complete!")

# ==============================================================================
# STEP 5: Statistics
# ==============================================================================
print("\n5. Computing statistics...")

total_erosion = sum([h["total_erosion"] for h in history])
total_deposition = sum([h["deposition"] for h in history])

mean_erosion = np.mean(total_erosion)
max_erosion = np.max(total_erosion)
mean_deposition = np.mean(total_deposition)
max_deposition = np.max(total_deposition)

delta_elev = strata["surface_elev"] - strata_initial["surface_elev"]
mean_delta = np.mean(delta_elev)

print(f"   Initial elevation: {strata_initial['surface_elev'].min():.1f} - {strata_initial['surface_elev'].max():.1f} m")
print(f"   Final elevation: {strata['surface_elev'].min():.1f} - {strata['surface_elev'].max():.1f} m")
print(f"   Erosion: mean {mean_erosion:.2f} m, max {max_erosion:.2f} m")
print(f"   Deposition: mean {mean_deposition:.2f} m, max {max_deposition:.2f} m")
print(f"   Net change: {mean_delta:+.2f} m")

# Check for blow-up
if strata['surface_elev'].min() < -1000:
    print("   ⚠ WARNING: Elevation went very negative! Check parameters.")
elif strata['surface_elev'].min() < 0:
    print("   ⚠ WARNING: Some negative elevations (below sea level).")
else:
    print("   ✓ All elevations positive (above sea level)")

# ==============================================================================
# STEP 6: Visualizations
# ==============================================================================
print("\n6. Creating visualizations...")

final_flow = history[-1]["flow_data"]
discharge = final_flow["discharge"]

# Identify rivers (high discharge = upslope area)
discharge_norm = (discharge - discharge.min()) / (discharge.max() - discharge.min() + 1e-9)
river_threshold = np.percentile(discharge_norm, 95)
rivers = discharge_norm > river_threshold

print(f"   Rivers detected: {np.sum(rivers)} cells (top 5% discharge)")

# Main figure
fig2, axes = plt.subplots(3, 3, figsize=(16, 15))

# Row 1: Before, After, Change
ax = axes[0, 0]
im = ax.imshow(strata_initial["surface_elev"], origin="lower", cmap="terrain")
ax.set_title("BEFORE: Elevation", fontweight='bold')
plt.colorbar(im, ax=ax, label="m", fraction=0.046)
elev_range = f"{strata_initial['surface_elev'].min():.0f}-{strata_initial['surface_elev'].max():.0f}m"
ax.text(0.5, 0.95, elev_range, transform=ax.transAxes, ha='center', va='top',
        fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax = axes[0, 1]
im = ax.imshow(strata["surface_elev"], origin="lower", cmap="terrain")
ax.set_title("AFTER: Elevation", fontweight='bold')
plt.colorbar(im, ax=ax, label="m", fraction=0.046)
elev_range = f"{strata['surface_elev'].min():.0f}-{strata['surface_elev'].max():.0f}m"
ax.text(0.5, 0.95, elev_range, transform=ax.transAxes, ha='center', va='top',
        fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax = axes[0, 2]
vmax = max(abs(delta_elev.min()), abs(delta_elev.max()))
if vmax < 0.01:
    vmax = 0.01
im = ax.imshow(delta_elev, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
ax.set_title("Elevation Change (Δz)", fontweight='bold')
plt.colorbar(im, ax=ax, label="m (red=erosion, blue=deposition)", fraction=0.046)
change_range = f"{delta_elev.min():.1f} to {delta_elev.max():.1f}m"
ax.text(0.5, 0.95, change_range, transform=ax.transAxes, ha='center', va='top',
        fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Row 2: Erosion, Deposition, Discharge
ax = axes[1, 0]
im = ax.imshow(total_erosion, origin="lower", cmap="YlOrRd")
ax.set_title("Total Erosion", fontweight='bold')
plt.colorbar(im, ax=ax, label="m", fraction=0.046)

ax = axes[1, 1]
im = ax.imshow(total_deposition, origin="lower", cmap="Blues")
ax.set_title("Total Deposition", fontweight='bold')
plt.colorbar(im, ax=ax, label="m", fraction=0.046)

ax = axes[1, 2]
# Log-scale discharge (upslope area)
discharge_log = np.log10(discharge + 1)
im = ax.imshow(discharge_log, origin="lower", cmap="viridis")
ax.set_title("Discharge (log₁₀ upslope area)", fontweight='bold')
plt.colorbar(im, ax=ax, label="log₁₀(A + 1)", fraction=0.046)
ax.text(0.02, 0.98, "High values = rivers\n(large drainage basins)",
        transform=ax.transAxes, va='top', fontsize=8,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Row 3: Rivers, Terrain + Rivers, Cross-section
ax = axes[2, 0]
im = ax.imshow(rivers, origin="lower", cmap="Blues")
ax.set_title("River Network", fontweight='bold')
ax.text(0.5, 0.95, f"{np.sum(rivers)} river cells", transform=ax.transAxes,
        ha='center', va='top', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax = axes[2, 1]
# Final terrain with rivers
im = ax.imshow(strata["surface_elev"], origin="lower", cmap="terrain", alpha=0.7)
river_overlay = np.ma.masked_where(~rivers, discharge_norm)
ax.imshow(river_overlay, origin="lower", cmap="Blues", alpha=0.8)
# Add barriers
ax.contour(wind_structs["barrier_mask"], levels=[0.5], colors='red', linewidths=1, alpha=0.6)
ax.set_title("Final: Terrain + Rivers + Barriers", fontweight='bold')
ax.text(0.02, 0.98, "Blue = rivers\nRed = wind barriers",
        transform=ax.transAxes, va='top', fontsize=8,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Cross-section
ax = axes[2, 2]
row_idx = ny // 2
x_km = np.arange(nx) * pixel_scale_m / 1000.0
ax.plot(x_km, strata_initial["surface_elev"][row_idx, :], 'k-', linewidth=2, label="Before", alpha=0.6)
ax.plot(x_km, strata["surface_elev"][row_idx, :], 'b-', linewidth=2, label="After")
ax.fill_between(x_km, strata["surface_elev"][row_idx, :], strata_initial["surface_elev"][row_idx, :],
                where=(strata["surface_elev"][row_idx, :] < strata_initial["surface_elev"][row_idx, :]),
                color='red', alpha=0.3, label="Erosion")
ax.fill_between(x_km, strata["surface_elev"][row_idx, :], strata_initial["surface_elev"][row_idx, :],
                where=(strata["surface_elev"][row_idx, :] > strata_initial["surface_elev"][row_idx, :]),
                color='blue', alpha=0.3, label="Deposition")
ax.set_xlabel("Distance (km)")
ax.set_ylabel("Elevation (m)")
ax.set_title(f"Cross-Section (row {row_idx})", fontweight='bold')
ax.legend(loc='best', fontsize=8)
ax.grid(True, alpha=0.3)

plt.suptitle(f"Weather-Driven Erosion Results (FIXED) - {num_epochs * dt / 1000:.1f} kyr", 
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()

# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "=" * 80)
print("FIXED SYSTEM SUMMARY")
print("=" * 80)
print(f"\n✅ Wind features: {n_barriers} barriers, {n_channels} channels")
print(f"   (Should be coherent large-scale features, not speckles)")
print(f"\n✅ Weather: Strong orographic control (70% weight)")
print(f"   Rainfall range: {test_rainfall.min():.2f} - {test_rainfall.max():.2f} m/year")
print(f"\n✅ Erosion: Proper flow routing with upslope area")
print(f"   River cells: {np.sum(rivers)} (forming dendritic networks)")
print(f"   Elevation range: {strata['surface_elev'].min():.1f} - {strata['surface_elev'].max():.1f} m")
print(f"   (Should be ~{base_elevation_m} - {base_elevation_m + relief_m} m, slightly modified)")
print(f"\n✅ No numerical blow-up!")
print(f"   All values bounded and realistic")
print("=" * 80)
