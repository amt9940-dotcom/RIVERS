"""
CELL 3: PROPER FLUVIAL EROSION DEMO

Demonstrates the corrected erosion model with:
- Half-loss rule (50% of eroded material removed → net volume loss)
- Proper sediment transport (only deposits where capacity is exceeded)
- No random uphill deposition
- Valleys can deepen over time
"""
import numpy as np
import matplotlib.pyplot as plt
import copy

print("=" * 80)
print("PROPER FLUVIAL EROSION DEMO")
print("=" * 80)

# ==============================================================================
# PARAMETERS
# ==============================================================================

# Grid
N = 512
pixel_scale_m = 10.0

# Simulation
num_epochs = 10  # 10 epochs
dt = 100.0  # 100 years per epoch
# TOTAL: 1,000 years with extreme rain boost

# Seed
seed = 42

print(f"\n📐 SETUP:")
print(f"   Grid: {N}×{N} at {pixel_scale_m}m/pixel")
print(f"   Epochs: {num_epochs}")
print(f"   Time step: {dt} years per epoch")
print(f"   Total simulation: {num_epochs * dt:.0f} years")
print(f"   Rain boost: {RAIN_BOOST}× (extreme erosion!)")

# ==============================================================================
# GENERATE TERRAIN
# ==============================================================================

print("\n🏔️  GENERATING TERRAIN...")

# Try to use user's quantum_seeded_topography
try:
    z_norm, rng = quantum_seeded_topography(N=N, beta=3.0, random_seed=seed)
    print(f"   ✓ Terrain generated: {N}×{N}")
except (NameError, TypeError):
    # Try with scale parameter instead
    try:
        z_norm, rng = quantum_seeded_topography(N=N, scale=3.0, random_seed=seed)
        print(f"   ✓ Terrain generated: {N}×{N} (using scale parameter)")
    except Exception as e:
        print(f"   ✗ Error generating terrain: {e}")
        print(f"   Creating simple test terrain...")
        z_norm = np.random.randn(N, N)
        from scipy.ndimage import gaussian_filter
        z_norm = gaussian_filter(z_norm, sigma=5.0)
        z_norm = (z_norm - z_norm.min()) / (z_norm.max() - z_norm.min())

# Generate stratigraphy
try:
    strata = generate_stratigraphy(
        z_norm, rng,
        pixel_scale_m=pixel_scale_m,
        elev_range_m=700.0
    )
    print(f"   ✓ Stratigraphy generated")
    print(f"      Elevation range: {strata['surface_elev'].min():.1f} - {strata['surface_elev'].max():.1f} m")
except Exception as e:
    print(f"   ✗ Error generating stratigraphy: {e}")
    print(f"   Creating simple stratigraphy...")
    surface_elev = z_norm * 700.0
    strata = {
        "surface_elev": surface_elev,
    }

# Store initial state
strata_initial = copy.deepcopy(strata)
elevation_initial = strata_initial["surface_elev"].copy()

print(f"\n✓ INITIAL STATE:")
print(f"   Elevation: {elevation_initial.min():.1f} - {elevation_initial.max():.1f} m")
print(f"   Relief: {elevation_initial.max() - elevation_initial.min():.1f} m")

# ==============================================================================
# RAINFALL FUNCTION
# ==============================================================================

def simple_orographic_rain(strata, epoch):
    """
    Simple spatially-varying rainfall (orographic enhancement).
    
    Higher elevations get more rain (windward slopes).
    """
    elevation = strata["surface_elev"]
    
    # Base rain (uniform)
    base_rain = 1.0  # meters/year (before boost)
    
    # Orographic enhancement (higher elevations get more rain)
    elev_norm = (elevation - elevation.min()) / (elevation.max() - elevation.min() + 1e-6)
    orographic_factor = 1.0 + elev_norm * 0.5  # Up to 1.5× rain at peaks
    
    rain = base_rain * orographic_factor
    
    return rain


# ==============================================================================
# RUN EROSION
# ==============================================================================

print("\n" + "=" * 80)
print("RUNNING FLUVIAL EROSION (with HALF-LOSS RULE)")
print("=" * 80)

history = run_fluvial_simulation(
    strata,
    rain_func=simple_orographic_rain,
    pixel_scale_m=pixel_scale_m,
    num_epochs=num_epochs,
    dt=dt,
    apply_diffusion=True,
    verbose=True
)

elevation_final = strata["surface_elev"].copy()

print(f"\n✓ FINAL STATE:")
print(f"   Elevation: {elevation_final.min():.1f} - {elevation_final.max():.1f} m")
print(f"   Relief: {elevation_final.max() - elevation_final.min():.1f} m")

# Compute total change
total_change = elevation_final - elevation_initial
total_erosion_sum = sum([h["erosion"].sum() for h in history])
total_deposition_sum = sum([h["deposition"].sum() for h in history])
net_volume_change = total_deposition_sum - total_erosion_sum

print(f"\n📊 CUMULATIVE CHANGES:")
print(f"   Elevation change: {total_change.min():.2f} to {total_change.max():.2f} m")
print(f"   Avg change: {total_change.mean():.3f} m")
print(f"   Total erosion: {total_erosion_sum:.1f} m³")
print(f"   Total deposition: {total_deposition_sum:.1f} m³")
print(f"   NET VOLUME CHANGE: {net_volume_change:.1f} m³")
print(f"   Volume loss ratio: {net_volume_change / total_erosion_sum * 100:.1f}%")
print(f"   ⚠️  Expected: ~-50% (half-loss rule)")

# ==============================================================================
# VISUALIZATION
# ==============================================================================

print("\n🎨 CREATING VISUALIZATIONS...")

fig, axes = plt.subplots(2, 3, figsize=(20, 14))
fig.suptitle(f"PROPER FLUVIAL EROSION: {num_epochs * dt:.0f} Years with Half-Loss Rule\n" +
             f"(50% of eroded material removed → NET VOLUME LOSS!)", 
             fontsize=16, fontweight='bold')

# Row 1: Before, After, Change
ax = axes[0, 0]
im = ax.imshow(elevation_initial, cmap='terrain', interpolation='bilinear')
ax.set_title(f"1. BEFORE: Initial Elevation\nRange: {elevation_initial.min():.0f}-{elevation_initial.max():.0f} m", 
             fontsize=12, fontweight='bold')
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Elevation (m)')
ax.text(0.5, -0.08, "Your original terrain", 
        transform=ax.transAxes, ha='center', fontsize=9, style='italic', color='green')

ax = axes[0, 1]
im = ax.imshow(elevation_final, cmap='terrain', interpolation='bilinear')
ax.set_title(f"2. AFTER: Final Elevation\nRange: {elevation_final.min():.0f}-{elevation_final.max():.0f} m", 
             fontsize=12, fontweight='bold')
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Elevation (m)')
ax.text(0.5, -0.08, "After erosion (NOTE: should be LOWER overall!)", 
        transform=ax.transAxes, ha='center', fontsize=9, style='italic', color='blue')

ax = axes[0, 2]
# Use diverging colormap for change
vmax_change = max(abs(total_change.min()), abs(total_change.max()))
im = ax.imshow(total_change, cmap='RdBu_r', interpolation='bilinear',
               vmin=-vmax_change, vmax=vmax_change)
ax.set_title(f"3. TOTAL CHANGE (AFTER - BEFORE)\nRange: {total_change.min():.1f} to {total_change.max():.1f} m", 
             fontsize=12, fontweight='bold')
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Elevation Change (m)')
ax.text(0.5, -0.08, "RED = erosion (should be DOMINANT!), blue = deposition", 
        transform=ax.transAxes, ha='center', fontsize=9, style='italic', color='purple')

# Row 2: Erosion, deposition, discharge
cumulative_erosion = sum([h["erosion"] for h in history])
cumulative_deposition = sum([h["deposition"] for h in history])
final_Q = history[-1]["Q"]

ax = axes[1, 0]
im = ax.imshow(cumulative_erosion, cmap='Reds', interpolation='bilinear')
ax.set_title(f"4. CUMULATIVE EROSION\nTotal: {cumulative_erosion.sum():.0f} m³, Max: {cumulative_erosion.max():.1f} m", 
             fontsize=12, fontweight='bold')
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Erosion (m)')
ax.text(0.5, -0.08, f"Valleys, channels, slopes (avg: {cumulative_erosion.mean():.2f} m)", 
        transform=ax.transAxes, ha='center', fontsize=9, style='italic', color='darkred')

ax = axes[1, 1]
im = ax.imshow(cumulative_deposition, cmap='Blues', interpolation='bilinear')
ax.set_title(f"5. CUMULATIVE DEPOSITION\nTotal: {cumulative_deposition.sum():.0f} m³, Max: {cumulative_deposition.max():.1f} m", 
             fontsize=12, fontweight='bold')
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Deposition (m)')
ax.text(0.5, -0.08, f"Flats, basins, fans (avg: {cumulative_deposition.mean():.2f} m) - LESS than erosion!", 
        transform=ax.transAxes, ha='center', fontsize=9, style='italic', color='darkblue')

ax = axes[1, 2]
# Log scale for discharge (better visualization)
Q_plot = np.log10(final_Q + 1.0)
im = ax.imshow(Q_plot, cmap='Blues', interpolation='bilinear')
ax.set_title(f"6. DISCHARGE (log scale)\nShows river/channel network", 
             fontsize=12, fontweight='bold')
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='log10(Q+1)')
ax.text(0.5, -0.08, "High Q = rivers/channels (water accumulates downhill)", 
        transform=ax.transAxes, ha='center', fontsize=9, style='italic', color='navy')

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('proper_fluvial_erosion_results.png', dpi=150, bbox_inches='tight')
print(f"   ✓ Saved: proper_fluvial_erosion_results.png")
plt.show()

# Cross-section comparison
print("\n🔍 CROSS-SECTION COMPARISON...")

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Cross-Section: Note NET LOWERING (valleys deepen!)", fontsize=14, fontweight='bold')

# Select middle row
mid_row = N // 2
x_coords = np.arange(N) * pixel_scale_m / 1000.0  # Convert to km

ax = axes[0]
ax.plot(x_coords, elevation_initial[mid_row, :], 'k-', linewidth=2, label='BEFORE', alpha=0.7)
ax.plot(x_coords, elevation_final[mid_row, :], 'r-', linewidth=2, label='AFTER', alpha=0.7)
ax.fill_between(x_coords, elevation_initial[mid_row, :], elevation_final[mid_row, :],
                 where=(elevation_final[mid_row, :] < elevation_initial[mid_row, :]),
                 color='red', alpha=0.3, label='Erosion (NET LOSS)')
ax.fill_between(x_coords, elevation_initial[mid_row, :], elevation_final[mid_row, :],
                 where=(elevation_final[mid_row, :] > elevation_initial[mid_row, :]),
                 color='blue', alpha=0.3, label='Deposition (minor)')
ax.set_xlabel('Distance (km)', fontsize=12)
ax.set_ylabel('Elevation (m)', fontsize=12)
ax.set_title('Elevation Evolution (AFTER should be mostly BELOW BEFORE!)', fontsize=12, fontweight='bold')
ax.legend(loc='best', framealpha=0.9)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(x_coords, total_change[mid_row, :], 'purple', linewidth=2)
ax.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
ax.fill_between(x_coords, 0, total_change[mid_row, :],
                 where=(total_change[mid_row, :] < 0),
                 color='red', alpha=0.5, label='Erosion (should be DOMINANT)')
ax.fill_between(x_coords, 0, total_change[mid_row, :],
                 where=(total_change[mid_row, :] > 0),
                 color='blue', alpha=0.5, label='Deposition (minor)')
ax.set_xlabel('Distance (km)', fontsize=12)
ax.set_ylabel('Elevation Change (m)', fontsize=12)
ax.set_title('Erosion/Deposition Profile (MORE RED than BLUE!)', fontsize=12, fontweight='bold')
ax.legend(loc='best', framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('proper_fluvial_cross_section.png', dpi=150, bbox_inches='tight')
print(f"   ✓ Saved: proper_fluvial_cross_section.png")
plt.show()

print("\n" + "=" * 80)
print("✓ DEMO COMPLETE!")
print("=" * 80)
print(f"\nKEY RESULTS:")
print(f"  • Simulated {num_epochs * dt:.0f} years of erosion")
print(f"  • Rain boost: {RAIN_BOOST}×")
print(f"  • Total erosion: {cumulative_erosion.sum():.0f} m³")
print(f"  • Total deposition: {cumulative_deposition.sum():.0f} m³")
print(f"  • NET VOLUME LOSS: {net_volume_change:.0f} m³ ({net_volume_change / total_erosion_sum * 100:.1f}%)")
print(f"")
print(f"✅ EXPECTED BEHAVIOR:")
print(f"  • More RED than BLUE in change map")
print(f"  • Deposition ≈ 50% of erosion (half-loss rule)")
print(f"  • Valleys deepen (net volume loss)")
print(f"  • No random uphill deposition")
print(f"  • Sediment only deposits where capacity is exceeded (flats, basins)")
