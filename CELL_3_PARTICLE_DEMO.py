"""
CELL 3: PARTICLE EROSION DEMO

Demonstrates VISIBLE erosion using:
1. Time acceleration (100× faster erosion)
2. Particle-based Musgrave erosion
3. Your high-resolution terrain (N=512, 10m pixels)
4. Clear before/after visualization

IMPORTANT: This will show VISIBLE CHANGES because:
- Each sim year = 100 real years of erosion
- Particles aggressively erode and deposit
- Running for 50 sim years = 5,000 real years total
"""
import numpy as np
import matplotlib.pyplot as plt
import copy

# Assumes Cell 1 and Cell 2 are already run!
# Need: quantum_seeded_topography, generate_stratigraphy from Cell 1
# Need: run_particle_erosion_simulation from Cell 2

print("=" * 80)
print("PARTICLE EROSION DEMO - VISIBLE CHANGES!")
print("=" * 80)

# ==============================================================================
# PARAMETERS
# ==============================================================================

# Grid
N = 512
pixel_scale_m = 10.0

# Simulation
num_epochs = 5  # 5 epochs
dt = 10.0  # 10 sim years per epoch = 1000 real years per epoch
# TOTAL: 5 × 10 × 100 = 5,000 real years

# Particle parameters
num_particles_per_year = 10000  # 10k particles per sim year
erosion_strength = 2.0  # 2× erosion multiplier for visibility

# Seed
seed = 42

print(f"\n📐 SETUP:")
print(f"   Grid: {N}×{N} at {pixel_scale_m}m/pixel")
print(f"   Epochs: {num_epochs}")
print(f"   Time step: {dt} sim years = {dt * 100:.0f} real years")
print(f"   Total simulation: {num_epochs * dt * 100:.0f} real years")
print(f"   Particles per sim year: {num_particles_per_year}")
print(f"   Total particles: {num_epochs * dt * num_particles_per_year:.0f}")
print(f"   Erosion strength: {erosion_strength}×")

# ==============================================================================
# GENERATE TERRAIN
# ==============================================================================

print("\n🏔️  GENERATING TERRAIN...")

# Try to use user's quantum_seeded_topography
try:
    z_norm, rng = quantum_seeded_topography(N=N, beta=3.0, random_seed=seed)
    print(f"   ✓ Terrain generated: {N}×{N}")
except TypeError:
    # Try with scale parameter instead
    try:
        z_norm, rng = quantum_seeded_topography(N=N, scale=3.0, random_seed=seed)
        print(f"   ✓ Terrain generated: {N}×{N} (using scale parameter)")
    except Exception as e:
        print(f"   ✗ Error generating terrain: {e}")
        print(f"   Creating simple test terrain...")
        z_norm = np.random.randn(N, N)
        z_norm = (z_norm - z_norm.min()) / (z_norm.max() - z_norm.min())

# Generate stratigraphy
try:
    strata = generate_stratigraphy(
        z_norm, rng,
        pixel_scale_m=pixel_scale_m,
        elev_range_m=700.0
    )
    print(f"   ✓ Stratigraphy generated")
    print(f"      Layers: {strata['properties'].shape[0]}")
    print(f"      Elevation range: {strata['surface_elev'].min():.1f} - {strata['surface_elev'].max():.1f} m")
except Exception as e:
    print(f"   ✗ Error generating stratigraphy: {e}")
    print(f"   Creating simple stratigraphy...")
    surface_elev = z_norm * 700.0
    strata = {
        "surface_elev": surface_elev,
        "interfaces": np.zeros((1, N, N)),
        "thickness": np.ones((1, N, N)) * 100.0,
        "properties": np.array([{"name": "Bedrock", "K_erosion": 1.0}])
    }

# Store initial state
strata_initial = copy.deepcopy(strata)
elevation_initial = strata_initial["surface_elev"].copy()

print(f"\n✓ INITIAL STATE:")
print(f"   Elevation: {elevation_initial.min():.1f} - {elevation_initial.max():.1f} m")
print(f"   Relief: {elevation_initial.max() - elevation_initial.min():.1f} m")

# ==============================================================================
# RUN EROSION
# ==============================================================================

print("\n" + "=" * 80)
print("RUNNING PARTICLE EROSION")
print("=" * 80)

history = run_particle_erosion_simulation(
    strata, pixel_scale_m,
    num_epochs=num_epochs,
    dt=dt,
    num_particles_per_year=num_particles_per_year,
    erosion_strength=erosion_strength,
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

print(f"\n📊 CUMULATIVE CHANGES:")
print(f"   Elevation change: {total_change.min():.2f} to {total_change.max():.2f} m")
print(f"   Avg change: {total_change.mean():.3f} m")
print(f"   Total erosion: {total_erosion_sum:.1f} m³")
print(f"   Total deposition: {total_deposition_sum:.1f} m³")
print(f"   Balance: {(total_deposition_sum - total_erosion_sum) / total_erosion_sum * 100:.1f}%")

# ==============================================================================
# VISUALIZATION
# ==============================================================================

print("\n🎨 CREATING VISUALIZATIONS...")

fig, axes = plt.subplots(2, 3, figsize=(20, 14))
fig.suptitle(f"PARTICLE EROSION: {num_epochs * dt * 100:.0f} Real Years Simulated", 
             fontsize=16, fontweight='bold')

# Row 1: Before, After, Change
ax = axes[0, 0]
im = ax.imshow(elevation_initial, cmap='terrain', interpolation='bilinear')
ax.set_title(f"BEFORE: Elevation\nRange: {elevation_initial.min():.0f}-{elevation_initial.max():.0f} m", 
             fontsize=12, fontweight='bold')
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Elevation (m)')

ax = axes[0, 1]
im = ax.imshow(elevation_final, cmap='terrain', interpolation='bilinear')
ax.set_title(f"AFTER: Elevation\nRange: {elevation_final.min():.0f}-{elevation_final.max():.0f} m", 
             fontsize=12, fontweight='bold')
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Elevation (m)')

ax = axes[0, 2]
# Use diverging colormap for change
vmax_change = max(abs(total_change.min()), abs(total_change.max()))
im = ax.imshow(total_change, cmap='RdBu_r', interpolation='bilinear',
               vmin=-vmax_change, vmax=vmax_change)
ax.set_title(f"CHANGE (Δz)\nRange: {total_change.min():.1f} to {total_change.max():.1f} m", 
             fontsize=12, fontweight='bold')
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Elevation Change (m)')

# Add text annotation
change_pct = (elevation_final.max() - elevation_final.min()) / (elevation_initial.max() - elevation_initial.min()) * 100
ax.text(0.5, -0.05, f"Relief change: {change_pct:.1f}% of initial", 
        transform=ax.transAxes, ha='center', fontsize=10, style='italic')

# Row 2: Cumulative erosion, deposition, net
cumulative_erosion = sum([h["erosion"] for h in history])
cumulative_deposition = sum([h["deposition"] for h in history])

ax = axes[1, 0]
im = ax.imshow(cumulative_erosion, cmap='Reds', interpolation='bilinear')
ax.set_title(f"Cumulative EROSION\nTotal: {cumulative_erosion.sum():.0f} m³", 
             fontsize=12, fontweight='bold')
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Erosion (m)')

ax = axes[1, 1]
im = ax.imshow(cumulative_deposition, cmap='Blues', interpolation='bilinear')
ax.set_title(f"Cumulative DEPOSITION\nTotal: {cumulative_deposition.sum():.0f} m³", 
             fontsize=12, fontweight='bold')
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Deposition (m)')

ax = axes[1, 2]
net_change = cumulative_deposition - cumulative_erosion
vmax_net = max(abs(net_change.min()), abs(net_change.max()))
im = ax.imshow(net_change, cmap='RdBu_r', interpolation='bilinear',
               vmin=-vmax_net, vmax=vmax_net)
ax.set_title(f"NET CHANGE\n(Deposition - Erosion)", 
             fontsize=12, fontweight='bold')
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Net Change (m)')

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('/workspace/particle_erosion_results.png', dpi=150, bbox_inches='tight')
print(f"   ✓ Saved: particle_erosion_results.png")
plt.show()

# Cross-section comparison
print("\n🔍 CROSS-SECTION COMPARISON...")

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Cross-Section Analysis (Middle Row)", fontsize=14, fontweight='bold')

# Select middle row
mid_row = N // 2
x_coords = np.arange(N) * pixel_scale_m / 1000.0  # Convert to km

ax = axes[0]
ax.plot(x_coords, elevation_initial[mid_row, :], 'k-', linewidth=2, label='BEFORE', alpha=0.7)
ax.plot(x_coords, elevation_final[mid_row, :], 'r-', linewidth=2, label='AFTER', alpha=0.7)
ax.fill_between(x_coords, elevation_initial[mid_row, :], elevation_final[mid_row, :],
                 where=(elevation_final[mid_row, :] < elevation_initial[mid_row, :]),
                 color='red', alpha=0.3, label='Erosion')
ax.fill_between(x_coords, elevation_initial[mid_row, :], elevation_final[mid_row, :],
                 where=(elevation_final[mid_row, :] > elevation_initial[mid_row, :]),
                 color='blue', alpha=0.3, label='Deposition')
ax.set_xlabel('Distance (km)', fontsize=12)
ax.set_ylabel('Elevation (m)', fontsize=12)
ax.set_title('Elevation Evolution', fontsize=12, fontweight='bold')
ax.legend(loc='best', framealpha=0.9)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(x_coords, total_change[mid_row, :], 'purple', linewidth=2)
ax.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
ax.fill_between(x_coords, 0, total_change[mid_row, :],
                 where=(total_change[mid_row, :] < 0),
                 color='red', alpha=0.5, label='Erosion')
ax.fill_between(x_coords, 0, total_change[mid_row, :],
                 where=(total_change[mid_row, :] > 0),
                 color='blue', alpha=0.5, label='Deposition')
ax.set_xlabel('Distance (km)', fontsize=12)
ax.set_ylabel('Elevation Change (m)', fontsize=12)
ax.set_title('Erosion/Deposition Profile', fontsize=12, fontweight='bold')
ax.legend(loc='best', framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('/workspace/particle_erosion_cross_section.png', dpi=150, bbox_inches='tight')
print(f"   ✓ Saved: particle_erosion_cross_section.png")
plt.show()

print("\n" + "=" * 80)
print("✓ DEMO COMPLETE!")
print("=" * 80)
print(f"\nKEY RESULTS:")
print(f"  • Simulated {num_epochs * dt * 100:.0f} real years of erosion")
print(f"  • Used {num_epochs * dt * num_particles_per_year:.0f} raindrop particles")
print(f"  • Elevation change: {total_change.min():.1f} to {total_change.max():.1f} m")
print(f"  • Average change: {abs(total_change).mean():.2f} m")
print(f"  • Maximum erosion: {cumulative_erosion.max():.1f} m")
print(f"  • Maximum deposition: {cumulative_deposition.max():.1f} m")
print(f"\n👀 If you still don't see changes, increase:")
print(f"  • num_epochs (more simulation time)")
print(f"  • num_particles_per_year (more aggressive erosion)")
print(f"  • erosion_strength (multiplier on erosion rates)")
print(f"  • TIME_ACCELERATION in Cell 2 (currently 100×)")
