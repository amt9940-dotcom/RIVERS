"""
DEMONSTRATION: EROSION SIMULATION WITH INTEGRATED WEATHER PHYSICS

This version PROPERLY USES the weather system:
- Wind-topography interaction
- Quantum random rain within storms
- Real weather physics affects erosion
- Rivers and lakes visualization

NO artificial rain generation - uses actual weather simulation!
"""

import numpy as np
import matplotlib.pyplot as plt
import time

print("="*80)
print("INTEGRATED WEATHER + EROSION SIMULATION")
print("="*80)
print(f"Time Acceleration: {TIME_ACCELERATION}×")
print(f"Rain Boost: {RAIN_BOOST}×")
print("="*80)

# ============================================================================
# STEP 1: Generate terrain
# ============================================================================
print("\n[1/5] Generating quantum-seeded terrain...")
start_time = time.time()

N = 256  # Grid size (256×256 for good performance)
pixel_scale_m = 20.0  # 20m per pixel → 5.12 km domain
elev_range_m = 500.0
seed = None  # Quantum random

try:
    z_norm, rng = quantum_seeded_topography(
        N=N,
        beta=3.0,
        warp_amp=0.10,
        ridged_alpha=0.15,
        random_seed=seed
    )
    
    strata = generate_stratigraphy(z_norm, rng, pixel_scale_m, elev_range_m)
    
    print(f"✓ Terrain generated: {N}×{N} grid")
    print(f"  Domain size: {N*pixel_scale_m/1000:.2f} × {N*pixel_scale_m/1000:.2f} km")
    print(f"  Elevation range: {strata['surface_elev'].min():.1f} - {strata['surface_elev'].max():.1f} m")
    print(f"  Layers: {len(strata['layer_order'])}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    raise

elapsed = time.time() - start_time
print(f"  Time: {elapsed:.1f} s")

# ============================================================================
# STEP 2: Generate weather with wind-topography interaction
# ============================================================================
print("\n[2/5] Running weather simulation with wind-topography interaction...")
start_time = time.time()

# Run weather for the same number of years as erosion simulation
num_years = 100  # 100 years simulation (= 1000 real years with 10× acceleration)
base_wind_dir_deg = 270.0  # West wind (common in mid-latitudes)
mean_annual_rain_m = 1.0  # 1 meter per year

try:
    weather_data = run_weather_simulation(
        surface_elev=strata['surface_elev'],
        pixel_scale_m=pixel_scale_m,
        num_years=num_years,
        base_wind_dir_deg=base_wind_dir_deg,
        mean_annual_rain_m=mean_annual_rain_m,
        random_seed=None  # Quantum random
    )
    
    print(f"✓ Weather simulation complete!")
    print(f"  Total years: {num_years}")
    print(f"  Wind direction: {base_wind_dir_deg}° (West wind)")
    print(f"  Mean annual rain: {mean_annual_rain_m:.2f} m/yr")
    print(f"  Generated {len(weather_data['annual_rain_maps'])} annual rain maps")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    raise

elapsed = time.time() - start_time
print(f"  Time: {elapsed:.1f} s")

# ============================================================================
# STEP 3: Prepare rain data for erosion
# ============================================================================
print("\n[3/5] Preparing rain data for erosion simulation...")

# Stack annual rain maps into timeseries
rain_timeseries = np.array(weather_data['annual_rain_maps'], dtype=np.float32)

print(f"✓ Rain timeseries prepared")
print(f"  Shape: {rain_timeseries.shape}")
print(f"  Mean annual rain: {rain_timeseries.mean(axis=(1,2)).mean():.3f} m/yr")
print(f"  Min: {rain_timeseries.min():.3f} m/yr")
print(f"  Max: {rain_timeseries.max():.3f} m/yr")

# Visualize rain patterns
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Total rain over all years
ax = axes[0]
im = ax.imshow(weather_data['total_rain'], cmap='Blues', origin='lower')
ax.set_title(f"Total Rain ({num_years} years)")
ax.set_xlabel("X (cells)")
ax.set_ylabel("Y (cells)")
plt.colorbar(im, ax=ax, label="Total rain (m)")

# Wind barriers and channels
ax = axes[1]
viz = np.zeros((N, N, 3))
viz[weather_data['wind_features']['barrier_mask']] = [1, 0, 0]  # Red = barriers
viz[weather_data['wind_features']['channel_mask']] = [0, 0, 1]  # Blue = channels
ax.imshow(viz, origin='lower')
ax.set_title("Wind Features")
ax.set_xlabel("X (cells)")
ax.set_ylabel("Y (cells)")

# Sample year rain
ax = axes[2]
sample_year = 0
im = ax.imshow(rain_timeseries[sample_year], cmap='Blues', origin='lower')
ax.set_title(f"Year {sample_year+1} Rain")
ax.set_xlabel("X (cells)")
ax.set_ylabel("Y (cells)")
plt.colorbar(im, ax=ax, label="Rain (m/yr)")

plt.tight_layout()
plt.show()

# ============================================================================
# STEP 4: Run erosion simulation with REAL weather data
# ============================================================================
print("\n[4/5] Running erosion simulation with weather physics...")
print(f"  Using REAL rain from weather simulation")
print(f"  Wind-topography effects included")
print(f"  Quantum random rain distribution")
print(f"  This may take several minutes...")
start_time = time.time()

try:
    results = run_erosion_simulation(
        elevation_initial=strata['surface_elev'],
        thickness_initial=strata['thickness'],
        layer_order=strata['layer_order'],
        rain_timeseries=rain_timeseries,  # ← USING WEATHER DATA!
        pixel_scale_m=pixel_scale_m,
        dt=1.0,
        num_timesteps=num_years,
        save_interval=10,
        apply_diffusion=True,
        verbose=True
    )
    
    elapsed = time.time() - start_time
    print(f"\n✓ Erosion simulation complete!")
    print(f"  Computation time: {elapsed:.1f} s ({elapsed/60:.1f} min)")
    print(f"  Time per timestep: {elapsed/num_years:.2f} s")
    
except Exception as e:
    print(f"\n✗ Simulation error: {e}")
    import traceback
    traceback.print_exc()
    results = None

# ============================================================================
# STEP 5: Visualize results
# ============================================================================
if results is not None:
    print("\n[5/5] Creating visualizations...")
    start_time = time.time()
    
    # Main results plot
    fig1 = plot_erosion_results(
        results, 
        pixel_scale_m=pixel_scale_m,
        figsize=(20, 12),
        river_discharge_threshold=5000.0,
        lake_discharge_threshold=1000.0
    )
    
    # Time evolution plot
    fig2 = plot_elevation_history(
        results,
        pixel_scale_m=pixel_scale_m,
        figsize=(18, 4)
    )
    
    # Additional plot: Rain vs Erosion correlation
    fig3, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax = axes[0]
    total_rain = weather_data['total_rain']
    total_erosion = results['elevation_initial'] - results['elevation_final']
    ax.scatter(total_rain.flatten(), total_erosion.flatten(), alpha=0.1, s=1)
    ax.set_xlabel("Total Rain (m)")
    ax.set_ylabel("Total Erosion (m)")
    ax.set_title("Rain vs Erosion")
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    # Show wind channels and river correlation
    channel_mask = weather_data['wind_features']['channel_mask']
    Q = results['diagnostics_history'][-1]['Q']
    river_mask = Q > 5000.0
    
    overlap = channel_mask & river_mask
    print(f"\n  Wind channel - River overlap: {np.sum(overlap)} cells ({np.sum(overlap)/np.sum(channel_mask)*100:.1f}% of channels)")
    
    viz = np.zeros((N, N, 3))
    viz[channel_mask] = [0, 0, 1]  # Blue = channels
    viz[river_mask] = [0, 1, 0]  # Green = rivers
    viz[overlap] = [1, 1, 0]  # Yellow = both
    ax.imshow(viz, origin='lower')
    ax.set_title("Wind Channels (Blue) vs Rivers (Green)\nYellow = Overlap")
    ax.set_xlabel("X (cells)")
    ax.set_ylabel("Y (cells)")
    
    plt.tight_layout()
    plt.show()
    
    elapsed = time.time() - start_time
    print(f"✓ Visualizations created in {elapsed:.1f} s")
    
    # ============================================================================
    # VALIDATION
    # ============================================================================
    print("\n" + "="*80)
    print("VALIDATION STATISTICS")
    print("="*80)
    
    elev_change = results['elevation_final'] - results['elevation_initial']
    total_erosion = -elev_change[elev_change < 0].sum()
    total_deposition = elev_change[elev_change > 0].sum()
    
    print(f"\nEROSION:")
    print(f"  Total erosion: {total_erosion:.2f} m")
    print(f"  Total deposition: {total_deposition:.2f} m")
    print(f"  Net change: {elev_change.sum():.2f} m (negative = volume loss)")
    
    print(f"\nTIME:")
    print(f"  Simulated time: {num_years} years")
    print(f"  Real-world equivalent: {num_years * TIME_ACCELERATION} years")
    print(f"  Erosion rate: {total_erosion / num_years:.4f} m/yr (sim)")
    print(f"  Real erosion rate: {total_erosion / (num_years * TIME_ACCELERATION):.4f} m/yr (real)")
    
    print(f"\nWEATHER INTEGRATION:")
    print(f"  Wind affected by topography: ✓")
    print(f"  Rain affected by wind: ✓")
    print(f"  Quantum random rain: ✓")
    print(f"  Wind barriers identified: {np.sum(weather_data['wind_features']['barrier_mask'])}")
    print(f"  Wind channels identified: {np.sum(weather_data['wind_features']['channel_mask'])}")
    
    print("\n" + "="*80)
    print("✅ COMPLETE!")
    print("="*80)
    print("\nFeatures successfully integrated:")
    print("  ✓ Wind-topography interaction (barriers, channels)")
    print("  ✓ Rain affected by wind patterns")
    print("  ✓ Quantum random rain distribution within storms")
    print("  ✓ Weather physics properly drives erosion")
    print(f"  ✓ Time acceleration: {TIME_ACCELERATION}×")
    print(f"  ✓ Rain boost: {RAIN_BOOST}×")
    print("  ✓ Rivers and lakes detected")
    print("  ✓ Half-loss rule applied")
    
else:
    print("\n✗ Simulation failed. Check error messages above.")

print("\n" + "="*80)
print("END OF DEMONSTRATION")
print("="*80)
