#!/usr/bin/env python3
"""
Complete Erosion Simulation with Fast Realistic Weather

This combines:
- Fast, realistic weather generation (wind-driven, orographic)
- Multi-layer erosion simulation
- River and lake formation
- Time-stepped evolution

MUCH FASTER and MORE REALISTIC than previous versions!
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add workspace to path
workspace = Path(__file__).parent
if str(workspace) not in sys.path:
    sys.path.insert(0, str(workspace))

from erosion_simulation import ErosionSimulation, plot_simulation_summary
from fast_realistic_weather import FastWeatherSystem, create_weather_system


def generate_terrain(N=128, seed=42):
    """Generate realistic terrain."""
    if seed is not None:
        np.random.seed(seed)
    
    # Fractional Brownian motion
    kx = np.fft.fftfreq(N)
    ky = np.fft.rfftfreq(N)
    K = np.sqrt(kx[:, None]**2 + ky[None, :]**2)
    K[0, 0] = np.inf
    
    beta = 3.0
    amp = 1.0 / (K ** (beta/2))
    phase = np.random.uniform(0, 2*np.pi, size=(N, ky.size))
    spec = amp * (np.cos(phase) + 1j*np.sin(phase))
    spec[0, 0] = 0.0
    
    z = np.fft.irfftn(spec, s=(N, N))
    z = (z - z.min()) / (z.max() - z.min())
    
    # Add mountain range
    xx, yy = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    mountains = 0.4 * np.exp(-3 * ((xx-0.2)**2 + yy**2))
    
    z = 0.6 * z + 0.4 * mountains
    return z * 1500.0


def generate_layers(surface_elevation):
    """Generate geological layers."""
    ny, nx = surface_elevation.shape
    
    # Slope factor
    dy, dx = np.gradient(surface_elevation)
    slope = np.sqrt(dx**2 + dy**2)
    slope_factor = np.clip(1.0 - slope / 0.3, 0.2, 1.0)
    
    # Valley factor
    elev_norm = (surface_elevation - surface_elevation.min()) / \
                (surface_elevation.max() - surface_elevation.min() + 1e-9)
    valley_factor = 1.0 + 1.5 * (1.0 - elev_norm)
    
    layer_order = [
        "Topsoil",
        "Subsoil",
        "Colluvium",
        "Saprolite",
        "WeatheredBR",
        "Sandstone",
        "Shale",
        "Limestone",
        "Granite",
        "Basement"
    ]
    
    layer_interfaces = {}
    current_elev = surface_elevation.copy()
    
    # Surface layers (thin, slope-dependent)
    for name, base_thick in [("Topsoil", 1.5), ("Subsoil", 2.5)]:
        thickness = base_thick * slope_factor
        current_elev = current_elev - thickness
        layer_interfaces[name] = current_elev.copy()
    
    # Colluvium (thicker in valleys)
    thickness = 5.0 * slope_factor * valley_factor
    current_elev = current_elev - thickness
    layer_interfaces["Colluvium"] = current_elev.copy()
    
    # Weathered layers
    for name, thick in [("Saprolite", 10), ("WeatheredBR", 20)]:
        current_elev = current_elev - thick
        layer_interfaces[name] = current_elev.copy()
    
    # Sedimentary rocks
    for name, thick in [("Sandstone", 60), ("Shale", 80), ("Limestone", 100)]:
        current_elev = current_elev - thick
        layer_interfaces[name] = current_elev.copy()
    
    # Basement
    current_elev = current_elev - 300
    layer_interfaces["Granite"] = current_elev.copy()
    layer_interfaces["Basement"] = current_elev - 500
    
    return layer_interfaces, layer_order


def run_complete_simulation(
    N: int = 128,
    pixel_scale_m: float = 100.0,
    n_years: int = 100,
    climate: str = "temperate",
    wind_from: str = "west",
    random_seed: int = 42,
    plot_interval: int = 20,
):
    """
    Run complete erosion simulation with fast realistic weather.
    
    Parameters:
    -----------
    N : int
        Grid size (N×N)
    pixel_scale_m : float
        Spatial resolution (m/pixel)
    n_years : int
        Simulation duration (years)
    climate : str
        "arid", "semi-arid", "temperate", "wet", "tropical"
    wind_from : str
        "north", "south", "east", "west", "northwest", etc.
    random_seed : int
        For reproducibility
    plot_interval : int
        Plot every N years (0 = no plots during sim)
    """
    
    print("\n" + "=" * 80)
    print("COMPLETE EROSION SIMULATION WITH FAST REALISTIC WEATHER")
    print("=" * 80)
    
    print(f"\nConfiguration:")
    print(f"  Grid size: {N}×{N} cells")
    print(f"  Resolution: {pixel_scale_m} m/pixel")
    print(f"  Domain: {N*pixel_scale_m/1000:.1f} × {N*pixel_scale_m/1000:.1f} km")
    print(f"  Duration: {n_years} years")
    print(f"  Climate: {climate}")
    print(f"  Wind from: {wind_from}")
    print(f"  Random seed: {random_seed}")
    
    # ========================================================================
    # STEP 1: Generate Terrain
    # ========================================================================
    print("\n" + "-" * 80)
    print("STEP 1: Generate Terrain")
    print("-" * 80)
    
    surface_elevation = generate_terrain(N=N, seed=random_seed)
    
    print(f"✓ Terrain generated")
    print(f"  Elevation: {surface_elevation.min():.1f} to {surface_elevation.max():.1f} m")
    print(f"  Relief: {surface_elevation.max() - surface_elevation.min():.1f} m")
    
    # ========================================================================
    # STEP 2: Generate Layers
    # ========================================================================
    print("\n" + "-" * 80)
    print("STEP 2: Generate Geological Layers")
    print("-" * 80)
    
    layer_interfaces, layer_order = generate_layers(surface_elevation)
    
    print(f"✓ {len(layer_interfaces)} layers created")
    print(f"  Layers: {', '.join(layer_order[:5])}...")
    
    # ========================================================================
    # STEP 3: Initialize Weather System
    # ========================================================================
    print("\n" + "-" * 80)
    print("STEP 3: Initialize Fast Weather System")
    print("-" * 80)
    
    weather = create_weather_system(
        terrain=surface_elevation,
        pixel_scale_m=pixel_scale_m,
        climate=climate,
        wind_from=wind_from
    )
    
    print(f"✓ Weather system initialized")
    print(f"  Mean annual rainfall: {weather.mean_annual_rain:.0f} mm/year")
    print(f"  Storm frequency: {weather.storm_frequency:.1f} storms/year")
    print(f"  Wind speed: {weather.wind_speed:.1f} m/s")
    print(f"  Wind direction: {weather.prevailing_wind_dir:.0f}° (from {wind_from})")
    
    # ========================================================================
    # STEP 4: Initialize Erosion Simulation
    # ========================================================================
    print("\n" + "-" * 80)
    print("STEP 4: Initialize Erosion Simulation")
    print("-" * 80)
    
    sim = ErosionSimulation(
        surface_elevation=surface_elevation,
        layer_interfaces=layer_interfaces,
        layer_order=layer_order,
        pixel_scale_m=pixel_scale_m,
        uplift_rate=0.0001  # 0.1 mm/year
    )
    
    print(f"✓ Erosion simulation initialized")
    
    # ========================================================================
    # STEP 5: Run Coupled Weather-Erosion Simulation
    # ========================================================================
    print("\n" + "-" * 80)
    print("STEP 5: Run Coupled Simulation")
    print("-" * 80)
    print(f"Simulating {n_years} years...")
    print()
    
    # Storage for time series
    time_data = {
        'years': [],
        'erosion': [],
        'deposition': [],
        'rivers': [],
        'lakes': [],
        'mean_rain': []
    }
    
    # Random number generator
    rng = np.random.default_rng(random_seed)
    
    for year in range(n_years):
        # Generate weather for this year
        annual_rainfall = weather.generate_annual_rainfall(year=year, rng=rng)
        
        # Apply erosion
        sim.step(dt=1.0, rainfall_map=annual_rainfall)
        
        # Update weather system with new terrain (every 5 years for speed)
        if year % 5 == 0:
            weather.update_terrain(sim.elevation)
        
        # Record data
        time_data['years'].append(year + 1)
        time_data['erosion'].append(sim.get_total_erosion() / 1e6)  # km³
        time_data['deposition'].append(sim.get_total_deposition() / 1e6)
        time_data['rivers'].append(np.sum(sim.river_mask))
        time_data['lakes'].append(np.sum(sim.lake_mask))
        time_data['mean_rain'].append(annual_rainfall.mean())
        
        # Progress report
        if (year + 1) % max(1, n_years // 10) == 0:
            progress = 100 * (year + 1) / n_years
            print(f"  {progress:5.1f}% - Year {year+1:3d}: "
                  f"Rain={annual_rainfall.mean():6.1f}mm, "
                  f"Erosion={time_data['erosion'][-1]:6.2f}km³, "
                  f"Rivers={time_data['rivers'][-1]:4d}, "
                  f"Lakes={time_data['lakes'][-1]:3d}")
        
        # Intermediate plots
        if plot_interval > 0 and (year + 1) % plot_interval == 0:
            fig = plot_simulation_summary(sim)
            plt.suptitle(f'Year {year+1}/{n_years}', fontsize=16)
            plt.savefig(f'erosion_year{year+1:04d}.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
    
    print(f"\n✓ Simulation complete!")
    
    # ========================================================================
    # STEP 6: Final Results and Visualization
    # ========================================================================
    print("\n" + "-" * 80)
    print("STEP 6: Final Results")
    print("-" * 80)
    
    print(f"\nFinal Statistics:")
    print(f"  Duration: {sim.current_time:.1f} years")
    print(f"  Total erosion: {sim.get_total_erosion()/1e9:.4f} km³")
    print(f"  Total deposition: {sim.get_total_deposition()/1e9:.4f} km³")
    print(f"  River cells: {np.sum(sim.river_mask)} ({100*np.sum(sim.river_mask)/sim.river_mask.size:.2f}%)")
    print(f"  Lake cells: {np.sum(sim.lake_mask)} ({100*np.sum(sim.lake_mask)/sim.lake_mask.size:.2f}%)")
    print(f"  Mean elevation change: {(sim.elevation - surface_elevation).mean():.2f} m")
    print(f"  Max erosion: {(surface_elevation - sim.elevation).max():.2f} m")
    
    # Create comprehensive visualization
    print("\nCreating visualizations...")
    
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Row 1: Terrain evolution
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(surface_elevation, origin='lower', cmap='terrain')
    plt.colorbar(im1, ax=ax1, label='Elevation (m)', shrink=0.8)
    ax1.set_title('Initial Terrain')
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(sim.elevation, origin='lower', cmap='terrain')
    plt.colorbar(im2, ax=ax2, label='Elevation (m)', shrink=0.8)
    ax2.set_title(f'Final Terrain (Year {n_years})')
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Y (pixels)')
    
    ax3 = fig.add_subplot(gs[0, 2])
    change = sim.elevation - surface_elevation
    vmax = np.abs(change).max()
    im3 = ax3.imshow(change, origin='lower', cmap='RdBu', vmin=-vmax, vmax=vmax)
    plt.colorbar(im3, ax=ax3, label='Change (m)', shrink=0.8)
    ax3.set_title('Elevation Change\n(Red=Dep, Blue=Erosion)')
    ax3.set_xlabel('X (pixels)')
    ax3.set_ylabel('Y (pixels)')
    
    # Row 2: Weather and water features
    ax4 = fig.add_subplot(gs[1, 0])
    # Show orographic rainfall pattern
    demo_rain = weather.generate_base_rainfall_pattern()
    im4 = ax4.imshow(demo_rain, origin='lower', cmap='Blues')
    plt.colorbar(im4, ax=ax4, label='Multiplier', shrink=0.8)
    ax4.set_title('Orographic Rainfall Pattern\n(Wind-Driven)')
    ax4.set_xlabel('X (pixels)')
    ax4.set_ylabel('Y (pixels)')
    
    # Add wind arrow
    wind_rad = np.deg2rad(weather.prevailing_wind_dir)
    arrow_len = N * 0.2
    arrow_x = N/2 - arrow_len * np.cos(wind_rad)
    arrow_y = N/2 - arrow_len * np.sin(wind_rad)
    ax4.arrow(arrow_x, arrow_y, 
             arrow_len * np.cos(wind_rad), arrow_len * np.sin(wind_rad),
             color='red', width=2, head_width=8, head_length=6, 
             length_includes_head=True, label='Wind')
    
    ax5 = fig.add_subplot(gs[1, 1])
    flow_log = np.log10(sim.flow_accumulation + 1)
    im5 = ax5.imshow(flow_log, origin='lower', cmap='viridis')
    plt.colorbar(im5, ax=ax5, label='log10(Flow)', shrink=0.8)
    ax5.set_title('Drainage Network')
    ax5.set_xlabel('X (pixels)')
    ax5.set_ylabel('Y (pixels)')
    
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.imshow(sim.elevation, origin='lower', cmap='gray', alpha=0.3)
    water = np.zeros_like(sim.elevation)
    water[sim.river_mask] = 1
    water[sim.lake_mask] = 2
    if np.any(water > 0):
        water_masked = np.ma.masked_where(water == 0, water)
        im6 = ax6.imshow(water_masked, origin='lower', cmap='Blues', alpha=0.7)
        plt.colorbar(im6, ax=ax6, label='Water\n1=River\n2=Lake', shrink=0.8, ticks=[1, 2])
    ax6.set_title('Rivers and Lakes')
    ax6.set_xlabel('X (pixels)')
    ax6.set_ylabel('Y (pixels)')
    
    # Row 3: Time series
    ax7 = fig.add_subplot(gs[2, :2])
    ax7_twin = ax7.twinx()
    
    line1 = ax7.plot(time_data['years'], time_data['erosion'],
                    'b-', linewidth=2, label='Total Erosion')
    line2 = ax7.plot(time_data['years'], time_data['deposition'],
                    'r-', linewidth=2, label='Total Deposition')
    line3 = ax7_twin.plot(time_data['years'], time_data['mean_rain'],
                          'g-', linewidth=2, alpha=0.7, label='Annual Rainfall')
    
    ax7.set_xlabel('Year', fontsize=11)
    ax7.set_ylabel('Volume (km³)', fontsize=11)
    ax7_twin.set_ylabel('Rainfall (mm/year)', fontsize=11, color='g')
    ax7.set_title('Erosion and Climate Over Time', fontsize=12)
    ax7.grid(True, alpha=0.3)
    
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax7.legend(lines, labels, loc='upper left')
    
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.plot(time_data['years'], time_data['rivers'],
            'b-', linewidth=2, label='River cells')
    ax8.plot(time_data['years'], time_data['lakes'],
            'c-', linewidth=2, label='Lake cells')
    ax8.set_xlabel('Year', fontsize=11)
    ax8.set_ylabel('Number of cells', fontsize=11)
    ax8.set_title('Water Feature Evolution', fontsize=12)
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    plt.suptitle(f'Complete Erosion Simulation with Fast Realistic Weather\n'
                f'{n_years} years, {N}×{N} grid, {climate} climate, wind from {wind_from}',
                fontsize=14, fontweight='bold', y=0.995)
    
    plt.savefig('complete_erosion_final.png', dpi=200, bbox_inches='tight')
    print("✓ Saved: complete_erosion_final.png")
    
    plt.show()
    
    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE!")
    print("=" * 80)
    print("\n✓ Features used:")
    print("  • Fast realistic weather (wind-driven storms)")
    print("  • Orographic precipitation (mountains get more rain)")
    print("  • Rain shadows (leeward sides get less)")
    print("  • Topographic steering (terrain deflects storms)")
    print("  • Multi-layer erosion (different rock types)")
    print("  • River and lake formation")
    
    return sim, weather, time_data


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("FAST REALISTIC WEATHER + EROSION SIMULATION")
    print("=" * 80)
    print("\nThis simulation features:")
    print("  ✓ FAST weather generation (vectorized, optimized)")
    print("  ✓ REALISTIC wind-topography interactions")
    print("  ✓ Orographic lift and rain shadows")
    print("  ✓ Storm tracks affected by terrain")
    print("  ✓ Multi-layer erosion")
    print("  ✓ River and lake formation")
    
    # Run simulation
    try:
        sim, weather, time_data = run_complete_simulation(
            N=128,                  # Grid size (128=fast, 256=detailed)
            pixel_scale_m=100.0,    # 100m resolution
            n_years=100,            # 100 years
            climate="temperate",    # "arid", "semi-arid", "temperate", "wet", "tropical"
            wind_from="west",       # Wind direction
            random_seed=42,         # Reproducible
            plot_interval=0         # Plot interval (0=only final)
        )
        
        print("\n✓ SUCCESS!")
        print("\nOutput: complete_erosion_final.png")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
