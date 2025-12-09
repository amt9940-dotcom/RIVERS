#!/usr/bin/env python3
"""
Simple Working Erosion Simulation

This version works WITHOUT trying to import from Rivers new.
It uses the erosion simulation with realistic (but simplified) weather patterns.

If you want to integrate with YOUR Rivers new weather later, you can
modify this template.
"""

import numpy as np
import matplotlib.pyplot as plt
from erosion_simulation import (
    ErosionSimulation,
    plot_simulation_summary
)

# Try to import terrain generation from Rivers new
# If it fails, use fallback
try:
    # This is a safer way to import specific functions
    import sys
    from pathlib import Path
    
    # You would manually import the functions you need here
    # For now, we use a simple fallback
    USE_QUANTUM_TERRAIN = False
    
except:
    USE_QUANTUM_TERRAIN = False


def simple_terrain_generator(N=256, seed=42):
    """Generate realistic terrain using fractional Brownian motion."""
    if seed is not None:
        np.random.seed(seed)
    
    # Generate power-law spectrum terrain
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
    
    # Add mountain features
    xx, yy = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    mountains = 0.3 * np.exp(-2 * (xx**2 + yy**2))
    
    z = 0.7 * z + 0.3 * mountains
    
    # Scale to realistic elevations
    return z * 1500.0


def generate_simple_layers(surface_elevation):
    """Generate simplified but realistic layer stack."""
    ny, nx = surface_elevation.shape
    
    # Generate noise for thickness variation
    def noise_field():
        kx = np.fft.fftfreq(nx)
        ky = np.fft.rfftfreq(ny)
        K = np.sqrt(kx[:, None]**2 + ky[None, :]**2)
        K[0, 0] = np.inf
        amp = 1.0 / (K ** 1.5)
        phase = np.random.uniform(0, 2*np.pi, size=(ny, ky.size))
        spec = amp * (np.cos(phase) + 1j*np.sin(phase))
        spec[0, 0] = 0.0
        z = np.fft.irfftn(spec, s=(ny, nx))
        return (z - z.min()) / (z.max() - z.min() + 1e-9)
    
    # Compute slope factor
    dy, dx = np.gradient(surface_elevation)
    slope = np.sqrt(dx**2 + dy**2)
    slope_factor = np.clip(1.0 - slope / 0.3, 0.2, 1.0)
    
    # Elevation factor (thicker sediments in valleys)
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
    
    # Topsoil (0.5-2m)
    thickness = (0.5 + 1.5 * noise_field()) * slope_factor
    current_elev = current_elev - thickness
    layer_interfaces["Topsoil"] = current_elev.copy()
    
    # Subsoil (1-4m)
    thickness = (1.0 + 3.0 * noise_field()) * slope_factor
    current_elev = current_elev - thickness
    layer_interfaces["Subsoil"] = current_elev.copy()
    
    # Colluvium (2-10m)
    thickness = (2.0 + 8.0 * noise_field()) * slope_factor * valley_factor
    current_elev = current_elev - thickness
    layer_interfaces["Colluvium"] = current_elev.copy()
    
    # Saprolite (5-20m)
    thickness = 5.0 + 15.0 * noise_field()
    current_elev = current_elev - thickness
    layer_interfaces["Saprolite"] = current_elev.copy()
    
    # Weathered bedrock (10-30m)
    thickness = 10.0 + 20.0 * noise_field()
    current_elev = current_elev - thickness
    layer_interfaces["WeatheredBR"] = current_elev.copy()
    
    # Sandstone (20-80m)
    thickness = 20.0 + 60.0 * noise_field()
    current_elev = current_elev - thickness
    layer_interfaces["Sandstone"] = current_elev.copy()
    
    # Shale (30-100m)
    thickness = 30.0 + 70.0 * noise_field()
    current_elev = current_elev - thickness
    layer_interfaces["Shale"] = current_elev.copy()
    
    # Limestone (40-120m)
    thickness = 40.0 + 80.0 * noise_field()
    current_elev = current_elev - thickness
    layer_interfaces["Limestone"] = current_elev.copy()
    
    # Granite (100-400m)
    thickness = 100.0 + 300.0 * noise_field()
    current_elev = current_elev - thickness
    layer_interfaces["Granite"] = current_elev.copy()
    
    # Basement
    layer_interfaces["Basement"] = current_elev - 500.0
    
    return layer_interfaces, layer_order


def generate_realistic_rainfall(surface_elevation, pixel_scale_m, 
                                mean_annual_mm=1200.0, storm_frequency=0.15):
    """
    Generate realistic rainfall patterns with orographic effects and storms.
    
    This is simplified but realistic. You can replace this with your
    actual weather generation from Rivers new.
    """
    ny, nx = surface_elevation.shape
    
    # Orographic pattern (more rain on mountains)
    elev_norm = (surface_elevation - surface_elevation.min()) / \
                (surface_elevation.max() - surface_elevation.min() + 1e-9)
    orographic_pattern = 0.5 + 1.0 * elev_norm
    
    # Windward/leeward effects
    dy, dx = np.gradient(surface_elevation)
    aspect = np.arctan2(dy, dx)
    
    # Assume SW wind (225 degrees)
    wind_dir = np.deg2rad(225)
    windward_factor = 0.7 + 0.6 * (0.5 + 0.5 * np.cos(aspect - wind_dir))
    
    # Combined pattern
    rainfall_pattern = orographic_pattern * windward_factor
    rainfall_pattern = rainfall_pattern / rainfall_pattern.mean()
    
    def rainfall_for_year(year, rng=None):
        """Generate rainfall for one year."""
        if rng is None:
            rng = np.random.default_rng(int(year * 1000))
        
        # Base rainfall
        year_rainfall = mean_annual_mm * rainfall_pattern
        
        # Add storms
        if rng.random() < storm_frequency:
            # Storm location
            storm_row = rng.integers(0, ny)
            storm_col = rng.integers(0, nx)
            
            # Storm intensity and size
            storm_intensity = rng.uniform(3.0, 8.0)
            storm_radius = rng.uniform(25, 60)
            
            # Create storm pattern
            rows = np.arange(ny)
            cols = np.arange(nx)
            R, C = np.meshgrid(rows, cols, indexing='ij')
            
            dist_sq = (R - storm_row)**2 + (C - storm_col)**2
            storm_pattern = storm_intensity * np.exp(-dist_sq / (2 * storm_radius**2))
            
            year_rainfall = year_rainfall * (1.0 + storm_pattern)
        
        # Add variability
        noise = 0.8 + 0.4 * rng.random(size=(ny, nx))
        year_rainfall = year_rainfall * noise
        
        return year_rainfall
    
    return rainfall_for_year


def run_simple_erosion_simulation(
    N=128,
    pixel_scale_m=100.0,
    n_years=50,
    mean_annual_rain_mm=1200.0,
    storm_frequency=0.2,
    random_seed=42,
):
    """
    Run erosion simulation with realistic but simplified weather.
    
    This works WITHOUT needing to import Rivers new.
    """
    
    print("\n" + "=" * 80)
    print("SIMPLE EROSION SIMULATION (Working Version)")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Grid: {N}×{N}")
    print(f"  Resolution: {pixel_scale_m} m/pixel")
    print(f"  Duration: {n_years} years")
    print(f"  Mean rainfall: {mean_annual_rain_mm} mm/year")
    print(f"  Storm frequency: {storm_frequency} per year")
    
    # Generate terrain
    print("\n1. Generating terrain...")
    surface_elevation = simple_terrain_generator(N=N, seed=random_seed)
    print(f"   ✓ Elevation range: {surface_elevation.min():.1f} to {surface_elevation.max():.1f} m")
    
    # Generate layers
    print("\n2. Generating geological layers...")
    layer_interfaces, layer_order = generate_simple_layers(surface_elevation)
    print(f"   ✓ Created {len(layer_interfaces)} layers")
    print(f"   Layers: {', '.join(layer_order)}")
    
    # Setup rainfall generator
    print("\n3. Setting up rainfall generator...")
    rainfall_generator = generate_realistic_rainfall(
        surface_elevation,
        pixel_scale_m,
        mean_annual_rain_mm,
        storm_frequency
    )
    print(f"   ✓ Rainfall generator ready")
    
    # Initialize erosion simulation
    print("\n4. Initializing erosion simulation...")
    sim = ErosionSimulation(
        surface_elevation=surface_elevation,
        layer_interfaces=layer_interfaces,
        layer_order=layer_order,
        pixel_scale_m=pixel_scale_m,
        uplift_rate=0.0001  # 0.1 mm/year
    )
    print(f"   ✓ Simulation initialized")
    
    # Run simulation
    print(f"\n5. Running {n_years}-year simulation...")
    
    for year in range(n_years):
        # Generate rainfall for this year
        rainfall_map = rainfall_generator(year)
        
        # Apply erosion
        sim.step(dt=1.0, rainfall_map=rainfall_map)
        
        # Progress report
        if (year + 1) % max(1, n_years // 10) == 0:
            progress = 100 * (year + 1) / n_years
            print(f"   {progress:5.1f}% - Year {year+1:3d}: "
                  f"Erosion={sim.get_total_erosion()/1e6:.2f} km³, "
                  f"Rivers={np.sum(sim.river_mask):4d} cells, "
                  f"Lakes={np.sum(sim.lake_mask):3d} cells")
    
    print(f"   ✓ Simulation complete!")
    
    # Results
    print("\n6. Final Results:")
    print(f"   Total erosion: {sim.get_total_erosion()/1e9:.4f} km³")
    print(f"   Total deposition: {sim.get_total_deposition()/1e9:.4f} km³")
    print(f"   River cells: {np.sum(sim.river_mask)} ({100*np.sum(sim.river_mask)/sim.river_mask.size:.2f}%)")
    print(f"   Lake cells: {np.sum(sim.lake_mask)} ({100*np.sum(sim.lake_mask)/sim.lake_mask.size:.2f}%)")
    print(f"   Mean elevation change: {(sim.elevation - surface_elevation).mean():.2f} m")
    
    # Visualization
    print("\n7. Creating visualization...")
    fig = plot_simulation_summary(sim)
    plt.savefig('erosion_simple_working.png', dpi=200, bbox_inches='tight')
    print("   ✓ Saved: erosion_simple_working.png")
    plt.show()
    
    print("\n" + "=" * 80)
    print("✓ SUCCESS")
    print("=" * 80)
    
    return sim


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("SIMPLE WORKING EROSION SIMULATION")
    print("=" * 80)
    print("\nThis version works WITHOUT importing from 'Rivers new'.")
    print("It uses realistic (but simplified) weather patterns.")
    print("\nIf you want to use YOUR Rivers new weather, you'll need to")
    print("manually integrate the specific functions you need.")
    
    try:
        sim = run_simple_erosion_simulation(
            N=128,                      # Grid size
            pixel_scale_m=100.0,        # Resolution
            n_years=50,                 # Duration
            mean_annual_rain_mm=1200.0, # Rainfall
            storm_frequency=0.2,        # Storms per year
            random_seed=42              # Reproducible
        )
        
        print("\n✓ Simulation completed successfully!")
        print("\nOutput: erosion_simple_working.png")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
