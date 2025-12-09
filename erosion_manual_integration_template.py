#!/usr/bin/env python3
"""
Manual Integration Template

Copy the functions YOU need from "Rivers new" into this file,
then use them with the erosion simulation.

INSTRUCTIONS:
1. Copy your terrain generation function from Rivers new → paste below
2. Copy your weather generation function from Rivers new → paste below  
3. Run this file
"""

import numpy as np
import matplotlib.pyplot as plt
from erosion_simulation import ErosionSimulation, plot_simulation_summary


# =============================================================================
# STEP 1: COPY YOUR TERRAIN GENERATION FROM "RIVERS NEW" HERE
# =============================================================================

# TODO: Copy your quantum_seeded_topography function here
# For now, using a placeholder:

def my_terrain_function(N=128, random_seed=42):
    """
    REPLACE THIS with your actual terrain generation from Rivers new.
    
    Copy quantum_seeded_topography() and any helper functions it needs.
    """
    # Placeholder: simple fractional Brownian motion
    if random_seed:
        np.random.seed(random_seed)
    
    kx = np.fft.fftfreq(N)
    ky = np.fft.rfftfreq(N)
    K = np.sqrt(kx[:, None]**2 + ky[None, :]**2)
    K[0, 0] = np.inf
    
    amp = 1.0 / (K ** 1.5)
    phase = np.random.uniform(0, 2*np.pi, size=(N, ky.size))
    spec = amp * (np.cos(phase) + 1j*np.sin(phase))
    spec[0, 0] = 0.0
    
    z = np.fft.irfftn(spec, s=(N, N))
    z_norm = (z - z.min()) / (z.max() - z.min())
    
    return z_norm * 1500.0  # Scale to meters


# =============================================================================
# STEP 2: COPY YOUR LAYER GENERATION FROM "RIVERS NEW" HERE
# =============================================================================

# TODO: Copy your generate_stratigraphy function here
# For now, using a placeholder:

def my_layer_function(surface_elevation):
    """
    REPLACE THIS with your actual layer generation from Rivers new.
    
    Copy generate_stratigraphy() or your layer creation code.
    """
    # Placeholder: simple layers
    layer_interfaces = {
        "Topsoil": surface_elevation - 2,
        "Sandstone": surface_elevation - 50,
        "Shale": surface_elevation - 150,
        "Granite": surface_elevation - 300,
        "Basement": surface_elevation - 1000,
    }
    
    layer_order = ["Topsoil", "Sandstone", "Shale", "Granite", "Basement"]
    
    return layer_interfaces, layer_order


# =============================================================================
# STEP 3: COPY YOUR WEATHER GENERATION FROM "RIVERS NEW" HERE
# =============================================================================

# TODO: Copy your weather/storm functions here
# For now, using a placeholder:

def my_weather_function(surface_elevation, year, mean_annual_mm=1200.0):
    """
    REPLACE THIS with your actual weather generation from Rivers new.
    
    Copy these functions:
    - generate_storm_schedule_for_year()
    - generate_storm_weather_fields()
    - accumulate_rain_for_storm()
    
    Or create a simple wrapper that calls them.
    """
    # Placeholder: orographic rainfall
    elev_norm = (surface_elevation - surface_elevation.min()) / \
                (surface_elevation.max() - surface_elevation.min() + 1e-9)
    
    base_rain = mean_annual_mm * (0.5 + 0.5 * elev_norm)
    
    # Add random storm
    if np.random.random() < 0.15:
        ny, nx = surface_elevation.shape
        storm_row = np.random.randint(0, ny)
        storm_col = np.random.randint(0, nx)
        
        rows = np.arange(ny)
        cols = np.arange(nx)
        R, C = np.meshgrid(rows, cols, indexing='ij')
        
        dist_sq = (R - storm_row)**2 + (C - storm_col)**2
        storm_pattern = 5.0 * np.exp(-dist_sq / (2 * 40**2))
        
        base_rain = base_rain * (1.0 + storm_pattern)
    
    return base_rain


# =============================================================================
# STEP 4: RUN EROSION SIMULATION USING YOUR FUNCTIONS
# =============================================================================

def run_my_integrated_simulation(
    N=128,
    n_years=50,
    pixel_scale_m=100.0,
    random_seed=42
):
    """
    Run erosion simulation using YOUR terrain and weather.
    """
    
    print("\n" + "=" * 80)
    print("MY INTEGRATED EROSION SIMULATION")
    print("=" * 80)
    
    # Use YOUR terrain function
    print("\n1. Generating terrain...")
    surface_elevation = my_terrain_function(N=N, random_seed=random_seed)
    print(f"   ✓ Terrain: {surface_elevation.min():.1f} to {surface_elevation.max():.1f} m")
    
    # Use YOUR layer function
    print("\n2. Generating layers...")
    layer_interfaces, layer_order = my_layer_function(surface_elevation)
    print(f"   ✓ {len(layer_interfaces)} layers: {', '.join(layer_order)}")
    
    # Initialize erosion
    print("\n3. Initializing erosion...")
    sim = ErosionSimulation(
        surface_elevation=surface_elevation,
        layer_interfaces=layer_interfaces,
        layer_order=layer_order,
        pixel_scale_m=pixel_scale_m,
        uplift_rate=0.0001
    )
    print(f"   ✓ Initialized")
    
    # Run simulation with YOUR weather
    print(f"\n4. Running {n_years}-year simulation with YOUR weather...")
    for year in range(n_years):
        # Use YOUR weather function
        rainfall_map = my_weather_function(sim.elevation, year)
        
        # Apply erosion
        sim.step(dt=1.0, rainfall_map=rainfall_map)
        
        # Progress
        if (year + 1) % max(1, n_years // 10) == 0:
            print(f"   {100*(year+1)/n_years:5.1f}% - Year {year+1}: "
                  f"Erosion={sim.get_total_erosion()/1e6:.2f} km³, "
                  f"Rivers={np.sum(sim.river_mask)} cells")
    
    print(f"   ✓ Complete!")
    
    # Results
    print("\n5. Results:")
    print(f"   Total erosion: {sim.get_total_erosion()/1e9:.4f} km³")
    print(f"   Rivers: {np.sum(sim.river_mask)} cells")
    print(f"   Lakes: {np.sum(sim.lake_mask)} cells")
    
    # Visualize
    print("\n6. Visualization...")
    fig = plot_simulation_summary(sim)
    plt.savefig('my_integrated_erosion.png', dpi=200, bbox_inches='tight')
    print("   ✓ Saved: my_integrated_erosion.png")
    plt.show()
    
    print("\n" + "=" * 80)
    print("✓ SUCCESS")
    print("=" * 80)
    
    return sim


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("MANUAL INTEGRATION TEMPLATE")
    print("=" * 80)
    print("\nINSTRUCTIONS:")
    print("1. Edit this file")
    print("2. Copy your terrain function from Rivers new → my_terrain_function()")
    print("3. Copy your layer function from Rivers new → my_layer_function()")
    print("4. Copy your weather function from Rivers new → my_weather_function()")
    print("5. Run: python3 erosion_manual_integration_template.py")
    print("\n" + "=" * 80)
    
    # Run with current (placeholder) functions
    sim = run_my_integrated_simulation(
        N=128,
        n_years=50,
        pixel_scale_m=100.0,
        random_seed=42
    )
    
    print("\n✓ Template works! Now customize it with YOUR functions.")
