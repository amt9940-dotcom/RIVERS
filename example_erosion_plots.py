"""
Example: Erosion Plotting Functions

This script demonstrates the new erosion visualization functions:
- plot_erosion_analysis(): Comprehensive multi-panel erosion analysis
- plot_erosion_rate_map(): Erosion rate with optional river overlay
"""

import numpy as np
import matplotlib.pyplot as plt

# Import the landscape evolution framework
from landscape_evolution import (
    WorldState,
    LandscapeEvolutionSimulator,
    TectonicUplift,
    WeatherGenerator,
    FlowRouter,
    plot_erosion_analysis,
    plot_erosion_rate_map
)
from landscape_evolution.terrain_generation import (
    quantum_seeded_topography,
    denormalize_elevation
)
from landscape_evolution.initial_stratigraphy import (
    create_slope_dependent_stratigraphy
)


def main():
    """Run a simple simulation and demonstrate erosion plotting."""
    
    print("="*60)
    print("Landscape Evolution Simulator - Erosion Plotting Demo")
    print("="*60)
    
    # Parameters
    N = 256
    pixel_scale_m = 100.0
    elev_range_m = (0.0, 1000.0)
    
    print("\n1. Generating initial terrain...")
    z_norm, rng = quantum_seeded_topography(
        N=N,
        beta=3.1,
        warp_amp=0.12,
        ridged_alpha=0.18,
        random_seed=42
    )
    surface_elev = denormalize_elevation(z_norm, elev_range_m)
    print(f"   Terrain: {N}×{N}, elevation range: [{surface_elev.min():.1f}, {surface_elev.max():.1f}] m")
    
    # Define layers
    layer_names = [
        "Topsoil",
        "Colluvium", 
        "Saprolite",
        "WeatheredBR",
        "Sandstone",
        "Shale",
        "Basement"
    ]
    
    print("\n2. Initializing world state with stratigraphy...")
    world = WorldState(N, N, pixel_scale_m, layer_names)
    create_slope_dependent_stratigraphy(
        world,
        surface_elev=surface_elev,
        pixel_scale_m=pixel_scale_m,
        base_regolith_m=2.0,
        base_saprolite_m=5.0,
        bedrock_thickness_m=100.0
    )
    
    print("\n3. Setting up external forcing...")
    # Tectonic uplift
    tectonics = TectonicUplift(N, N, pixel_scale_m)
    tectonics.set_uniform_uplift(1e-3)  # 1 mm/yr
    
    # Weather/climate
    weather = WeatherGenerator(
        N, N, pixel_scale_m,
        mean_annual_precip_m=1.0,
        wind_direction_deg=270.0,
        orographic_factor=0.5
    )
    
    print("\n4. Creating simulator...")
    simulator = LandscapeEvolutionSimulator(
        world=world,
        tectonics=tectonics,
        weather=weather,
        snapshot_interval=50,
        verbose=True
    )
    
    print("\n5. Running simulation...")
    total_time = 5000.0  # 5,000 years
    dt = 10.0  # 10-year time steps
    
    history = simulator.run(total_time=total_time, dt=dt)
    
    print("\n6. Computing flow routing for river overlay...")
    flow_router = FlowRouter(pixel_scale_m)
    flow_dir, slope, flow_accum = flow_router.compute_flow(world.surface_elev)
    
    print("\n7. Creating erosion plots...\n")
    
    # Get cumulative erosion from history
    cumulative_erosion = history.get_total_erosion()
    
    # Plot 1: Comprehensive erosion analysis
    print("   Plotting comprehensive erosion analysis...")
    plot_erosion_analysis(
        erosion=cumulative_erosion,
        surface_elev=world.surface_elev,
        pixel_scale_m=pixel_scale_m,
        row_for_profile=N//2,
        save_path='/workspace/erosion_analysis.png'
    )
    
    # Plot 2: Erosion rate map (approximate from cumulative)
    # Note: In a real application, you'd track instantaneous rates
    print("\n   Plotting erosion rate map with rivers...")
    erosion_rate = cumulative_erosion / total_time  # Approximate average rate
    plot_erosion_rate_map(
        erosion_rate=erosion_rate,
        pixel_scale_m=pixel_scale_m,
        flow_accum=flow_accum,
        save_path='/workspace/erosion_rate_map.png'
    )
    
    print("\n" + "="*60)
    print("Done! Plots saved to:")
    print("  - /workspace/erosion_analysis.png")
    print("  - /workspace/erosion_rate_map.png")
    print("="*60)


if __name__ == "__main__":
    main()
