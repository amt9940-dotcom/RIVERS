#!/usr/bin/env python3
"""
Example: Comprehensive Erosion Simulation

This script demonstrates how to use the erosion simulation with the quantum-seeded
terrain generation, multi-layer stratigraphy, and weather systems from "Rivers new".

It creates a complete erosion model that:
1. Generates realistic terrain with multiple geological layers
2. Simulates weather patterns with storms and rainfall
3. Erodes the terrain realistically based on material properties
4. Forms rivers and lakes over time
5. Produces visualizations of the evolving landscape
"""

import numpy as np
import matplotlib.pyplot as plt
from erosion_simulation import (
    ErosionSimulation,
    run_erosion_simulation,
    plot_simulation_summary,
    plot_topography,
    ERODIBILITY
)

# Import terrain generation from "Rivers new" if available
# (These are simplified versions if the full module isn't imported)

def simple_terrain_generator(N=256, seed=None):
    """
    Simple terrain generator (fallback if quantum seeding not available).
    Creates realistic multi-scale terrain.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate fractional Brownian motion terrain
    kx = np.fft.fftfreq(N)
    ky = np.fft.rfftfreq(N)
    K = np.sqrt(kx[:, None]**2 + ky[None, :]**2)
    K[0, 0] = np.inf
    
    beta = 3.0  # power law exponent
    amp = 1.0 / (K ** (beta/2))
    phase = np.random.uniform(0, 2*np.pi, size=(N, ky.size))
    spec = amp * (np.cos(phase) + 1j*np.sin(phase))
    spec[0, 0] = 0.0
    
    z = np.fft.irfftn(spec, s=(N, N))
    
    # Normalize and scale
    z = (z - z.min()) / (z.max() - z.min())
    
    # Add some mountain features
    xx, yy = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    mountains = 0.3 * np.exp(-2 * (xx**2 + yy**2))
    
    z = 0.7 * z + 0.3 * mountains
    
    # Scale to realistic elevations (0-1200m)
    z = z * 1200.0
    
    return z


def generate_layer_stack(surface_elevation, base_depth=-1000.0):
    """
    Generate a realistic geological layer stack beneath the surface.
    
    Parameters:
    -----------
    surface_elevation : ndarray
        Surface elevation map (m)
    base_depth : float
        Depth of basement (m below sea level)
    
    Returns:
    --------
    layer_interfaces : dict
        Dictionary mapping layer name to elevation of top surface
    layer_order : list
        List of layer names from top to bottom
    """
    ny, nx = surface_elevation.shape
    
    # Layer order from top to bottom
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
    
    # Generate some noise for layer thickness variation
    def noise_field(scale=0.3):
        kx = np.fft.fftfreq(nx)
        ky = np.fft.rfftfreq(ny)
        K = np.sqrt(kx[:, None]**2 + ky[None, :]**2)
        K[0, 0] = np.inf
        amp = 1.0 / (K ** 1.5)
        phase = np.random.uniform(0, 2*np.pi, size=(ny, ky.size))
        spec = amp * (np.cos(phase) + 1j*np.sin(phase))
        spec[0, 0] = 0.0
        z = np.fft.irfftn(spec, s=(ny, nx))
        z = (z - z.min()) / (z.max() - z.min() + 1e-9)
        return z * scale
    
    # Compute slope for erosion-dependent thickness
    dy, dx = np.gradient(surface_elevation)
    slope = np.sqrt(dx**2 + dy**2)
    slope_factor = np.clip(1.0 - slope / 0.3, 0.2, 1.0)
    
    # Initialize layer interfaces
    layer_interfaces = {}
    
    # Start from surface and work down
    current_elev = surface_elevation.copy()
    
    # Surface layers (thinner, vary with slope and elevation)
    # Topsoil (0.5-2m)
    thickness = (0.5 + 1.5 * noise_field()) * slope_factor
    current_elev = current_elev - thickness
    layer_interfaces["Topsoil"] = surface_elevation.copy()
    
    # Subsoil (1-4m)
    thickness = (1.0 + 3.0 * noise_field()) * slope_factor
    current_elev = current_elev - thickness
    layer_interfaces["Subsoil"] = current_elev.copy()
    
    # Colluvium (2-10m, more in valleys)
    elev_norm = (surface_elevation - surface_elevation.min()) / (surface_elevation.max() - surface_elevation.min() + 1e-9)
    valley_factor = 1.0 + 2.0 * (1.0 - elev_norm)  # thicker in valleys
    thickness = (2.0 + 8.0 * noise_field()) * slope_factor * valley_factor
    current_elev = current_elev - thickness
    layer_interfaces["Colluvium"] = current_elev.copy()
    
    # Saprolite (5-20m)
    thickness = (5.0 + 15.0 * noise_field())
    current_elev = current_elev - thickness
    layer_interfaces["Saprolite"] = current_elev.copy()
    
    # Weathered bedrock (10-30m)
    thickness = (10.0 + 20.0 * noise_field())
    current_elev = current_elev - thickness
    layer_interfaces["WeatheredBR"] = current_elev.copy()
    
    # Sedimentary layers (more variable)
    # Sandstone (20-100m)
    thickness = (20.0 + 80.0 * noise_field(scale=0.5))
    current_elev = current_elev - thickness
    layer_interfaces["Sandstone"] = current_elev.copy()
    
    # Shale (30-100m)
    thickness = (30.0 + 70.0 * noise_field(scale=0.5))
    current_elev = current_elev - thickness
    layer_interfaces["Shale"] = current_elev.copy()
    
    # Limestone (40-120m)
    thickness = (40.0 + 80.0 * noise_field(scale=0.5))
    current_elev = current_elev - thickness
    layer_interfaces["Limestone"] = current_elev.copy()
    
    # Granite (variable, down to basement)
    thickness = np.abs(current_elev - base_depth) * (0.5 + 0.5 * noise_field(scale=0.3))
    current_elev = current_elev - thickness
    layer_interfaces["Granite"] = current_elev.copy()
    
    # Basement floor
    layer_interfaces["Basement"] = np.full_like(surface_elevation, base_depth)
    
    return layer_interfaces, layer_order


def create_rainfall_generator(
    surface_elevation,
    pixel_scale_m=100.0,
    mean_annual_rainfall_mm=1000.0,
    storm_frequency=0.1,  # storms per year
):
    """
    Create a rainfall generator function that produces spatially-varying rainfall
    based on topography and random storm events.
    
    Parameters:
    -----------
    surface_elevation : ndarray
        Terrain elevation map (m)
    pixel_scale_m : float
        Spatial resolution (m/pixel)
    mean_annual_rainfall_mm : float
        Average annual rainfall (mm/year)
    storm_frequency : float
        Average number of major storms per year
    
    Returns:
    --------
    rainfall_generator : callable
        Function that takes time (years) and returns rainfall map (mm)
    """
    ny, nx = surface_elevation.shape
    
    # Normalize elevation for orographic effects
    elev_norm = (surface_elevation - surface_elevation.min()) / \
                (surface_elevation.max() - surface_elevation.min() + 1e-9)
    
    # Base rainfall pattern (orographic: more rain at high elevations)
    base_pattern = 0.5 + 1.0 * elev_norm
    
    # Windward/leeward effects (simplified)
    dy, dx = np.gradient(surface_elevation)
    aspect = np.arctan2(dy, dx)
    
    # Assume prevailing wind from west (aspect = 0 or π)
    windward_factor = 0.7 + 0.6 * np.cos(aspect)
    
    # Combined base pattern
    base_pattern = base_pattern * windward_factor
    base_pattern = base_pattern / base_pattern.mean()
    
    def rainfall_generator(time_years):
        """
        Generate rainfall map for a given time.
        
        Parameters:
        -----------
        time_years : float
            Current simulation time in years
        
        Returns:
        --------
        rainfall_mm : ndarray
            Rainfall amount in mm for this time step
        """
        # Base rainfall (distributed according to pattern)
        base_rain = mean_annual_rainfall_mm * base_pattern
        
        # Add stochastic storm events
        # Use time as seed for reproducibility
        rng = np.random.default_rng(int(time_years * 1000))
        
        # Check if storm occurs
        if rng.random() < storm_frequency:
            # Storm center (random location)
            storm_row = rng.integers(0, ny)
            storm_col = rng.integers(0, nx)
            
            # Storm intensity (variable)
            storm_intensity = rng.uniform(2.0, 5.0)
            
            # Storm size (in pixels)
            storm_radius = rng.uniform(20, 50)
            
            # Create storm pattern (Gaussian)
            rows = np.arange(ny)
            cols = np.arange(nx)
            R, C = np.meshgrid(rows, cols, indexing='ij')
            
            dist_sq = (R - storm_row)**2 + (C - storm_col)**2
            storm_pattern = storm_intensity * np.exp(-dist_sq / (2 * storm_radius**2))
            
            # Add storm to base rainfall
            rainfall = base_rain * (1.0 + storm_pattern)
        else:
            # No storm, just base rainfall
            rainfall = base_rain.copy()
        
        # Add small-scale variability
        noise = 0.8 + 0.4 * rng.random(size=(ny, nx))
        rainfall = rainfall * noise
        
        return rainfall
    
    return rainfall_generator


# ============================================================================
# EXAMPLE 1: Basic erosion simulation with simple terrain
# ============================================================================

def example_basic_erosion(plot=True):
    """
    Basic erosion simulation example with simple synthetic terrain.
    """
    print("=" * 70)
    print("EXAMPLE 1: Basic Erosion Simulation")
    print("=" * 70)
    
    # Parameters
    N = 128  # grid size
    pixel_scale_m = 100.0  # 100m resolution
    n_years = 500.0  # simulate 500 years
    dt = 5.0  # 5 year time steps
    
    print(f"\nGenerating terrain ({N}x{N} grid, {pixel_scale_m}m resolution)...")
    
    # Generate terrain
    surface_elevation = simple_terrain_generator(N=N, seed=42)
    
    # Generate layers
    print("Generating geological layers...")
    layer_interfaces, layer_order = generate_layer_stack(surface_elevation)
    
    print(f"Layers created: {', '.join(layer_order)}")
    
    # Create rainfall generator
    rainfall_gen = create_rainfall_generator(
        surface_elevation,
        pixel_scale_m=pixel_scale_m,
        mean_annual_rainfall_mm=1200.0,
        storm_frequency=0.2
    )
    
    # Run simulation
    print(f"\nRunning simulation for {n_years} years...")
    sim = run_erosion_simulation(
        surface_elevation=surface_elevation,
        layer_interfaces=layer_interfaces,
        layer_order=layer_order,
        n_years=n_years,
        dt=dt,
        pixel_scale_m=pixel_scale_m,
        uplift_rate=0.0001,  # slow uplift: 0.1 mm/year
        plot_interval=20,  # plot every 20 steps
        rainfall_generator=rainfall_gen
    )
    
    # Final visualization
    if plot:
        print("\nCreating final summary plots...")
        fig = plot_simulation_summary(sim)
        plt.savefig('erosion_final.png', dpi=200, bbox_inches='tight')
        plt.show()
        
        # Additional detail plot: surface materials
        print("Plotting surface materials...")
        fig, ax = plt.subplots(figsize=(12, 10))
        
        materials = sim.get_surface_material()
        unique_materials = np.unique(materials)
        
        # Create color map for materials
        material_colors = {}
        cmap = plt.cm.get_cmap('tab20')
        for i, mat in enumerate(unique_materials):
            material_colors[mat] = cmap(i / len(unique_materials))
        
        # Convert to numeric for plotting
        material_numeric = np.zeros(materials.shape)
        for i, mat in enumerate(unique_materials):
            material_numeric[materials == mat] = i
        
        im = ax.imshow(
            material_numeric,
            origin='lower',
            cmap='tab20',
            extent=[0, sim.nx * pixel_scale_m / 1000,
                    0, sim.ny * pixel_scale_m / 1000]
        )
        
        # Create legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=material_colors[mat], label=mat)
                          for mat in unique_materials]
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        
        ax.set_xlabel('Distance (km)')
        ax.set_ylabel('Distance (km)')
        ax.set_title(f'Surface Geology After {n_years} Years of Erosion')
        plt.tight_layout()
        plt.savefig('surface_geology.png', dpi=200, bbox_inches='tight')
        plt.show()
    
    return sim


# ============================================================================
# EXAMPLE 2: Long-term erosion with varying climate
# ============================================================================

def example_climate_change_erosion(plot=True):
    """
    Erosion simulation with varying climate conditions over time.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Long-term Erosion with Climate Variation")
    print("=" * 70)
    
    # Parameters
    N = 128
    pixel_scale_m = 100.0
    n_years = 2000.0  # 2000 years
    dt = 10.0  # 10 year time steps
    
    print(f"\nGenerating terrain ({N}x{N} grid, {pixel_scale_m}m resolution)...")
    surface_elevation = simple_terrain_generator(N=N, seed=123)
    
    print("Generating geological layers...")
    layer_interfaces, layer_order = generate_layer_stack(surface_elevation)
    
    # Create climate-varying rainfall generator
    def climate_varying_rainfall(time_years):
        """Rainfall that varies with climate cycles."""
        base_gen = create_rainfall_generator(
            surface_elevation,
            pixel_scale_m=pixel_scale_m,
            mean_annual_rainfall_mm=1000.0,
            storm_frequency=0.15
        )
        
        # Add long-term climate cycle (500-year period)
        climate_factor = 0.7 + 0.6 * np.sin(2 * np.pi * time_years / 500.0)
        
        rainfall = base_gen(time_years)
        return rainfall * climate_factor
    
    # Run simulation
    print(f"\nRunning simulation for {n_years} years with climate variation...")
    sim = run_erosion_simulation(
        surface_elevation=surface_elevation,
        layer_interfaces=layer_interfaces,
        layer_order=layer_order,
        n_years=n_years,
        dt=dt,
        pixel_scale_m=pixel_scale_m,
        uplift_rate=0.0002,  # moderate uplift: 0.2 mm/year
        plot_interval=0,  # don't plot during simulation
        rainfall_generator=climate_varying_rainfall
    )
    
    # Final visualization
    if plot:
        print("\nCreating final summary plots...")
        fig = plot_simulation_summary(sim)
        plt.savefig('erosion_climate_final.png', dpi=200, bbox_inches='tight')
        plt.show()
    
    return sim


# ============================================================================
# EXAMPLE 3: Comparative study - different rock types
# ============================================================================

def example_rock_type_comparison(plot=True):
    """
    Compare erosion patterns in different rock types.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Rock Type Comparison")
    print("=" * 70)
    
    # Create two terrains with different dominant rock types
    N = 128
    pixel_scale_m = 100.0
    n_years = 1000.0
    dt = 5.0
    
    print("\nGenerating base terrain...")
    surface_elevation = simple_terrain_generator(N=N, seed=456)
    
    # Scenario 1: Soft sedimentary rocks (high erodibility)
    print("\nScenario 1: Soft sedimentary terrain (shale, sandstone)...")
    layer_interfaces_soft, _ = generate_layer_stack(surface_elevation)
    layer_order_soft = ["Topsoil", "Subsoil", "Shale", "Sandstone", "Basement"]
    
    rainfall_gen = create_rainfall_generator(
        surface_elevation,
        pixel_scale_m=pixel_scale_m,
        mean_annual_rainfall_mm=1200.0,
        storm_frequency=0.2
    )
    
    sim_soft = run_erosion_simulation(
        surface_elevation=surface_elevation.copy(),
        layer_interfaces=layer_interfaces_soft,
        layer_order=layer_order_soft,
        n_years=n_years,
        dt=dt,
        pixel_scale_m=pixel_scale_m,
        uplift_rate=0.0,
        plot_interval=0,
        rainfall_generator=rainfall_gen
    )
    
    # Scenario 2: Hard crystalline rocks (low erodibility)
    print("\nScenario 2: Hard crystalline terrain (granite, gneiss)...")
    layer_interfaces_hard, _ = generate_layer_stack(surface_elevation)
    layer_order_hard = ["Topsoil", "WeatheredBR", "Granite", "Basement"]
    
    sim_hard = run_erosion_simulation(
        surface_elevation=surface_elevation.copy(),
        layer_interfaces=layer_interfaces_hard,
        layer_order=layer_order_hard,
        n_years=n_years,
        dt=dt,
        pixel_scale_m=pixel_scale_m,
        uplift_rate=0.0,
        plot_interval=0,
        rainfall_generator=rainfall_gen
    )
    
    # Comparison plot
    if plot:
        print("\nCreating comparison plots...")
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Soft rocks
        plot_topography(sim_soft, title='Soft Sedimentary - Final Topography', 
                       ax=axes[0, 0])
        
        change_soft = sim_soft.elevation - sim_soft.original_elevation
        im1 = axes[0, 1].imshow(change_soft, origin='lower', cmap='RdBu',
                                vmin=-50, vmax=50)
        plt.colorbar(im1, ax=axes[0, 1], label='Elevation Change (m)')
        axes[0, 1].set_title('Soft Sedimentary - Elevation Change')
        
        axes[0, 2].text(0.5, 0.5, 
                       f'Soft Sedimentary Rocks\n\n'
                       f'Total Erosion: {sim_soft.get_total_erosion()/1e6:.2f} km³\n'
                       f'Total Deposition: {sim_soft.get_total_deposition()/1e6:.2f} km³\n'
                       f'Rivers: {np.sum(sim_soft.river_mask)} cells\n'
                       f'Lakes: {np.sum(sim_soft.lake_mask)} cells\n'
                       f'Avg. Elevation Change: {change_soft.mean():.2f} m',
                       transform=axes[0, 2].transAxes,
                       fontsize=12, verticalalignment='center',
                       horizontalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[0, 2].axis('off')
        
        # Hard rocks
        plot_topography(sim_hard, title='Hard Crystalline - Final Topography',
                       ax=axes[1, 0])
        
        change_hard = sim_hard.elevation - sim_hard.original_elevation
        im2 = axes[1, 1].imshow(change_hard, origin='lower', cmap='RdBu',
                                vmin=-50, vmax=50)
        plt.colorbar(im2, ax=axes[1, 1], label='Elevation Change (m)')
        axes[1, 1].set_title('Hard Crystalline - Elevation Change')
        
        axes[1, 2].text(0.5, 0.5,
                       f'Hard Crystalline Rocks\n\n'
                       f'Total Erosion: {sim_hard.get_total_erosion()/1e6:.2f} km³\n'
                       f'Total Deposition: {sim_hard.get_total_deposition()/1e6:.2f} km³\n'
                       f'Rivers: {np.sum(sim_hard.river_mask)} cells\n'
                       f'Lakes: {np.sum(sim_hard.lake_mask)} cells\n'
                       f'Avg. Elevation Change: {change_hard.mean():.2f} m',
                       transform=axes[1, 2].transAxes,
                       fontsize=12, verticalalignment='center',
                       horizontalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('rock_type_comparison.png', dpi=200, bbox_inches='tight')
        plt.show()
    
    return sim_soft, sim_hard


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("EROSION SIMULATION EXAMPLES")
    print("=" * 70)
    print("\nThis script demonstrates comprehensive erosion modeling with:")
    print("  • Realistic multi-layer geology")
    print("  • Spatially-varying rainfall patterns")
    print("  • Material-dependent erodibility")
    print("  • River and lake formation")
    print("  • Long-term landscape evolution")
    print("\n" + "=" * 70)
    
    # Run examples
    try:
        # Example 1: Basic erosion
        sim1 = example_basic_erosion(plot=True)
        
        # Example 2: Climate variation (commented out for speed, uncomment to run)
        # sim2 = example_climate_change_erosion(plot=True)
        
        # Example 3: Rock type comparison (commented out for speed, uncomment to run)
        # sim_soft, sim_hard = example_rock_type_comparison(plot=True)
        
        print("\n" + "=" * 70)
        print("Examples completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
