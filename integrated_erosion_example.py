#!/usr/bin/env python3
"""
Integrated Erosion Simulation with Rivers New Components

This script integrates the erosion simulation with the quantum-seeded terrain generation,
multi-layer stratigraphy, and weather systems from "Rivers new".

It demonstrates:
1. Using quantum-seeded terrain generation
2. Full stratigraphy with realistic geological layers
3. Weather-driven erosion with storm patterns
4. River and lake formation from realistic rainfall
5. Time-stepped simulation with visualization
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import our erosion simulation
from erosion_simulation import (
    ErosionSimulation,
    plot_simulation_summary,
    plot_topography,
    ERODIBILITY
)

# Try to import components from "Rivers new"
try:
    # Add the workspace directory to path if needed
    workspace_path = Path(__file__).parent
    if str(workspace_path) not in sys.path:
        sys.path.insert(0, str(workspace_path))
    
    # Import the key functions from Rivers new
    # Note: You may need to adjust these imports based on the actual structure
    print("Attempting to import from 'Rivers new'...")
    
    # We'll use exec to load functions from the file
    with open(workspace_path / "Rivers new", "r") as f:
        rivers_code = f.read()
    
    # Extract key functions using exec in a namespace
    rivers_namespace = {}
    exec(rivers_code, rivers_namespace)
    
    # Get the functions we need
    quantum_seeded_topography = rivers_namespace.get('quantum_seeded_topography')
    generate_stratigraphy = rivers_namespace.get('generate_stratigraphy')
    generate_storm_weather_fields = rivers_namespace.get('generate_storm_weather_fields')
    
    RIVERS_AVAILABLE = quantum_seeded_topography is not None
    
    if RIVERS_AVAILABLE:
        print("✓ Successfully imported from 'Rivers new'")
    else:
        print("✗ Could not find required functions in 'Rivers new'")
        RIVERS_AVAILABLE = False
        
except Exception as e:
    print(f"✗ Could not import from 'Rivers new': {e}")
    RIVERS_AVAILABLE = False
    quantum_seeded_topography = None
    generate_stratigraphy = None
    generate_storm_weather_fields = None


# ============================================================================
# INTEGRATED SIMULATION
# ============================================================================

def run_integrated_erosion_simulation(
    N: int = 256,
    pixel_scale_m: float = 100.0,
    n_years: float = 1000.0,
    dt: float = 1.0,
    use_quantum_terrain: bool = True,
    random_seed: int = None,
    save_plots: bool = True,
):
    """
    Run a complete integrated erosion simulation using all available components.
    
    Parameters:
    -----------
    N : int
        Grid size (N x N)
    pixel_scale_m : float
        Spatial resolution in meters per pixel
    n_years : float
        Total simulation time in years
    dt : float
        Time step in years
    use_quantum_terrain : bool
        Use quantum-seeded terrain if available
    random_seed : int
        Random seed for reproducibility
    save_plots : bool
        Save plots to files
    """
    print("\n" + "=" * 80)
    print("INTEGRATED EROSION SIMULATION")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Grid size: {N} x {N}")
    print(f"  Resolution: {pixel_scale_m} m/pixel")
    print(f"  Domain size: {N * pixel_scale_m / 1000:.1f} x {N * pixel_scale_m / 1000:.1f} km")
    print(f"  Simulation time: {n_years} years")
    print(f"  Time step: {dt} years")
    print(f"  Random seed: {random_seed if random_seed else 'None (random)'}")
    
    # ========================================================================
    # STEP 1: Generate terrain
    # ========================================================================
    print("\n" + "-" * 80)
    print("STEP 1: Terrain Generation")
    print("-" * 80)
    
    if use_quantum_terrain and RIVERS_AVAILABLE and quantum_seeded_topography is not None:
        print("Using quantum-seeded terrain generation...")
        try:
            # Generate quantum terrain
            z_norm, rng = quantum_seeded_topography(
                N=N,
                beta=3.1,
                warp_amp=0.12,
                ridged_alpha=0.18,
                random_seed=random_seed
            )
            
            # Scale to realistic elevations
            # Map [0, 1] to elevation range
            elev_min = 0.0  # sea level
            elev_max = 1500.0  # max elevation
            surface_elevation = elev_min + (elev_max - elev_min) * z_norm
            
            print(f"✓ Generated quantum terrain: {surface_elevation.min():.1f} to {surface_elevation.max():.1f} m")
            
        except Exception as e:
            print(f"✗ Quantum terrain generation failed: {e}")
            print("Falling back to classical terrain generation...")
            use_quantum_terrain = False
    
    if not use_quantum_terrain or not RIVERS_AVAILABLE:
        print("Using classical terrain generation...")
        if random_seed:
            np.random.seed(random_seed)
        
        # Generate fractional Brownian motion terrain
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
        
        # Scale to elevations
        surface_elevation = z * 1500.0
        
        print(f"✓ Generated classical terrain: {surface_elevation.min():.1f} to {surface_elevation.max():.1f} m")
    
    # ========================================================================
    # STEP 2: Generate geological layers
    # ========================================================================
    print("\n" + "-" * 80)
    print("STEP 2: Geological Layer Generation")
    print("-" * 80)
    
    # Generate layer stack
    ny, nx = surface_elevation.shape
    
    # Layer order from top to bottom
    layer_order = [
        "Topsoil",
        "Subsoil",
        "Colluvium",
        "Saprolite",
        "WeatheredBR",
        "Clay",
        "Silt",
        "Sand",
        "Sandstone",
        "Conglomerate",
        "Shale",
        "Mudstone",
        "Limestone",
        "Granite",
        "Gneiss",
        "Basement"
    ]
    
    print(f"Creating {len(layer_order)} geological layers...")
    
    # Generate noise for layer thickness variation
    def noise_field(scale=0.3, seed_offset=0):
        if random_seed:
            np.random.seed(random_seed + seed_offset)
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
    
    # Compute slope-dependent thickness factor
    dy, dx = np.gradient(surface_elevation)
    slope = np.sqrt(dx**2 + dy**2)
    slope_factor = np.clip(1.0 - slope / 0.3, 0.2, 1.0)
    
    # Elevation-dependent factor (thicker sediments in valleys)
    elev_norm = (surface_elevation - surface_elevation.min()) / \
                (surface_elevation.max() - surface_elevation.min() + 1e-9)
    valley_factor = 1.0 + 1.5 * (1.0 - elev_norm)
    
    # Initialize layer interfaces
    layer_interfaces = {}
    current_elev = surface_elevation.copy()
    
    # Create layers with varying thickness
    layer_thicknesses = {
        "Topsoil": (0.5, 2.0),
        "Subsoil": (1.0, 4.0),
        "Colluvium": (2.0, 10.0),
        "Saprolite": (5.0, 20.0),
        "WeatheredBR": (10.0, 30.0),
        "Clay": (2.0, 8.0),
        "Silt": (3.0, 10.0),
        "Sand": (2.0, 8.0),
        "Sandstone": (20.0, 80.0),
        "Conglomerate": (15.0, 60.0),
        "Shale": (30.0, 100.0),
        "Mudstone": (20.0, 80.0),
        "Limestone": (40.0, 120.0),
        "Granite": (100.0, 400.0),
        "Gneiss": (100.0, 300.0),
        "Basement": (500.0, 500.0),  # thick basement
    }
    
    for i, layer_name in enumerate(layer_order):
        if layer_name in layer_thicknesses:
            min_thick, max_thick = layer_thicknesses[layer_name]
            
            # Apply different factors for different layer types
            if layer_name in ["Topsoil", "Subsoil", "Colluvium"]:
                # Surface layers: affected by slope and valleys
                thickness = (min_thick + (max_thick - min_thick) * noise_field(seed_offset=i)) * \
                           slope_factor * valley_factor
            elif layer_name in ["Clay", "Silt", "Sand"]:
                # Unconsolidated sediments: prefer valleys
                thickness = (min_thick + (max_thick - min_thick) * noise_field(seed_offset=i)) * \
                           valley_factor
            else:
                # Deeper layers: more uniform but with some variation
                thickness = min_thick + (max_thick - min_thick) * noise_field(scale=0.5, seed_offset=i)
            
            current_elev = current_elev - thickness
            layer_interfaces[layer_name] = current_elev.copy()
    
    print(f"✓ Created {len(layer_interfaces)} layer interfaces")
    
    # Print layer properties
    print("\nLayer erodibility coefficients:")
    for layer in layer_order[:10]:  # Show first 10
        if layer in ERODIBILITY:
            print(f"  {layer:15s}: K = {ERODIBILITY[layer]:.6f}")
    print("  ...")
    
    # ========================================================================
    # STEP 3: Setup weather/rainfall system
    # ========================================================================
    print("\n" + "-" * 80)
    print("STEP 3: Weather System Setup")
    print("-" * 80)
    
    # Create rainfall generator with orographic effects
    mean_annual_rainfall_mm = 1200.0
    storm_frequency_per_year = 0.15
    
    # Orographic rainfall pattern (more rain on windward slopes)
    elev_norm = (surface_elevation - surface_elevation.min()) / \
                (surface_elevation.max() - surface_elevation.min() + 1e-9)
    
    # Base pattern: higher elevations get more rain
    orographic_pattern = 0.5 + 1.0 * elev_norm
    
    # Add windward/leeward effect
    dy, dx = np.gradient(surface_elevation)
    aspect = np.arctan2(dy, dx)
    
    # Assume prevailing wind from SW (225°)
    wind_dir = np.deg2rad(225)
    windward_factor = 0.7 + 0.6 * (0.5 + 0.5 * np.cos(aspect - wind_dir))
    
    # Combined rainfall pattern
    rainfall_pattern = orographic_pattern * windward_factor
    rainfall_pattern = rainfall_pattern / rainfall_pattern.mean()
    
    print(f"Mean annual rainfall: {mean_annual_rainfall_mm} mm/year")
    print(f"Storm frequency: {storm_frequency_per_year} storms/year")
    print(f"Rainfall multiplier range: {rainfall_pattern.min():.2f} to {rainfall_pattern.max():.2f}")
    
    def rainfall_generator(time_years):
        """Generate spatially-varying rainfall for a given time."""
        if random_seed:
            rng = np.random.default_rng(int(random_seed * 1000 + time_years * 1000))
        else:
            rng = np.random.default_rng(int(time_years * 1000))
        
        # Base rainfall
        base_rain = mean_annual_rainfall_mm * rainfall_pattern * dt
        
        # Add storm events
        if rng.random() < storm_frequency_per_year * dt:
            # Storm parameters
            storm_row = rng.integers(0, ny)
            storm_col = rng.integers(0, nx)
            storm_intensity = rng.uniform(3.0, 8.0)
            storm_radius = rng.uniform(25, 60)
            
            # Storm pattern
            rows = np.arange(ny)
            cols = np.arange(nx)
            R, C = np.meshgrid(rows, cols, indexing='ij')
            
            dist_sq = (R - storm_row)**2 + (C - storm_col)**2
            storm_pattern = storm_intensity * np.exp(-dist_sq / (2 * storm_radius**2))
            
            rainfall = base_rain * (1.0 + storm_pattern)
        else:
            rainfall = base_rain
        
        # Add small-scale variability
        noise = 0.8 + 0.4 * rng.random(size=(ny, nx))
        rainfall = rainfall * noise
        
        return rainfall
    
    # ========================================================================
    # STEP 4: Initialize erosion simulation
    # ========================================================================
    print("\n" + "-" * 80)
    print("STEP 4: Initialize Erosion Simulation")
    print("-" * 80)
    
    sim = ErosionSimulation(
        surface_elevation=surface_elevation,
        layer_interfaces=layer_interfaces,
        layer_order=layer_order,
        pixel_scale_m=pixel_scale_m,
        uplift_rate=0.0001  # 0.1 mm/year tectonic uplift
    )
    
    print(f"✓ Simulation initialized")
    print(f"  Initial elevation range: {sim.elevation.min():.1f} to {sim.elevation.max():.1f} m")
    print(f"  Uplift rate: {sim.uplift_rate * 1000:.3f} mm/year")
    
    # ========================================================================
    # STEP 5: Run simulation
    # ========================================================================
    print("\n" + "-" * 80)
    print("STEP 5: Run Erosion Simulation")
    print("-" * 80)
    
    n_steps = int(n_years / dt)
    print(f"Running {n_steps} time steps...")
    
    # Storage for time series data
    time_series = {
        'time': [],
        'total_erosion': [],
        'total_deposition': [],
        'n_rivers': [],
        'n_lakes': [],
        'mean_elevation': [],
    }
    
    for step in range(n_steps):
        # Get rainfall
        rainfall_map = rainfall_generator(sim.current_time)
        
        # Perform time step
        sim.step(dt=dt, rainfall_map=rainfall_map)
        
        # Record data
        time_series['time'].append(sim.current_time)
        time_series['total_erosion'].append(sim.get_total_erosion() / 1e6)  # km³
        time_series['total_deposition'].append(sim.get_total_deposition() / 1e6)  # km³
        time_series['n_rivers'].append(np.sum(sim.river_mask))
        time_series['n_lakes'].append(np.sum(sim.lake_mask))
        time_series['mean_elevation'].append(sim.elevation.mean())
        
        # Progress report
        if (step + 1) % max(1, n_steps // 20) == 0:
            progress = 100 * (step + 1) / n_steps
            print(f"  {progress:5.1f}% complete (t={sim.current_time:7.1f} yr): "
                  f"Erosion={time_series['total_erosion'][-1]:6.2f} km³, "
                  f"Rivers={time_series['n_rivers'][-1]:5d} cells, "
                  f"Lakes={time_series['n_lakes'][-1]:5d} cells")
        
        # Save intermediate plots
        if save_plots and n_steps > 10 and (step + 1) % (n_steps // 5) == 0:
            fig = plot_simulation_summary(sim)
            plt.savefig(f'erosion_t{sim.current_time:06.0f}yr.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
    
    print(f"✓ Simulation complete!")
    
    # ========================================================================
    # STEP 6: Visualization and Analysis
    # ========================================================================
    print("\n" + "-" * 80)
    print("STEP 6: Results and Visualization")
    print("-" * 80)
    
    # Summary statistics
    print(f"\nFinal Statistics:")
    print(f"  Simulation time: {sim.current_time:.1f} years")
    print(f"  Total erosion: {sim.get_total_erosion() / 1e9:.4f} km³")
    print(f"  Total deposition: {sim.get_total_deposition() / 1e9:.4f} km³")
    print(f"  Net volume change: {(sim.get_total_deposition() - sim.get_total_erosion()) / 1e9:.4f} km³")
    print(f"  River cells: {np.sum(sim.river_mask)} ({100*np.sum(sim.river_mask)/sim.river_mask.size:.2f}%)")
    print(f"  Lake cells: {np.sum(sim.lake_mask)} ({100*np.sum(sim.lake_mask)/sim.lake_mask.size:.2f}%)")
    print(f"  Mean elevation change: {(sim.elevation - surface_elevation).mean():.2f} m")
    print(f"  Max erosion depth: {(surface_elevation - sim.elevation).max():.2f} m")
    print(f"  Max deposition depth: {(sim.elevation - surface_elevation).max():.2f} m")
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Row 1: Topography evolution
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(surface_elevation, origin='lower', cmap='terrain')
    plt.colorbar(im1, ax=ax1, label='Elevation (m)')
    ax1.set_title('Initial Topography')
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(sim.elevation, origin='lower', cmap='terrain')
    plt.colorbar(im2, ax=ax2, label='Elevation (m)')
    ax2.set_title(f'Final Topography (t={sim.current_time:.0f} yr)')
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Y (pixels)')
    
    ax3 = fig.add_subplot(gs[0, 2])
    elevation_change = sim.elevation - surface_elevation
    vmax = np.abs(elevation_change).max()
    im3 = ax3.imshow(elevation_change, origin='lower', cmap='RdBu', vmin=-vmax, vmax=vmax)
    plt.colorbar(im3, ax=ax3, label='Change (m)')
    ax3.set_title('Elevation Change\n(Red=Deposition, Blue=Erosion)')
    ax3.set_xlabel('X (pixels)')
    ax3.set_ylabel('Y (pixels)')
    
    # Row 2: Water features and erosion
    ax4 = fig.add_subplot(gs[1, 0])
    flow_log = np.log10(sim.flow_accumulation + 1)
    im4 = ax4.imshow(flow_log, origin='lower', cmap='viridis')
    plt.colorbar(im4, ax=ax4, label='log10(Flow)')
    ax4.set_title('Drainage Network')
    ax4.set_xlabel('X (pixels)')
    ax4.set_ylabel('Y (pixels)')
    
    ax5 = fig.add_subplot(gs[1, 1])
    # Show rivers and lakes
    water_display = np.zeros(sim.elevation.shape)
    water_display[sim.river_mask] = 1
    water_display[sim.lake_mask] = 2
    im5 = ax5.imshow(sim.elevation, origin='lower', cmap='gray', alpha=0.3)
    if np.any(water_display > 0):
        water_masked = np.ma.masked_where(water_display == 0, water_display)
        im5b = ax5.imshow(water_masked, origin='lower', cmap='Blues', alpha=0.7)
        plt.colorbar(im5b, ax=ax5, label='Water\n(1=River, 2=Lake)', ticks=[1, 2])
    ax5.set_title('Rivers and Lakes')
    ax5.set_xlabel('X (pixels)')
    ax5.set_ylabel('Y (pixels)')
    
    ax6 = fig.add_subplot(gs[1, 2])
    K_map = sim.get_erodibility_map()
    im6 = ax6.imshow(K_map, origin='lower', cmap='YlOrRd')
    plt.colorbar(im6, ax=ax6, label='Erodibility K')
    ax6.set_title('Surface Material Erodibility')
    ax6.set_xlabel('X (pixels)')
    ax6.set_ylabel('Y (pixels)')
    
    # Row 3: Time series
    ax7 = fig.add_subplot(gs[2, :2])
    ax7_twin = ax7.twinx()
    
    line1 = ax7.plot(time_series['time'], time_series['total_erosion'], 
                     'b-', linewidth=2, label='Total Erosion')
    line2 = ax7.plot(time_series['time'], time_series['total_deposition'],
                     'r-', linewidth=2, label='Total Deposition')
    line3 = ax7_twin.plot(time_series['time'], time_series['mean_elevation'],
                          'g-', linewidth=2, label='Mean Elevation')
    
    ax7.set_xlabel('Time (years)', fontsize=11)
    ax7.set_ylabel('Volume (km³)', fontsize=11, color='k')
    ax7_twin.set_ylabel('Mean Elevation (m)', fontsize=11, color='g')
    ax7.set_title('Erosion and Deposition Over Time', fontsize=12)
    ax7.grid(True, alpha=0.3)
    
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax7.legend(lines, labels, loc='upper left')
    
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.plot(time_series['time'], time_series['n_rivers'], 
             'b-', linewidth=2, label='River cells')
    ax8.plot(time_series['time'], time_series['n_lakes'],
             'c-', linewidth=2, label='Lake cells')
    ax8.set_xlabel('Time (years)', fontsize=11)
    ax8.set_ylabel('Number of cells', fontsize=11)
    ax8.set_title('Water Feature Evolution', fontsize=12)
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    if save_plots:
        plt.savefig('integrated_erosion_final.png', dpi=200, bbox_inches='tight')
        print(f"\n✓ Saved final visualization to 'integrated_erosion_final.png'")
    
    plt.show()
    
    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE")
    print("=" * 80)
    
    return sim, time_series


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("INTEGRATED EROSION SIMULATION WITH RIVERS NEW")
    print("=" * 80)
    
    if RIVERS_AVAILABLE:
        print("\n✓ Rivers new components available")
        print("  Using quantum-seeded terrain generation")
    else:
        print("\n⚠ Rivers new components not fully available")
        print("  Using classical terrain generation fallback")
    
    print("\nThis simulation combines:")
    print("  • Realistic terrain generation (quantum or classical)")
    print("  • Multi-layer geological stratigraphy")
    print("  • Material-dependent erosion rates")
    print("  • Orographic rainfall patterns")
    print("  • Water flow and sediment transport")
    print("  • River and lake formation")
    print("  • Long-term landscape evolution")
    
    # Run the simulation
    try:
        sim, time_series = run_integrated_erosion_simulation(
            N=128,  # Grid size (128x128 for speed, use 256+ for detail)
            pixel_scale_m=100.0,  # 100m resolution
            n_years=500.0,  # 500 years simulation
            dt=2.0,  # 2 year time steps
            use_quantum_terrain=True,
            random_seed=42,  # for reproducibility
            save_plots=True
        )
        
        print("\n✓ Simulation completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Simulation interrupted by user")
    except Exception as e:
        print(f"\n\n✗ Simulation failed with error: {e}")
        import traceback
        traceback.print_exc()
