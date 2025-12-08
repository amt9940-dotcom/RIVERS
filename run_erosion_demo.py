#!/usr/bin/env python3
"""
Demo script for running the erosion model simulation.

This script demonstrates how to:
1. Initialize the erosion model
2. Generate terrain and layers
3. Run erosion simulation over time
4. Visualize results including rivers and lakes
"""

import numpy as np
import matplotlib.pyplot as plt
from erosion_model import ErosionModel

def main():
    """Run erosion simulation demo."""
    print("=" * 70)
    print("EROSION MODEL SIMULATION DEMO")
    print("=" * 70)
    print()
    
    # ============================================================
    # Configuration
    # ============================================================
    grid_size = 256          # Grid resolution (256x256 cells)
    pixel_scale_m = 10.0     # Each pixel = 10 meters
    elev_range_m = 700.0     # Elevation range: 0-700 meters
    random_seed = 42         # For reproducibility
    
    # Simulation parameters
    num_years = 20.0         # Simulate 20 years
    annual_rainfall_mm = 1200.0  # 1200 mm/year average rainfall
    time_step_years = 0.1   # 0.1 year time steps
    snapshot_interval = 2.0  # Save snapshot every 2 years
    
    print("Configuration:")
    print(f"  Grid size: {grid_size}x{grid_size}")
    print(f"  Pixel scale: {pixel_scale_m} m")
    print(f"  Elevation range: 0-{elev_range_m} m")
    print(f"  Simulation time: {num_years} years")
    print(f"  Annual rainfall: {annual_rainfall_mm} mm")
    print(f"  Time step: {time_step_years} years")
    print()
    
    # ============================================================
    # Initialize Model
    # ============================================================
    print("Initializing erosion model...")
    model = ErosionModel(
        grid_size=grid_size,
        pixel_scale_m=pixel_scale_m,
        elev_range_m=elev_range_m,
        random_seed=random_seed
    )
    
    # Generate initial terrain
    model.generate_initial_terrain()
    print()
    
    # ============================================================
    # Run Simulation
    # ============================================================
    print("Running erosion simulation...")
    print("-" * 70)
    
    results = model.simulate(
        num_years=num_years,
        annual_rainfall_mm=annual_rainfall_mm,
        time_step_years=time_step_years,
        save_snapshots=True,
        snapshot_interval_years=snapshot_interval
    )
    
    print("-" * 70)
    print()
    
    # ============================================================
    # Create Visualizations
    # ============================================================
    print("Creating visualizations...")
    
    # 1. Initial vs Final Topography Comparison
    print("  1. Initial vs Final Topography...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    model.plot_topography(
        elevation=results["initial_elevation"],
        title="Initial Topography",
        ax=axes[0]
    )
    
    model.plot_topography(
        elevation=results["final_elevation"],
        rivers=results["final_rivers"],
        lakes=results["final_lakes"],
        title=f"Final Topography (after {num_years} years)",
        ax=axes[1]
    )
    
    plt.tight_layout()
    plt.savefig("erosion_comparison.png", dpi=150, bbox_inches='tight')
    print("     Saved: erosion_comparison.png")
    
    # 2. Total Erosion Map
    print("  2. Erosion Map...")
    fig, ax = plt.subplots(figsize=(12, 10))
    model.plot_erosion_map(
        results["total_erosion"],
        title=f"Total Erosion over {num_years} years",
        ax=ax
    )
    plt.tight_layout()
    plt.savefig("erosion_map.png", dpi=150, bbox_inches='tight')
    print("     Saved: erosion_map.png")
    
    # 3. Time Series Evolution
    print("  3. Time Series Evolution...")
    model.plot_time_series(results, save_path="erosion_timeseries.png")
    print("     Saved: erosion_timeseries.png")
    
    # 4. River and Lake Statistics
    print("  4. River and Lake Statistics...")
    if results["final_rivers"] is not None:
        num_river_cells = results["final_rivers"].sum()
        river_area_km2 = num_river_cells * (pixel_scale_m / 1000.0) ** 2
        print(f"    River cells: {num_river_cells}")
        print(f"    River area: {river_area_km2:.2f} km²")
    
    if results["final_lakes"] is not None:
        num_lake_cells = results["final_lakes"].sum()
        lake_area_km2 = num_lake_cells * (pixel_scale_m / 1000.0) ** 2
        print(f"    Lake cells: {num_lake_cells}")
        print(f"    Lake area: {lake_area_km2:.2f} km²")
    
    # 5. Elevation Statistics
    print("  5. Elevation Statistics...")
    initial_max = results["initial_elevation"].max()
    final_max = results["final_elevation"].max()
    initial_min = results["initial_elevation"].min()
    final_min = results["final_elevation"].min()
    
    print(f"    Initial elevation range: {initial_min:.1f} - {initial_max:.1f} m")
    print(f"    Final elevation range: {final_min:.1f} - {final_max:.1f} m")
    print(f"    Maximum erosion: {results['total_erosion'].max():.2f} m")
    print(f"    Average erosion: {results['total_erosion'].mean():.2f} m")
    
    # 6. Create detailed analysis plot
    print("  6. Detailed Analysis Plot...")
    fig = plt.figure(figsize=(20, 12))
    
    # Top row: Initial, Final, Erosion
    ax1 = plt.subplot(2, 3, 1)
    model.plot_topography(
        elevation=results["initial_elevation"],
        title="Initial Topography",
        ax=ax1
    )
    
    ax2 = plt.subplot(2, 3, 2)
    model.plot_topography(
        elevation=results["final_elevation"],
        rivers=results["final_rivers"],
        lakes=results["final_lakes"],
        title="Final Topography",
        ax=ax2
    )
    
    ax3 = plt.subplot(2, 3, 3)
    model.plot_erosion_map(
        results["total_erosion"],
        title="Total Erosion",
        ax=ax3
    )
    
    # Bottom row: Flow accumulation, Rivers, Lakes
    ax4 = plt.subplot(2, 3, 4)
    if len(results["snapshots"]) > 0:
        last_snapshot = results["snapshots"][-1]
        im = ax4.imshow(
            last_snapshot["flow_accumulation"],
            cmap='Blues',
            origin='lower',
            extent=[0, grid_size * pixel_scale_m / 1000,
                   0, grid_size * pixel_scale_m / 1000]
        )
        ax4.set_xlabel('Distance (km)')
        ax4.set_ylabel('Distance (km)')
        ax4.set_title('Flow Accumulation')
        plt.colorbar(im, ax=ax4, label='Flow (relative)')
    
    ax5 = plt.subplot(2, 3, 5)
    if results["final_rivers"] is not None:
        im = ax5.imshow(
            results["final_elevation"],
            cmap='terrain',
            origin='lower',
            extent=[0, grid_size * pixel_scale_m / 1000,
                   0, grid_size * pixel_scale_m / 1000]
        )
        river_y, river_x = np.where(results["final_rivers"])
        if len(river_y) > 0:
            ax5.scatter(
                river_x * pixel_scale_m / 1000,
                river_y * pixel_scale_m / 1000,
                c='blue', s=0.5, alpha=0.8
            )
        ax5.set_xlabel('Distance (km)')
        ax5.set_ylabel('Distance (km)')
        ax5.set_title('River Network')
        plt.colorbar(im, ax=ax5, label='Elevation (m)')
    
    ax6 = plt.subplot(2, 3, 6)
    if results["final_lakes"] is not None:
        im = ax6.imshow(
            results["final_elevation"],
            cmap='terrain',
            origin='lower',
            extent=[0, grid_size * pixel_scale_m / 1000,
                   0, grid_size * pixel_scale_m / 1000]
        )
        lake_y, lake_x = np.where(results["final_lakes"])
        if len(lake_y) > 0:
            ax6.scatter(
                lake_x * pixel_scale_m / 1000,
                lake_y * pixel_scale_m / 1000,
                c='cyan', s=2, alpha=0.7, marker='s'
            )
        ax6.set_xlabel('Distance (km)')
        ax6.set_ylabel('Distance (km)')
        ax6.set_title('Lakes')
        plt.colorbar(im, ax=ax6, label='Elevation (m)')
    
    plt.tight_layout()
    plt.savefig("erosion_analysis.png", dpi=150, bbox_inches='tight')
    print("     Saved: erosion_analysis.png")
    
    print()
    print("=" * 70)
    print("SIMULATION COMPLETE!")
    print("=" * 70)
    print()
    print("Generated files:")
    print("  - erosion_comparison.png  : Initial vs Final topography")
    print("  - erosion_map.png         : Total erosion map")
    print("  - erosion_timeseries.png  : Evolution over time")
    print("  - erosion_analysis.png    : Detailed analysis")
    print()
    print("You can now:")
    print("  1. Adjust simulation parameters (years, rainfall, etc.)")
    print("  2. Change layer erodibility values")
    print("  3. Modify time step for different temporal resolution")
    print("  4. Experiment with different terrain sizes")
    print()
    
    # Show plots
    plt.show()


if __name__ == "__main__":
    main()
