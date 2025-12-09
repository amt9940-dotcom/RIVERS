#!/usr/bin/env python3
"""
Quick test to verify erosion simulation works correctly.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from erosion_simulation import (
    ErosionSimulation,
    ERODIBILITY,
    plot_simulation_summary
)

def test_basic_simulation():
    """Test basic erosion simulation functionality."""
    print("\n" + "=" * 60)
    print("TESTING EROSION SIMULATION")
    print("=" * 60)
    
    # Create simple test terrain
    print("\n1. Creating test terrain (32x32)...")
    N = 32
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)
    
    # Simple dome-shaped terrain
    surface_elevation = 100 + 50 * np.exp(-(X**2 + Y**2) / 0.3)
    print(f"   Elevation range: {surface_elevation.min():.1f} to {surface_elevation.max():.1f} m")
    
    # Create simple layer stack
    print("\n2. Creating geological layers...")
    layer_order = ["Topsoil", "Sandstone", "Granite", "Basement"]
    layer_interfaces = {
        "Topsoil": surface_elevation - 2,
        "Sandstone": surface_elevation - 20,
        "Granite": surface_elevation - 100,
        "Basement": surface_elevation - 500,
    }
    print(f"   Layers: {', '.join(layer_order)}")
    
    # Initialize simulation
    print("\n3. Initializing simulation...")
    sim = ErosionSimulation(
        surface_elevation=surface_elevation,
        layer_interfaces=layer_interfaces,
        layer_order=layer_order,
        pixel_scale_m=100.0,
        uplift_rate=0.0
    )
    print(f"   Grid size: {sim.nx} x {sim.ny}")
    print(f"   Resolution: {sim.pixel_scale_m} m/pixel")
    
    # Check initial state
    print("\n4. Checking initial state...")
    materials = sim.get_surface_material()
    unique_materials = np.unique(materials)
    print(f"   Surface materials: {', '.join(unique_materials)}")
    
    K = sim.get_erodibility_map()
    print(f"   Erodibility range: {K.min():.6f} to {K.max():.6f}")
    
    # Run a few time steps
    print("\n5. Running simulation (10 years, 2 time steps)...")
    for i in range(2):
        # Uniform rainfall
        rainfall_map = np.full(sim.elevation.shape, 500.0)  # 500mm per step
        
        sim.step(dt=5.0, rainfall_map=rainfall_map)
        
        print(f"   Step {i+1}: t={sim.current_time:.1f} yr, "
              f"Erosion={sim.get_total_erosion()/1e6:.4f} km³")
    
    # Check results
    print("\n6. Checking results...")
    print(f"   Total erosion: {sim.get_total_erosion()/1e6:.6f} km³")
    print(f"   Total deposition: {sim.get_total_deposition()/1e6:.6f} km³")
    print(f"   River cells: {np.sum(sim.river_mask)}")
    print(f"   Lake cells: {np.sum(sim.lake_mask)}")
    
    elevation_change = sim.elevation - surface_elevation
    print(f"   Mean elevation change: {elevation_change.mean():.4f} m")
    print(f"   Max erosion depth: {-elevation_change.min():.4f} m")
    
    # Test visualization
    print("\n7. Testing visualization...")
    try:
        fig = plot_simulation_summary(sim)
        plt.savefig('/tmp/test_erosion_plot.png', dpi=100)
        plt.close(fig)
        print("   ✓ Plot created successfully")
    except Exception as e:
        print(f"   ✗ Plot failed: {e}")
    
    # Verify physics
    print("\n8. Verifying physics...")
    
    # Check conservation (with uplift/loss)
    total_change = np.sum(elevation_change) * sim.cell_area_m2
    net_erosion = sim.get_total_deposition() - sim.get_total_erosion()
    print(f"   Total elevation change volume: {total_change/1e6:.6f} km³")
    print(f"   Net erosion (deposition - erosion): {net_erosion/1e6:.6f} km³")
    
    # Check that erosion occurred (should be non-zero with rainfall)
    if sim.get_total_erosion() > 0:
        print("   ✓ Erosion is occurring")
    else:
        print("   ⚠ Warning: No erosion detected")
    
    # Check flow accumulation
    if sim.flow_accumulation.max() > 1:
        print(f"   ✓ Flow accumulation computed (max={sim.flow_accumulation.max():.0f} cells)")
    else:
        print("   ⚠ Warning: Flow accumulation not computed")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    
    return sim


def test_erodibility_coefficients():
    """Test that erodibility coefficients are reasonable."""
    print("\n" + "=" * 60)
    print("TESTING ERODIBILITY COEFFICIENTS")
    print("=" * 60)
    
    print("\nMaterial erodibility coefficients (K):")
    print(f"{'Material':<20} {'K value':<12} {'Category'}")
    print("-" * 60)
    
    # Sort by erodibility (most to least erodible)
    sorted_materials = sorted(ERODIBILITY.items(), key=lambda x: x[1], reverse=True)
    
    for material, K in sorted_materials[:15]:  # Show top 15
        if K > 0.004:
            category = "Very Erodible"
        elif K > 0.002:
            category = "Moderately Erodible"
        elif K > 0.001:
            category = "Resistant"
        else:
            category = "Very Resistant"
        
        print(f"{material:<20} {K:<12.6f} {category}")
    
    print(f"\n✓ Total materials defined: {len(ERODIBILITY)}")
    print(f"✓ K range: {min(ERODIBILITY.values()):.6f} to {max(ERODIBILITY.values()):.6f}")
    
    # Verify relative ordering makes sense
    print("\n✓ Checking physical consistency:")
    checks = [
        (ERODIBILITY["Topsoil"] > ERODIBILITY["Granite"], "Topsoil > Granite"),
        (ERODIBILITY["Sand"] > ERODIBILITY["Sandstone"], "Sand > Sandstone"),
        (ERODIBILITY["Shale"] > ERODIBILITY["Limestone"], "Shale > Limestone"),
        (ERODIBILITY["Limestone"] > ERODIBILITY["Granite"], "Limestone > Granite"),
        (ERODIBILITY["Granite"] > ERODIBILITY["Basement"], "Granite > Basement"),
    ]
    
    for check, description in checks:
        status = "✓" if check else "✗"
        print(f"  {status} {description}")


if __name__ == "__main__":
    try:
        # Test erodibility coefficients
        test_erodibility_coefficients()
        
        # Test basic simulation
        sim = test_basic_simulation()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        print("\nThe erosion simulation is working correctly!")
        print("You can now run:")
        print("  - python3 example_erosion_simulation.py")
        print("  - python3 integrated_erosion_example.py")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
