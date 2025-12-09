"""
Quick test script to verify the landscape_evolution package imports correctly.
"""

import sys
import traceback

def test_imports():
    """Test that all modules can be imported."""
    
    print("Testing landscape_evolution package imports...\n")
    
    modules_to_test = [
        ("Core state", "landscape_evolution.world_state", ["WorldState", "MaterialProperties"]),
        ("Forcing", "landscape_evolution.forcing", ["TectonicUplift", "WeatherGenerator"]),
        ("Hydrology", "landscape_evolution.hydrology", ["FlowRouter"]),
        ("Geomorphic", "landscape_evolution.geomorphic_processes", ["GeomorphicEngine"]),
        ("Stratigraphy", "landscape_evolution.stratigraphy", ["StratigraphyUpdater"]),
        ("Simulator", "landscape_evolution.simulator", ["LandscapeEvolutionSimulator"]),
        ("Terrain", "landscape_evolution.terrain_generation", ["quantum_seeded_topography"]),
        ("Initial strat", "landscape_evolution.initial_stratigraphy", ["initialize_world_from_stratigraphy"]),
        ("Visualization", "landscape_evolution.visualization", ["plot_evolution_summary"]),
    ]
    
    success_count = 0
    fail_count = 0
    
    for name, module_path, classes in modules_to_test:
        try:
            module = __import__(module_path, fromlist=classes)
            for cls in classes:
                if not hasattr(module, cls):
                    raise ImportError(f"{cls} not found in {module_path}")
            print(f"✓ {name:20s} ({module_path})")
            success_count += 1
        except Exception as e:
            print(f"✗ {name:20s} ({module_path})")
            print(f"  Error: {str(e)}")
            fail_count += 1
    
    print(f"\n{'='*60}")
    print(f"Results: {success_count} passed, {fail_count} failed")
    
    if fail_count == 0:
        print("\n✓ All imports successful! Package is ready to use.")
        return True
    else:
        print("\n✗ Some imports failed. Check the errors above.")
        return False


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
