#!/usr/bin/env python3
"""
Diagnostic script to identify integration issues with Rivers new
"""

import sys
from pathlib import Path

print("=" * 80)
print("DIAGNOSTIC: Testing Rivers new Integration")
print("=" * 80)

# Test 1: Check files exist
print("\n1. Checking files...")
workspace = Path("/workspace")

rivers_file = workspace / "Rivers new"
erosion_file = workspace / "erosion_simulation.py"

if rivers_file.exists():
    print(f"  ✓ Rivers new found ({rivers_file.stat().st_size} bytes)")
else:
    print(f"  ✗ Rivers new NOT FOUND")
    sys.exit(1)

if erosion_file.exists():
    print(f"  ✓ erosion_simulation.py found")
else:
    print(f"  ✗ erosion_simulation.py NOT FOUND")
    sys.exit(1)

# Test 2: Check imports
print("\n2. Testing imports...")
try:
    import numpy as np
    print("  ✓ numpy imported")
except ImportError as e:
    print(f"  ✗ numpy import failed: {e}")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    print("  ✓ matplotlib imported")
except ImportError as e:
    print(f"  ✗ matplotlib import failed: {e}")
    sys.exit(1)

try:
    from erosion_simulation import ErosionSimulation
    print("  ✓ erosion_simulation imported")
except ImportError as e:
    print(f"  ✗ erosion_simulation import failed: {e}")
    sys.exit(1)

# Test 3: Load Rivers new
print("\n3. Loading Rivers new...")
try:
    with open(rivers_file, "r") as f:
        rivers_code = f.read()
    print(f"  ✓ Rivers new loaded ({len(rivers_code)} characters)")
except Exception as e:
    print(f"  ✗ Failed to load Rivers new: {e}")
    sys.exit(1)

# Test 4: Execute Rivers new
print("\n4. Executing Rivers new code...")
try:
    rivers_ns = {}
    exec(rivers_code, rivers_ns)
    print(f"  ✓ Rivers new executed ({len(rivers_ns)} objects in namespace)")
except Exception as e:
    print(f"  ✗ Failed to execute Rivers new: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Check for required functions
print("\n5. Checking for required functions...")

required_funcs = [
    'quantum_seeded_topography',
    'generate_stratigraphy',
    'generate_storm_weather_fields',
    'generate_storm_schedule_for_year',
    'build_wind_structures',
    'compute_orographic_low_pressure',
]

found_funcs = []
missing_funcs = []

for func_name in required_funcs:
    if func_name in rivers_ns and callable(rivers_ns[func_name]):
        found_funcs.append(func_name)
        print(f"  ✓ {func_name}")
    else:
        missing_funcs.append(func_name)
        print(f"  ✗ {func_name} NOT FOUND")

# Test 6: Try simple terrain generation
if 'quantum_seeded_topography' in found_funcs:
    print("\n6. Testing terrain generation...")
    try:
        quantum_seeded_topography = rivers_ns['quantum_seeded_topography']
        z_norm, rng = quantum_seeded_topography(N=32, random_seed=42)
        print(f"  ✓ Terrain generated: shape={z_norm.shape}, range=[{z_norm.min():.3f}, {z_norm.max():.3f}]")
    except Exception as e:
        print(f"  ✗ Terrain generation failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\n6. SKIPPED: quantum_seeded_topography not available")

# Test 7: Try stratigraphy
if 'generate_stratigraphy' in found_funcs and 'quantum_seeded_topography' in found_funcs:
    print("\n7. Testing stratigraphy generation...")
    try:
        quantum_seeded_topography = rivers_ns['quantum_seeded_topography']
        generate_stratigraphy = rivers_ns['generate_stratigraphy']
        
        z_norm, rng = quantum_seeded_topography(N=32, random_seed=42)
        surface_elev = z_norm * 1000.0
        
        strata = generate_stratigraphy(
            surface_elev=surface_elev,
            pixel_scale_m=100.0,
            rng=rng
        )
        
        n_layers = len(strata.get('interfaces', {}))
        print(f"  ✓ Stratigraphy generated: {n_layers} layers")
        
    except Exception as e:
        print(f"  ✗ Stratigraphy generation failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\n7. SKIPPED: generate_stratigraphy not available")

# Summary
print("\n" + "=" * 80)
print("DIAGNOSTIC SUMMARY")
print("=" * 80)

if missing_funcs:
    print(f"\n✗ MISSING FUNCTIONS ({len(missing_funcs)}):")
    for func in missing_funcs:
        print(f"  - {func}")
    print("\nThese functions are required but not found in 'Rivers new'.")
    print("This might mean:")
    print("  1. They have different names in your code")
    print("  2. They haven't been defined yet")
    print("  3. There's a syntax error preventing execution")
else:
    print(f"\n✓ ALL REQUIRED FUNCTIONS FOUND ({len(found_funcs)}/{len(required_funcs)})")
    print("\nIntegration should work!")

print("\n" + "=" * 80)
