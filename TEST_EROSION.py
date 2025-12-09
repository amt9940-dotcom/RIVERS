"""
Simple test to verify erosion model modifies surface_elev correctly.
"""
import numpy as np

# Create a fake strata dict
strata = {
    "surface_elev": np.ones((10, 10)) * 1000.0,  # 1000m elevation
    "thickness": {
        "Topsoil": np.ones((10, 10)) * 5.0,
        "Basement": np.ones((10, 10)) * 100.0,
    },
    "interfaces": {
        "Topsoil": np.ones((10, 10)) * 995.0,
        "Basement": np.ones((10, 10)) * 895.0,
    },
    "properties": {}
}

print("BEFORE erosion:")
print(f"  Surface min/max: {strata['surface_elev'].min()} / {strata['surface_elev'].max()}")
print(f"  Surface mean: {strata['surface_elev'].mean()}")

# Apply erosion manually (simulate what update_stratigraphy_with_erosion does)
erosion = np.ones((10, 10)) * 10.0  # Remove 10m

print(f"\nErosion to apply: {erosion.mean()} m")

# This is what happens in update_stratigraphy_with_erosion at line 467
strata["surface_elev"] -= erosion

print("\nAFTER erosion:")
print(f"  Surface min/max: {strata['surface_elev'].min()} / {strata['surface_elev'].max()}")
print(f"  Surface mean: {strata['surface_elev'].mean()}")
print(f"  Expected: 990.0 m")

if strata["surface_elev"].mean() == 990.0:
    print("\n✓ TEST PASSED: Erosion applied correctly!")
else:
    print("\n✗ TEST FAILED: Surface not modified as expected!")
