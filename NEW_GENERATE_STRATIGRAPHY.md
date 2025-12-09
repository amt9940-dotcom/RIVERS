# Constraint-Based generate_stratigraphy() Implementation

## Summary

The current `generate_stratigraphy()` function (lines 1049-1877, **828 lines**) needs to be completely replaced with a constraint-based approach that:

1. **Works from surface downward** (not upward from basement)
2. **Computes erosion intensity from surface geometry**
3. **Uses explicit constraints** to prevent deep layers appearing inappropriately
4. **Only exposes basement in structural highs** with sufficient erosion

## Required Changes

### 1. Insert Helper Functions (before line 1049)

Add these functions from `constraint_stratigraphy.py`:
- `normalize(arr)` - if not already defined
- `box_blur(arr, k)` - if not already defined as `_box_blur`
- `compute_erosion_intensity(surface_elev, pixel_scale_m, ...)`
- `generate_structural_uplift(N, rng, pixel_scale_m, ...)`
- `compute_structural_high_mask(U, z_norm, ...)`
- `compute_cover_thicknesses(E, slope_norm, wetness, curvature, z_norm)`
- `enforce_minimum_sediment_cover(interfaces, surface_elev, structural_high_mask, ...)`
- `constrain_deep_layer_exposure(interfaces, surface_elev, E, structural_high_mask, ...)`
- `apply_progressive_stripping(interfaces, surface_elev, E, structural_high_mask, ...)`
- `enforce_ordering(interfaces, layer_order, eps)`

### 2. Replace generate_stratigraphy() (lines 1049-1877)

The new function should be ~200-300 lines instead of 828, following this structure:

```python
def generate_stratigraphy(
    z_norm,
    rng,
    elev_range_m=700.0,
    pixel_scale_m=10.0,
    n_anticlines=3,
    sediment_min_depth=300.0,
    E_sandstone_threshold=0.5,
    E_shale_threshold=0.6,
    E_limestone_threshold=0.7,
    E_basement_threshold=0.85,
    **kwargs  # Ignore unused legacy parameters
):
    """
    Generate stratigraphy using constraint-based approach.
    
    Key innovation: Work from surface DOWNWARD, applying explicit constraints
    about where deep layers can appear based on erosion intensity and structural position.
    """
    
    N = z_norm.shape[0]
    surface_elev = z_norm * elev_range_m
    
    # ============ STEP 1: Compute erosion intensity from surface ============
    E, E_components = compute_erosion_intensity(surface_elev, pixel_scale_m)
    slope_norm = E_components["slope_norm"]
    wetness = E_components["wetness"]
    curvature = E_components["curvature"]
    
    # ============ STEP 2: Generate structural uplift ============
    U = generate_structural_uplift(N, rng, pixel_scale_m, n_anticlines)
    structural_high_mask = compute_structural_high_mask(U, z_norm)
    
    # ============ STEP 3: Compute cover layer thicknesses ============
    cover_thick = compute_cover_thicknesses(E, slope_norm, wetness, curvature, z_norm)
    
    # ============ STEP 4: Build interfaces from surface DOWNWARD ============
    interfaces = {}
    
    # Cover layers (unconsolidated)
    interfaces["Topsoil"] = surface_elev - cover_thick["soil"] * 0.4
    interfaces["Subsoil"] = interfaces["Topsoil"] - cover_thick["soil"] * 0.6
    interfaces["Colluvium"] = interfaces["Subsoil"] - cover_thick["colluvium"]
    interfaces["Alluvium"] = interfaces["Colluvium"] - cover_thick["alluvium"]
    interfaces["Saprolite"] = interfaces["Alluvium"] - cover_thick["saprolite"]
    interfaces["WeatheredBR"] = interfaces["Saprolite"] - cover_thick["weathered"]
    
    # Valley fill sediments (modern)
    clay_thick = compute_clay_thickness(slope_norm, wetness, z_norm)
    silt_thick = compute_silt_thickness(slope_norm, wetness, z_norm)
    sand_thick = compute_sand_thickness(slope_norm, wetness, z_norm)
    
    interfaces["Clay"] = interfaces["WeatheredBR"] - clay_thick
    interfaces["Silt"] = interfaces["Clay"] - silt_thick
    interfaces["Sand"] = interfaces["Silt"] - sand_thick
    
    # Rock layers - initial positions based on uplift
    rock_base = interfaces["Sand"] - 50.0  # Gap below unconsolidated
    rock_base += U  # Uplift brings rock up
    
    # Default rock thicknesses (will be constrained later)
    interfaces["Sandstone"] = rock_base
    interfaces["Conglomerate"] = interfaces["Sandstone"] - 80.0
    interfaces["Shale"] = interfaces["Conglomerate"] - 30.0
    interfaces["Mudstone"] = interfaces["Shale"] - 150.0
    interfaces["Siltstone"] = interfaces["Mudstone"] - 80.0
    interfaces["Limestone"] = interfaces["Siltstone"] - 60.0
    interfaces["Dolomite"] = interfaces["Limestone"] - 100.0
    interfaces["Evaporite"] = interfaces["Dolomite"] - 30.0
    
    # Basement and crystalline
    interfaces["Granite"] = interfaces["Evaporite"]
    interfaces["Gneiss"] = interfaces["Granite"] - 5.0
    interfaces["Basalt"] = interfaces["Gneiss"] - 5.0
    interfaces["AncientCrust"] = interfaces["Basalt"] - 2.0
    interfaces["Basement"] = interfaces["AncientCrust"]
    interfaces["BasementFloor"] = interfaces["Basement"] - 500.0
    
    # ============ STEP 5: Apply constraints ============
    
    # 5a. Minimum sediment cover outside structural highs
    interfaces = enforce_minimum_sediment_cover(
        interfaces, surface_elev, structural_high_mask, sediment_min_depth
    )
    
    # 5b. Constrain deep layer exposure
    interfaces = constrain_deep_layer_exposure(
        interfaces, surface_elev, E, structural_high_mask,
        E_sandstone_threshold, E_shale_threshold, 
        E_limestone_threshold, E_basement_threshold
    )
    
    # 5c. Progressive stripping of cover
    interfaces = apply_progressive_stripping(
        interfaces, surface_elev, E, structural_high_mask
    )
    
    # ============ STEP 6: Enforce ordering ============
    layer_order = [
        "Topsoil", "Subsoil", "Colluvium", "Alluvium",
        "Saprolite", "WeatheredBR", "Clay", "Silt", "Sand",
        "Sandstone", "Conglomerate", "Shale", "Mudstone", "Siltstone",
        "Limestone", "Dolomite", "Evaporite",
        "Granite", "Gneiss", "Basalt", "AncientCrust",
        "Basement", "BasementFloor"
    ]
    
    interfaces = enforce_ordering(interfaces, layer_order, eps=0.01)
    
    # ============ STEP 7: Ensure no interface above surface ============
    for layer in interfaces:
        interfaces[layer] = np.minimum(interfaces[layer], surface_elev - 0.01)
    
    # ============ STEP 8: Compute thicknesses ============
    thickness = {}
    for i in range(len(layer_order) - 1):
        layer = layer_order[i]
        below = layer_order[i + 1]
        if layer in interfaces and below in interfaces:
            thickness[layer] = np.maximum(
                interfaces[layer] - interfaces[below], 
                0.0
            )
    
    # Add final layer
    if "BasementFloor" in interfaces:
        z_floor = float(interfaces["BasementFloor"].min() - 100.0)
        thickness["BasementFloor"] = np.maximum(
            interfaces["BasementFloor"] - z_floor,
            0.0
        )
    
    # ============ STEP 9: Compute color properties (simplified) ============
    properties = {}
    for layer in thickness:
        # Use default colors for now
        properties[layer] = np.ones_like(thickness[layer])
    
    # ============ STEP 10: Return results ============
    return {
        "surface_elev": surface_elev,
        "interfaces": interfaces,
        "thickness": thickness,
        "properties": properties,
        "meta": {
            "elev_range_m": elev_range_m,
            "pixel_scale_m": pixel_scale_m,
            "erosion_intensity": E,
            "structural_uplift": U,
            "structural_high_mask": structural_high_mask,
            "E_components": E_components,
        }
    }
```

### 3. Add Small Helper Functions

These compute valley fill thickness:

```python
def compute_clay_thickness(slope_norm, wetness, z_norm, max_thick=20.0):
    """Clay in flat, wet, low areas"""
    valley = (slope_norm < 0.1) & (wetness > 0.6) & (z_norm < 0.3)
    return max_thick * valley.astype(float)

def compute_silt_thickness(slope_norm, wetness, z_norm, max_thick=15.0):
    """Silt in gentle, moderately wet areas"""
    gentle_wet = (slope_norm < 0.2) & (wetness > 0.4) & (z_norm < 0.4)
    return max_thick * gentle_wet.astype(float) * wetness

def compute_sand_thickness(slope_norm, wetness, z_norm, max_thick=25.0):
    """Sand in channels (high wetness + moderate slope)"""
    channel = (slope_norm > 0.05) & (slope_norm < 0.3) & (wetness > 0.5)
    return max_thick * channel.astype(float) * wetness
```

## Implementation Steps

1. ✅ Create `constraint_stratigraphy.py` with all helper functions
2. ⏳ Insert helper functions into main file before line 1049
3. ⏳ Replace lines 1049-1877 with new ~300 line function
4. ⏳ Test and validate
5. ⏳ Update documentation

## Expected Benefits

- **Much shorter**: 300 lines vs 828 lines
- **Clearer logic**: Surface → constraints → layers
- **Geologically correct**: Basement only in structural highs
- **No more "sand blanket"**: Explicit rules prevent inappropriate exposure
- **Valleys mud-dominated**: Erosion intensity drives facies

## Key Architectural Changes

| Old Approach | New Approach |
|--------------|--------------|
| Build upward from basement | Build downward from surface |
| Basement independent of surface | Basement constrained by surface + structure |
| Thickness from complex facies belts | Thickness from erosion intensity |
| No explicit exposure rules | Explicit E thresholds for each layer |
| 828 lines of complex logic | 300 lines of clear constraints |

---

**Status**: Specification complete. Ready for implementation.

The complete code is in `constraint_stratigraphy.py` and needs to be integrated into the main file.
