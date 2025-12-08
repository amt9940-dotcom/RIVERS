# Geological Layer Constraint System

## Overview

This document describes a constraint-based approach to generating realistic stratigraphic layers that properly respond to surface topography, erosion, and structural geology. The key innovation is **working backwards from the surface** rather than forwards from arbitrary structural planes.

---

## Part 1: The Problem with Current Approach

### Current Flaws

The existing code has these issues:

1. **Floor values guarantee minimum thickness everywhere**
   ```python
   base_sand = sed_total * f_sand * (0.4 + 0.6 * sand_env)  # 40% minimum!
   ```

2. **`sed_total` increases on mountains** (isostatic model gives thicker crust under high topo)

3. **Erosion shifts layers together** without actually removing material

4. **No explicit "no sediments on mountains" rule**

### The Fix: Constraint-Based Layer Generation

Instead of computing thicknesses and hoping they work out, we:
1. **Start from the surface** and work down
2. **Compute erosion intensity** from surface geometry
3. **Apply explicit constraints** that prevent deep layers from appearing where they shouldn't
4. **Only expose basement in valid structural zones** with sufficient erosion

---

## Part 2: Data Model

### Required Arrays

```python
# Primary terrain
surface_elev[x, y]      # Current ground surface elevation (meters)
z_norm[x, y]            # Normalized terrain from generator (0-1)

# Layer interfaces (elevation of TOP of each layer)
interfaces = {
    "Topsoil": ndarray,
    "Subsoil": ndarray,
    "Colluvium": ndarray,
    "Alluvium": ndarray,
    "Saprolite": ndarray,
    "WeatheredBR": ndarray,
    "Clay": ndarray,
    "Silt": ndarray,
    "Sand": ndarray,
    "Sandstone": ndarray,
    "Shale": ndarray,
    "Limestone": ndarray,
    "Basement": ndarray,
    "BasementFloor": ndarray,
}

# Derived thicknesses
thickness[layer_name][x, y] = interfaces[layer_above] - interfaces[layer]
```

### Geo-Derivative Fields (Computed from `surface_elev`)

```python
slope[x, y]           # Surface slope magnitude (degrees)
curvature[x, y]       # Surface curvature (+ = concave/valley, - = convex/ridge)
relief_local[x, y]    # Local relief (cell elevation minus neighborhood mean)
wetness[x, y]         # Topographic wetness index / catchment proxy
dist_to_channel[x, y] # Distance to nearest drainage channel (optional)
```

### Structural Fields

```python
structural_uplift[x, y]     # Uplift field (anticlines, domes)
structural_high_mask[x, y]  # Boolean: True where deep layers CAN reach surface
erosion_intensity[x, y]     # E in [0, 1]: how much erosion has occurred here
```

---

## Part 3: Layer Ordering Rules

### Rule 1: No Interface Above Surface

Every layer interface must be at or below the surface:

```python
eps = 0.01  # Small separation to prevent z-fighting

for layer_name in interfaces:
    interfaces[layer_name] = np.minimum(
        interfaces[layer_name], 
        surface_elev - eps
    )
```

### Rule 2: Vertical Ordering Must Be Maintained

Layers must stack in correct order (top to bottom):

```python
layer_order = [
    "Topsoil",
    "Subsoil", 
    "Colluvium",
    "Alluvium",
    "Saprolite",
    "WeatheredBR",
    "Clay",
    "Silt", 
    "Sand",
    "Sandstone",
    "Shale",
    "Limestone",
    "Basement",
    "BasementFloor",
]

def enforce_ordering(interfaces, layer_order, eps=0.01):
    """Push each layer down to be below the one above it."""
    for i in range(1, len(layer_order)):
        above = layer_order[i - 1]
        here = layer_order[i]
        if above in interfaces and here in interfaces:
            interfaces[here] = np.minimum(
                interfaces[here],
                interfaces[above] - eps
            )
    return interfaces
```

---

## Part 4: Erosion Detection from Surface Geometry

### Rule 3: Compute Erosion Intensity Field

Erosion intensity `E[x, y]` represents how much erosion has occurred at each location.

```python
def compute_erosion_intensity(
    surface_elev,
    pixel_scale_m,
    w_slope=0.35,
    w_relief=0.35,
    w_channel=0.20,
    w_elevation=0.10,
    channel_max_dist=500.0,  # meters
):
    """
    Compute erosion intensity E[x,y] in [0, 1].
    
    High E = strong erosion = thin cover, deep layers exposed
    Low E = weak erosion = thick cover, deep layers buried
    """
    N = surface_elev.shape[0]
    
    # 1. Slope component (steeper = more erosion)
    dEy, dEx = np.gradient(surface_elev, pixel_scale_m, pixel_scale_m)
    slope_mag = np.hypot(dEx, dEy)
    slope_deg = np.rad2deg(np.arctan(slope_mag))
    E_slope = np.clip(slope_deg / 45.0, 0, 1)  # Normalize to 45° max
    
    # 2. Local relief component (valleys cutting deep = more erosion)
    k_relief = max(31, int(0.1 * N) | 1)
    elev_smooth = box_blur(surface_elev, k=k_relief)
    relief_local = surface_elev - elev_smooth
    relief_range = np.percentile(np.abs(relief_local), 98)
    E_relief = np.clip(np.abs(relief_local) / (relief_range + 1e-9), 0, 1)
    
    # 3. Channel proximity (near channels = more erosion)
    # Simplified: use catchment/wetness proxy
    catch = box_blur(box_blur(1.0 - normalize(slope_mag), k=7), k=13)
    wetness = normalize(catch)
    # Channels are where wetness is high AND slope is moderate
    channel_indicator = wetness * (1 - E_slope * 0.5)
    E_channel = normalize(channel_indicator)
    
    # 4. Elevation component (higher = more erosion potential)
    E_elev = normalize(surface_elev)
    
    # Combine with weights
    E = (w_slope * E_slope + 
         w_relief * E_relief + 
         w_channel * E_channel +
         w_elevation * E_elev)
    
    # Smooth to avoid pixel-scale noise
    E = box_blur(E, k=7)
    E = np.clip(E, 0, 1)
    
    return E, {
        "E_slope": E_slope,
        "E_relief": E_relief,
        "E_channel": E_channel,
        "E_elev": E_elev,
    }
```

### Interpretation of Erosion Intensity

| Location | E Value | Meaning |
|----------|---------|---------|
| Valley floors | 0.6-0.9 | High erosion, channels cutting deep |
| Ridge tops | 0.4-0.7 | Moderate erosion, exposed to weathering |
| Gentle uplands | 0.1-0.3 | Low erosion, thick soil develops |
| Steep cliffs | 0.8-1.0 | Very high erosion, bedrock exposed |
| Flat basins | 0.0-0.2 | Deposition zone, sediments accumulate |

---

## Part 5: Cover Layer Constraints (Soil, Colluvium, Alluvium)

### Rule 4: Soil Thickness vs Erosion

Soil is thinner where erosion is strong:

```python
def compute_soil_thickness(E, slope_norm, soil_thick_min=0.3, soil_thick_max=2.0):
    """
    Soil thickness decreases with erosion intensity.
    Also reduced on steep slopes.
    """
    # Base thickness from erosion
    soil_target = soil_thick_max - (soil_thick_max - soil_thick_min) * E
    
    # Further reduce on steep slopes
    slope_factor = 1.0 - slope_norm ** 2
    soil_target *= slope_factor
    
    # Minimum viable soil
    soil_target = np.maximum(soil_target, soil_thick_min * 0.5)
    
    return soil_target
```

### Rule 5: Colluvium Location and Thickness

Colluvium accumulates on slopes with convergent flow:

```python
def compute_colluvium_thickness(
    slope_norm, 
    curvature, 
    wetness,
    E,
    coll_min=0.0,
    coll_max=20.0,
):
    """
    Colluvium is thickest where:
    - Slope is moderate to steep (transport zone)
    - Curvature is concave (flow convergence / hollows)
    - Wetness is moderate (not main channel, but collecting)
    
    Reduced where erosion is extremely high (active removal).
    """
    # Slope factor: peaks at moderate slopes (15-30°)
    slope_opt = 0.4  # ~18° optimal
    slope_factor = 1.0 - np.abs(slope_norm - slope_opt) / 0.6
    slope_factor = np.clip(slope_factor, 0, 1)
    
    # Curvature factor: concave (positive) = more colluvium
    curv_norm = normalize(curvature)
    hollow_factor = np.clip(curv_norm, 0, 1)  # Only concave areas
    
    # Wetness factor: moderate wetness preferred
    wetness_factor = wetness * (1 - wetness)  # Peaks at 0.5
    wetness_factor = normalize(wetness_factor)
    
    # Combine into colluvium index
    C = 0.4 * slope_factor + 0.4 * hollow_factor + 0.2 * wetness_factor
    C = normalize(C)
    
    # Erosion suppression: very high erosion removes colluvium
    erosion_suppress = np.clip(1.0 - (E - 0.7) / 0.3, 0, 1)
    C *= erosion_suppress
    
    thickness = coll_min + (coll_max - coll_min) * C
    return thickness
```

### Rule 6: Alluvium Restricted to Valley Floors

Alluvium only exists in flat, wet valley bottoms:

```python
def compute_alluvium_thickness(
    slope_norm,
    wetness,
    E,
    z_norm,
    alluv_min=0.0,
    alluv_max=30.0,
    slope_threshold=0.15,
    wetness_threshold=0.6,
):
    """
    Alluvium ONLY in valley floors:
    - Low slope (flat)
    - High wetness (channels/floodplains)
    - Low elevation (valleys, not upland flats)
    """
    # Valley floor mask
    is_flat = slope_norm < slope_threshold
    is_wet = wetness > wetness_threshold
    is_low = z_norm < 0.4  # Lower 40% of elevation
    
    valley_mask = is_flat & is_wet & is_low
    
    # Strength within valley (how "valley-like")
    valley_strength = np.zeros_like(slope_norm)
    valley_strength[valley_mask] = (
        (1 - slope_norm[valley_mask] / slope_threshold) *
        ((wetness[valley_mask] - wetness_threshold) / (1 - wetness_threshold)) *
        (1 - z_norm[valley_mask] / 0.4)
    )
    valley_strength = np.clip(valley_strength, 0, 1)
    
    # Thickness
    thickness = alluv_max * valley_strength
    
    return thickness, valley_mask
```

---

## Part 6: Deep Layer Constraints (Sandstone, Shale, Limestone, Basement)

### Rule 7: Minimum Sediment Cover Outside Structural Highs

**This is the key rule that prevents basement from appearing everywhere:**

```python
def enforce_minimum_sediment_cover(
    interfaces,
    surface_elev,
    structural_high_mask,
    sediment_min_depth=300.0,  # Minimum meters of sediment outside structural highs
):
    """
    Outside structural high zones, enforce minimum depth to basement.
    This prevents basement from appearing under every small hill.
    """
    # Current sediment thickness
    current_sed_depth = surface_elev - interfaces["Basement"]
    
    # Where sediment is too thin AND not in structural high
    needs_deepening = (current_sed_depth < sediment_min_depth) & (~structural_high_mask)
    
    # Push basement down
    target_basement = surface_elev - sediment_min_depth
    interfaces["Basement"] = np.where(
        needs_deepening,
        target_basement,
        interfaces["Basement"]
    )
    
    # Also push down BasementFloor
    interfaces["BasementFloor"] = np.minimum(
        interfaces["BasementFloor"],
        interfaces["Basement"] - 100.0  # At least 100m of basement thickness
    )
    
    return interfaces
```

### Rule 8: Deep Layers Only Exposed in Valid Structural Zones

```python
def constrain_deep_layer_exposure(
    interfaces,
    surface_elev,
    E,
    structural_high_mask,
    E_sandstone_threshold=0.5,
    E_shale_threshold=0.6,
    E_limestone_threshold=0.7,
    E_basement_threshold=0.85,
):
    """
    Deep layers can only reach near-surface where:
    1. structural_high_mask is True (valid uplift zone)
    2. Erosion intensity exceeds threshold for that layer
    
    Otherwise, push the layer down.
    """
    cover_layers = ["Topsoil", "Subsoil", "Colluvium", "Alluvium", 
                    "Saprolite", "WeatheredBR", "Clay", "Silt", "Sand"]
    
    # Compute total cover thickness
    cover_bottom = interfaces["Sand"]  # Bottom of unconsolidated cover
    
    # --- SANDSTONE ---
    # Can approach surface only if structural high + sufficient erosion
    sandstone_allowed_near_surface = structural_high_mask & (E >= E_sandstone_threshold)
    
    # Where NOT allowed, enforce minimum cover
    min_cover_for_sandstone = 20.0  # At least 20m of cover
    sandstone_max_elev = surface_elev - min_cover_for_sandstone
    
    interfaces["Sandstone"] = np.where(
        sandstone_allowed_near_surface,
        interfaces["Sandstone"],  # Keep as-is
        np.minimum(interfaces["Sandstone"], sandstone_max_elev)
    )
    
    # --- SHALE ---
    shale_allowed_near_surface = structural_high_mask & (E >= E_shale_threshold)
    min_cover_for_shale = 50.0
    shale_max_elev = surface_elev - min_cover_for_shale
    
    interfaces["Shale"] = np.where(
        shale_allowed_near_surface,
        interfaces["Shale"],
        np.minimum(interfaces["Shale"], shale_max_elev)
    )
    
    # --- LIMESTONE ---
    limestone_allowed_near_surface = structural_high_mask & (E >= E_limestone_threshold)
    min_cover_for_limestone = 100.0
    limestone_max_elev = surface_elev - min_cover_for_limestone
    
    interfaces["Limestone"] = np.where(
        limestone_allowed_near_surface,
        interfaces["Limestone"],
        np.minimum(interfaces["Limestone"], limestone_max_elev)
    )
    
    # --- BASEMENT ---
    basement_allowed_near_surface = structural_high_mask & (E >= E_basement_threshold)
    min_cover_for_basement = 300.0
    basement_max_elev = surface_elev - min_cover_for_basement
    
    interfaces["Basement"] = np.where(
        basement_allowed_near_surface,
        interfaces["Basement"],
        np.minimum(interfaces["Basement"], basement_max_elev)
    )
    
    return interfaces
```

---

## Part 7: Structural Uplift and Anticlines

### Rule 9: Structural Uplift Field

```python
def generate_structural_uplift(
    N,
    rng,
    pixel_scale_m,
    n_anticlines=3,
    uplift_amp_range=(50.0, 200.0),
    uplift_sigma_range=(0.1, 0.3),  # As fraction of domain
):
    """
    Generate structural uplift field with anticlines/domes.
    This pushes rock layers UP in certain zones.
    """
    U = np.zeros((N, N), dtype=np.float64)
    
    domain_size = N * pixel_scale_m
    
    for _ in range(n_anticlines):
        # Random center
        cx = rng.uniform(0.2, 0.8) * N
        cy = rng.uniform(0.2, 0.8) * N
        
        # Random amplitude and width
        amp = rng.uniform(*uplift_amp_range)
        sigma = rng.uniform(*uplift_sigma_range) * N
        
        # Gaussian dome
        ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
        r2 = (ii - cx)**2 + (jj - cy)**2
        U += amp * np.exp(-r2 / (2 * sigma**2))
    
    # Add some long-wavelength variation
    from your_terrain_module import fractional_surface
    regional = fractional_surface(N, beta=4.0, rng=rng) * 50.0
    U += regional
    
    return U

def apply_structural_uplift(interfaces, U, rock_layers_only=True):
    """
    Apply uplift to rock interfaces (not soil/cover).
    """
    rock_layers = ["Sandstone", "Shale", "Limestone", "Basement", "BasementFloor"]
    
    for layer in interfaces:
        if rock_layers_only and layer not in rock_layers:
            continue
        interfaces[layer] = interfaces[layer] + U
    
    return interfaces
```

### Rule 10: Derive Structural High Mask

```python
def compute_structural_high_mask(
    U,
    z_norm,
    uplift_threshold=50.0,
    elevation_threshold=0.5,
):
    """
    Structural highs are where:
    1. Uplift exceeds threshold, OR
    2. Elevation is high AND some uplift present
    
    These are the only places where deep layers can reach surface.
    """
    # Direct uplift zones
    uplift_highs = U > uplift_threshold
    
    # High elevation + moderate uplift
    elev_highs = (z_norm > elevation_threshold) & (U > uplift_threshold * 0.3)
    
    # Combine
    structural_high_mask = uplift_highs | elev_highs
    
    # Smooth to avoid pixel-scale boundaries
    structural_high_mask = box_blur(structural_high_mask.astype(float), k=11) > 0.3
    
    return structural_high_mask
```

---

## Part 8: Progressive Stripping Logic

### Rule 11: Strip Cover Layers Based on Erosion

```python
def apply_progressive_stripping(
    interfaces,
    surface_elev,
    E,
    structural_high_mask,
    cover_max_total=50.0,  # Max total cover thickness at E=0
    cover_min_total=2.0,   # Min total cover thickness at E=1
):
    """
    As erosion increases, strip away cover layers progressively.
    
    E ~ 0: Thick soil, colluvium, alluvium
    E ~ 0.5: Thin soil, some colluvium, little alluvium
    E ~ 1.0: Almost no cover, rock at surface (if structural high)
    """
    # Target total cover thickness based on erosion
    cover_target = cover_max_total - (cover_max_total - cover_min_total) * E
    
    # Current cover thickness
    cover_layers = ["Topsoil", "Subsoil", "Colluvium", "Alluvium", 
                    "Saprolite", "WeatheredBR"]
    
    # Find bottom of cover (top of rock)
    rock_top = interfaces["Sandstone"]  # Or whatever is the top rock layer
    
    # Current total cover
    current_cover = surface_elev - rock_top
    
    # Where cover exceeds target, compress it
    compression_factor = np.clip(cover_target / (current_cover + 1e-6), 0, 1)
    
    # Apply compression to cover layers
    for i, layer in enumerate(cover_layers):
        if layer not in interfaces:
            continue
        
        # Distance from surface
        depth_from_surface = surface_elev - interfaces[layer]
        
        # Compressed depth
        new_depth = depth_from_surface * compression_factor
        
        # New interface position
        interfaces[layer] = surface_elev - new_depth
    
    # In very high erosion + structural high: allow rock at surface
    rock_at_surface = (E > 0.8) & structural_high_mask
    
    if np.any(rock_at_surface):
        # Set cover to minimum viable
        min_cover = 0.5  # Just a veneer
        for layer in cover_layers:
            if layer not in interfaces:
                continue
            interfaces[layer] = np.where(
                rock_at_surface,
                surface_elev - min_cover,
                interfaces[layer]
            )
    
    return interfaces
```

---

## Part 9: Complete Pipeline Integration

### Master Function

```python
def generate_stratigraphy_constrained(
    z_norm,
    rng,
    elev_range_m=700.0,
    pixel_scale_m=10.0,
    # Cover layer params
    soil_thick_range=(0.3, 2.0),
    colluvium_max=20.0,
    alluvium_max=30.0,
    # Rock layer params
    sandstone_thickness=80.0,
    shale_thickness=100.0,
    limestone_thickness=60.0,
    # Constraint params
    sediment_min_depth=300.0,
    n_anticlines=3,
    # Erosion thresholds
    E_sandstone_threshold=0.5,
    E_basement_threshold=0.85,
):
    """
    Generate stratigraphy using constraint-based approach.
    """
    N = z_norm.shape[0]
    surface_elev = z_norm * elev_range_m
    
    # ============ STEP 1: Compute surface derivatives ============
    dEy, dEx = np.gradient(surface_elev, pixel_scale_m, pixel_scale_m)
    slope_mag = np.hypot(dEx, dEy)
    slope_deg = np.rad2deg(np.arctan(slope_mag))
    slope_norm = normalize(slope_mag)
    
    # Curvature
    d2x, _ = np.gradient(dEx)
    _, d2y = np.gradient(dEy)
    curvature = d2x + d2y
    
    # Wetness proxy
    catch = box_blur(box_blur(1.0 - slope_norm, k=7), k=13)
    wetness = normalize(catch - slope_norm)
    
    # ============ STEP 2: Compute erosion intensity ============
    E, E_components = compute_erosion_intensity(
        surface_elev, pixel_scale_m
    )
    
    # ============ STEP 3: Generate structural uplift ============
    U = generate_structural_uplift(N, rng, pixel_scale_m, n_anticlines)
    structural_high_mask = compute_structural_high_mask(U, z_norm)
    
    # ============ STEP 4: Compute cover layer thicknesses ============
    soil_thick = compute_soil_thickness(E, slope_norm, *soil_thick_range)
    colluv_thick = compute_colluvium_thickness(slope_norm, curvature, wetness, E, 0, colluvium_max)
    alluv_thick, valley_mask = compute_alluvium_thickness(slope_norm, wetness, E, z_norm, 0, alluvium_max)
    
    # Saprolite and weathered rind (simplified)
    saprolite_thick = 5.0 * (1 - E) * (1 - slope_norm)
    weathered_thick = 2.0 * (1 - E * 0.5)
    
    # ============ STEP 5: Build initial interfaces (top-down from surface) ============
    interfaces = {}
    
    # Cover layers
    interfaces["Topsoil"] = surface_elev - soil_thick * 0.4
    interfaces["Subsoil"] = interfaces["Topsoil"] - soil_thick * 0.6
    interfaces["Colluvium"] = interfaces["Subsoil"] - colluv_thick
    interfaces["Alluvium"] = interfaces["Colluvium"] - alluv_thick
    interfaces["Saprolite"] = interfaces["Alluvium"] - saprolite_thick
    interfaces["WeatheredBR"] = interfaces["Saprolite"] - weathered_thick
    
    # Rock layers - start from bottom of weathered material
    cover_bottom = interfaces["WeatheredBR"]
    
    # Apply structural uplift to determine rock layer positions
    rock_base_ref = cover_bottom - 50.0  # Some gap
    rock_base_ref += U  # Uplift brings rock up
    
    interfaces["Sandstone"] = rock_base_ref
    interfaces["Shale"] = interfaces["Sandstone"] - sandstone_thickness
    interfaces["Limestone"] = interfaces["Shale"] - shale_thickness
    interfaces["Basement"] = interfaces["Limestone"] - limestone_thickness
    interfaces["BasementFloor"] = interfaces["Basement"] - 500.0
    
    # ============ STEP 6: Apply constraints ============
    
    # 6a. Enforce minimum sediment cover outside structural highs
    interfaces = enforce_minimum_sediment_cover(
        interfaces, surface_elev, structural_high_mask, sediment_min_depth
    )
    
    # 6b. Constrain deep layer exposure
    interfaces = constrain_deep_layer_exposure(
        interfaces, surface_elev, E, structural_high_mask,
        E_sandstone_threshold=E_sandstone_threshold,
        E_basement_threshold=E_basement_threshold,
    )
    
    # 6c. Apply progressive stripping
    interfaces = apply_progressive_stripping(
        interfaces, surface_elev, E, structural_high_mask
    )
    
    # ============ STEP 7: Enforce ordering ============
    interfaces = enforce_ordering(interfaces, layer_order)
    
    # ============ STEP 8: Ensure no interface above surface ============
    for layer in interfaces:
        interfaces[layer] = np.minimum(interfaces[layer], surface_elev - 0.01)
    
    # ============ STEP 9: Compute thicknesses ============
    thickness = {}
    for i, layer in enumerate(layer_order[:-1]):
        below = layer_order[i + 1]
        if layer in interfaces and below in interfaces:
            thickness[layer] = np.maximum(interfaces[layer] - interfaces[below], 0.0)
    
    # ============ STEP 10: Return results ============
    return {
        "surface_elev": surface_elev,
        "interfaces": interfaces,
        "thickness": thickness,
        "erosion_intensity": E,
        "structural_uplift": U,
        "structural_high_mask": structural_high_mask,
        "meta": {
            "elev_range_m": elev_range_m,
            "pixel_scale_m": pixel_scale_m,
        }
    }
```

---

## Part 10: Visual Verification

### Expected Results

After implementing these constraints:

| Location | Cover Thickness | Rock Exposure | Basement |
|----------|-----------------|---------------|----------|
| Flat basin | Thick (30-50m) | None visible | Deep (300m+) |
| Gentle upland | Moderate (10-30m) | None visible | Deep (300m+) |
| Valley floor | Thick alluvium (20-40m) | None visible | Deep (300m+) |
| Moderate slope | Thin soil (2-5m) | Sandstone may show | Deep unless structural high |
| Steep ridge | Very thin (<2m) | Sandstone/Shale if structural high | 300m+ unless extreme erosion |
| Mountain peak (structural high) | Veneer (<1m) | Limestone/Basement possible | Can approach surface if E > 0.85 |

### Cross-Section Appearance

```
                    BEFORE (current code)                    AFTER (constrained)
                    
Mountain:          Thick sediments inside              Only thin veneer, basement near surface
    /\                  /\                                      /\
   /SS\                /  \                                    /BB\
  /SHSH\              /    \                                  /BBBB\
 /LIMELI\            /      \                                /BBBBBB\

Basin:             Same as mountain                    Thick sediments preserved  
  ____                ____                                    ____
 |SSSS|              |SSSS|                                  |SSSS|
 |SHSH|              |SHSH|                                  |SHSH|
 |LIME|              |LIME|                                  |LIME|
 |BASE|              |BASE|                                  |BASE|

SS=Sandstone, SH=Shale, LI=Limestone, BB=Basement
```

---

## Part 11: Implementation Checklist

### Step-by-Step Integration

- [ ] Add helper functions: `normalize()`, `box_blur()`, `lerp()`
- [ ] Implement `compute_erosion_intensity()` 
- [ ] Implement cover layer thickness functions
- [ ] Implement `generate_structural_uplift()` and `compute_structural_high_mask()`
- [ ] Implement constraint functions:
  - [ ] `enforce_minimum_sediment_cover()`
  - [ ] `constrain_deep_layer_exposure()`
  - [ ] `apply_progressive_stripping()`
- [ ] Implement `enforce_ordering()`
- [ ] Modify `generate_stratigraphy()` to use new constraint pipeline
- [ ] Test with cross-section visualization
- [ ] Verify basin/mountain thickness ratios

### Key Parameters to Tune

| Parameter | Suggested Range | Effect |
|-----------|-----------------|--------|
| `sediment_min_depth` | 200-500m | How deep basement stays in basins |
| `E_basement_threshold` | 0.8-0.95 | How much erosion needed to expose basement |
| `E_sandstone_threshold` | 0.4-0.6 | When sandstone can appear at surface |
| `cover_max_total` | 30-80m | Maximum cover thickness in low-erosion areas |
| `n_anticlines` | 2-5 | Number of structural high zones |
| `uplift_amp_range` | (50, 200)m | How much uplift in anticlines |

---

## Summary

The key insight is: **Don't compute thicknesses and hope they work out. Instead:**

1. **Start from the surface** and work down
2. **Compute WHERE erosion is strong** from surface shape
3. **Explicitly constrain** which layers can appear where
4. **Only expose basement** in designated structural high zones with sufficient erosion
5. **Enforce ordering** and surface constraints at the end

This produces geologically realistic stratigraphy where:
- Basins have thick sedimentary cover
- Mountains expose basement only where structurally justified
- Erosion progressively strips cover layers
- No arbitrary thick sediments inside mountains
