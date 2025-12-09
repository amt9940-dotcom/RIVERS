"""
CELL 1: TERRAIN GENERATOR (FIXED)

Major fixes:
- Wind barrier/channel detection uses larger-scale analysis
- Fewer, more coherent features
- Better orographic low-pressure computation
"""
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 1. QUANTUM-SEEDED TOPOGRAPHY
# ==============================================================================

def quantum_seeded_topography(N=50, random_seed=42, scale=3.0, octaves=6):
    """
    Generate quantum-seeded terrain using multi-scale noise.
    
    Args:
        N: grid size
        random_seed: for reproducibility
        scale: controls feature size
        octaves: number of noise scales
    
    Returns:
        z_norm: normalized elevation (0-1)
        rng: random number generator
    """
    rng = np.random.default_rng(random_seed)
    
    # Multi-octave noise
    elevation = np.zeros((N, N))
    amplitude = 1.0
    frequency = 1.0
    
    for _ in range(octaves):
        noise = rng.random((N, N)) - 0.5
        # Smooth at current frequency
        from scipy.ndimage import gaussian_filter
        smoothed = gaussian_filter(noise, sigma=scale / frequency, mode='wrap')
        elevation += amplitude * smoothed
        amplitude *= 0.5
        frequency *= 2.0
    
    # Normalize to 0-1
    z_norm = (elevation - elevation.min()) / (elevation.max() - elevation.min() + 1e-9)
    
    return z_norm, rng


def generate_stratigraphy(elevation_normalized, pixel_scale_m=1000.0, 
                          base_elevation_m=1000.0, relief_m=400.0):
    """
    Generate stratigraphy from normalized elevation.
    
    Args:
        elevation_normalized: 2D array (0-1)
        pixel_scale_m: cell size
        base_elevation_m: minimum elevation
        relief_m: elevation range
    
    Returns:
        strata: dict with surface_elev, interfaces, thickness, properties
    """
    ny, nx = elevation_normalized.shape
    
    # Convert to actual elevation
    surface_elev = base_elevation_m + elevation_normalized * relief_m
    
    # Simple layered stratigraphy
    strata = {
        "surface_elev": surface_elev.copy(),
        "interfaces": {},
        "thickness": {},
        "properties": {},
    }
    
    # Define layers (top to bottom)
    layers = {
        "Topsoil": {"thickness": 5.0, "erodibility": 1.0},
        "Saprolite": {"thickness": 20.0, "erodibility": 0.8},
        "Sandstone": {"thickness": 50.0, "erodibility": 0.5},
        "Basement": {"thickness": 100.0, "erodibility": 0.1},
    }
    
    current_interface = surface_elev.copy()
    
    for layer_name, props in layers.items():
        thickness = np.ones((ny, nx)) * props["thickness"]
        interface = current_interface - thickness
        
        strata["thickness"][layer_name] = thickness
        strata["interfaces"][layer_name] = interface
        strata["properties"][layer_name] = {"erodibility": props["erodibility"]}
        
        current_interface = interface
    
    # BasementFloor (for reference)
    strata["interfaces"]["BasementFloor"] = current_interface - 50000.0
    
    return strata


# ==============================================================================
# 2. TOPOGRAPHIC ANALYSIS (FIXED)
# ==============================================================================

def compute_topo_fields(E, pixel_scale_m=1000.0):
    """Compute slope, aspect, curvature."""
    from scipy.ndimage import gaussian_filter
    
    # Smooth slightly to reduce noise
    E_smooth = gaussian_filter(E, sigma=1.0, mode='wrap')
    
    dy, dx = np.gradient(E_smooth, pixel_scale_m)
    slope_mag = np.sqrt(dx**2 + dy**2)
    
    # Aspect (direction of steepest descent)
    aspect = np.arctan2(-dy, -dx)  # radians, -π to π
    
    # Curvature (laplacian)
    dxx = np.gradient(dx, pixel_scale_m, axis=1)
    dyy = np.gradient(dy, pixel_scale_m, axis=0)
    laplacian = dxx + dyy
    
    return {
        "dEx": dx,
        "dEy": dy,
        "slope": slope_mag,
        "aspect": aspect,
        "laplacian": laplacian,
    }


def classify_wind_barriers_FIXED(E, slope_mag, dEx, dEy, base_wind_dir_deg,
                                   prominence_window=15, slope_thresh=0.10,
                                   aspect_tolerance_deg=60):
    """
    FIXED: Detect major ridges/walls that block wind.
    
    Criteria (STRICTER):
    - Locally high (top 20% in large window)
    - Steep (slope > threshold)
    - Faces into wind (aspect within tolerance)
    - Connected features (morphological closing)
    
    Returns boolean mask with fewer, larger features.
    """
    from scipy.ndimage import maximum_filter, gaussian_filter, binary_closing
    
    ny, nx = E.shape
    
    # 1. LOCAL PROMINENCE (stricter: top 20% in large window)
    E_smooth = gaussian_filter(E, sigma=2.0, mode='wrap')
    local_max = maximum_filter(E_smooth, size=prominence_window, mode='wrap')
    prominence = E_smooth - local_max
    
    # Only keep top 20% (was too lenient before)
    prominence_thresh = np.percentile(prominence, 80)
    is_high = prominence > prominence_thresh
    
    # 2. STEEP (stricter threshold)
    is_steep = slope_mag > slope_thresh
    
    # 3. FACES WIND
    wind_dir_rad = np.deg2rad(base_wind_dir_deg)
    wind_vec = np.array([np.cos(wind_dir_rad), np.sin(wind_dir_rad)])
    
    # Slope faces wind if dot(gradient, wind) > 0
    dot_product = dEx * wind_vec[0] + dEy * wind_vec[1]
    
    # Normalize by magnitude
    dot_norm = dot_product / (slope_mag + 1e-9)
    
    # Aspect tolerance
    cos_tolerance = np.cos(np.deg2rad(aspect_tolerance_deg))
    faces_wind = dot_norm > cos_tolerance
    
    # 4. COMBINE
    barrier_mask = is_high & is_steep & faces_wind
    
    # 5. MORPHOLOGICAL CLOSING (connect nearby features, remove isolated pixels)
    structure = np.ones((5, 5))
    barrier_mask = binary_closing(barrier_mask, structure=structure, iterations=2)
    
    # 6. SIZE FILTER (remove tiny features)
    from scipy.ndimage import label
    labeled, num_features = label(barrier_mask)
    for i in range(1, num_features + 1):
        feature_size = np.sum(labeled == i)
        if feature_size < 10:  # Remove features smaller than 10 pixels
            barrier_mask[labeled == i] = False
    
    return barrier_mask


def classify_wind_channels_FIXED(E, slope_mag, laplacian, dEx, dEy, base_wind_dir_deg,
                                   depression_window=15, slope_min=0.02, slope_max=0.25,
                                   curvature_thresh=-0.02, alignment_thresh_deg=45):
    """
    FIXED: Detect valleys/passes that guide wind.
    
    Criteria (LESS STRICT):
    - Locally low (bottom 30% in large window)
    - Moderate slope
    - Concave (negative laplacian)
    - Aligned with wind
    - Connected features
    
    Returns boolean mask with more, connected valleys.
    """
    from scipy.ndimage import minimum_filter, gaussian_filter, binary_closing
    
    ny, nx = E.shape
    
    # 1. LOCAL DEPRESSION (less strict: bottom 30%)
    E_smooth = gaussian_filter(E, sigma=2.0, mode='wrap')
    local_min = minimum_filter(E_smooth, size=depression_window, mode='wrap')
    depression = E_smooth - local_min
    
    # Bottom 30%
    depression_thresh = np.percentile(depression, 30)
    is_low = depression < depression_thresh
    
    # 2. MODERATE SLOPE (not too flat, not too steep)
    is_moderate_slope = (slope_mag > slope_min) & (slope_mag < slope_max)
    
    # 3. CONCAVE (negative laplacian = valley)
    is_concave = laplacian < curvature_thresh
    
    # 4. ALIGNED WITH WIND
    wind_dir_rad = np.deg2rad(base_wind_dir_deg)
    wind_vec = np.array([np.cos(wind_dir_rad), np.sin(wind_dir_rad)])
    
    # Valley axis is perpendicular to gradient
    # Check if gradient is roughly perpendicular to wind (valley parallel to wind)
    dot_product = dEx * wind_vec[0] + dEy * wind_vec[1]
    dot_norm = np.abs(dot_product) / (slope_mag + 1e-9)
    
    # Aligned if gradient is NOT parallel to wind (i.e., valley IS parallel)
    cos_thresh = np.cos(np.deg2rad(alignment_thresh_deg))
    is_aligned = dot_norm < (1 - cos_thresh)  # Inverted logic for valleys
    
    # 5. COMBINE
    channel_mask = is_low & is_moderate_slope & is_concave & is_aligned
    
    # 6. MORPHOLOGICAL CLOSING
    structure = np.ones((5, 5))
    channel_mask = binary_closing(channel_mask, structure=structure, iterations=2)
    
    # 7. SIZE FILTER
    from scipy.ndimage import label
    labeled, num_features = label(channel_mask)
    for i in range(1, num_features + 1):
        feature_size = np.sum(labeled == i)
        if feature_size < 10:
            channel_mask[labeled == i] = False
    
    return channel_mask


def classify_windward_leeward(E, dEx, dEy, slope_mag, base_wind_dir_deg, slope_thresh=0.05):
    """Classify windward vs leeward slopes."""
    wind_dir_rad = np.deg2rad(base_wind_dir_deg)
    wind_vec = np.array([np.cos(wind_dir_rad), np.sin(wind_dir_rad)])
    
    # Dot product: positive = faces wind (windward)
    dot_product = dEx * wind_vec[0] + dEy * wind_vec[1]
    dot_norm = dot_product / (slope_mag + 1e-9)
    
    windward = (dot_norm > 0.3) & (slope_mag > slope_thresh)
    leeward = (dot_norm < -0.3) & (slope_mag > slope_thresh)
    
    return windward, leeward


def build_wind_structures(E, pixel_scale_m, base_wind_dir_deg):
    """Build wind structure masks (FIXED)."""
    topo = compute_topo_fields(E, pixel_scale_m)
    
    # FIXED detection functions
    barrier_mask = classify_wind_barriers_FIXED(
        E, topo["slope"], topo["dEx"], topo["dEy"], base_wind_dir_deg,
        prominence_window=15, slope_thresh=0.10, aspect_tolerance_deg=60
    )
    
    channel_mask = classify_wind_channels_FIXED(
        E, topo["slope"], topo["laplacian"], topo["dEx"], topo["dEy"], base_wind_dir_deg,
        depression_window=15, slope_min=0.02, slope_max=0.25,
        curvature_thresh=-0.02, alignment_thresh_deg=45
    )
    
    windward_mask, leeward_mask = classify_windward_leeward(
        E, topo["dEx"], topo["dEy"], topo["slope"], base_wind_dir_deg
    )
    
    # Basin detection (unchanged)
    from scipy.ndimage import minimum_filter
    local_min = minimum_filter(E, size=11, mode='wrap')
    is_local_min = (E == local_min)
    basin_mask = is_local_min
    
    return {
        "barrier_mask": barrier_mask,
        "channel_mask": channel_mask,
        "windward_mask": windward_mask,
        "leeward_mask": leeward_mask,
        "basin_mask": basin_mask,
        "topo_fields": topo,
    }


def compute_orographic_low_pressure(E, rng, pixel_scale_m, base_wind_dir_deg,
                                      wind_structs, scale_factor=1.5, orographic_weight=0.7):
    """
    FIXED: Compute low-pressure likelihood with stronger topographic control.
    """
    from scipy.ndimage import gaussian_filter
    
    ny, nx = E.shape
    
    # Base random field (less weight)
    base_random = rng.random((ny, nx))
    base_random = gaussian_filter(base_random, sigma=3.0, mode='wrap')
    
    # Topographic contribution (more weight)
    topo_contribution = np.zeros((ny, nx))
    
    # Strong signal at barriers (forced ascent)
    topo_contribution += wind_structs["barrier_mask"].astype(float) * 2.0
    
    # Strong signal in channels (convergence)
    topo_contribution += wind_structs["channel_mask"].astype(float) * 1.5
    
    # Moderate signal on windward slopes
    topo_contribution += wind_structs["windward_mask"].astype(float) * 0.8
    
    # Weak signal in basins (convection)
    topo_contribution += wind_structs["basin_mask"].astype(float) * 0.3
    
    # Smooth topographic contribution
    topo_contribution = gaussian_filter(topo_contribution, sigma=2.0, mode='wrap')
    
    # Combine with stronger orographic weight
    low_pressure = (1 - orographic_weight) * base_random + orographic_weight * topo_contribution
    
    # Normalize to 0-1
    low_pressure = (low_pressure - low_pressure.min()) / (low_pressure.max() - low_pressure.min() + 1e-9)
    
    return {
        "low_pressure_likelihood": low_pressure,
        "topo_contribution": topo_contribution,
        "base_random": base_random,
    }


print("✓ Terrain generator (FIXED) loaded successfully!")
print("  Key improvements:")
print("    - Stricter barrier detection (top 20% prominence)")
print("    - Less strict channel detection (bottom 30% depression)")
print("    - Morphological operations to connect features")
print("    - Size filtering to remove tiny speckles")
print("    - Stronger orographic weight in weather (70%)")
