"""
Constraint-Based Stratigraphic Layer Generation

This module implements a constraint-based approach where layers are built
from the surface downward, with explicit rules about where deep layers can appear.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def normalize(arr):
    """Normalize array to [0, 1]"""
    arr_min = float(np.min(arr))
    arr_max = float(np.max(arr))
    if arr_max - arr_min < 1e-12:
        return np.zeros_like(arr)
    return (arr - arr_min) / (arr_max - arr_min)


def box_blur(arr, k=5):
    """Simple box blur / moving average"""
    from scipy.ndimage import uniform_filter
    return uniform_filter(arr.astype(np.float64), size=k, mode='reflect')


def compute_erosion_intensity(
    surface_elev,
    pixel_scale_m,
    w_slope=0.35,
    w_relief=0.35,
    w_channel=0.20,
    w_elevation=0.10,
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
    slope_norm = normalize(slope_mag)
    catch = box_blur(box_blur(1.0 - slope_norm, k=7), k=13)
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
        "slope_norm": slope_norm,
        "wetness": wetness,
        "curvature": dEx + dEy,  # Simplified curvature
    }


def generate_structural_uplift(
    N,
    rng,
    pixel_scale_m,
    n_anticlines=3,
    uplift_amp_range=(50.0, 200.0),
    uplift_sigma_range=(0.1, 0.3),
):
    """
    Generate structural uplift field with anticlines/domes.
    This pushes rock layers UP in certain zones.
    """
    U = np.zeros((N, N), dtype=np.float64)
    
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
    
    # Add some long-wavelength variation (simple version)
    regional = rng.standard_normal((N, N)) * 30.0
    regional = gaussian_filter(regional, sigma=N*0.15)
    U += regional
    
    # Smooth the uplift field
    U = box_blur(U, k=max(31, int(0.1 * N) | 1))
    
    return U


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


def compute_cover_thicknesses(E, slope_norm, wetness, curvature, z_norm):
    """
    Compute thickness of cover layers (soil, colluvium, alluvium, saprolite).
    All respond to erosion intensity E.
    """
    # Soil: thinner where erosion is strong
    soil_thick_min, soil_thick_max = 0.5, 3.0
    soil_target = soil_thick_max - (soil_thick_max - soil_thick_min) * E
    slope_factor = 1.0 - slope_norm ** 2
    soil_thickness = np.maximum(soil_target * slope_factor, soil_thick_min * 0.3)
    
    # Colluvium: peaks at moderate slopes, concave areas
    slope_opt = 0.4  # ~18° optimal
    slope_factor_coll = 1.0 - np.abs(slope_norm - slope_opt) / 0.6
    slope_factor_coll = np.clip(slope_factor_coll, 0, 1)
    
    curv_norm = normalize(curvature)
    hollow_factor = np.clip(curv_norm, 0, 1)  # Only concave
    
    wetness_factor = wetness * (1 - wetness)  # Peaks at 0.5
    wetness_factor = normalize(wetness_factor)
    
    C = 0.4 * slope_factor_coll + 0.4 * hollow_factor + 0.2 * wetness_factor
    C = normalize(C)
    
    # Erosion suppression
    erosion_suppress = np.clip(1.0 - (E - 0.7) / 0.3, 0, 1)
    C *= erosion_suppress
    
    colluvium_thickness = 25.0 * C  # Up to 25m
    
    # Alluvium: ONLY in valley floors
    slope_threshold = 0.15
    wetness_threshold = 0.6
    
    is_flat = slope_norm < slope_threshold
    is_wet = wetness > wetness_threshold
    is_low = z_norm < 0.4
    
    valley_mask = is_flat & is_wet & is_low
    
    valley_strength = np.zeros_like(slope_norm)
    valley_strength[valley_mask] = (
        (1 - slope_norm[valley_mask] / slope_threshold) *
        ((wetness[valley_mask] - wetness_threshold) / (1 - wetness_threshold)) *
        (1 - z_norm[valley_mask] / 0.4)
    )
    valley_strength = np.clip(valley_strength, 0, 1)
    
    alluvium_thickness = 40.0 * valley_strength  # Up to 40m
    
    # Saprolite: thick where stable (low E, low slope)
    saprolite_base = 8.0
    saprolite_thickness = saprolite_base * (1 - E) * (1 - slope_norm**2)
    saprolite_thickness = np.clip(saprolite_thickness, 0.5, 30.0)
    
    # Weathered bedrock rind
    weathered_thickness = 3.0 * (1 - E * 0.5)
    weathered_thickness = np.clip(weathered_thickness, 0.5, 8.0)
    
    return {
        "soil": soil_thickness,
        "colluvium": colluvium_thickness,
        "alluvium": alluvium_thickness,
        "saprolite": saprolite_thickness,
        "weathered": weathered_thickness,
    }


def enforce_minimum_sediment_cover(
    interfaces,
    surface_elev,
    structural_high_mask,
    sediment_min_depth=300.0,
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
        interfaces["Basement"] - 100.0
    )
    
    return interfaces


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
    """
    
    # SANDSTONE
    sandstone_allowed = structural_high_mask & (E >= E_sandstone_threshold)
    min_cover_sandstone = 20.0
    sandstone_max_elev = surface_elev - min_cover_sandstone
    
    interfaces["Sandstone"] = np.where(
        sandstone_allowed,
        interfaces["Sandstone"],
        np.minimum(interfaces["Sandstone"], sandstone_max_elev)
    )
    
    # SHALE
    shale_allowed = structural_high_mask & (E >= E_shale_threshold)
    min_cover_shale = 50.0
    shale_max_elev = surface_elev - min_cover_shale
    
    interfaces["Shale"] = np.where(
        shale_allowed,
        interfaces["Shale"],
        np.minimum(interfaces["Shale"], shale_max_elev)
    )
    
    # LIMESTONE
    limestone_allowed = structural_high_mask & (E >= E_limestone_threshold)
    min_cover_limestone = 100.0
    limestone_max_elev = surface_elev - min_cover_limestone
    
    interfaces["Limestone"] = np.where(
        limestone_allowed,
        interfaces["Limestone"],
        np.minimum(interfaces["Limestone"], limestone_max_elev)
    )
    
    # BASEMENT
    basement_allowed = structural_high_mask & (E >= E_basement_threshold)
    min_cover_basement = 300.0
    basement_max_elev = surface_elev - min_cover_basement
    
    interfaces["Basement"] = np.where(
        basement_allowed,
        interfaces["Basement"],
        np.minimum(interfaces["Basement"], basement_max_elev)
    )
    
    return interfaces


def apply_progressive_stripping(
    interfaces,
    surface_elev,
    E,
    structural_high_mask,
    cover_max_total=50.0,
    cover_min_total=2.0,
):
    """
    As erosion increases, strip away cover layers progressively.
    """
    # Target total cover thickness based on erosion
    cover_target = cover_max_total - (cover_max_total - cover_min_total) * E
    
    cover_layers = ["Topsoil", "Subsoil", "Colluvium", "Alluvium", 
                    "Saprolite", "WeatheredBR"]
    
    # Find bottom of cover (top of rock)
    if "Sand" in interfaces:
        rock_top = interfaces["Sand"]
    elif "Sandstone" in interfaces:
        rock_top = interfaces["Sandstone"]
    else:
        return interfaces
    
    # Current total cover
    current_cover = surface_elev - rock_top
    
    # Where cover exceeds target, compress it
    compression_factor = np.clip(cover_target / (current_cover + 1e-6), 0, 1)
    
    # Apply compression to cover layers
    for layer in cover_layers:
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
        min_cover = 0.5
        for layer in cover_layers:
            if layer not in interfaces:
                continue
            interfaces[layer] = np.where(
                rock_at_surface,
                surface_elev - min_cover,
                interfaces[layer]
            )
    
    return interfaces


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
