#!/usr/bin/env python3
"""
🔒 TOPOGRAPHY GENERATOR: LOCKED (DO NOT MODIFY)

Realistic terrain + stratigraphy with CONSTRAINT-BASED layer generation:
- Surface elevation map
- Stratigraphic cross-sections along X and Y axes
- Layers generated using constraint-based approach (work from surface down)

CONSTRAINT-BASED APPROACH:
- Compute erosion intensity from surface geometry
- Generate structural uplift zones
- Build layers DOWNWARD from surface
- Apply explicit constraints (basement only in structural highs)
- Valleys are mud-dominated, mountains expose basement

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import gaussian_filter, uniform_filter


# ============================================================================
# 🔒 TOPOGRAPHY GENERATOR - LOCKED (DO NOT MODIFY)
# ============================================================================

def qrng_uint32(n, nbits=32):
    """Quasi-random number generator using van der Corput sequence."""
    mask = (1 << nbits) - 1
    out = np.zeros(n, dtype=np.uint32)
    for i in range(n):
        bits = 0
        val = i + 1
        for shift in range(nbits):
            if val & 1:
                bits |= (1 << (nbits - 1 - shift))
            val >>= 1
        out[i] = bits & mask
    return out

def rng_from_qrng(n_seeds=4, random_seed=None):
    """Create numpy RNG from quasi-random seeds."""
    if random_seed is not None:
        np.random.seed(random_seed)
        seeds = np.random.randint(0, 2**31, size=n_seeds, dtype=np.int64)
    else:
        qseeds = qrng_uint32(n_seeds)
        seeds = qseeds.astype(np.int64)
    
    from numpy.random import PCG64
    pcg = PCG64(seeds[0])
    for s in seeds[1:]:
        pcg = pcg.advance(int(s))
    return np.random.Generator(pcg)

def fractional_surface(N, beta=3.1, rng=None):
    """Generate fractional Brownian motion surface."""
    if rng is None:
        rng = np.random.default_rng()
    
    kx = np.fft.fftfreq(N, d=1.0/N)
    ky = np.fft.fftfreq(N, d=1.0/N)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K = np.sqrt(KX**2 + KY**2)
    K[0, 0] = 1.0
    
    power = K**(-beta/2.0)
    power[0, 0] = 0.0
    
    phase = rng.uniform(0, 2*np.pi, size=(N, N))
    F = power * np.exp(1j * phase)
    z = np.fft.ifft2(F).real
    
    z = (z - np.min(z)) / (np.max(z) - np.min(z) + 1e-12)
    return z

def bilinear_sample(img, X, Y):
    """Bilinear interpolation sampling."""
    N = img.shape[0]
    X = np.clip(X, 0, N-1.001)
    Y = np.clip(Y, 0, N-1.001)
    
    x0, y0 = np.floor(X).astype(int), np.floor(Y).astype(int)
    x1, y1 = x0 + 1, y0 + 1
    x1 = np.clip(x1, 0, N-1)
    y1 = np.clip(y1, 0, N-1)
    
    fx, fy = X - x0, Y - y0
    
    return (img[x0, y0] * (1-fx) * (1-fy) +
            img[x1, y0] * fx * (1-fy) +
            img[x0, y1] * (1-fx) * fy +
            img[x1, y1] * fx * fy)

def domain_warp(z, rng, amp=0.12, beta=3.0):
    """Apply domain warping to terrain."""
    N = z.shape[0]
    dx = fractional_surface(N, beta=beta, rng=rng) * 2 - 1
    dy = fractional_surface(N, beta=beta, rng=rng) * 2 - 1
    
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    X = ii + amp * N * dx
    Y = jj + amp * N * dy
    
    return bilinear_sample(z, X, Y)

def ridged_mix(z, alpha=0.18):
    """Apply ridged multifractal mixing."""
    z_ridge = 1.0 - np.abs(2*z - 1)
    return (1 - alpha) * z + alpha * z_ridge

def lowpass2d(z, cutoff=None, rolloff=0.08):
    """Apply lowpass filter."""
    N = z.shape[0]
    if cutoff is None:
        cutoff = 0.4 * N
    
    kx = np.fft.fftfreq(N)
    ky = np.fft.fftfreq(N)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K = np.sqrt(KX**2 + KY**2) * N
    
    H = 1.0 / (1.0 + (K / cutoff)**(1.0 / rolloff))
    
    F = np.fft.fft2(z)
    F_filt = F * H
    z_filt = np.fft.ifft2(F_filt).real
    
    return z_filt

def gaussian_blur(z, sigma=None):
    """Apply Gaussian blur."""
    if sigma is None:
        sigma = z.shape[0] * 0.02
    return gaussian_filter(z.astype(np.float64), sigma=sigma, mode='reflect')

def quantum_seeded_topography(
    N=512,
    elev_range_m=700.0,
    n_qrng_seeds=4,
    random_seed=None,
    beta_base=3.1,
    warp_amp=0.12,
    warp_beta=3.0,
    ridge_alpha=0.18,
    smooth_sigma_frac=0.02,
):
    """
    🔒 LOCKED: Generate quantum-seeded realistic topography.
    DO NOT MODIFY THIS FUNCTION.
    """
    rng = rng_from_qrng(n_seeds=n_qrng_seeds, random_seed=random_seed)
    
    z = fractional_surface(N, beta=beta_base, rng=rng)
    z = domain_warp(z, rng, amp=warp_amp, beta=warp_beta)
    z = ridged_mix(z, alpha=ridge_alpha)
    z = lowpass2d(z, cutoff=0.4*N, rolloff=0.08)
    z = gaussian_blur(z, sigma=N * smooth_sigma_frac)
    
    z = (z - np.min(z)) / (np.max(z) - np.min(z) + 1e-12)
    
    return z, rng


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _box_blur(a, k=5):
    """Box blur using uniform filter."""
    return uniform_filter(a.astype(np.float64), size=k, mode='reflect')

def _normalize(x, eps=1e-12):
    """Normalize array to [0, 1]."""
    x_min, x_max = float(np.min(x)), float(np.max(x))
    if x_max - x_min < eps:
        return np.zeros_like(x)
    return (x - x_min) / (x_max - x_min)


# ============================================================================
# CONSTRAINT-BASED STRATIGRAPHY FUNCTIONS
# ============================================================================

def compute_erosion_intensity(
    surface_elev,
    pixel_scale_m,
    w_slope=0.35,
    w_relief=0.35,
    w_channel=0.20,
    w_elevation=0.10,
):
    """
    Compute erosion intensity E[x,y] in [0, 1] from surface geometry.
    
    High E = strong erosion = thin cover, deep layers can be exposed
    Low E = weak erosion = thick cover, deep layers buried
    
    Components:
    - Slope: steeper = more erosion
    - Relief: high local relief = more erosion
    - Channel: near channels = more erosion
    - Elevation: higher = more erosion potential
    """
    N = surface_elev.shape[0]
    
    # 1. Slope component
    dEy, dEx = np.gradient(surface_elev, pixel_scale_m, pixel_scale_m)
    slope_mag = np.hypot(dEx, dEy)
    slope_deg = np.rad2deg(np.arctan(slope_mag))
    E_slope = np.clip(slope_deg / 45.0, 0, 1)
    
    # 2. Local relief component
    k_relief = max(31, int(0.1 * N) | 1)
    elev_smooth = _box_blur(surface_elev, k=k_relief)
    relief_local = surface_elev - elev_smooth
    relief_range = np.percentile(np.abs(relief_local), 98)
    E_relief = np.clip(np.abs(relief_local) / (relief_range + 1e-9), 0, 1)
    
    # 3. Channel/wetness component
    slope_norm = _normalize(slope_mag)
    catch = _box_blur(_box_blur(1.0 - slope_norm, k=7), k=13)
    wetness = _normalize(catch)
    channel_indicator = wetness * (1 - E_slope * 0.5)
    E_channel = _normalize(channel_indicator)
    
    # 4. Elevation component
    E_elev = _normalize(surface_elev)
    
    # Combine
    E = (w_slope * E_slope + 
         w_relief * E_relief + 
         w_channel * E_channel +
         w_elevation * E_elev)
    
    E = _box_blur(E, k=7)
    E = np.clip(E, 0, 1)
    
    return E, {
        "slope_norm": slope_norm,
        "wetness": wetness,
        "curvature": dEx + dEy,
        "E_slope": E_slope,
        "E_relief": E_relief,
    }


def generate_structural_uplift(N, rng, pixel_scale_m, n_anticlines=3):
    """
    Generate structural uplift field with anticlines/domes.
    These are zones where rock layers are pushed up.
    """
    U = np.zeros((N, N), dtype=np.float64)
    
    for _ in range(n_anticlines):
        cx = rng.uniform(0.2, 0.8) * N
        cy = rng.uniform(0.2, 0.8) * N
        amp = rng.uniform(50.0, 200.0)
        sigma = rng.uniform(0.1, 0.3) * N
        
        ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
        r2 = (ii - cx)**2 + (jj - cy)**2
        U += amp * np.exp(-r2 / (2 * sigma**2))
    
    # Regional variation
    regional = rng.standard_normal((N, N)) * 30.0
    regional = gaussian_filter(regional, sigma=N*0.15)
    U += regional
    
    U = _box_blur(U, k=max(31, int(0.1 * N) | 1))
    return U


def compute_structural_high_mask(U, z_norm, uplift_threshold=50.0):
    """
    Define structural highs: only places where basement can reach surface.
    
    Returns boolean mask: True = basement allowed near surface
    """
    uplift_highs = U > uplift_threshold
    elev_highs = (z_norm > 0.5) & (U > uplift_threshold * 0.3)
    
    structural_high_mask = uplift_highs | elev_highs
    structural_high_mask = _box_blur(structural_high_mask.astype(float), k=11) > 0.3
    
    return structural_high_mask


def compute_cover_thicknesses(E, slope_norm, wetness, curvature, z_norm):
    """
    Compute thickness of cover layers based on erosion intensity.
    """
    # Soil: thinner where erosion is strong
    soil_thick_min, soil_thick_max = 0.5, 3.0
    soil_target = soil_thick_max - (soil_thick_max - soil_thick_min) * E
    slope_factor = 1.0 - slope_norm ** 2
    soil_thickness = np.maximum(soil_target * slope_factor, soil_thick_min * 0.3)
    
    # Colluvium: peaks at moderate slopes, concave areas
    slope_opt = 0.4
    slope_factor_coll = 1.0 - np.abs(slope_norm - slope_opt) / 0.6
    slope_factor_coll = np.clip(slope_factor_coll, 0, 1)
    
    curv_norm = _normalize(curvature)
    hollow_factor = np.clip(curv_norm, 0, 1)
    
    wetness_factor = wetness * (1 - wetness)
    wetness_factor = _normalize(wetness_factor)
    
    C = 0.4 * slope_factor_coll + 0.4 * hollow_factor + 0.2 * wetness_factor
    C = _normalize(C)
    
    erosion_suppress = np.clip(1.0 - (E - 0.7) / 0.3, 0, 1)
    C *= erosion_suppress
    
    colluvium_thickness = 25.0 * C
    
    # Alluvium: ONLY in valley floors
    is_flat = slope_norm < 0.15
    is_wet = wetness > 0.6
    is_low = z_norm < 0.4
    valley_mask = is_flat & is_wet & is_low
    
    valley_strength = np.zeros_like(slope_norm)
    valley_strength[valley_mask] = (
        (1 - slope_norm[valley_mask] / 0.15) *
        ((wetness[valley_mask] - 0.6) / 0.4) *
        (1 - z_norm[valley_mask] / 0.4)
    )
    valley_strength = np.clip(valley_strength, 0, 1)
    
    alluvium_thickness = 40.0 * valley_strength
    
    # Saprolite: thick where stable
    saprolite_thickness = 8.0 * (1 - E) * (1 - slope_norm**2)
    saprolite_thickness = np.clip(saprolite_thickness, 0.5, 30.0)
    
    # Weathered bedrock
    weathered_thickness = 3.0 * (1 - E * 0.5)
    weathered_thickness = np.clip(weathered_thickness, 0.5, 8.0)
    
    return {
        "soil": soil_thickness,
        "colluvium": colluvium_thickness,
        "alluvium": alluvium_thickness,
        "saprolite": saprolite_thickness,
        "weathered": weathered_thickness,
    }


def enforce_minimum_sediment_cover(interfaces, surface_elev, structural_high_mask, sediment_min_depth=300.0):
    """
    Outside structural highs, enforce minimum depth to basement.
    This prevents basement from appearing under every small hill.
    """
    current_sed_depth = surface_elev - interfaces["Basement"]
    needs_deepening = (current_sed_depth < sediment_min_depth) & (~structural_high_mask)
    
    target_basement = surface_elev - sediment_min_depth
    interfaces["Basement"] = np.where(needs_deepening, target_basement, interfaces["Basement"])
    interfaces["BasementFloor"] = np.minimum(interfaces["BasementFloor"], interfaces["Basement"] - 100.0)
    
    return interfaces


def constrain_deep_layer_exposure(interfaces, surface_elev, E, structural_high_mask,
                                   E_sandstone=0.5, E_shale=0.6, E_limestone=0.7, E_basement=0.85):
    """
    Deep layers can only reach near-surface where:
    1. structural_high_mask is True
    2. Erosion intensity E exceeds threshold
    """
    # Sandstone
    sandstone_allowed = structural_high_mask & (E >= E_sandstone)
    sandstone_max_elev = surface_elev - 20.0
    interfaces["Sandstone"] = np.where(sandstone_allowed, interfaces["Sandstone"], 
                                       np.minimum(interfaces["Sandstone"], sandstone_max_elev))
    
    # Shale
    shale_allowed = structural_high_mask & (E >= E_shale)
    shale_max_elev = surface_elev - 50.0
    interfaces["Shale"] = np.where(shale_allowed, interfaces["Shale"],
                                    np.minimum(interfaces["Shale"], shale_max_elev))
    
    # Limestone
    limestone_allowed = structural_high_mask & (E >= E_limestone)
    limestone_max_elev = surface_elev - 100.0
    interfaces["Limestone"] = np.where(limestone_allowed, interfaces["Limestone"],
                                       np.minimum(interfaces["Limestone"], limestone_max_elev))
    
    # Basement
    basement_allowed = structural_high_mask & (E >= E_basement)
    basement_max_elev = surface_elev - 300.0
    interfaces["Basement"] = np.where(basement_allowed, interfaces["Basement"],
                                      np.minimum(interfaces["Basement"], basement_max_elev))
    
    return interfaces


def apply_progressive_stripping(interfaces, surface_elev, E, structural_high_mask):
    """
    As erosion increases, progressively strip cover layers.
    """
    cover_max, cover_min = 50.0, 2.0
    cover_target = cover_max - (cover_max - cover_min) * E
    
    cover_layers = ["Topsoil", "Subsoil", "Colluvium", "Alluvium", "Saprolite", "WeatheredBR"]
    
    rock_top = interfaces.get("Sand", interfaces.get("Sandstone", surface_elev))
    current_cover = surface_elev - rock_top
    
    compression_factor = np.clip(cover_target / (current_cover + 1e-6), 0, 1)
    
    for layer in cover_layers:
        if layer not in interfaces:
            continue
        depth_from_surface = surface_elev - interfaces[layer]
        new_depth = depth_from_surface * compression_factor
        interfaces[layer] = surface_elev - new_depth
    
    # Very high erosion zones
    rock_at_surface = (E > 0.8) & structural_high_mask
    if np.any(rock_at_surface):
        for layer in cover_layers:
            if layer in interfaces:
                interfaces[layer] = np.where(rock_at_surface, surface_elev - 0.5, interfaces[layer])
    
    return interfaces


def enforce_ordering(interfaces, layer_order, eps=0.01):
    """Enforce stratigraphic order: each layer below the one above."""
    for i in range(1, len(layer_order)):
        above = layer_order[i - 1]
        here = layer_order[i]
        if above in interfaces and here in interfaces:
            interfaces[here] = np.minimum(interfaces[here], interfaces[above] - eps)
    return interfaces


# ============================================================================
# MAIN STRATIGRAPHY GENERATION (CONSTRAINT-BASED)
# ============================================================================

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
    **kwargs
):
    """
    Generate stratigraphy using CONSTRAINT-BASED approach.
    
    Key innovation: Work from surface DOWNWARD, applying explicit constraints.
    Basement only appears in structural highs with sufficient erosion.
    """
    N = z_norm.shape[0]
    surface_elev = z_norm * elev_range_m
    
    print(f"Generating constraint-based stratigraphy for {N}x{N} grid...")
    
    # ============ STEP 1: Compute erosion intensity ============
    E, E_comp = compute_erosion_intensity(surface_elev, pixel_scale_m)
    slope_norm = E_comp["slope_norm"]
    wetness = E_comp["wetness"]
    curvature = E_comp["curvature"]
    
    print(f"Erosion intensity: min={E.min():.3f}, mean={E.mean():.3f}, max={E.max():.3f}")
    
    # ============ STEP 2: Generate structural uplift ============
    U = generate_structural_uplift(N, rng, pixel_scale_m, n_anticlines)
    structural_high_mask = compute_structural_high_mask(U, z_norm)
    
    pct_structural_high = 100.0 * np.mean(structural_high_mask)
    print(f"Structural highs cover {pct_structural_high:.1f}% of terrain")
    
    # ============ STEP 3: Compute cover thicknesses ============
    cover_thick = compute_cover_thicknesses(E, slope_norm, wetness, curvature, z_norm)
    
    # ============ STEP 4: Build interfaces DOWNWARD from surface ============
    interfaces = {}
    
    # Cover layers
    interfaces["Topsoil"] = surface_elev - cover_thick["soil"] * 0.4
    interfaces["Subsoil"] = interfaces["Topsoil"] - cover_thick["soil"] * 0.6
    interfaces["Colluvium"] = interfaces["Subsoil"] - cover_thick["colluvium"]
    interfaces["Alluvium"] = interfaces["Colluvium"] - cover_thick["alluvium"]
    interfaces["Saprolite"] = interfaces["Alluvium"] - cover_thick["saprolite"]
    interfaces["WeatheredBR"] = interfaces["Saprolite"] - cover_thick["weathered"]
    
    # Valley fill
    clay_thick = 20.0 * ((slope_norm < 0.1) & (wetness > 0.6) & (z_norm < 0.3)).astype(float)
    silt_thick = 15.0 * ((slope_norm < 0.2) & (wetness > 0.4) & (z_norm < 0.4)).astype(float) * wetness
    sand_thick = 25.0 * ((slope_norm > 0.05) & (slope_norm < 0.3) & (wetness > 0.5)).astype(float) * wetness
    
    interfaces["Clay"] = interfaces["WeatheredBR"] - clay_thick
    interfaces["Silt"] = interfaces["Clay"] - silt_thick
    interfaces["Sand"] = interfaces["Silt"] - sand_thick
    
    # Rock layers - initial positions (will be constrained)
    rock_base = interfaces["Sand"] - 50.0
    rock_base += U  # Uplift brings rock up
    
    interfaces["Sandstone"] = rock_base
    interfaces["Conglomerate"] = interfaces["Sandstone"] - 80.0
    interfaces["Shale"] = interfaces["Conglomerate"] - 30.0
    interfaces["Mudstone"] = interfaces["Shale"] - 150.0
    interfaces["Siltstone"] = interfaces["Mudstone"] - 80.0
    interfaces["Limestone"] = interfaces["Siltstone"] - 60.0
    interfaces["Dolomite"] = interfaces["Limestone"] - 100.0
    interfaces["Evaporite"] = interfaces["Dolomite"] - 30.0
    
    # Crystalline basement
    interfaces["Granite"] = interfaces["Evaporite"]
    interfaces["Gneiss"] = interfaces["Granite"] - 5.0
    interfaces["Basalt"] = interfaces["Gneiss"] - 5.0
    interfaces["AncientCrust"] = interfaces["Basalt"] - 2.0
    interfaces["Basement"] = interfaces["AncientCrust"]
    interfaces["BasementFloor"] = interfaces["Basement"] - 500.0
    
    # ============ STEP 5: Apply constraints ============
    print("Applying constraints...")
    
    interfaces = enforce_minimum_sediment_cover(interfaces, surface_elev, structural_high_mask, sediment_min_depth)
    interfaces = constrain_deep_layer_exposure(interfaces, surface_elev, E, structural_high_mask,
                                                E_sandstone_threshold, E_shale_threshold, 
                                                E_limestone_threshold, E_basement_threshold)
    interfaces = apply_progressive_stripping(interfaces, surface_elev, E, structural_high_mask)
    
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
            thickness[layer] = np.maximum(interfaces[layer] - interfaces[below], 0.0)
    
    z_floor = float(interfaces["BasementFloor"].min() - 100.0)
    thickness["BasementFloor"] = np.maximum(interfaces["BasementFloor"] - z_floor, 0.0)
    
    # ============ STEP 9: Properties (simplified) ============
    properties = {}
    for layer in thickness:
        properties[layer] = np.ones_like(thickness[layer])
    
    print("Stratigraphy generation complete!")
    
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
            "z_floor": z_floor,
        }
    }


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def compute_top_material_map(strata, min_thick=0.05):
    """Determine the topmost material at each location."""
    thickness = strata["thickness"]
    N = list(thickness.values())[0].shape[0]
    
    layer_order = [
        "Topsoil", "Subsoil", "Clay", "Silt", "Sand", "Colluvium",
        "Saprolite", "WeatheredBR", "Sandstone", "Conglomerate",
        "Shale", "Mudstone", "Siltstone", "Limestone", "Dolomite", "Evaporite",
        "Granite", "Gneiss", "Basalt", "AncientCrust"
    ]
    
    top_material = np.full((N, N), -1, dtype=np.int32)
    
    for i, layer_name in enumerate(layer_order):
        if layer_name in thickness:
            t = thickness[layer_name]
            mask = t >= min_thick
            top_material[mask] = i
    
    return top_material, layer_order


def plot_cross_section(ax, x_vals, strata, slice_idx, axis='x', vertical_exag=1.0, title="Cross-Section"):
    """Plot a cross-section showing all layers."""
    interfaces = strata["interfaces"]
    meta = strata["meta"]
    surface_elev = strata["surface_elev"]
    pixel_scale_m = meta["pixel_scale_m"]
    z_floor = meta["z_floor"]
    
    N = surface_elev.shape[0]
    
    if axis == 'x':
        surf_line = surface_elev[slice_idx, :]
        get_interface = lambda name: interfaces[name][slice_idx, :] if name in interfaces else None
    else:
        surf_line = surface_elev[:, slice_idx]
        get_interface = lambda name: interfaces[name][:, slice_idx] if name in interfaces else None
    
    # Layer order and colors
    layers_plot = [
        ("BasementFloor", "#1a1a1a"),
        ("Basement", "#2d2d2d"),
        ("AncientCrust", "#404040"),
        ("Basalt", "#8b4513"),
        ("Gneiss", "#708090"),
        ("Granite", "#a9a9a9"),
        ("Evaporite", "#ffb6c1"),
        ("Dolomite", "#dda0dd"),
        ("Limestone", "#98fb98"),
        ("Siltstone", "#d2b48c"),
        ("Mudstone", "#8b7355"),
        ("Shale", "#556b2f"),
        ("Conglomerate", "#cd853f"),
        ("Sandstone", "#ffa500"),
        ("Sand", "#f4a460"),
        ("Silt", "#e6d8ad"),
        ("Clay", "#8fbc8f"),
        ("WeatheredBR", "#bdb76b"),
        ("Saprolite", "#daa520"),
        ("Colluvium", "#bc8f8f"),
        ("Subsoil", "#8b7765"),
        ("Topsoil", "#654321"),
    ]
    
    # Plot from bottom to top
    for layer_name, color in layers_plot:
        top_interface = get_interface(layer_name)
        
        if top_interface is None:
            continue
        
        # Find bottom interface
        idx = next((i for i, (name, _) in enumerate(layers_plot) if name == layer_name), None)
        if idx is None or idx == 0:
            bottom_interface = np.full_like(top_interface, z_floor)
        else:
            bottom_name = layers_plot[idx - 1][0]
            bottom_interface = get_interface(bottom_name)
            if bottom_interface is None:
                bottom_interface = np.full_like(top_interface, z_floor)
        
        ax.fill_between(x_vals, bottom_interface * vertical_exag, top_interface * vertical_exag,
                        color=color, edgecolor='none', label=layer_name, linewidth=0)
    
    # Plot surface
    ax.plot(x_vals, surf_line * vertical_exag, 'k-', linewidth=1.5, label='Surface')
    
    ax.set_xlabel(f"{'X' if axis == 'x' else 'Y'} Distance (m)", fontsize=10)
    ax.set_ylabel("Elevation (m)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_xlim(x_vals[0], x_vals[-1])


def plot_topography_map(ax, surface_elev, pixel_scale_m, title="Surface Topography"):
    """Plot topography map."""
    N = surface_elev.shape[0]
    extent = [0, N * pixel_scale_m, 0, N * pixel_scale_m]
    
    im = ax.imshow(surface_elev, origin='lower', extent=extent, cmap='terrain', aspect='equal')
    ax.set_xlabel("X (m)", fontsize=10)
    ax.set_ylabel("Y (m)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label="Elevation (m)", fraction=0.046, pad=0.04)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("CONSTRAINT-BASED GEOLOGICAL LAYER GENERATION")
    print("="*70)
    
    # Generate topography (LOCKED - DO NOT MODIFY)
    z, rng = quantum_seeded_topography(
        N=512,
        elev_range_m=700.0,
        n_qrng_seeds=4,
        random_seed=42,
        beta_base=3.1,
        warp_amp=0.12,
        ridge_alpha=0.18,
    )
    
    # Generate stratigraphy (CONSTRAINT-BASED)
    strata = generate_stratigraphy(
        z_norm=z,
        rng=rng,
        elev_range_m=700.0,
        pixel_scale_m=10.0,
        n_anticlines=3,
        sediment_min_depth=300.0,
        E_sandstone_threshold=0.5,
        E_shale_threshold=0.6,
        E_limestone_threshold=0.7,
        E_basement_threshold=0.85,
    )
    
    # Print statistics
    print("\n" + "="*70)
    print("LAYER THICKNESS STATISTICS")
    print("="*70)
    
    thickness = strata["thickness"]
    for layer_name in ["Topsoil", "Subsoil", "Clay", "Silt", "Sand", "Colluvium",
                       "Saprolite", "WeatheredBR", "Sandstone", "Conglomerate",
                       "Shale", "Mudstone", "Siltstone", "Limestone", "Dolomite",
                       "Granite", "Gneiss", "Basalt", "AncientCrust"]:
        if layer_name in thickness:
            t = thickness[layer_name]
            print(f"{layer_name:15s}: min={t.min():6.2f}  mean={t.mean():6.2f}  max={t.max():6.2f} m")
    
    # Validation
    E = strata["meta"]["erosion_intensity"]
    structural_high_mask = strata["meta"]["structural_high_mask"]
    
    print("\n" + "="*70)
    print("CONSTRAINT VALIDATION")
    print("="*70)
    print(f"Erosion intensity: min={E.min():.3f}, mean={E.mean():.3f}, max={E.max():.3f}")
    print(f"Structural highs: {100*np.mean(structural_high_mask):.1f}% of terrain")
    
    # Check basement exposure
    surface_elev = strata["surface_elev"]
    basement_depth = surface_elev - strata["interfaces"]["Basement"]
    basement_exposed = basement_depth < 50.0  # Within 50m of surface
    
    print(f"Basement near surface (<50m): {100*np.mean(basement_exposed):.1f}% of terrain")
    print(f"  - In structural highs: {100*np.mean(basement_exposed & structural_high_mask):.1f}%")
    print(f"  - Outside structural highs: {100*np.mean(basement_exposed & ~structural_high_mask):.1f}%")
    
    # Plotting
    print("\nGenerating plots...")
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Topography map
    ax_topo = fig.add_subplot(gs[0, :])
    plot_topography_map(ax_topo, surface_elev, 10.0, "Surface Topography")
    
    # Cross-sections
    N = surface_elev.shape[0]
    slice_y = N // 2
    slice_x = N // 2
    
    x_vals = np.arange(N) * 10.0
    
    ax_xs_x = fig.add_subplot(gs[1, :])
    plot_cross_section(ax_xs_x, x_vals, strata, slice_y, axis='x', vertical_exag=1.0,
                      title=f"X Cross-Section (Y = {slice_y * 10.0:.0f} m)")
    
    ax_xs_y = fig.add_subplot(gs[2, :])
    plot_cross_section(ax_xs_y, x_vals, strata, slice_x, axis='y', vertical_exag=1.0,
                      title=f"Y Cross-Section (X = {slice_x * 10.0:.0f} m)")
    
    # Add legend
    handles, labels = ax_xs_x.get_legend_handles_labels()
    unique_labels = {}
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels[label] = handle
    
    fig.legend(unique_labels.values(), unique_labels.keys(), 
              loc='upper left', bbox_to_anchor=(0.02, 0.98),
              ncol=4, fontsize=8, framealpha=0.9)
    
    plt.savefig("constraint_based_stratigraphy.png", dpi=150, bbox_inches='tight')
    print("Saved: constraint_based_stratigraphy.png")
    
    plt.show()
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
