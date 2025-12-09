#!/usr/bin/env python3
"""
FACIES-BASED STRATIGRAPHIC GENERATION

Each layer type appears ONLY where terrain conditions are appropriate.
Thickness allocated based on environmental favorability.

Key concept: Compute favorability fields for each facies, then allocate
basin_depth proportionally. Result: layers are zero where inappropriate.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, uniform_filter

# ============================================================================
# TOPOGRAPHY GENERATOR (LOCKED)
# ============================================================================

def qrng_uint32(n, nbits=32):
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

def quantum_seeded_topography(N=512, random_seed=None):
    rng = rng_from_qrng(n_seeds=4, random_seed=random_seed)
    z = fractional_surface(N, beta=3.1, rng=rng)
    # Simple smoothing
    z = gaussian_filter(z, sigma=N*0.02)
    z = (z - np.min(z)) / (np.max(z) - np.min(z) + 1e-12)
    return z, rng

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def normalize(x):
    x_min, x_max = float(np.min(x)), float(np.max(x))
    if x_max - x_min < 1e-12:
        return np.zeros_like(x)
    return (x - x_min) / (x_max - x_min)

def box_blur(a, k=5):
    return uniform_filter(a.astype(np.float64), size=k, mode='reflect')

# ============================================================================
# TERRAIN ANALYSIS
# ============================================================================

def compute_terrain_derivatives(surface_elev, pixel_scale_m):
    """
    Compute all terrain derivatives needed for facies rules.
    
    Returns dict with:
    - elev: absolute elevation
    - rel_elev: relative to local neighborhood
    - slope: gradient magnitude (normalized 0-1)
    - curv: curvature (positive = concave/valley, negative = convex/ridge)
    - flow_accum: drainage accumulation proxy
    """
    N = surface_elev.shape[0]
    
    # Absolute elevation
    elev = surface_elev
    elev_norm = normalize(elev)
    
    # Relative elevation (high vs low in local neighborhood)
    k_local = max(31, int(0.1 * N) | 1)
    elev_smooth = box_blur(elev, k=k_local)
    rel_elev = elev - elev_smooth
    rel_elev_norm = normalize(rel_elev)
    
    # Slope
    dEy, dEx = np.gradient(elev, pixel_scale_m, pixel_scale_m)
    slope_mag = np.hypot(dEx, dEy)
    slope_deg = np.rad2deg(np.arctan(slope_mag))
    slope_norm = normalize(slope_mag)
    
    # Curvature (Laplacian: positive = concave, negative = convex)
    d2Ex = np.gradient(dEx, pixel_scale_m, axis=1)
    d2Ey = np.gradient(dEy, pixel_scale_m, axis=0)
    curv = d2Ex + d2Ey
    curv_norm = normalize(curv)  # 0 = most convex, 1 = most concave
    
    # Flow accumulation proxy (inverse slope, smoothed)
    catchment = 1.0 - slope_norm
    catchment = box_blur(catchment, k=7)
    catchment = box_blur(catchment, k=13)
    flow_accum = normalize(catchment)
    
    return {
        "elev": elev,
        "elev_norm": elev_norm,
        "rel_elev": rel_elev,
        "rel_elev_norm": rel_elev_norm,
        "slope_mag": slope_mag,
        "slope_deg": slope_deg,
        "slope_norm": slope_norm,
        "curv": curv,
        "curv_norm": curv_norm,
        "flow_accum": flow_accum,
    }

def compute_basin_depth(elev_norm, rng):
    """
    Compute allowed sediment thickness from basin subsidence model.
    Deep in structural lows, thin on structural highs.
    """
    N = elev_norm.shape[0]
    
    # Structural basin field (independent of current topography)
    k_basin = max(63, int(0.3 * N) | 1)
    basin_noise = fractional_surface(N, beta=4.5, rng=rng)
    basin_field = box_blur(basin_noise, k=k_basin)
    basin_field = box_blur(basin_field, k=max(31, int(0.15 * N) | 1))
    basins = normalize(1.0 - basin_field)  # High = deep basin
    
    # Basin depth: 20m minimum, 600m maximum in deepest basins
    basin_depth = 20.0 + 580.0 * basins
    
    return basin_depth, basins

# ============================================================================
# FACIES FAVORABILITY FUNCTIONS
# ============================================================================

def favorability_conglomerate(terrain):
    """
    Conglomerate: alluvial fans, proximal braided rivers, near steep sources.
    
    Favors:
    - Near margins of basement highs (high rel_elev nearby)
    - Moderate-high slope
    - Convex or straight curvature
    
    Avoids:
    - Deep basin center (concave, very low slope)
    """
    # Near highs (distance from high rel_elev)
    high_mask = terrain["rel_elev_norm"] > 0.7
    dist_to_high = box_blur(high_mask.astype(float), k=15)
    near_high = normalize(dist_to_high)
    
    # Moderate-high slope
    slope_favor = np.clip(terrain["slope_norm"], 0.3, 0.8)
    slope_favor = (slope_favor - 0.3) / 0.5
    
    # Convex/straight curvature (avoid concave)
    curv_favor = 1.0 - terrain["curv_norm"]  # Low = convex
    curv_favor = np.clip(curv_favor, 0.3, 1.0)
    
    # Combine
    favor = near_high * slope_favor * curv_favor
    favor = normalize(favor)
    
    # Smooth
    favor = box_blur(favor, k=7)
    
    return favor

def favorability_sandstone(terrain, basins):
    """
    Sandstone: rivers, deltas, shorelines, dunes, shallow shelves.
    
    Favors:
    - Basin margins (intermediate basin depth)
    - Fluvial belts (moderate flow_accum)
    - Broad low-relief areas
    
    Avoids:
    - Deepest basin centers (too quiet)
    - Very steep mountain cores
    """
    # Basin margins (not too deep, not too shallow)
    margin_favor = 1.0 - np.abs(basins - 0.4) / 0.6
    margin_favor = np.clip(margin_favor, 0, 1)
    
    # Fluvial influence
    fluvial_favor = terrain["flow_accum"] ** 0.5  # Moderate flow
    
    # Avoid very steep
    slope_favor = 1.0 - terrain["slope_norm"] ** 2
    
    # Combine
    favor = margin_favor * (0.5 + 0.5 * fluvial_favor) * slope_favor
    favor = normalize(favor)
    
    # Smooth
    favor = box_blur(favor, k=11)
    
    return favor

def favorability_shale(terrain, basins):
    """
    Shale/Mudstone: low-energy offshore, deep lake, floodplain.
    
    Favors:
    - Concave (valleys, depressions)
    - Low slope
    - Deep basins
    
    Avoids:
    - Steep slopes
    - High rel_elev
    """
    # Concave areas
    concave_favor = terrain["curv_norm"]  # High = concave
    
    # Low slope
    slope_favor = 1.0 - terrain["slope_norm"]
    
    # Deep basins
    basin_favor = basins ** 1.5  # Strongly favor deep basins
    
    # Combine
    favor = concave_favor * slope_favor * basin_favor
    favor = normalize(favor)
    
    # Smooth
    favor = box_blur(favor, k=11)
    
    return favor

def favorability_limestone(terrain, basins):
    """
    Limestone: shallow, warm, clear marine platforms.
    
    Favors:
    - Low slope
    - Low clastic input (away from high flow_accum)
    - Moderate basin depth (platforms, not deep)
    
    Avoids:
    - Steep mountains
    - High-energy fluvial belts
    """
    # Low slope
    slope_favor = 1.0 - terrain["slope_norm"] ** 2
    
    # Low clastic input (avoid high flow_accum)
    clastic_avoid = 1.0 - terrain["flow_accum"] ** 2
    
    # Moderate basin depth (platforms)
    platform_favor = 1.0 - np.abs(basins - 0.5) / 0.5
    platform_favor = np.clip(platform_favor, 0, 1)
    
    # Combine
    favor = slope_favor * clastic_avoid * platform_favor
    favor = normalize(favor)
    
    # Smooth
    favor = box_blur(favor, k=11)
    
    return favor

def favorability_evaporite(terrain, basins):
    """
    Evaporite: closed, arid basins with high evaporation.
    
    Favors:
    - Very low rel_elev
    - Strongly concave
    - Internally drained (low flow outlets)
    
    Avoids:
    - Open slopes
    - Through-flowing valleys
    """
    # Very low areas
    low_favor = 1.0 - terrain["elev_norm"]
    low_favor = low_favor ** 2
    
    # Strongly concave
    concave_favor = terrain["curv_norm"] ** 2
    
    # Deep basins
    basin_favor = basins ** 2
    
    # Combine
    favor = low_favor * concave_favor * basin_favor
    favor = normalize(favor)
    
    # Very restrictive - only top 10%
    threshold = np.percentile(favor, 90)
    favor = np.where(favor > threshold, favor, 0.0)
    favor = normalize(favor)
    
    # Smooth
    favor = box_blur(favor, k=7)
    
    return favor

# ============================================================================
# FACIES ALLOCATION
# ============================================================================

def allocate_sedimentary_facies(terrain, basin_depth, basins, rng):
    """
    Allocate basin_depth among facies based on favorability.
    
    Returns thickness dict with conglomerate, sandstone, shale, limestone, evaporite.
    """
    # Compute favorability for each facies
    favor_conglom = favorability_conglomerate(terrain)
    favor_sand = favorability_sandstone(terrain, basins)
    favor_shale = favorability_shale(terrain, basins)
    favor_lime = favorability_limestone(terrain, basins)
    favor_evap = favorability_evaporite(terrain, basins)
    
    # Subdivide shale into mudstone/siltstone
    favor_mudstone = favor_shale * 0.4
    favor_siltstone = favor_shale * 0.3
    favor_shale = favor_shale * 0.3
    
    # Subdivide limestone into dolomite
    favor_dolomite = favor_lime * 0.25
    favor_lime = favor_lime * 0.75
    
    # Total favorability
    total_favor = (favor_conglom + favor_sand + favor_shale + favor_mudstone + 
                   favor_siltstone + favor_lime + favor_dolomite + favor_evap + 1e-9)
    
    # Allocate basin_depth proportionally
    thickness = {}
    thickness["Conglomerate"] = basin_depth * (favor_conglom / total_favor)
    thickness["Sandstone"] = basin_depth * (favor_sand / total_favor)
    thickness["Shale"] = basin_depth * (favor_shale / total_favor)
    thickness["Mudstone"] = basin_depth * (favor_mudstone / total_favor)
    thickness["Siltstone"] = basin_depth * (favor_siltstone / total_favor)
    thickness["Limestone"] = basin_depth * (favor_lime / total_favor)
    thickness["Dolomite"] = basin_depth * (favor_dolomite / total_favor)
    thickness["Evaporite"] = basin_depth * (favor_evap / total_favor)
    
    return thickness

def add_unconsolidated_sediments(terrain):
    """
    Add unconsolidated sediments (Gravel, Sand, Silt, Clay) in active environments.
    Localized, meters to tens of meters.
    """
    thickness = {}
    
    # Gravel: active channels, high energy
    channel = (terrain["flow_accum"] > 0.7) & (terrain["slope_norm"] > 0.2) & (terrain["slope_norm"] < 0.6)
    thickness["Gravel"] = np.where(channel, 8.0 * terrain["flow_accum"], 0.0)
    
    # Sand: moderate energy zones
    sand_zone = (terrain["flow_accum"] > 0.5) & (terrain["slope_norm"] < 0.3)
    thickness["Sand"] = np.where(sand_zone, 15.0 * terrain["flow_accum"], 0.0)
    
    # Silt: quiet floodplains
    silt_zone = (terrain["slope_norm"] < 0.15) & (terrain["flow_accum"] > 0.4) & (terrain["flow_accum"] < 0.7)
    thickness["Silt"] = np.where(silt_zone, 10.0 * terrain["flow_accum"], 0.0)
    
    # Clay: standing water
    clay_zone = (terrain["slope_norm"] < 0.1) & (terrain["curv_norm"] > 0.6) & (terrain["elev_norm"] < 0.4)
    thickness["Clay"] = np.where(clay_zone, 12.0, 0.0)
    
    return thickness

def add_regolith_layers(terrain):
    """
    Add regolith based on slope and curvature.
    """
    thickness = {}
    
    # WeatheredBR: everywhere, 1-8m
    thickness["WeatheredBR"] = 2.0 + 6.0 * (1 - terrain["slope_norm"] ** 2)
    
    # Saprolite: low-moderate slope, 5-40m
    stable = (1 - terrain["slope_norm"]) * (1 - np.abs(terrain["curv_norm"] - 0.5))
    thickness["Saprolite"] = 5.0 + 35.0 * stable
    
    # Colluvium: concave footslopes
    colluvial = terrain["curv_norm"] * (terrain["slope_norm"] > 0.1).astype(float) * (terrain["slope_norm"] < 0.5).astype(float)
    thickness["Colluvium"] = 30.0 * colluvial
    
    # Subsoil: gentle slopes
    thickness["Subsoil"] = 1.0 * (1 - terrain["slope_norm"])
    
    # Topsoil: everywhere except steep
    thickness["Topsoil"] = 0.5 * (1 - terrain["slope_norm"] ** 3)
    
    return thickness

# ============================================================================
# MAIN GENERATION
# ============================================================================

def generate_stratigraphy_facies_based(z_norm, rng, elev_range_m=700.0, pixel_scale_m=10.0):
    """
    Generate stratigraphy using facies-based allocation.
    Each layer appears only where terrain conditions are favorable.
    """
    N = z_norm.shape[0]
    surface_elev = z_norm * elev_range_m
    
    print(f"\n{'='*70}")
    print("FACIES-BASED STRATIGRAPHY GENERATION")
    print(f"{'='*70}\n")
    
    # Step 1: Compute terrain derivatives
    print("Computing terrain derivatives...")
    terrain = compute_terrain_derivatives(surface_elev, pixel_scale_m)
    
    # Step 2: Compute basin depth
    print("Computing basin depth field...")
    basin_depth, basins = compute_basin_depth(z_norm, rng)
    print(f"Basin depth: min={basin_depth.min():.1f}m, mean={basin_depth.mean():.1f}m, max={basin_depth.max():.1f}m")
    
    # Step 3: Allocate sedimentary facies
    print("Allocating sedimentary facies based on terrain...")
    sed_thickness = allocate_sedimentary_facies(terrain, basin_depth, basins, rng)
    
    # Step 4: Add unconsolidated sediments
    print("Adding unconsolidated sediments...")
    unconsol_thickness = add_unconsolidated_sediments(terrain)
    
    # Step 5: Add regolith
    print("Adding regolith layers...")
    regolith_thickness = add_regolith_layers(terrain)
    
    # Step 6: Build interfaces from surface downward
    print("Building stratigraphic interfaces...")
    interfaces = {}
    thickness_all = {}
    
    # Start from surface
    current_top = surface_elev.copy()
    
    # Regolith (top to bottom)
    for layer in ["Topsoil", "Subsoil", "Colluvium"]:
        thickness_all[layer] = regolith_thickness[layer]
        interfaces[layer] = current_top
        current_top = current_top - thickness_all[layer]
    
    # Saprolite and weathered
    for layer in ["Saprolite", "WeatheredBR"]:
        thickness_all[layer] = regolith_thickness[layer]
        interfaces[layer] = current_top
        current_top = current_top - thickness_all[layer]
    
    # Unconsolidated sediments
    for layer in ["Clay", "Silt", "Sand", "Gravel"]:
        thickness_all[layer] = unconsol_thickness.get(layer, np.zeros_like(surface_elev))
        interfaces[layer] = current_top
        current_top = current_top - thickness_all[layer]
    
    # Sedimentary rocks
    for layer in ["Sandstone", "Conglomerate", "Shale", "Mudstone", "Siltstone", 
                  "Limestone", "Dolomite", "Evaporite"]:
        thickness_all[layer] = sed_thickness[layer]
        interfaces[layer] = current_top
        current_top = current_top - thickness_all[layer]
    
    # Crystalline basement
    for layer, thick in [("Granite", 5.0), ("Gneiss", 5.0), ("Basalt", 2.0), ("AncientCrust", 2.0)]:
        thickness_all[layer] = np.full_like(surface_elev, thick)
        interfaces[layer] = current_top
        current_top = current_top - thick
    
    interfaces["Basement"] = current_top
    interfaces["BasementFloor"] = current_top - 500.0
    thickness_all["Basement"] = np.full_like(surface_elev, 500.0)
    
    # Step 7: Print statistics
    print(f"\n{'='*70}")
    print("LAYER THICKNESS STATISTICS")
    print(f"{'='*70}\n")
    
    for layer in ["Topsoil", "Subsoil", "Colluvium", "Saprolite", "WeatheredBR",
                  "Clay", "Silt", "Sand", "Gravel",
                  "Sandstone", "Conglomerate", "Shale", "Mudstone", "Siltstone",
                  "Limestone", "Dolomite", "Evaporite"]:
        if layer in thickness_all:
            t = thickness_all[layer]
            pct_present = 100.0 * np.mean(t > 0.1)
            print(f"{layer:15s}: min={t.min():6.2f}  mean={t.mean():6.2f}  max={t.max():6.2f}  present={pct_present:5.1f}%")
    
    return {
        "surface_elev": surface_elev,
        "interfaces": interfaces,
        "thickness": thickness_all,
        "terrain": terrain,
        "basin_depth": basin_depth,
        "meta": {
            "elev_range_m": elev_range_m,
            "pixel_scale_m": pixel_scale_m,
        }
    }

# ============================================================================
# PLOTTING
# ============================================================================

def plot_cross_section(ax, x_vals, strata, slice_idx, axis='x'):
    """Plot cross-section."""
    interfaces = strata["interfaces"]
    surface_elev = strata["surface_elev"]
    
    if axis == 'x':
        surf_line = surface_elev[slice_idx, :]
        get_interface = lambda name: interfaces[name][slice_idx, :] if name in interfaces else None
    else:
        surf_line = surface_elev[:, slice_idx]
        get_interface = lambda name: interfaces[name][:, slice_idx] if name in interfaces else None
    
    z_floor = interfaces["BasementFloor"][0, 0]
    
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
        ("Gravel", "#a0522d"),
        ("Sand", "#f4a460"),
        ("Silt", "#e6d8ad"),
        ("Clay", "#8fbc8f"),
        ("WeatheredBR", "#bdb76b"),
        ("Saprolite", "#daa520"),
        ("Colluvium", "#bc8f8f"),
        ("Subsoil", "#8b7765"),
        ("Topsoil", "#654321"),
    ]
    
    for layer_name, color in layers_plot:
        top_interface = get_interface(layer_name)
        if top_interface is None:
            continue
        
        idx = next((i for i, (name, _) in enumerate(layers_plot) if name == layer_name), None)
        if idx is None or idx == 0:
            bottom_interface = np.full_like(top_interface, z_floor)
        else:
            bottom_name = layers_plot[idx - 1][0]
            bottom_interface = get_interface(bottom_name)
            if bottom_interface is None:
                bottom_interface = np.full_like(top_interface, z_floor)
        
        ax.fill_between(x_vals, bottom_interface, top_interface,
                        color=color, edgecolor='none', linewidth=0)
    
    ax.plot(x_vals, surf_line, 'k-', linewidth=1.5)
    ax.set_xlabel(f"{'X' if axis == 'x' else 'Y'} Distance (m)")
    ax.set_ylabel("Elevation (m)")
    ax.grid(True, alpha=0.3)

if __name__ == "__main__":
    # Generate
    z, rng = quantum_seeded_topography(N=512, random_seed=42)
    strata = generate_stratigraphy_facies_based(z, rng, elev_range_m=700.0, pixel_scale_m=10.0)
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    N = z.shape[0]
    x_vals = np.arange(N) * 10.0
    
    plot_cross_section(axes[0], x_vals, strata, N//2, axis='x')
    axes[0].set_title("X Cross-Section (Facies-Based)", fontweight='bold')
    
    plot_cross_section(axes[1], x_vals, strata, N//2, axis='y')
    axes[1].set_title("Y Cross-Section (Facies-Based)", fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("facies_based_stratigraphy.png", dpi=150)
    print("\nSaved: facies_based_stratigraphy.png")
    plt.show()
