#!/usr/bin/env python3
"""
Realistic Erosion Model with Rivers and Lakes

This module creates an erosion simulation that:
- Generates random terrain using fractal surfaces (from Rivers new)
- Creates realistic geological layers with different erodibilities
- Generates weather/rain storms over time
- Erodes layers realistically based on water flow and material properties
- Creates rivers and lakes from accumulated water
- Provides visualization of evolving topography

Author: Based on Rivers new code, extended with erosion physics
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch
from typing import Dict, Any, Optional, Tuple, List
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


# =============================================================================
# SECTION 1: TERRAIN GENERATION (from Rivers new)
# =============================================================================

def rng_from_seed(random_seed: Optional[int] = None) -> np.random.Generator:
    """Create a numpy random generator with optional seed."""
    if random_seed is not None:
        return np.random.default_rng(int(random_seed))
    import os, time, hashlib
    mix = os.urandom(16) + int(time.time_ns()).to_bytes(8, "little")
    h = hashlib.blake2b(mix, digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(h, "little"))


def fractional_surface(N: int, beta: float = 3.1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Generate a fractal surface using power-law spectrum. Higher beta = smoother terrain."""
    rng = rng or np.random.default_rng()
    kx = np.fft.fftfreq(N)
    ky = np.fft.rfftfreq(N)
    K = np.sqrt(kx[:, None]**2 + ky[None, :]**2)
    K[0, 0] = np.inf
    amp = 1.0 / (K ** (beta/2))
    phase = rng.uniform(0, 2*np.pi, size=(N, ky.size))
    spec = amp * (np.cos(phase) + 1j*np.sin(phase))
    spec[0, 0] = 0.0
    z = np.fft.irfftn(spec, s=(N, N), axes=(0, 1))
    lo, hi = np.percentile(z, [2, 98])
    return np.clip((z - lo)/(hi - lo + 1e-12), 0, 1)


def bilinear_sample(img: np.ndarray, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Bilinear interpolation sampling of an image."""
    N = img.shape[0]
    x0 = np.floor(X).astype(int) % N
    y0 = np.floor(Y).astype(int) % N
    x1 = (x0+1) % N
    y1 = (y0+1) % N
    dx = X - np.floor(X)
    dy = Y - np.floor(Y)
    return ((1-dx)*(1-dy)*img[x0,y0] + dx*(1-dy)*img[x1,y0] +
            (1-dx)*dy*img[x0,y1] + dx*dy*img[x1,y1])


def domain_warp(z: np.ndarray, rng: np.random.Generator, amp: float = 0.12, beta: float = 3.0) -> np.ndarray:
    """Apply coordinate distortion to create gnarlier micro-relief."""
    N = z.shape[0]
    u = fractional_surface(N, beta=beta, rng=rng)*2 - 1
    v = fractional_surface(N, beta=beta, rng=rng)*2 - 1
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    Xw = (ii + amp*N*u) % N
    Yw = (jj + amp*N*v) % N
    return bilinear_sample(z, Xw, Yw)


def ridged_mix(z: np.ndarray, alpha: float = 0.18) -> np.ndarray:
    """Apply ridge/valley sharpening. Higher alpha = craggier terrain."""
    ridged = 1.0 - np.abs(2.0*z - 1.0)
    out = (1-alpha)*z + alpha*ridged
    lo, hi = np.percentile(out, [2, 98])
    return np.clip((out - lo)/(hi - lo + 1e-12), 0, 1)


def _box_blur(a: np.ndarray, k: int = 5) -> np.ndarray:
    """Apply box blur smoothing."""
    if k <= 1:
        return a
    out = a.copy()
    for axis in (0, 1):
        tmp = out
        s = np.zeros_like(tmp)
        for i in range(-(k//2), k//2+1):
            s += np.roll(tmp, i, axis=axis)
        out = s / float(k)
    return out


def _normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize array to 0-1 range using percentiles."""
    lo, hi = np.percentile(x, [2, 98])
    return np.clip((x - lo)/(hi - lo + eps), 0.0, 1.0)


def generate_terrain(
    N: int = 256,
    beta: float = 3.1,
    warp_amp: float = 0.12,
    ridged_alpha: float = 0.18,
    random_seed: Optional[int] = None,
    elev_range_m: float = 500.0,
) -> Tuple[np.ndarray, np.random.Generator]:
    """
    Generate quantum-style seeded terrain.
    
    Parameters
    ----------
    N : int
        Grid size (NxN)
    beta : float
        Power-law exponent for terrain smoothness
    warp_amp : float
        Domain warping amplitude
    ridged_alpha : float
        Ridge sharpening factor
    random_seed : int, optional
        Seed for reproducibility
    elev_range_m : float
        Total elevation range in meters
        
    Returns
    -------
    elevation : np.ndarray
        Elevation array in meters (shape NxN)
    rng : np.random.Generator
        Random generator for further use
    """
    rng = rng_from_seed(random_seed)
    
    # Generate base terrain
    base_low = fractional_surface(N, beta=beta, rng=rng)
    base_high = fractional_surface(N, beta=beta-0.4, rng=rng)
    z = 0.65*base_low + 0.35*base_high
    
    # Apply transformations
    z = domain_warp(z, rng=rng, amp=warp_amp, beta=beta)
    z = ridged_mix(z, alpha=ridged_alpha)
    
    # Convert to meters
    elevation = z * elev_range_m
    
    return elevation, rng


# =============================================================================
# SECTION 2: GEOLOGICAL LAYER GENERATION
# =============================================================================

# Layer properties: erodibility (0-1, higher = easier to erode), density, color
LAYER_PROPERTIES = {
    "Topsoil": {"erodibility": 1.00, "density": 1600, "color": "sienna"},
    "Subsoil": {"erodibility": 0.85, "density": 1700, "color": "peru"},
    "Colluvium": {"erodibility": 0.90, "density": 1750, "color": "burlywood"},
    "Alluvium": {"erodibility": 0.95, "density": 1700, "color": "tan"},
    "Clay": {"erodibility": 0.80, "density": 1850, "color": "lightcoral"},
    "Silt": {"erodibility": 0.90, "density": 1750, "color": "thistle"},
    "Sand": {"erodibility": 0.85, "density": 1700, "color": "navajowhite"},
    "Saprolite": {"erodibility": 0.70, "density": 1900, "color": "khaki"},
    "WeatheredBR": {"erodibility": 0.55, "density": 2100, "color": "darkkhaki"},
    "Shale": {"erodibility": 0.45, "density": 2300, "color": "slategray"},
    "Mudstone": {"erodibility": 0.45, "density": 2300, "color": "rosybrown"},
    "Siltstone": {"erodibility": 0.35, "density": 2350, "color": "lightsteelblue"},
    "Sandstone": {"erodibility": 0.30, "density": 2200, "color": "orange"},
    "Conglomerate": {"erodibility": 0.25, "density": 2300, "color": "chocolate"},
    "Limestone": {"erodibility": 0.28, "density": 2400, "color": "lightgray"},
    "Dolomite": {"erodibility": 0.24, "density": 2450, "color": "gainsboro"},
    "Granite": {"erodibility": 0.15, "density": 2700, "color": "lightpink"},
    "Gneiss": {"erodibility": 0.16, "density": 2750, "color": "violet"},
    "Basalt": {"erodibility": 0.12, "density": 2950, "color": "royalblue"},
    "Basement": {"erodibility": 0.10, "density": 2850, "color": "dimgray"},
}

# Layer order from top to bottom
LAYER_ORDER = [
    "Topsoil", "Subsoil", "Colluvium", "Alluvium",
    "Clay", "Silt", "Sand", "Saprolite", "WeatheredBR",
    "Shale", "Mudstone", "Siltstone", "Sandstone", "Conglomerate",
    "Limestone", "Dolomite", "Granite", "Gneiss", "Basalt", "Basement"
]


def soil_thickness_from_slope(z_norm: np.ndarray, soil_range_m: Tuple[float, float] = (0.3, 1.8)) -> np.ndarray:
    """Calculate soil thickness based on slope - thinner on steep slopes."""
    dzdx, dzdy = np.gradient(z_norm)
    slope_mag = np.sqrt(dzdx**2 + dzdy**2)
    slope_n = _normalize(slope_mag)
    t = soil_range_m[1] - (soil_range_m[1] - soil_range_m[0]) * slope_n
    return _box_blur(t, k=5)


def generate_layer_thicknesses(
    elevation: np.ndarray,
    rng: np.random.Generator,
    total_depth_m: float = 200.0,
) -> Dict[str, np.ndarray]:
    """
    Generate thickness fields for each geological layer.
    
    Parameters
    ----------
    elevation : np.ndarray
        Surface elevation in meters
    rng : np.random.Generator
        Random generator
    total_depth_m : float
        Total depth of all layers
        
    Returns
    -------
    thicknesses : dict
        Dictionary mapping layer names to thickness arrays (meters)
    """
    N = elevation.shape[0]
    z_norm = _normalize(elevation)
    
    # Compute terrain derivatives
    dzdx, dzdy = np.gradient(z_norm)
    slope_mag = np.sqrt(dzdx**2 + dzdy**2)
    slope_n = _normalize(slope_mag)
    gentle = 1.0 - slope_n
    
    # Compute curvature for valley/ridge detection
    d2x, _ = np.gradient(dzdx)
    _, d2y = np.gradient(dzdy)
    curv = d2x + d2y
    
    # Basin vs high areas
    k_smooth = max(5, int(0.1 * N) | 1)
    z_smooth = _box_blur(z_norm, k=k_smooth)
    basins = _normalize(1.0 - z_smooth)
    highs = _normalize(z_smooth)
    
    thicknesses = {}
    
    # Near-surface regolith - thicker in basins and gentle slopes
    soil_total = soil_thickness_from_slope(z_norm)
    thicknesses["Topsoil"] = 0.4 * soil_total
    thicknesses["Subsoil"] = 0.6 * soil_total
    
    # Colluvium - slope wash material, thickest at slope bases
    hollows = _normalize(np.maximum(curv, 0.0))
    coll_index = _normalize(0.35*gentle + 0.30*hollows + 0.20*basins + 0.15*_normalize(1-slope_n))
    noise = rng.lognormal(mean=0.0, sigma=0.2, size=coll_index.shape)
    thicknesses["Colluvium"] = 0.5 + _normalize(coll_index * noise) * 15.0
    
    # Alluvium - river/floodplain deposits in low areas
    catch = _box_blur(_box_blur(1.0 - slope_n, k=7), k=13)
    wet = _normalize(catch - slope_n)
    thicknesses["Alluvium"] = np.where(wet > 0.6, rng.uniform(0.5, 3.0, size=wet.shape), 0.0)
    
    # Valley fill sediments
    basin_low = basins * np.clip(1.0 - z_norm, 0.0, 1.0)
    thicknesses["Clay"] = 15.0 * _normalize((1.0 - slope_n) * basin_low)
    thicknesses["Silt"] = 12.0 * _normalize((1.0 - slope_n * 0.7) * basin_low)
    thicknesses["Sand"] = 18.0 * _normalize(basin_low * slope_n)
    
    # Weathered rock
    inter_idx = _normalize(0.6*gentle + 0.4*highs)
    base_sap = np.exp(np.log(6.0) + 0.35 * rng.standard_normal(size=inter_idx.shape))
    thicknesses["Saprolite"] = np.clip(base_sap * (0.4 + 0.6*inter_idx), 0.5, 25.0)
    
    tex = fractional_surface(N, beta=3.0, rng=rng)
    tex = 1 - np.abs(2*tex - 1)
    base_rind = np.exp(np.log(1.8) + 0.25 * rng.standard_normal(size=tex.shape))
    thicknesses["WeatheredBR"] = np.clip(0.5*base_rind + 0.5*base_rind*tex, 0.4, 5.0)
    
    # Sedimentary rocks - controlled by basin/high environments
    sed_thickness_base = total_depth_m * 0.4
    
    # Shale - deepest basins
    shale_env = _normalize(basins)
    thicknesses["Shale"] = sed_thickness_base * 0.15 * (0.3 + 0.7 * shale_env)
    thicknesses["Mudstone"] = sed_thickness_base * 0.10 * (0.3 + 0.7 * shale_env)
    thicknesses["Siltstone"] = sed_thickness_base * 0.08 * (0.4 + 0.6 * _normalize(basins * (1-basins)))
    
    # Sandstone - basin margins
    sand_env = _normalize(basins * (1.0 - basins) * (0.5 + 0.5 * slope_n))
    thicknesses["Sandstone"] = sed_thickness_base * 0.20 * (0.4 + 0.6 * sand_env)
    thicknesses["Conglomerate"] = sed_thickness_base * 0.05 * (0.3 + 0.7 * slope_n)
    
    # Carbonates - gentle highs
    lime_env = _normalize(highs * gentle)
    thicknesses["Limestone"] = sed_thickness_base * 0.15 * (0.3 + 0.7 * lime_env)
    thicknesses["Dolomite"] = sed_thickness_base * 0.07 * (0.3 + 0.7 * lime_env)
    
    # Crystalline basement - fills remaining depth
    remaining = total_depth_m - sum(t.mean() for t in thicknesses.values())
    remaining = max(remaining, 20.0)
    
    basement_frac = 0.3 + 0.2 * z_norm + 0.1 * gentle
    thicknesses["Granite"] = remaining * 0.35 * _normalize(basement_frac)
    thicknesses["Gneiss"] = remaining * 0.25 * _normalize(basement_frac + 0.1*slope_n)
    thicknesses["Basalt"] = remaining * 0.10 * _normalize(basins * 0.5 + rng.random(size=basins.shape) * 0.5)
    thicknesses["Basement"] = remaining * 0.30 * np.ones_like(elevation)
    
    return thicknesses


def compute_layer_interfaces(
    elevation: np.ndarray,
    thicknesses: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Compute the elevation of the top of each layer.
    
    Parameters
    ----------
    elevation : np.ndarray
        Surface elevation
    thicknesses : dict
        Layer thicknesses
        
    Returns
    -------
    interfaces : dict
        Top elevation of each layer
    """
    interfaces = {"Surface": elevation.copy()}
    current_top = elevation.copy()
    
    for layer in LAYER_ORDER:
        if layer in thicknesses:
            current_top = current_top - thicknesses[layer]
            interfaces[layer] = current_top.copy()
    
    return interfaces


def get_surface_erodibility(
    elevation: np.ndarray,
    interfaces: Dict[str, np.ndarray],
    thicknesses: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    Get the erodibility of the currently exposed layer at each cell.
    
    The topmost layer with positive thickness determines the erodibility.
    
    Returns
    -------
    erodibility : np.ndarray
        Erodibility values (0-1) for each cell
    """
    erodibility = np.full_like(elevation, LAYER_PROPERTIES["Basement"]["erodibility"])
    
    # Go from bottom to top - the last layer with thickness > 0 wins
    for layer in reversed(LAYER_ORDER):
        if layer in thicknesses and layer in LAYER_PROPERTIES:
            mask = thicknesses[layer] > 0.01
            erodibility[mask] = LAYER_PROPERTIES[layer]["erodibility"]
    
    return erodibility


def get_top_layer_map(
    elevation: np.ndarray,
    thicknesses: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    Get the name of the topmost layer at each cell.
    
    Returns
    -------
    top_layer : np.ndarray
        String array with layer names
    """
    top_layer = np.full(elevation.shape, "Basement", dtype=object)
    
    for layer in reversed(LAYER_ORDER):
        if layer in thicknesses:
            mask = thicknesses[layer] > 0.01
            top_layer[mask] = layer
    
    return top_layer


# =============================================================================
# SECTION 3: WEATHER/RAINFALL GENERATION
# =============================================================================

def get_climate_params(year: int, terrain_relief_m: float = 500.0) -> Dict[str, float]:
    """
    Get climate parameters for a given year with terrain influence.
    
    Parameters
    ----------
    year : int
        Simulation year
    terrain_relief_m : float
        Terrain relief in meters (affects orographic rainfall)
        
    Returns
    -------
    params : dict
        Climate parameters including storms per year, rainfall, etc.
    """
    t = float(year)
    
    # Logistic trend function
    def logistic(t, low, high, t_mid=50.0, tau=30.0):
        x = (t - t_mid) / max(tau, 1e-9)
        return low + (high - low) / (1.0 + np.exp(-x))
    
    # Orographic factor based on terrain relief
    relief_factor = np.clip(terrain_relief_m / 800.0, 0.5, 2.0)
    
    return {
        "mean_storms_per_year": logistic(t, 20.0, 30.0) * (0.9 + 0.2 * relief_factor),
        "mean_storm_duration_days": 1.0 + 0.02 * t,
        "mean_storm_intensity": logistic(t, 1.0, 1.4),
        "mean_annual_rain_mm": logistic(t, 800.0, 1200.0) * relief_factor,
        "wind_dir_deg": 270.0 + 0.02 * t,  # Westerly winds with slight drift
    }


def generate_storm_schedule(
    year: int,
    rng: np.random.Generator,
    terrain_relief_m: float = 500.0,
    year_length_days: float = 365.0,
) -> List[Dict[str, Any]]:
    """
    Generate a schedule of storms for one year.
    
    Parameters
    ----------
    year : int
        Year index
    rng : np.random.Generator
        Random generator
    terrain_relief_m : float
        Terrain relief for climate calculation
    year_length_days : float
        Days in year
        
    Returns
    -------
    storms : list
        List of storm event dictionaries
    """
    params = get_climate_params(year, terrain_relief_m)
    
    storms = []
    t = 0.0
    storm_idx = 0
    
    # Mean gap between storms
    mean_gap = year_length_days / params["mean_storms_per_year"]
    
    while t < year_length_days:
        # Exponential inter-storm gap
        u = rng.random()
        gap = -mean_gap * np.log(1 - min(u, 0.9999))
        t += gap
        
        if t >= year_length_days:
            break
        
        # Storm duration (Weibull distributed)
        u_dur = rng.random()
        duration = params["mean_storm_duration_days"] * (-np.log(1 - min(u_dur, 0.9999)))**0.5
        duration = max(0.1, min(duration, 5.0))
        
        # Seasonal intensity variation
        frac_year = (t / year_length_days) % 1.0
        seasonal_mult = 1.0 + 0.4 * np.cos(2.0 * np.pi * frac_year)
        
        # Storm intensity
        intensity = params["mean_storm_intensity"] * seasonal_mult * (0.7 + 0.6 * rng.random())
        
        # Storm direction with seasonal and random variation
        base_dir = params["wind_dir_deg"]
        seasonal_shift = 20.0 * np.sin(2.0 * np.pi * frac_year)
        rand_shift = (rng.random() - 0.5) * 80.0
        direction = (base_dir + seasonal_shift + rand_shift) % 360.0
        
        storms.append({
            "year": year,
            "storm_index": storm_idx,
            "t_start_day": t,
            "duration_days": duration,
            "intensity": intensity,
            "wind_dir_deg": direction,
        })
        
        t += duration
        storm_idx += 1
    
    return storms


def generate_rain_field(
    elevation: np.ndarray,
    storm: Dict[str, Any],
    rng: np.random.Generator,
    pixel_scale_m: float = 100.0,
) -> np.ndarray:
    """
    Generate a rainfall intensity field for a storm.
    
    Parameters
    ----------
    elevation : np.ndarray
        Terrain elevation
    storm : dict
        Storm parameters
    rng : np.random.Generator
        Random generator
    pixel_scale_m : float
        Grid cell size in meters
        
    Returns
    -------
    rain : np.ndarray
        Rainfall intensity field (mm/day)
    """
    ny, nx = elevation.shape
    
    # Normalize elevation
    e_min, e_max = elevation.min(), elevation.max()
    e_norm = (elevation - e_min) / (e_max - e_min + 1e-9)
    
    # Compute gradients for windward/leeward effects
    dEy, dEx = np.gradient(elevation, pixel_scale_m, pixel_scale_m)
    slope_mag = np.sqrt(dEx**2 + dEy**2) + 1e-9
    
    # Wind direction (storm comes FROM this direction)
    theta = np.deg2rad(storm["wind_dir_deg"])
    u_hat, v_hat = np.cos(theta), np.sin(theta)
    
    # Windward/leeward factor: positive when slope faces wind
    dot = u_hat * dEx + v_hat * dEy
    windward = 0.5 * (np.clip(dot / slope_mag, -1, 1) + 1.0)  # 0 = lee, 1 = windward
    
    # Base orographic enhancement: more rain at higher elevations
    orographic = 0.5 + 0.5 * e_norm
    
    # Combine windward and orographic effects
    terrain_factor = orographic * (0.6 + 0.8 * windward)
    
    # Normalize so spatial mean ~ 1
    terrain_factor /= (terrain_factor.mean() + 1e-9)
    
    # Add storm core (moving Gaussian)
    x_coords = np.arange(nx)
    y_coords = np.arange(ny)
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # Storm enters from upwind edge
    if abs(u_hat) >= abs(v_hat):
        cx = 0.0 if u_hat > 0 else nx - 1
        cy = rng.uniform(0.2 * ny, 0.8 * ny)
    else:
        cy = 0.0 if v_hat > 0 else ny - 1
        cx = rng.uniform(0.2 * nx, 0.8 * nx)
    
    # Storm core radius
    core_radius = 0.35 * min(nx, ny)
    
    # Position after storm duration (approximate center)
    speed_cells = 0.15 * min(nx, ny) / storm["duration_days"]
    cx_mid = cx + u_hat * speed_cells * storm["duration_days"] * 0.5
    cy_mid = cy + v_hat * speed_cells * storm["duration_days"] * 0.5
    
    r2 = (X - cx_mid)**2 + (Y - cy_mid)**2
    core = np.exp(-r2 / (core_radius**2))
    
    # Combine core pattern with terrain effects
    rain_pattern = 0.5 * core + 0.5 * terrain_factor
    
    # Add noise
    noise = 1.0 + 0.3 * rng.standard_normal(size=elevation.shape)
    noise = np.clip(noise, 0.3, 2.0)
    
    # Scale by intensity
    rain = rain_pattern * noise * storm["intensity"]
    
    return np.maximum(rain, 0.0)


def accumulate_annual_rain(
    elevation: np.ndarray,
    year: int,
    rng: np.random.Generator,
    pixel_scale_m: float = 100.0,
) -> np.ndarray:
    """
    Accumulate total rainfall for one year.
    
    Returns
    -------
    total_rain : np.ndarray
        Total rainfall in mm for the year
    """
    relief = elevation.max() - elevation.min()
    storms = generate_storm_schedule(year, rng, relief)
    
    total_rain = np.zeros_like(elevation)
    
    for storm in storms:
        rain_field = generate_rain_field(elevation, storm, rng, pixel_scale_m)
        # Scale by storm duration to get total depth
        total_rain += rain_field * storm["duration_days"]
    
    return total_rain


# =============================================================================
# SECTION 4: WATER FLOW AND RIVER ROUTING
# =============================================================================

def compute_flow_direction_d8(elevation: np.ndarray) -> np.ndarray:
    """
    Compute D8 flow direction for each cell.
    
    Returns
    -------
    flow_dir : np.ndarray
        Flow direction encoded as 0-7 (8 neighbors), -1 for pit
    """
    ny, nx = elevation.shape
    flow_dir = np.full((ny, nx), -1, dtype=np.int8)
    
    # Neighbor offsets: 0=E, 1=NE, 2=N, 3=NW, 4=W, 5=SW, 6=S, 7=SE
    dx = [1, 1, 0, -1, -1, -1, 0, 1]
    dy = [0, -1, -1, -1, 0, 1, 1, 1]
    dist = [1.0, np.sqrt(2), 1.0, np.sqrt(2), 1.0, np.sqrt(2), 1.0, np.sqrt(2)]
    
    for j in range(ny):
        for i in range(nx):
            max_slope = 0.0
            best_dir = -1
            
            for d in range(8):
                ni, nj = i + dx[d], j + dy[d]
                
                # Handle boundaries - allow flow out at edges
                if ni < 0 or ni >= nx or nj < 0 or nj >= ny:
                    slope = elevation[j, i] / dist[d]  # Assume 0 at boundary
                else:
                    slope = (elevation[j, i] - elevation[nj, ni]) / dist[d]
                
                if slope > max_slope:
                    max_slope = slope
                    best_dir = d
            
            flow_dir[j, i] = best_dir
    
    return flow_dir


def compute_flow_accumulation(
    elevation: np.ndarray,
    flow_dir: np.ndarray,
    rain: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute flow accumulation (drainage area) using D8 algorithm.
    
    Parameters
    ----------
    elevation : np.ndarray
        Terrain elevation
    flow_dir : np.ndarray
        D8 flow directions
    rain : np.ndarray, optional
        Rainfall to accumulate (if None, counts cells)
        
    Returns
    -------
    accumulation : np.ndarray
        Flow accumulation (area or total rainfall)
    """
    ny, nx = elevation.shape
    
    if rain is None:
        accumulation = np.ones((ny, nx), dtype=np.float64)
    else:
        accumulation = rain.copy().astype(np.float64)
    
    # Neighbor offsets
    dx = [1, 1, 0, -1, -1, -1, 0, 1]
    dy = [0, -1, -1, -1, 0, 1, 1, 1]
    
    # Sort cells by elevation (highest first)
    flat_elev = elevation.flatten()
    sorted_indices = np.argsort(-flat_elev)
    
    for idx in sorted_indices:
        j, i = divmod(idx, nx)
        d = flow_dir[j, i]
        
        if d >= 0:
            ni, nj = i + dx[d], j + dy[d]
            if 0 <= ni < nx and 0 <= nj < ny:
                accumulation[nj, ni] += accumulation[j, i]
    
    return accumulation


def fill_depressions(elevation: np.ndarray, max_fill: float = 50.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fill depressions in the terrain to allow continuous flow.
    Uses priority-flood algorithm.
    
    Parameters
    ----------
    elevation : np.ndarray
        Input elevation
    max_fill : float
        Maximum fill depth allowed (creates lakes if exceeded)
        
    Returns
    -------
    filled : np.ndarray
        Filled elevation
    lake_depth : np.ndarray
        Depth of water in lakes/depressions
    """
    ny, nx = elevation.shape
    filled = elevation.copy()
    lake_depth = np.zeros_like(elevation)
    
    # Priority queue approach (simplified)
    import heapq
    
    # Initialize with boundary cells
    visited = np.zeros((ny, nx), dtype=bool)
    pq = []
    
    # Add boundary cells
    for i in range(nx):
        heapq.heappush(pq, (elevation[0, i], 0, i))
        heapq.heappush(pq, (elevation[ny-1, i], ny-1, i))
        visited[0, i] = True
        visited[ny-1, i] = True
    for j in range(1, ny-1):
        heapq.heappush(pq, (elevation[j, 0], j, 0))
        heapq.heappush(pq, (elevation[j, nx-1], j, nx-1))
        visited[j, 0] = True
        visited[j, nx-1] = True
    
    # Neighbor offsets (8-connected)
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    while pq:
        h, j, i = heapq.heappop(pq)
        
        for dj, di in neighbors:
            nj, ni = j + dj, i + di
            
            if 0 <= nj < ny and 0 <= ni < nx and not visited[nj, ni]:
                visited[nj, ni] = True
                
                # If neighbor is lower, fill it up to current level
                if elevation[nj, ni] < h:
                    fill_amount = h - elevation[nj, ni]
                    if fill_amount <= max_fill:
                        filled[nj, ni] = h
                        lake_depth[nj, ni] = fill_amount
                    else:
                        # Create a lake with max depth
                        filled[nj, ni] = elevation[nj, ni] + max_fill
                        lake_depth[nj, ni] = max_fill
                    heapq.heappush(pq, (filled[nj, ni], nj, ni))
                else:
                    heapq.heappush(pq, (elevation[nj, ni], nj, ni))
    
    return filled, lake_depth


def identify_rivers(
    flow_accumulation: np.ndarray,
    threshold_fraction: float = 0.05,
) -> np.ndarray:
    """
    Identify river cells based on flow accumulation threshold.
    
    Parameters
    ----------
    flow_accumulation : np.ndarray
        Flow accumulation values
    threshold_fraction : float
        Fraction of max accumulation to consider as river (default 5%)
        
    Returns
    -------
    river_mask : np.ndarray (bool)
        True where rivers exist
    """
    # Use percentage of maximum flow
    threshold = threshold_fraction * flow_accumulation.max()
    
    # But also don't be too restrictive - at least some cells should be rivers
    # Aim for ~5-15% of cells being rivers
    sorted_flow = np.sort(flow_accumulation.flatten())[::-1]
    target_river_count = max(int(0.05 * flow_accumulation.size), 10)
    
    if target_river_count < len(sorted_flow):
        alt_threshold = sorted_flow[target_river_count]
        threshold = min(threshold, alt_threshold)
    
    return flow_accumulation > threshold


def identify_lakes(lake_depth: np.ndarray, min_depth: float = 0.5) -> np.ndarray:
    """
    Identify lake cells based on water depth.
    
    Returns
    -------
    lake_mask : np.ndarray (bool)
        True where lakes exist
    """
    return lake_depth > min_depth


# =============================================================================
# SECTION 5: EROSION MODEL
# =============================================================================

def compute_stream_power_erosion(
    elevation: np.ndarray,
    flow_accumulation: np.ndarray,
    erodibility: np.ndarray,
    K_base: float = 1e-5,
    m: float = 0.5,
    n: float = 1.0,
    pixel_scale_m: float = 100.0,
) -> np.ndarray:
    """
    Compute stream power erosion rate.
    
    E = K * A^m * S^n
    
    Parameters
    ----------
    elevation : np.ndarray
        Terrain elevation
    flow_accumulation : np.ndarray
        Drainage area
    erodibility : np.ndarray
        Material erodibility field (0-1)
    K_base : float
        Base erosion coefficient
    m : float
        Area exponent
    n : float
        Slope exponent
    pixel_scale_m : float
        Grid cell size
        
    Returns
    -------
    erosion_rate : np.ndarray
        Erosion rate in meters per year
    """
    # Compute slope
    dy, dx = np.gradient(elevation, pixel_scale_m)
    slope = np.sqrt(dx**2 + dy**2)
    slope = np.maximum(slope, 1e-6)  # Prevent division by zero
    
    # Stream power erosion
    K = K_base * erodibility
    area_factor = flow_accumulation ** m
    slope_factor = slope ** n
    
    erosion = K * area_factor * slope_factor
    
    return erosion


def compute_hillslope_diffusion(
    elevation: np.ndarray,
    erodibility: np.ndarray,
    D_base: float = 1e-3,
    pixel_scale_m: float = 100.0,
) -> np.ndarray:
    """
    Compute hillslope diffusion (creep) erosion.
    
    Parameters
    ----------
    elevation : np.ndarray
        Terrain elevation
    erodibility : np.ndarray
        Material erodibility
    D_base : float
        Base diffusion coefficient
    pixel_scale_m : float
        Grid cell size
        
    Returns
    -------
    diffusion : np.ndarray
        Diffusive erosion/deposition rate
    """
    D = D_base * erodibility
    
    # Laplacian (second derivative)
    d2x = np.roll(elevation, -1, axis=1) + np.roll(elevation, 1, axis=1) - 2*elevation
    d2y = np.roll(elevation, -1, axis=0) + np.roll(elevation, 1, axis=0) - 2*elevation
    laplacian = (d2x + d2y) / (pixel_scale_m**2)
    
    return D * laplacian


def erode_layers(
    thicknesses: Dict[str, np.ndarray],
    erosion_depth: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Apply erosion to geological layers from top to bottom.
    
    Parameters
    ----------
    thicknesses : dict
        Current layer thicknesses
    erosion_depth : np.ndarray
        Total erosion depth to apply
        
    Returns
    -------
    new_thicknesses : dict
        Updated layer thicknesses
    """
    new_thicknesses = {k: v.copy() for k, v in thicknesses.items()}
    remaining_erosion = erosion_depth.copy()
    
    for layer in LAYER_ORDER:
        if layer not in new_thicknesses:
            continue
        
        # Erode this layer
        erode_from_layer = np.minimum(remaining_erosion, new_thicknesses[layer])
        new_thicknesses[layer] -= erode_from_layer
        new_thicknesses[layer] = np.maximum(new_thicknesses[layer], 0.0)
        remaining_erosion -= erode_from_layer
        
        # Stop if no more erosion needed
        if remaining_erosion.max() < 1e-6:
            break
    
    return new_thicknesses


# =============================================================================
# SECTION 6: MAIN EROSION SIMULATION CLASS
# =============================================================================

class ErosionSimulation:
    """
    Main erosion simulation class that ties everything together.
    
    Parameters
    ----------
    grid_size : int
        Size of the grid (NxN)
    pixel_scale_m : float
        Size of each grid cell in meters
    elev_range_m : float
        Total elevation range in meters
    total_depth_m : float
        Total depth of geological layers
    random_seed : int, optional
        Seed for reproducibility
    """
    
    def __init__(
        self,
        grid_size: int = 256,
        pixel_scale_m: float = 100.0,
        elev_range_m: float = 500.0,
        total_depth_m: float = 200.0,
        random_seed: Optional[int] = None,
    ):
        self.grid_size = grid_size
        self.pixel_scale_m = pixel_scale_m
        self.elev_range_m = elev_range_m
        self.total_depth_m = total_depth_m
        self.random_seed = random_seed
        
        # Initialize terrain and layers
        self._initialize()
        
        # History for tracking
        self.history = {
            "elevation": [self.elevation.copy()],
            "total_erosion": [np.zeros_like(self.elevation)],
            "years": [0],
        }
    
    def _initialize(self):
        """Initialize terrain, layers, and derived quantities."""
        # Generate terrain
        self.elevation, self.rng = generate_terrain(
            N=self.grid_size,
            elev_range_m=self.elev_range_m,
            random_seed=self.random_seed,
        )
        
        # Generate layers
        self.thicknesses = generate_layer_thicknesses(
            self.elevation, self.rng, self.total_depth_m
        )
        
        # Compute derived quantities
        self._update_derived()
    
    def _update_derived(self):
        """Update derived quantities after elevation changes."""
        self.interfaces = compute_layer_interfaces(self.elevation, self.thicknesses)
        self.erodibility = get_surface_erodibility(
            self.elevation, self.interfaces, self.thicknesses
        )
        self.top_layer = get_top_layer_map(self.elevation, self.thicknesses)
    
    def simulate_year(
        self,
        year: int,
        K_base: float = 1e-5,
        D_base: float = 1e-3,
        dt_years: float = 1.0,
    ) -> Dict[str, np.ndarray]:
        """
        Simulate one year of erosion.
        
        Parameters
        ----------
        year : int
            Current year index
        K_base : float
            Base stream power erosion coefficient
        D_base : float
            Base diffusion coefficient
        dt_years : float
            Time step in years
            
        Returns
        -------
        results : dict
            Simulation results for this year
        """
        # Generate rainfall
        rain = accumulate_annual_rain(
            self.elevation, year, self.rng, self.pixel_scale_m
        )
        
        # Fill depressions and identify water features
        filled_elev, lake_depth = fill_depressions(self.elevation)
        
        # Compute flow
        flow_dir = compute_flow_direction_d8(filled_elev)
        flow_accum = compute_flow_accumulation(filled_elev, flow_dir, rain)
        
        # Identify rivers and lakes
        river_mask = identify_rivers(flow_accum, threshold_fraction=0.05)
        lake_mask = identify_lakes(lake_depth, min_depth=0.5)
        
        # Compute erosion
        stream_erosion = compute_stream_power_erosion(
            self.elevation, flow_accum, self.erodibility,
            K_base=K_base, pixel_scale_m=self.pixel_scale_m,
        )
        
        diffusion = compute_hillslope_diffusion(
            self.elevation, self.erodibility,
            D_base=D_base, pixel_scale_m=self.pixel_scale_m,
        )
        
        # Total erosion (positive = material removed)
        total_erosion = (stream_erosion - diffusion) * dt_years
        total_erosion = np.maximum(total_erosion, 0.0)  # Only erosion, no deposition for now
        
        # Reduce erosion in lakes (underwater)
        total_erosion[lake_mask] *= 0.1
        
        # Apply erosion to elevation and layers
        self.elevation -= total_erosion
        self.thicknesses = erode_layers(self.thicknesses, total_erosion)
        
        # Update derived quantities
        self._update_derived()
        
        return {
            "rain": rain,
            "flow_accumulation": flow_accum,
            "river_mask": river_mask,
            "lake_mask": lake_mask,
            "lake_depth": lake_depth,
            "erosion": total_erosion,
        }
    
    def run_simulation(
        self,
        num_years: int,
        K_base: float = 1e-5,
        D_base: float = 1e-3,
        save_interval: int = 10,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run multi-year erosion simulation.
        
        Parameters
        ----------
        num_years : int
            Total simulation years
        K_base : float
            Base stream power coefficient
        D_base : float
            Base diffusion coefficient
        save_interval : int
            Save history every N years
        verbose : bool
            Print progress
            
        Returns
        -------
        final_results : dict
            Final simulation state and results
        """
        cumulative_erosion = np.zeros_like(self.elevation)
        
        for year in range(num_years):
            results = self.simulate_year(year, K_base, D_base)
            cumulative_erosion += results["erosion"]
            
            # Save history
            if (year + 1) % save_interval == 0 or year == num_years - 1:
                self.history["elevation"].append(self.elevation.copy())
                self.history["total_erosion"].append(cumulative_erosion.copy())
                self.history["years"].append(year + 1)
                
                if verbose:
                    print(f"Year {year + 1}/{num_years}: "
                          f"Erosion: {results['erosion'].mean()*1000:.2f} mm/yr mean, "
                          f"Rivers: {results['river_mask'].sum()} cells, "
                          f"Lakes: {results['lake_mask'].sum()} cells")
        
        # Final flow analysis (without rain for consistent drainage area)
        filled_elev, lake_depth = fill_depressions(self.elevation)
        flow_dir = compute_flow_direction_d8(filled_elev)
        flow_accum = compute_flow_accumulation(filled_elev, flow_dir)
        
        # Use consistent river threshold
        river_mask = identify_rivers(flow_accum, threshold_fraction=0.05)
        
        return {
            "elevation": self.elevation,
            "thicknesses": self.thicknesses,
            "flow_accumulation": flow_accum,
            "river_mask": river_mask,
            "lake_mask": identify_lakes(lake_depth),
            "lake_depth": lake_depth,
            "cumulative_erosion": cumulative_erosion,
            "history": self.history,
        }
    
    def plot_state(
        self,
        results: Optional[Dict] = None,
        title_prefix: str = "",
        figsize: Tuple[int, int] = (16, 12),
    ):
        """
        Plot the current simulation state.
        
        Parameters
        ----------
        results : dict, optional
            Results from simulate_year or run_simulation
        title_prefix : str
            Prefix for plot titles
        figsize : tuple
            Figure size
        """
        if results is None:
            # Generate current state
            filled_elev, lake_depth = fill_depressions(self.elevation)
            flow_dir = compute_flow_direction_d8(filled_elev)
            flow_accum = compute_flow_accumulation(filled_elev, flow_dir)
            results = {
                "flow_accumulation": flow_accum,
                "river_mask": identify_rivers(flow_accum),
                "lake_mask": identify_lakes(lake_depth),
                "lake_depth": lake_depth,
            }
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Topography with rivers and lakes
        ax = axes[0, 0]
        im = ax.imshow(self.elevation, cmap='terrain', origin='lower')
        plt.colorbar(im, ax=ax, label='Elevation (m)')
        
        # Overlay rivers
        if results.get("river_mask") is not None:
            river_display = np.ma.masked_where(~results["river_mask"], results["flow_accumulation"])
            ax.imshow(river_display, cmap='Blues', origin='lower', alpha=0.7)
        
        # Overlay lakes
        if results.get("lake_mask") is not None:
            lake_display = np.ma.masked_where(~results["lake_mask"], results["lake_depth"])
            ax.imshow(lake_display, cmap='Blues_r', origin='lower', alpha=0.8)
        
        ax.set_title(f'{title_prefix}Topography with Rivers & Lakes')
        ax.set_xlabel('X (grid cells)')
        ax.set_ylabel('Y (grid cells)')
        
        # 2. Surface geology
        ax = axes[0, 1]
        
        # Create numeric map for plotting
        layer_to_num = {layer: i for i, layer in enumerate(LAYER_ORDER)}
        layer_num = np.zeros_like(self.elevation)
        colors = []
        for i, layer in enumerate(LAYER_ORDER):
            if layer in LAYER_PROPERTIES:
                mask = self.top_layer == layer
                layer_num[mask] = i
                colors.append(LAYER_PROPERTIES[layer]["color"])
        
        cmap = LinearSegmentedColormap.from_list("geology", colors, N=len(colors))
        im = ax.imshow(layer_num, cmap=cmap, origin='lower', vmin=0, vmax=len(LAYER_ORDER)-1)
        
        # Create legend
        patches = [Patch(color=LAYER_PROPERTIES[layer]["color"], label=layer) 
                   for layer in LAYER_ORDER if layer in LAYER_PROPERTIES]
        ax.legend(handles=patches[:10], loc='upper left', fontsize=6, ncol=2)
        
        ax.set_title(f'{title_prefix}Surface Geology')
        ax.set_xlabel('X (grid cells)')
        ax.set_ylabel('Y (grid cells)')
        
        # 3. Erodibility map
        ax = axes[1, 0]
        im = ax.imshow(self.erodibility, cmap='RdYlGn_r', origin='lower', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label='Erodibility')
        ax.set_title(f'{title_prefix}Surface Erodibility')
        ax.set_xlabel('X (grid cells)')
        ax.set_ylabel('Y (grid cells)')
        
        # 4. Flow accumulation (drainage network)
        ax = axes[1, 1]
        log_accum = np.log10(results["flow_accumulation"] + 1)
        im = ax.imshow(log_accum, cmap='Blues', origin='lower')
        plt.colorbar(im, ax=ax, label='Log10(Flow Accumulation)')
        ax.set_title(f'{title_prefix}Drainage Network')
        ax.set_xlabel('X (grid cells)')
        ax.set_ylabel('Y (grid cells)')
        
        plt.tight_layout()
        plt.show()
    
    def plot_cross_section(
        self,
        row: Optional[int] = None,
        col: Optional[int] = None,
        figsize: Tuple[int, int] = (14, 6),
    ):
        """
        Plot a cross-section showing geological layers.
        
        Parameters
        ----------
        row : int, optional
            Row index for E-W cross-section
        col : int, optional
            Column index for N-S cross-section
        """
        if row is None and col is None:
            row = self.grid_size // 2
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if row is not None:
            x = np.arange(self.grid_size)
            surface = self.elevation[row, :]
            axis_label = "X (grid cells)"
            section_label = f"E-W Cross-Section at Y={row}"
        else:
            x = np.arange(self.grid_size)
            surface = self.elevation[:, col]
            axis_label = "Y (grid cells)"
            section_label = f"N-S Cross-Section at X={col}"
        
        # Draw layers from bottom to top
        current_top = surface.copy()
        
        for layer in LAYER_ORDER:
            if layer not in self.thicknesses:
                continue
            
            if row is not None:
                thickness = self.thicknesses[layer][row, :]
            else:
                thickness = self.thicknesses[layer][:, col]
            
            bottom = current_top - thickness
            
            if thickness.max() > 0.1:  # Only draw visible layers
                color = LAYER_PROPERTIES.get(layer, {}).get("color", "gray")
                ax.fill_between(x, bottom, current_top, color=color, 
                               alpha=0.8, label=layer, linewidth=0.5, edgecolor='black')
            
            current_top = bottom
        
        # Surface line
        ax.plot(x, surface, 'k-', linewidth=2, label='Surface')
        
        ax.set_xlabel(axis_label)
        ax.set_ylabel('Elevation (m)')
        ax.set_title(section_label)
        ax.legend(loc='upper left', fontsize=7, ncol=3)
        
        # Set reasonable y-limits
        y_min = current_top.min() - 10
        y_max = surface.max() + 10
        ax.set_ylim(y_min, y_max)
        
        plt.tight_layout()
        plt.show()
    
    def plot_evolution(self, figsize: Tuple[int, int] = (16, 8)):
        """Plot the evolution of terrain over time."""
        n_snapshots = len(self.history["elevation"])
        
        if n_snapshots < 2:
            print("Not enough history to plot evolution. Run simulation first.")
            return
        
        # Select up to 4 snapshots
        indices = [0]
        if n_snapshots > 2:
            indices.append(n_snapshots // 3)
        if n_snapshots > 3:
            indices.append(2 * n_snapshots // 3)
        indices.append(n_snapshots - 1)
        
        fig, axes = plt.subplots(2, len(indices), figsize=figsize)
        
        for i, idx in enumerate(indices):
            elev = self.history["elevation"][idx]
            erosion = self.history["total_erosion"][idx]
            year = self.history["years"][idx]
            
            # Elevation
            ax = axes[0, i]
            im = ax.imshow(elev, cmap='terrain', origin='lower')
            ax.set_title(f'Year {year}\nElevation')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.colorbar(im, ax=ax, shrink=0.8)
            
            # Cumulative erosion
            ax = axes[1, i]
            im = ax.imshow(erosion * 1000, cmap='Reds', origin='lower')  # Convert to mm
            ax.set_title(f'Year {year}\nCumulative Erosion (mm)')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.colorbar(im, ax=ax, shrink=0.8)
        
        plt.tight_layout()
        plt.show()


# =============================================================================
# SECTION 7: CONVENIENCE FUNCTIONS AND DEMO
# =============================================================================

def run_erosion_demo(
    grid_size: int = 128,
    num_years: int = 100,
    random_seed: Optional[int] = None,
    K_base: float = 5e-5,
    D_base: float = 1e-3,
):
    """
    Run a demo erosion simulation.
    
    Parameters
    ----------
    grid_size : int
        Grid size (NxN)
    num_years : int
        Years to simulate
    random_seed : int, optional
        Random seed for reproducibility
    K_base : float
        Stream power erosion coefficient
    D_base : float
        Diffusion coefficient
    """
    print("=" * 60)
    print("EROSION MODEL DEMONSTRATION")
    print("=" * 60)
    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Simulation years: {num_years}")
    print()
    
    # Initialize simulation
    print("Initializing terrain and layers...")
    sim = ErosionSimulation(
        grid_size=grid_size,
        pixel_scale_m=100.0,
        elev_range_m=500.0,
        total_depth_m=200.0,
        random_seed=random_seed,
    )
    
    # Show initial state
    print("\nInitial State:")
    print(f"  Elevation range: {sim.elevation.min():.1f} - {sim.elevation.max():.1f} m")
    print(f"  Mean erodibility: {sim.erodibility.mean():.3f}")
    
    # Plot initial state
    print("\nPlotting initial state...")
    sim.plot_state(title_prefix="Initial: ")
    
    # Plot initial cross-section
    print("Plotting initial cross-section...")
    sim.plot_cross_section()
    
    # Run simulation
    print(f"\nRunning {num_years}-year erosion simulation...")
    print("-" * 40)
    results = sim.run_simulation(
        num_years=num_years,
        K_base=K_base,
        D_base=D_base,
        save_interval=max(1, num_years // 10),
        verbose=True,
    )
    print("-" * 40)
    
    # Summary
    print("\nFinal State:")
    print(f"  Elevation range: {sim.elevation.min():.1f} - {sim.elevation.max():.1f} m")
    print(f"  Total erosion: {results['cumulative_erosion'].sum()/1e6:.2f} million m³")
    print(f"  Max erosion depth: {results['cumulative_erosion'].max()*1000:.1f} mm")
    print(f"  River cells: {results['river_mask'].sum()}")
    print(f"  Lake cells: {results['lake_mask'].sum()}")
    
    # Plot final state
    print("\nPlotting final state...")
    sim.plot_state(results, title_prefix=f"After {num_years} years: ")
    
    # Plot evolution
    print("Plotting terrain evolution...")
    sim.plot_evolution()
    
    # Plot final cross-section
    print("Plotting final cross-section...")
    sim.plot_cross_section()
    
    print("\nDemo complete!")
    return sim, results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Run demonstration
    sim, results = run_erosion_demo(
        grid_size=128,      # Smaller for faster demo
        num_years=50,       # 50 years of erosion
        random_seed=42,     # Reproducible
        K_base=1e-4,        # Higher erosion rate for visible changes
        D_base=5e-4,        # Moderate diffusion
    )
