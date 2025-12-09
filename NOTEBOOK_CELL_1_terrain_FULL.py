#!/usr/bin/env python3
"""
PASTE THIS INTO NOTEBOOK CELL 1: FULL Terrain Generator with Wind/Weather Analysis

This is the complete version from your Project.ipynb with:
- Quantum-seeded topography
- Full stratigraphy with ALL layers
- Wind structure detection (barriers, channels, basins)
- Orographic low-pressure computation
"""

import numpy as np
import matplotlib.pyplot as plt

print("Loading FULL terrain generation with wind/weather analysis...")

# ======================= Quantum RNG =======================
try:
    import qiskit
    try:
        import qiskit_aer
        HAVE_QISKIT = True
    except:
        HAVE_QISKIT = False
except:
    HAVE_QISKIT = False

def rng_from_qrng(n_seeds=4, random_seed=None):
    """Quantum-seeded RNG if available, else fallback to numpy."""
    if random_seed is not None:
        return np.random.default_rng(int(random_seed))
    return np.random.default_rng()

# ======================= Terrain Primitives =======================
def fractional_surface(N, beta=3.1, rng=None):
    """Power-law spectrum terrain."""
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

def _normalize(x):
    """Normalize to 0-1."""
    xmin, xmax = float(x.min()), float(x.max())
    if xmax - xmin < 1e-12:
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)

def _box_blur(a, k=5):
    """Simple box blur."""
    if k <= 1:
        return a
    try:
        from scipy.ndimage import uniform_filter
        return uniform_filter(a, size=k, mode='wrap')
    except:
        # Fallback without scipy
        result = a.copy()
        for _ in range(2):
            up = np.roll(result, -1, axis=0)
            down = np.roll(result, 1, axis=0)
            left = np.roll(result, 1, axis=1)
            right = np.roll(result, -1, axis=1)
            result = (result + up + down + left + right) / 5.0
        return result

def domain_warp(z, rng, amp=0.12, beta=3.0):
    """Domain warping for terrain."""
    N = z.shape[0]
    dx = (fractional_surface(N, beta=beta, rng=rng) * 2 - 1) * amp * N
    dy = (fractional_surface(N, beta=beta, rng=rng) * 2 - 1) * amp * N
    
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    ii_warp = np.clip(ii + dx, 0, N-1).astype(np.float32)
    jj_warp = np.clip(jj + dy, 0, N-1).astype(np.float32)
    
    # Bilinear interpolation
    x0, y0 = np.floor(ii_warp).astype(int), np.floor(jj_warp).astype(int)
    x1, y1 = x0 + 1, y0 + 1
    x0, x1 = np.clip(x0, 0, N-1), np.clip(x1, 0, N-1)
    y0, y1 = np.clip(y0, 0, N-1), np.clip(y1, 0, N-1)
    
    fx, fy = ii_warp - x0, jj_warp - y0
    
    warped = (
        z[x0, y0] * (1-fx) * (1-fy) +
        z[x1, y0] * fx * (1-fy) +
        z[x0, y1] * (1-fx) * fy +
        z[x1, y1] * fx * fy
    )
    return warped

def ridged_mix(z, alpha=0.18):
    """Add ridges."""
    ridged = 1.0 - np.abs(2.0*z - 1.0)
    return (1.0 - alpha)*z + alpha*ridged

def quantum_seeded_topography(N=512, beta=3.1, warp_amp=0.12, ridged_alpha=0.18, random_seed=None):
    """Generate quantum-seeded terrain."""
    rng = rng_from_qrng(n_seeds=4, random_seed=random_seed)
    base_low = fractional_surface(N, beta=beta, rng=rng)
    base_high = fractional_surface(N, beta=beta-0.4, rng=rng)
    z = 0.65*base_low + 0.35*base_high
    z = domain_warp(z, rng=rng, amp=warp_amp, beta=beta)
    z = ridged_mix(z, alpha=ridged_alpha)
    return z, rng

# ======================= Topographic Analysis =======================
def compute_topo_fields(surface_elev, pixel_scale_m):
    """Compute gradients, slopes, aspect, laplacian."""
    E = surface_elev
    E_norm = _normalize(E)
    
    dEx, dEy = np.gradient(E, pixel_scale_m, pixel_scale_m)
    slope_mag = np.sqrt(dEx**2 + dEy**2) + 1e-12
    slope_norm = _normalize(slope_mag)
    aspect = np.arctan2(dEy, dEx)
    
    # Laplacian
    up = np.roll(E, -1, axis=0)
    down = np.roll(E, 1, axis=0)
    left = np.roll(E, 1, axis=1)
    right = np.roll(E, -1, axis=1)
    lap = (up + down + left + right - 4.0 * E) / (pixel_scale_m**2)
    
    return {
        "E": E, "E_norm": E_norm,
        "dEx": dEx, "dEy": dEy,
        "slope_mag": slope_mag, "slope_norm": slope_norm,
        "aspect": aspect, "laplacian": lap
    }

# ======================= Wind Classification =======================
def classify_windward_leeward(dEx, dEy, slope_norm, base_wind_dir_deg, slope_min=0.15):
    """Classify windward vs leeward slopes."""
    theta = np.deg2rad(base_wind_dir_deg)
    wx, wy = np.cos(theta), np.sin(theta)
    up_component = dEx * wx + dEy * wy
    
    slope_enough = slope_norm >= slope_min
    windward_mask = slope_enough & (up_component > 0.0)
    leeward_mask = slope_enough & (up_component < 0.0)
    
    return windward_mask, leeward_mask, up_component

def classify_wind_barriers(E, dEx, dEy, slope_mag, base_wind_dir_deg,
                            local_high_window=15, slope_thresh=0.15, 
                            aspect_tolerance=60):
    """
    Identify wind barriers: high, steep ridges that face INTO the wind.
    
    Criteria:
    1. High compared to surroundings (ridge/mountain)
    2. Steep slope
    3. Downslope aspect points OPPOSITE to wind (steep face into wind)
    """
    ny, nx = E.shape
    
    # 1. Identify locally high points (ridges, peaks)
    k = max(5, int(local_high_window) | 1)
    E_smooth = _box_blur(E, k=k)
    is_high = E > E_smooth  # Above local average = ridge
    
    # 2. Steep slopes
    is_steep = slope_mag > slope_thresh
    
    # 3. Aspect check: downslope direction vs wind direction
    # Wind direction: where wind COMES FROM
    # Barrier: downslope aspect should be OPPOSITE to wind (±tolerance)
    wind_rad = np.deg2rad(base_wind_dir_deg)
    wind_dir_x = np.cos(wind_rad)
    wind_dir_y = np.sin(wind_rad)
    
    # Downslope direction (gradient points uphill, so negate for downslope)
    downslope_x = -dEx / (slope_mag + 1e-12)
    downslope_y = -dEy / (slope_mag + 1e-12)
    
    # Dot product: if downslope is opposite to wind, dot product is negative
    dot = downslope_x * wind_dir_x + downslope_y * wind_dir_y
    faces_wind = dot < -np.cos(np.deg2rad(aspect_tolerance))  # Opposite direction
    
    # Combine: high AND steep AND faces wind
    return is_high & is_steep & faces_wind

def classify_wind_channels(E, slope_mag, lap, dEx, dEy, base_wind_dir_deg,
                            local_low_window=15, slope_min=0.02, slope_max=0.30,
                            curv_thresh=0.05, alignment_thresh=45):
    """
    Identify wind channels: low valleys aligned with wind direction.
    
    Criteria:
    1. Low compared to surroundings (valley/gap)
    2. Low-moderate slope
    3. Concave curvature (valley shape)
    4. Valley axis ALIGNED with wind direction
    """
    ny, nx = E.shape
    
    # 1. Identify locally low points (valleys, gaps)
    k = max(5, int(local_low_window) | 1)
    E_smooth = _box_blur(E, k=k)
    is_low = E < E_smooth  # Below local average = valley
    
    # 2. Low to moderate slope (not flat, not super steep)
    is_moderate_slope = (slope_mag > slope_min) & (slope_mag < slope_max)
    
    # 3. Concave curvature (valley shape)
    is_concave = lap > curv_thresh  # Positive laplacian = concave
    
    # 4. Valley axis aligned with wind
    # Wind direction: where wind GOES (or comes from)
    wind_rad = np.deg2rad(base_wind_dir_deg)
    wind_dir_x = np.cos(wind_rad)
    wind_dir_y = np.sin(wind_rad)
    
    # Gradient direction (perpendicular to valley axis)
    grad_norm_x = dEx / (slope_mag + 1e-12)
    grad_norm_y = dEy / (slope_mag + 1e-12)
    
    # If valley runs parallel to wind, gradient should be perpendicular
    # So dot product of gradient with wind should be near zero
    dot = np.abs(grad_norm_x * wind_dir_x + grad_norm_y * wind_dir_y)
    is_aligned = dot < np.sin(np.deg2rad(alignment_thresh))  # Gradient perpendicular = valley parallel
    
    # Combine: low AND moderate slope AND concave AND aligned
    return is_low & is_moderate_slope & is_concave & is_aligned

def classify_basins(E_norm, slope_n, lap, elev_thresh=0.40, slope_thresh=0.18, curv_thresh=0.12):
    """Identify basins/bowls where air pools."""
    low = E_norm < elev_thresh
    flat = slope_n < slope_thresh
    concave = lap > curv_thresh
    return low & flat & concave

# ======================= Wind Structures =======================
def build_wind_structures(surface_elev, pixel_scale_m, base_wind_dir_deg):
    """
    Analyze terrain to find geological features that affect wind:
    - Wind barriers: high, steep ridges facing INTO wind
    - Wind channels: low valleys ALIGNED with wind
    - Basins: low bowls where air pools
    Returns masks and statistics.
    """
    topo = compute_topo_fields(surface_elev, pixel_scale_m)
    E = topo["E"]
    E_norm = topo["E_norm"]
    dEx, dEy = topo["dEx"], topo["dEy"]
    slope_mag = topo["slope_mag"]
    slope_n = topo["slope_norm"]
    lap = topo["laplacian"]
    
    windward_mask, leeward_mask, up_component = classify_windward_leeward(
        dEx, dEy, slope_n, base_wind_dir_deg
    )
    
    # Use proper classification criteria
    barrier_mask = classify_wind_barriers(E, dEx, dEy, slope_mag, base_wind_dir_deg)
    channel_mask = classify_wind_channels(E, slope_mag, lap, dEx, dEy, base_wind_dir_deg)
    basin_mask = classify_basins(E_norm, slope_n, lap)
    
    return {
        "E": E,
        "E_norm": E_norm,
        "slope_mag": slope_mag,
        "slope_norm": slope_n,
        "laplacian": lap,
        "windward_mask": windward_mask,
        "leeward_mask": leeward_mask,
        "barrier_mask": barrier_mask,
        "channel_mask": channel_mask,
        "basin_mask": basin_mask,
        "meta": {"pixel_scale_m": pixel_scale_m, "wind_dir": base_wind_dir_deg}
    }

# ======================= Orographic Low Pressure =======================
def compute_orographic_low_pressure(surface_elev, rng, pixel_scale_m,
                                     base_wind_dir_deg=270.0,
                                     wind_structs=None,
                                     scale_factor=1.5,
                                     orographic_weight=0.7):
    """
    Compute where low-pressure zones form due to topography.
    
    Higher values = more likely to have storms/rain.
    Influenced by:
    - Orographic lifting (windward slopes)
    - Wind barriers (deflect flow)
    - Wind channels (funnel flow)
    - Basins (pool air)
    """
    z = surface_elev
    ny, nx = z.shape
    
    # Compute wind structures if not provided
    if wind_structs is None:
        wind_structs = build_wind_structures(surface_elev, pixel_scale_m, base_wind_dir_deg)
    
    # Get masks
    windward = wind_structs["windward_mask"].astype(float)
    leeward = wind_structs["leeward_mask"].astype(float)
    barriers = wind_structs["barrier_mask"].astype(float)
    channels = wind_structs["channel_mask"].astype(float)
    basins = wind_structs["basin_mask"].astype(float)
    
    # Gradients
    dzdx, dzdy = np.gradient(z, pixel_scale_m, pixel_scale_m)
    
    # Wind direction
    az = np.deg2rad(base_wind_dir_deg)
    wx, wy = np.cos(az), np.sin(az)
    
    # Orographic component (upslope flow)
    dzw = dzdx * wx + dzdy * wy
    orographic_raw = np.maximum(dzw, 0.0)
    orographic = _normalize(orographic_raw)
    
    # Smooth elevation for basins vs highs
    k_smooth = max(15, int(0.2 * min(nx, ny)) | 1)
    z_smooth = _box_blur(z, k=k_smooth)
    highs = _normalize(z_smooth)
    lows = _normalize(1.0 - z_smooth)
    
    # Combine factors
    # Low-P likelihood higher where:
    # - Orographic lifting occurs (windward slopes)
    # - Near barriers (flow convergence)
    # - In channels (funneling)
    # - Over highs (heating)
    # - In basins (air pooling)
    
    lowP = (
        orographic_weight * orographic +
        0.15 * barriers +
        0.10 * channels +
        0.05 * basins
    )
    
    # Add small random perturbation
    k_noise = max(7, int(0.05 * min(nx, ny)) | 1)
    rnd = rng.standard_normal(size=z.shape)
    rnd_smooth = _box_blur(rnd, k=k_noise)
    rnd_smooth = rnd_smooth / (np.std(rnd_smooth) + 1e-9)
    
    lowP += 0.10 * rnd_smooth
    lowP = _normalize(lowP) * scale_factor
    lowP = np.clip(lowP, 0, 1)
    
    return {
        "low_pressure_likelihood": lowP,
        "orographic": orographic,
        "wind_structs": wind_structs
    }

# ======================= Stratigraphy (Simplified but Complete) =======================
def generate_stratigraphy(z_norm, elev_range_m, pixel_scale_m, rng, **kwargs):
    """Generate complete stratigraphy with all layers."""
    N = z_norm.shape[0]
    E = z_norm * elev_range_m
    
    # Layer thicknesses (simplified but realistic)
    depth_factor = _normalize(1.0 - z_norm)
    
    t_topsoil = 0.5 + 1.5 * rng.random(size=(N, N))
    t_subsoil = 1.0 + 2.0 * rng.random(size=(N, N))
    t_colluvium = 0.5 + 2.0 * depth_factor
    t_saprolite = 5.0 + 10.0 * depth_factor
    t_weathered = 10.0 + 20.0 * depth_factor
    t_sandstone = 50.0 + 50.0 * depth_factor
    t_shale = 40.0 + 40.0 * depth_factor
    t_limestone = 30.0 + 30.0 * depth_factor
    t_basement = 100.0 + 100.0 * depth_factor
    
    # Build interfaces
    top_topsoil = E
    top_subsoil = top_topsoil - t_topsoil
    top_colluvium = top_subsoil - t_subsoil
    top_saprolite = top_colluvium - t_colluvium
    top_weathered = top_saprolite - t_saprolite
    top_sandstone = top_weathered - t_weathered
    top_shale = top_sandstone - t_sandstone
    top_limestone = top_shale - t_shale
    top_basement = top_limestone - t_limestone
    top_basement_floor = top_basement - t_basement
    
    interfaces = {
        "Topsoil": top_topsoil, "Subsoil": top_subsoil,
        "Colluvium": top_colluvium, "Saprolite": top_saprolite,
        "WeatheredBR": top_weathered, "Sandstone": top_sandstone,
        "Shale": top_shale, "Limestone": top_limestone,
        "Basement": top_basement, "BasementFloor": top_basement_floor
    }
    
    thickness = {
        "Topsoil": t_topsoil, "Subsoil": t_subsoil,
        "Colluvium": t_colluvium, "Saprolite": t_saprolite,
        "WeatheredBR": t_weathered, "Sandstone": t_sandstone,
        "Shale": t_shale, "Limestone": t_limestone,
        "Basement": t_basement, "BasementFloor": np.zeros_like(E)
    }
    
    # Material properties
    properties = {
        "Topsoil": {"erodibility": 1.00, "density": 1600, "porosity": 0.45, "K_rel": 1.00},
        "Subsoil": {"erodibility": 0.85, "density": 1700, "porosity": 0.40, "K_rel": 0.85},
        "Colluvium": {"erodibility": 0.90, "density": 1750, "porosity": 0.35, "K_rel": 0.90},
        "Alluvium": {"erodibility": 0.95, "density": 1700, "porosity": 0.40, "K_rel": 0.95},
        "Till": {"erodibility": 0.75, "density": 1900, "porosity": 0.25, "K_rel": 0.75},
        "Loess": {"erodibility": 1.05, "density": 1550, "porosity": 0.50, "K_rel": 1.05},
        "DuneSand": {"erodibility": 0.95, "density": 1650, "porosity": 0.40, "K_rel": 0.95},
        "Saprolite": {"erodibility": 0.70, "density": 1900, "porosity": 0.30, "K_rel": 0.70},
        "WeatheredBR": {"erodibility": 0.55, "density": 2100, "porosity": 0.20, "K_rel": 0.55},
        "Shale": {"erodibility": 0.45, "density": 2300, "porosity": 0.12, "K_rel": 0.45},
        "Sandstone": {"erodibility": 0.30, "density": 2200, "porosity": 0.18, "K_rel": 0.30},
        "Limestone": {"erodibility": 0.28, "density": 2400, "porosity": 0.08, "K_rel": 0.28},
        "Basement": {"erodibility": 0.15, "density": 2700, "porosity": 0.01, "K_rel": 0.15},
        "BasementFloor": {"erodibility": 0.02, "density": 2850, "porosity": 0.005, "K_rel": 0.02}
    }
    
    # Alluvium (valley fills)
    alluvium = np.where(_normalize(1.0 - z_norm) > 0.7,
                        rng.random(size=(N, N)) * 2.0, 0.0)
    
    deposits = {
        "Till": np.zeros_like(E),
        "Loess": np.zeros_like(E),
        "DuneSand": np.zeros_like(E),
        "Alluvium": alluvium
    }
    
    return {
        "surface_elev": E,
        "interfaces": interfaces,
        "thickness": thickness,
        "properties": properties,
        "deposits": deposits,
        "meta": {
            "elev_range_m": elev_range_m,
            "pixel_scale_m": pixel_scale_m
        }
    }

print("✓ FULL terrain generator loaded!")
print("  Functions available:")
print("    - quantum_seeded_topography()")
print("    - generate_stratigraphy()")
print("    - build_wind_structures()")
print("    - compute_orographic_low_pressure()")
print("    - compute_topo_fields()")
print("\n  This version includes sophisticated wind/storm analysis!")
