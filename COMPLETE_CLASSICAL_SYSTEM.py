"""
COMPLETE EROSION SYSTEM - FULLY CLASSICAL (NO QUANTUM)

Single block containing:
- Terrain generation (classical random)
- 6-layer stratigraphy
- Wind-rain physics
- Erosion simulation (all components)
- Water flow simulation
- Epoch-by-epoch visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List
import time

print("\n" + "="*80)
print("COMPLETE CLASSICAL EROSION SYSTEM")
print("="*80)

# ==============================================================================
# TERRAIN GENERATION (CLASSICAL)
# ==============================================================================

def fractional_surface(N, beta=3.0, rng=None):
    """Generate fractional Brownian surface."""
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


def bilinear_sample(img, X, Y):
    """Bilinear interpolation."""
    N = img.shape[0]
    x0 = np.floor(X).astype(int) % N
    y0 = np.floor(Y).astype(int) % N
    x1 = (x0+1) % N
    y1 = (y0+1) % N
    dx = X - np.floor(X)
    dy = Y - np.floor(Y)
    return ((1-dx)*(1-dy)*img[x0,y0] + dx*(1-dy)*img[x1,y0] +
            (1-dx)*dy*img[x0,y1] + dx*dy*img[x1,y1])


def domain_warp(z, rng, amp=0.10, beta=3.0):
    """Apply domain warping."""
    N = z.shape[0]
    u = fractional_surface(N, beta=beta, rng=rng)*2 - 1
    v = fractional_surface(N, beta=beta, rng=rng)*2 - 1
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    Xw = (ii + amp*N*u) % N
    Yw = (jj + amp*N*v) % N
    return bilinear_sample(z, Xw, Yw)


def ridged_mix(z, alpha=0.15):
    """Mix with ridged noise."""
    ridged = 1.0 - np.abs(2.0*z - 1.0)
    out = (1-alpha)*z + alpha*ridged
    lo, hi = np.percentile(out, [2, 98])
    return np.clip((out - lo)/(hi - lo + 1e-12), 0, 1)


def classical_topography(N=256, beta=3.0, warp_amp=0.10, ridged_alpha=0.15, random_seed=None):
    """Generate topography with classical randomness."""
    rng = np.random.default_rng(random_seed)
    
    base_low = fractional_surface(N, beta=beta, rng=rng)
    base_high = fractional_surface(N, beta=beta-0.4, rng=rng)
    z = 0.65*base_low + 0.35*base_high
    
    z = domain_warp(z, rng=rng, amp=warp_amp, beta=beta)
    z = ridged_mix(z, alpha=ridged_alpha)
    
    return z, rng


# ==============================================================================
# LAYER GENERATION
# ==============================================================================

def generate_stratigraphy_with_layers(z_norm, rng, pixel_scale_m=20.0, elev_range_m=500.0):
    """Generate 6-layer stratigraphy."""
    ny, nx = z_norm.shape
    surface_elev = z_norm * elev_range_m
    
    layer_order = ['Topsoil', 'Subsoil', 'Colluvium', 'Saprolite', 'WeatheredBR', 'Basement']
    thickness = {}
    
    # Compute slope
    grad_y, grad_x = np.gradient(z_norm, pixel_scale_m / elev_range_m)
    slope_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Compute curvature
    grad2_y, grad2_x = np.gradient(slope_mag)
    curvature = grad2_y + grad2_x
    
    # Topsoil
    gentle_mask = slope_mag < 0.1
    steep_mask = slope_mag > 0.3
    topsoil = np.ones((ny, nx)) * 1.5
    topsoil[gentle_mask] += 1.0
    topsoil[steep_mask] -= 0.8
    topsoil += rng.uniform(-0.3, 0.3, size=(ny, nx))
    thickness['Topsoil'] = np.maximum(0.2, topsoil)
    
    # Subsoil
    mid_elev_mask = (z_norm > 0.3) & (z_norm < 0.7)
    subsoil = np.ones((ny, nx)) * 3.0
    subsoil[mid_elev_mask] += 2.0
    subsoil[gentle_mask] += 1.0
    subsoil += rng.uniform(-0.5, 0.5, size=(ny, nx))
    thickness['Subsoil'] = np.maximum(0.5, subsoil)
    
    # Colluvium
    valley_mask = curvature < -0.0001
    low_elev_mask = z_norm < 0.4
    colluvium = np.zeros((ny, nx))
    colluvium[valley_mask] = 5.0 + rng.uniform(0, 5, size=np.sum(valley_mask))
    colluvium[valley_mask & low_elev_mask] += 3.0
    colluvium[steep_mask & low_elev_mask] += 2.0
    thickness['Colluvium'] = np.maximum(0.0, colluvium)
    
    # Saprolite
    ridge_mask = curvature > 0.0001
    high_elev_mask = z_norm > 0.5
    saprolite = np.ones((ny, nx)) * 8.0
    saprolite[ridge_mask & gentle_mask] += 10.0
    saprolite[high_elev_mask] += 5.0
    saprolite[valley_mask] *= 0.3
    saprolite[steep_mask] *= 0.5
    saprolite += rng.uniform(-2, 2, size=(ny, nx))
    thickness['Saprolite'] = np.maximum(0.5, saprolite)
    
    # Weathered Bedrock
    patch_pattern = fractional_surface(ny, beta=3.0, rng=rng)
    weathered_br = 3.0 + 4.0 * patch_pattern
    weathered_br[high_elev_mask] += 2.0
    weathered_br[valley_mask] *= 0.5
    thickness['WeatheredBR'] = np.maximum(0.5, weathered_br)
    
    # Basement
    thickness['Basement'] = np.ones((ny, nx)) * 1000.0
    
    return {
        'surface_elev': surface_elev,
        'thickness': thickness,
        'layer_order': layer_order,
        'pixel_scale_m': pixel_scale_m,
        'slope_mag': slope_mag,
        'curvature': curvature
    }


def compute_top_layer_map(thickness: Dict[str, np.ndarray], layer_order: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Determine which layer is exposed at surface."""
    N, M = next(iter(thickness.values())).shape
    top_idx = -np.ones((N, M), dtype=int)
    top_name = np.empty((N, M), dtype=object)
    
    for k, layer in enumerate(layer_order):
        th = thickness[layer]
        mask = (th > 1e-3) & (top_idx == -1)
        top_idx[mask] = k
        top_name[mask] = layer
    
    top_name[top_idx == -1] = "Basement"
    return top_idx, top_name


# ==============================================================================
# WIND-RAIN PHYSICS
# ==============================================================================

def classify_wind_features(surface_elev, pixel_scale_m, wind_dir_deg=90.0):
    """Classify terrain for wind interaction."""
    ny, nx = surface_elev.shape
    
    grad_y, grad_x = np.gradient(surface_elev, pixel_scale_m)
    
    wind_rad = np.radians(wind_dir_deg)
    wind_x = np.cos(wind_rad)
    wind_y = np.sin(wind_rad)
    wind_vector = (wind_x, wind_y)
    
    slope_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    grad2_y, grad2_x = np.gradient(slope_mag, pixel_scale_m)
    curvature = np.sqrt(grad2_x**2 + grad2_y**2)
    
    elev_norm = (surface_elev - surface_elev.min()) / (surface_elev.max() - surface_elev.min() + 1e-9)
    curv_norm = (curvature - curvature.min()) / (curvature.max() - curvature.min() + 1e-9)
    
    barrier_score = 0.5 * curv_norm + 0.5 * elev_norm
    barrier_score = barrier_score ** 2
    barrier_score = np.clip(barrier_score, 0, 1)
    
    channel_score = (1 - elev_norm) * (1 - curv_norm)
    channel_score = channel_score ** 2
    channel_score = np.clip(channel_score, 0, 1)
    
    total_score = barrier_score + channel_score + 1e-9
    barrier_score = barrier_score / total_score
    channel_score = channel_score / total_score
    
    return {
        'barrier_score': barrier_score.astype(np.float32),
        'channel_score': channel_score.astype(np.float32),
        'slope_vectors': (grad_x.astype(np.float32), grad_y.astype(np.float32)),
        'wind_vector': wind_vector,
        'wind_dir_deg': wind_dir_deg
    }


def apply_wind_rain_physics(base_rain, wind_features, k_windward=0.8, k_lee=0.6, k_channel=0.5):
    """Apply barrier and channel physics to rain."""
    ny, nx = base_rain.shape
    
    slope_x, slope_y = wind_features['slope_vectors']
    wind_x, wind_y = wind_features['wind_vector']
    barrier_score = wind_features['barrier_score']
    channel_score = wind_features['channel_score']
    
    slope_mag = np.sqrt(slope_x**2 + slope_y**2) + 1e-9
    cos_theta = (slope_x * wind_x + slope_y * wind_y) / slope_mag
    
    barrier_factor = np.ones((ny, nx), dtype=np.float32)
    
    windward_mask = cos_theta > 0
    barrier_factor[windward_mask] = 1.0 + k_windward * cos_theta[windward_mask] * barrier_score[windward_mask]
    
    leeward_mask = cos_theta < 0
    barrier_factor[leeward_mask] = 1.0 - k_lee * (-cos_theta[leeward_mask]) * barrier_score[leeward_mask]
    
    barrier_factor = np.clip(barrier_factor, 0.2, 2.5)
    
    channel_factor = 1.0 + k_channel * channel_score
    
    rain = base_rain * barrier_factor * channel_factor
    
    return rain


def generate_storm_with_classical_rain(surface_elev, wind_features, storm_center_ij,
                                        storm_radius_cells, base_intensity_m_per_hour,
                                        duration_hours, pixel_scale_m, rng):
    """Generate storm with classical random rain."""
    ny, nx = surface_elev.shape
    ci, cj = storm_center_ij
    
    ii, jj = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    di = ii - ci
    dj = jj - cj
    
    di = np.where(di > ny/2, di - ny, di)
    di = np.where(di < -ny/2, di + ny, di)
    dj = np.where(dj > nx/2, dj - nx, dj)
    dj = np.where(dj < -nx/2, dj + nx, dj)
    
    dist = np.sqrt(di**2 + dj**2)
    base_storm = np.exp(-(dist / storm_radius_cells)**2)
    
    storm_cells = base_storm > 0.1
    n_storm_cells = np.sum(storm_cells)
    
    if n_storm_cells > 0:
        # Classical random multipliers (instead of quantum)
        classical_mults = rng.uniform(0, 1, size=n_storm_cells)
        classical_mults = np.exp((classical_mults - 0.5) * 1.5)
        
        base_storm_random = base_storm.copy()
        base_storm_random[storm_cells] *= classical_mults
        base_storm = base_storm_random
    
    base_rain = base_intensity_m_per_hour * duration_hours * base_storm
    rain = apply_wind_rain_physics(base_rain, wind_features, k_windward=0.8, k_lee=0.6, k_channel=0.5)
    
    return rain


def run_weather_simulation(surface_elev, pixel_scale_m, num_years=10, wind_dir_deg=90.0,
                           mean_annual_rain_m=1.0, random_seed=None):
    """Run multi-year weather simulation."""
    print(f"\nGenerating {num_years} years of weather...")
    print(f"  Wind direction: {wind_dir_deg}° (EAST)")
    
    ny, nx = surface_elev.shape
    rng = np.random.default_rng(random_seed)
    
    wind_features = classify_wind_features(surface_elev, pixel_scale_m, wind_dir_deg)
    
    annual_rain_maps = []
    
    for year in range(num_years):
        year_rain = np.zeros((ny, nx), dtype=np.float32)
        
        n_storms = rng.integers(5, 16)
        
        for storm_idx in range(n_storms):
            ci = rng.integers(0, ny)
            cj = rng.integers(0, nx)
            radius = rng.integers(10, 30)
            intensity = rng.uniform(0.001, 0.005)
            duration = rng.uniform(4, 24)
            
            storm_rain = generate_storm_with_classical_rain(
                surface_elev, wind_features,
                (ci, cj), radius, intensity, duration,
                pixel_scale_m, rng
            )
            
            year_rain += storm_rain
        
        scale_factor = mean_annual_rain_m / (year_rain.mean() + 1e-9)
        year_rain *= scale_factor
        
        annual_rain_maps.append(year_rain)
        
        if (year + 1) % max(1, num_years // 5) == 0:
            print(f"  Year {year+1}/{num_years}: {year_rain.mean():.3f} m/yr")
    
    total_rain = np.sum(annual_rain_maps, axis=0)
    
    print(f"✓ Weather simulation complete")
    
    return {
        'annual_rain_maps': annual_rain_maps,
        'wind_features': wind_features,
        'total_rain': total_rain,
        'num_years': num_years
    }


# ==============================================================================
# EROSION CONSTANTS
# ==============================================================================

TIME_ACCELERATION = 10.0
RAIN_BOOST = 100.0
BASE_K = 0.001
MAX_ERODE_PER_STEP = 0.5
FLAT_K = 0.0005
SLOPE_THRESHOLD = 0.001
M_DISCHARGE = 0.5
N_SLOPE = 1.0
HALF_LOSS_FRACTION = 0.5
CAPACITY_K = 0.01
CAPACITY_M = 0.5
CAPACITY_N = 1.0
INFILTRATION_FRACTION = 0.3
DIFFUSION_K = 0.01

ERODIBILITY_MAP = {
    "Topsoil": 2.0,
    "Subsoil": 1.5,
    "Colluvium": 1.8,
    "Alluvium": 2.0,
    "Saprolite": 1.2,
    "WeatheredBR": 0.8,
    "Sandstone": 0.6,
    "Shale": 1.0,
    "Limestone": 0.7,
    "Basement": 0.3,
    "BasementFloor": 0.1,
}

print("\n✓ Erosion constants defined")

# ==============================================================================
# FLOW DIRECTION (D8)
# ==============================================================================

def compute_flow_direction_d8(elevation, pixel_scale_m):
    """Compute D8 flow direction."""
    ny, nx = elevation.shape
    flow_dir = np.full((ny, nx), -1, dtype=np.int32)
    receivers = np.zeros((ny, nx, 2), dtype=np.int32)
    distances = np.zeros((ny, nx), dtype=np.float32)
    slopes = np.zeros((ny, nx), dtype=np.float32)
    
    offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    diag_dist = pixel_scale_m * np.sqrt(2)
    card_dist = pixel_scale_m
    
    for i in range(ny):
        for j in range(nx):
            best_slope = 0
            best_idx = -1
            best_receiver = (i, j)
            best_dist = pixel_scale_m
            
            for idx, (di, dj) in enumerate(offsets):
                ni, nj = i + di, j + dj
                if 0 <= ni < ny and 0 <= nj < nx:
                    dist = diag_dist if (di != 0 and dj != 0) else card_dist
                    slope = (elevation[i,j] - elevation[ni,nj]) / dist
                    if slope > best_slope:
                        best_slope = slope
                        best_idx = idx
                        best_receiver = (ni, nj)
                        best_dist = dist
            
            flow_dir[i,j] = best_idx
            receivers[i,j] = best_receiver
            distances[i,j] = best_dist
            slopes[i,j] = best_slope
    
    return flow_dir, receivers, slopes


# ==============================================================================
# DISCHARGE
# ==============================================================================

def compute_runoff(rain, infiltration_fraction=0.3):
    """Compute runoff."""
    infiltration = rain * infiltration_fraction
    runoff = rain - infiltration
    return np.maximum(runoff, 0.0)


def compute_discharge(elevation, flow_dir, receivers, runoff, pixel_scale_m):
    """Compute discharge Q."""
    ny, nx = elevation.shape
    Q = np.zeros_like(elevation, dtype=np.float64)
    
    indices = [(i,j) for i in range(ny) for j in range(nx)]
    indices_sorted = sorted(indices, key=lambda idx: elevation[idx], reverse=True)
    
    for (i, j) in indices_sorted:
        Q[i,j] += runoff[i,j]
        
        if flow_dir[i,j] >= 0:
            ni, nj = receivers[i,j]
            Q[ni,nj] += Q[i,j]
    
    return Q


# ==============================================================================
# EROSION PASS A
# ==============================================================================

def get_erodibility_grid(top_layer_name, erodibility_map):
    """Create erodibility grid."""
    ny, nx = top_layer_name.shape
    erodibility = np.ones((ny, nx), dtype=np.float32)
    
    for layer_name, erod_val in erodibility_map.items():
        mask = (top_layer_name == layer_name)
        erodibility[mask] = erod_val
    
    return erodibility


def compute_erosion_pass_a(elevation, Q, slope, flow_dir, erodibility,
                           base_k, flat_k, max_erode_per_step, slope_threshold,
                           half_loss_fraction, m_discharge=0.5, n_slope=1.0, dt=1.0):
    """Erosion Pass A."""
    ny, nx = elevation.shape
    elevation_new = elevation.copy()
    sediment_out = np.zeros((ny, nx), dtype=np.float32)
    
    Q_safe = np.maximum(Q, 1e-6)
    
    for i in range(ny):
        for j in range(nx):
            is_downslope = (slope[i, j] > slope_threshold) and (flow_dir[i, j] >= 0)
            
            if is_downslope:
                erosion_power = (base_k * (Q_safe[i, j] ** m_discharge) *
                                (slope[i, j] ** n_slope) * erodibility[i, j] * dt)
                dz_erosion = -min(max_erode_per_step * dt, erosion_power)
            else:
                if Q[i, j] > 1000.0:
                    erosion_power = (flat_k * (Q_safe[i, j] ** m_discharge) *
                                    erodibility[i, j] * dt)
                    dz_erosion = -min(max_erode_per_step * dt, erosion_power)
                else:
                    dz_erosion = 0.0
            
            elevation_new[i, j] += dz_erosion
            eroded_material = -dz_erosion
            sediment_to_move = (1.0 - half_loss_fraction) * eroded_material
            sediment_out[i, j] = sediment_to_move
    
    return elevation_new, sediment_out


# ==============================================================================
# SEDIMENT TRANSPORT
# ==============================================================================

def compute_sediment_transport_pass_b(elevation, Q, slope, flow_dir, receivers,
                                     sediment_out, capacity_k, capacity_m,
                                     capacity_n, pixel_scale_m):
    """Sediment Transport Pass B."""
    ny, nx = elevation.shape
    elevation_new = elevation.copy()
    sediment_in = np.zeros((ny, nx), dtype=np.float32)
    
    Q_safe = np.maximum(Q, 1e-6)
    slope_safe = np.maximum(slope, 1e-9)
    
    indices = [(i, j) for i in range(ny) for j in range(nx)]
    indices_sorted = sorted(indices, key=lambda idx: elevation[idx], reverse=True)
    
    for (i, j) in indices_sorted:
        total_sediment = sediment_in[i, j] + sediment_out[i, j]
        
        capacity = capacity_k * (Q_safe[i, j] ** capacity_m) * (slope_safe[i, j] ** capacity_n)
        
        if slope[i, j] < 1e-6:
            capacity = capacity * 0.1
        
        if total_sediment > capacity:
            deposit = total_sediment - capacity
            elevation_new[i, j] += deposit
            sediment_to_downstream = capacity
        else:
            sediment_to_downstream = total_sediment
        
        if flow_dir[i, j] >= 0:
            ni, nj = receivers[i, j]
            sediment_in[ni, nj] += sediment_to_downstream
    
    return elevation_new


# ==============================================================================
# HILLSLOPE DIFFUSION
# ==============================================================================

def apply_hillslope_diffusion_8neighbor(elevation, diffusion_k, pixel_scale_m, dt=1.0):
    """Apply hillslope diffusion."""
    ny, nx = elevation.shape
    dz = np.zeros_like(elevation)
    
    offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    diag_dist = pixel_scale_m * np.sqrt(2)
    card_dist = pixel_scale_m
    
    for i in range(ny):
        for j in range(nx):
            for (di, dj) in offsets:
                ni, nj = i + di, j + dj
                if 0 <= ni < ny and 0 <= nj < nx:
                    dist = diag_dist if (di != 0 and dj != 0) else card_dist
                    height_diff = elevation[i,j] - elevation[ni,nj]
                    if height_diff > 0:
                        flux = diffusion_k * height_diff / (dist * dist)
                        dz[i,j] -= flux * dt
                        dz[ni,nj] += flux * dt
    
    return elevation + dz


# ==============================================================================
# RIVER/LAKE DETECTION
# ==============================================================================

def detect_rivers(discharge, pixel_scale_m, discharge_threshold=None):
    """Detect rivers based on discharge."""
    if discharge_threshold is None:
        discharge_threshold = np.percentile(discharge, 95)
    
    river_mask = discharge > discharge_threshold
    return river_mask


def detect_lakes(elevation, discharge, pixel_scale_m, min_lake_area_m2=100):
    """Detect lakes."""
    grad_y, grad_x = np.gradient(elevation, pixel_scale_m)
    slope = np.sqrt(grad_x**2 + grad_y**2)
    
    flat_mask = slope < 0.01
    high_discharge_mask = discharge > np.percentile(discharge, 80)
    
    lake_mask = flat_mask & high_discharge_mask
    lake_labels = np.zeros_like(elevation, dtype=np.int32)
    
    return lake_mask, lake_labels


# ==============================================================================
# MAIN SIMULATION
# ==============================================================================

def run_erosion_timestep(elevation, thickness, layer_order, rain_map, pixel_scale_m,
                        dt=1.0, apply_diffusion=True, verbose=False):
    """Run one erosion timestep."""
    rain_boosted = rain_map * RAIN_BOOST * TIME_ACCELERATION
    runoff = compute_runoff(rain_boosted, INFILTRATION_FRACTION)
    
    flow_dir, receivers, slopes = compute_flow_direction_d8(elevation, pixel_scale_m)
    Q = compute_discharge(elevation, flow_dir, receivers, runoff, pixel_scale_m)
    
    top_idx, top_name = compute_top_layer_map(thickness, layer_order)
    erodibility = get_erodibility_grid(top_name, ERODIBILITY_MAP)
    
    elevation_after_erosion, sediment_out = compute_erosion_pass_a(
        elevation, Q, slopes, flow_dir, erodibility,
        BASE_K, FLAT_K, MAX_ERODE_PER_STEP, SLOPE_THRESHOLD,
        HALF_LOSS_FRACTION, M_DISCHARGE, N_SLOPE, dt
    )
    
    elevation_after_transport = compute_sediment_transport_pass_b(
        elevation_after_erosion, Q, slopes, flow_dir, receivers,
        sediment_out, CAPACITY_K, CAPACITY_M, CAPACITY_N, pixel_scale_m
    )
    
    if apply_diffusion:
        elevation_final = apply_hillslope_diffusion_8neighbor(
            elevation_after_transport, DIFFUSION_K, pixel_scale_m, dt
        )
    else:
        elevation_final = elevation_after_transport
    
    dz_total = elevation_final - elevation
    thickness_new = {k: v.copy() for k, v in thickness.items()}
    
    for i in range(elevation.shape[0]):
        for j in range(elevation.shape[1]):
            dz = dz_total[i, j]
            
            if dz < 0:
                remaining_erosion = -dz
                for layer in layer_order:
                    if remaining_erosion <= 0:
                        break
                    available = thickness_new[layer][i, j]
                    if available > 1e-6:
                        removed = min(available, remaining_erosion)
                        thickness_new[layer][i, j] -= removed
                        remaining_erosion -= removed
            
            elif dz > 0:
                if "Alluvium" in thickness_new:
                    thickness_new["Alluvium"][i, j] += dz
                elif "Topsoil" in thickness_new:
                    thickness_new["Topsoil"][i, j] += dz
                else:
                    thickness_new[layer_order[0]][i, j] += dz
    
    diagnostics = {
        "Q": Q,
        "flow_dir": flow_dir,
        "receivers": receivers,
        "slopes": slopes,
        "dz_total": dz_total,
        "runoff": runoff,
        "top_layer": top_name,
        "erodibility": erodibility
    }
    
    return elevation_final, thickness_new, diagnostics


def run_erosion_simulation(elevation_initial, thickness_initial, layer_order,
                           rain_timeseries, pixel_scale_m, dt=1.0,
                           num_timesteps=100, save_interval=10,
                           apply_diffusion=True, verbose=True):
    """Run full erosion simulation."""
    elevation = elevation_initial.copy()
    thickness = {k: v.copy() for k, v in thickness_initial.items()}
    
    history = {
        'elevation': [elevation.copy()],
        'years': [0]
    }
    
    for t in range(num_timesteps):
        rain_map = rain_timeseries[t % len(rain_timeseries)]
        
        elevation, thickness, diag = run_erosion_timestep(
            elevation, thickness, layer_order, rain_map, pixel_scale_m,
            dt, apply_diffusion, verbose=False
        )
        
        if (t + 1) % save_interval == 0:
            history['elevation'].append(elevation.copy())
            history['years'].append(t + 1)
            if verbose:
                print(f"  Year {t+1}/{num_timesteps} complete")
    
    return {
        'elevation_final': elevation,
        'thickness_final': thickness,
        'elevation_initial': elevation_initial,
        'history': history
    }


# ==============================================================================
# GENERATE INITIAL TERRAIN AND WEATHER
# ==============================================================================

print("\n" + "="*80)
print("GENERATING TERRAIN WITH REALISTIC LAYERS")
print("="*80)

N = 256
pixel_scale_m = 20.0
elev_range_m = 500.0
num_weather_years = 100
wind_dir_deg = 90.0
mean_annual_rain_m = 1.0

print(f"\nConfiguration:")
print(f"  Grid: {N}×{N}, Pixel: {pixel_scale_m}m, Elevation: {elev_range_m}m")
print(f"  Weather: {num_weather_years} years, Wind: {wind_dir_deg}° (EAST)")
print(f"  Classical randomness (no quantum)")

print("\nGenerating terrain...")
start_time = time.time()

z_norm, rng = classical_topography(N=N, beta=3.0, warp_amp=0.10, ridged_alpha=0.15, random_seed=None)

GLOBAL_STRATA = generate_stratigraphy_with_layers(z_norm, rng, pixel_scale_m, elev_range_m)

print(f"✓ Terrain generated in {time.time() - start_time:.1f} s")
print(f"  Elevation: {GLOBAL_STRATA['surface_elev'].min():.1f} - {GLOBAL_STRATA['surface_elev'].max():.1f} m")
print(f"  Layers: {GLOBAL_STRATA['layer_order']}")

start_time = time.time()

GLOBAL_WEATHER_DATA = run_weather_simulation(
    surface_elev=GLOBAL_STRATA['surface_elev'],
    pixel_scale_m=pixel_scale_m,
    num_years=num_weather_years,
    wind_dir_deg=wind_dir_deg,
    mean_annual_rain_m=mean_annual_rain_m,
    random_seed=None
)

print(f"✓ Weather generated in {time.time() - start_time:.1f} s")

GLOBAL_RAIN_TIMESERIES = np.array(GLOBAL_WEATHER_DATA['annual_rain_maps'], dtype=np.float32)

print("\n" + "="*80)
print("✅ DATA READY")
print("="*80)

# ==============================================================================
# RUN EROSION WITH EPOCH SNAPSHOTS
# ==============================================================================

print("\n" + "="*80)
print("RUNNING EROSION SIMULATION")
print("="*80)

elevation_initial = GLOBAL_STRATA['surface_elev'].copy()
thickness_initial = {k: v.copy() for k, v in GLOBAL_STRATA['thickness'].items()}
layer_order = GLOBAL_STRATA['layer_order'].copy()
pixel_scale_m = GLOBAL_STRATA['pixel_scale_m']
rain_timeseries = GLOBAL_RAIN_TIMESERIES.copy()

num_epochs = 5
years_per_epoch = 20
total_years = num_epochs * years_per_epoch

print(f"\nSimulation plan:")
print(f"  Total years: {total_years}")
print(f"  Epochs: {num_epochs}")
print(f"  Years per epoch: {years_per_epoch}")
print(f"  Real-world equivalent: {total_years * TIME_ACCELERATION:.0f} years")

epoch_elevations = []
epoch_layers = []
epoch_years = []

elevation = elevation_initial.copy()
thickness = {k: v.copy() for k, v in thickness_initial.items()}

epoch_elevations.append(elevation.copy())
top_idx, top_name = compute_top_layer_map(thickness, layer_order)
epoch_layers.append(top_name.copy())
epoch_years.append(0)

print(f"\nEpoch 0: Initial state")

for epoch in range(1, num_epochs + 1):
    print(f"\nEpoch {epoch}/{num_epochs}: Years {(epoch-1)*years_per_epoch} → {epoch*years_per_epoch}")
    
    start_time = time.time()
    
    results = run_erosion_simulation(
        elevation_initial=elevation,
        thickness_initial=thickness,
        layer_order=layer_order,
        rain_timeseries=rain_timeseries[(epoch-1)*years_per_epoch:epoch*years_per_epoch],
        pixel_scale_m=pixel_scale_m,
        dt=1.0,
        num_timesteps=years_per_epoch,
        save_interval=years_per_epoch,
        apply_diffusion=True,
        verbose=True
    )
    
    elapsed = time.time() - start_time
    
    elevation = results['elevation_final']
    thickness = results['thickness_final']
    
    epoch_elevations.append(elevation.copy())
    top_idx, top_name = compute_top_layer_map(thickness, layer_order)
    epoch_layers.append(top_name.copy())
    epoch_years.append(epoch * years_per_epoch)
    
    print(f"✓ Epoch {epoch} complete in {elapsed:.1f} s")

print("\n✅ EROSION SIMULATION COMPLETE")

# ==============================================================================
# WATER FLOW SIMULATION
# ==============================================================================

print("\n" + "="*80)
print("SIMULATING WATER FLOW")
print("="*80)

NEIGHBOR_OFFSETS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

def simulate_water_flow(height, rain_amount=1.0, steps=200, flow_factor=0.5):
    """Simulate water flowing downhill."""
    height = height.astype(np.float64)
    rows, cols = height.shape
    
    water = np.full_like(height, rain_amount, dtype=np.float64)
    
    print(f"  Starting water flow simulation...")
    print(f"    Initial rain: {rain_amount:.3f} m per cell")
    print(f"    Flow iterations: {steps}")
    
    for step in range(steps):
        surface = height + water
        dwater = np.zeros_like(water)
        
        for i in range(rows):
            for j in range(cols):
                current_surface = surface[i, j]
                lowest_surface = current_surface
                lowest_pos = None
                
                for di, dj in NEIGHBOR_OFFSETS:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        ns = surface[ni, nj]
                        if ns < lowest_surface:
                            lowest_surface = ns
                            lowest_pos = (ni, nj)
                
                if lowest_pos is None:
                    continue
                
                drop = current_surface - lowest_surface
                if drop <= 0:
                    continue
                
                outflow = min(water[i, j], drop * flow_factor)
                if outflow > 0:
                    dwater[i, j] -= outflow
                    dwater[lowest_pos] += outflow
        
        water += dwater
        
        if (step + 1) % 50 == 0:
            print(f"    Iteration {step+1}/{steps}: max water depth = {water.max():.3f} m")
    
    print(f"  ✓ Water flow complete")
    print(f"    Final max depth: {water.max():.3f} m")
    
    return water


def classify_water_cells(height, water, min_depth=0.01):
    """Classify water cells into lakes vs rivers."""
    height = height.astype(np.float64)
    surface = height + water
    rows, cols = surface.shape
    
    lakes = np.zeros_like(surface, dtype=bool)
    rivers = np.zeros_like(surface, dtype=bool)
    
    for i in range(rows):
        for j in range(cols):
            if water[i, j] < min_depth:
                continue
            
            current_surface = surface[i, j]
            has_downhill = False
            
            for di, dj in NEIGHBOR_OFFSETS:
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    if surface[ni, nj] < current_surface - 1e-9:
                        has_downhill = True
                        break
            
            if has_downhill:
                rivers[i, j] = True
            else:
                lakes[i, j] = True
    
    return lakes, rivers


elevation_final = epoch_elevations[-1].copy()

water_depth = simulate_water_flow(
    height=elevation_final,
    rain_amount=1.0,
    steps=300,
    flow_factor=0.4
)

lakes_mask, rivers_mask = classify_water_cells(
    height=elevation_final,
    water=water_depth,
    min_depth=0.02
)

num_river_cells = np.sum(rivers_mask)
num_lake_cells = np.sum(lakes_mask)
ny, nx = elevation_final.shape

print(f"\n✓ Water classification complete:")
print(f"  River cells: {num_river_cells} ({100*num_river_cells/(ny*nx):.2f}%)")
print(f"  Lake cells: {num_lake_cells} ({100*num_lake_cells/(ny*nx):.2f}%)")

print("\n✅ WATER FLOW SIMULATION COMPLETE")

# ==============================================================================
# VISUALIZATION
# ==============================================================================

print("\nGenerating visualizations...")

# Plot 1: Epoch evolution
fig, axes = plt.subplots(3, num_epochs + 1, figsize=(4*(num_epochs+1), 12))

all_elevs = np.concatenate([e.flatten() for e in epoch_elevations])
vmin, vmax = np.percentile(all_elevs, [1, 99])

initial_elev = epoch_elevations[0]

# Row 1: Elevation maps
for i, (elev, year) in enumerate(zip(epoch_elevations, epoch_years)):
    ax = axes[0, i]
    im = ax.imshow(elev, cmap='terrain', vmin=vmin, vmax=vmax, origin='lower')
    ax.set_title(f"Year {year}\n({year * TIME_ACCELERATION:.0f} real years)")
    ax.axis('off')
    if i == num_epochs:
        plt.colorbar(im, ax=ax, label='Elevation (m)', fraction=0.046)

axes[0, 0].set_ylabel("ELEVATION", fontsize=12, fontweight='bold')

# Row 2: Surface material
layer_colors_map = {
    'Topsoil': 0, 'Subsoil': 1, 'Colluvium': 2,
    'Saprolite': 3, 'WeatheredBR': 4, 'Basement': 5
}

for i, (layer_map, year) in enumerate(zip(epoch_layers, epoch_years)):
    ax = axes[1, i]
    
    ny_plot, nx_plot = layer_map.shape
    color_map = np.zeros((ny_plot, nx_plot))
    for ii in range(ny_plot):
        for jj in range(nx_plot):
            color_map[ii, jj] = layer_colors_map.get(layer_map[ii, jj], 5)
    
    im = ax.imshow(color_map, cmap='tab10', vmin=0, vmax=5, origin='lower')
    ax.set_title(f"Year {year}")
    ax.axis('off')
    
    if i == num_epochs:
        cbar = plt.colorbar(im, ax=ax, ticks=[0,1,2,3,4,5], fraction=0.046)
        cbar.set_ticklabels(['Topsoil', 'Subsoil', 'Colluvium', 'Saprolite', 'W.BR', 'Basement'])

axes[1, 0].set_ylabel("SURFACE MATERIAL", fontsize=12, fontweight='bold')

# Row 3: Erosion depth
for i, (elev, year) in enumerate(zip(epoch_elevations, epoch_years)):
    ax = axes[2, i]
    
    erosion_depth = initial_elev - elev
    
    im = ax.imshow(erosion_depth, cmap='hot_r', vmin=0, vmax=None, origin='lower')
    ax.set_title(f"Year {year}")
    ax.axis('off')
    
    if i == num_epochs:
        plt.colorbar(im, ax=ax, label='Erosion (m)', fraction=0.046)

axes[2, 0].set_ylabel("EROSION DEPTH", fontsize=12, fontweight='bold')

plt.suptitle(f"Erosion Evolution Over {total_years} Years ({total_years * TIME_ACCELERATION:.0f} Real Years)", 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()

# Plot 2: Water features
fig = plt.figure(figsize=(20, 12))

ax1 = plt.subplot(2, 3, 1)
ax2 = plt.subplot(2, 3, 2)
ax3 = plt.subplot(2, 3, 3)
ax4 = plt.subplot(2, 3, 4)
ax5 = plt.subplot(2, 3, 5)
ax6 = plt.subplot(2, 3, 6)

# Plot 1: Final elevation
im1 = ax1.imshow(elevation_final, cmap='terrain', origin='lower')
ax1.set_title("Final Terrain Elevation", fontsize=14, fontweight='bold')
ax1.axis('off')
plt.colorbar(im1, ax=ax1, label='Elevation (m)', fraction=0.046)

# Plot 2: Lakes Only
water_to_show = np.where(water_depth > 0.02, water_depth, np.nan)
im2 = ax2.imshow(water_to_show, cmap='Blues', origin='lower')
ax2.set_title("Lakes Only", fontsize=14, fontweight='bold')
ax2.axis('off')
plt.colorbar(im2, ax=ax2, label='Depth (m)', fraction=0.046)

# Plot 3: Rivers only
rivers_depth = np.where(rivers_mask, water_depth, np.nan)
im3 = ax3.imshow(rivers_depth, cmap='Blues', origin='lower')
ax3.set_title("Rivers Only", fontsize=14, fontweight='bold')
ax3.axis('off')
plt.colorbar(im3, ax=ax3, label='Depth (m)', fraction=0.046)

# Plot 4: Lakes only (classified)
lakes_depth = np.where(lakes_mask, water_depth, np.nan)
im4 = ax4.imshow(lakes_depth, cmap='Blues', origin='lower')
ax4.set_title("Lakes Only (Classified)", fontsize=14, fontweight='bold')
ax4.axis('off')
plt.colorbar(im4, ax=ax4, label='Depth (m)', fraction=0.046)

# Plot 5: Terrain + Rivers + Lakes
terrain_rgb = plt.cm.terrain((elevation_final - elevation_final.min()) / 
                             (elevation_final.max() - elevation_final.min() + 1e-9))[:, :, :3]

water_overlay = np.zeros((ny, nx, 4))
water_overlay[:, :, 3] = 0.0

water_overlay[rivers_mask, 0] = 0.0
water_overlay[rivers_mask, 1] = 0.4
water_overlay[rivers_mask, 2] = 1.0
water_overlay[rivers_mask, 3] = 0.7

water_overlay[lakes_mask, 0] = 0.0
water_overlay[lakes_mask, 1] = 0.7
water_overlay[lakes_mask, 2] = 1.0
water_overlay[lakes_mask, 3] = 0.85

ax5.imshow(terrain_rgb, origin='lower')
ax5.imshow(water_overlay, origin='lower')
ax5.set_title("🌊 TERRAIN + RIVERS + LAKES 🌊", fontsize=14, fontweight='bold', color='blue')
ax5.axis('off')

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='blue', alpha=0.7, label=f'Rivers ({num_river_cells} cells)'),
    Patch(facecolor='cyan', alpha=0.85, label=f'Lakes ({num_lake_cells} cells)')
]
ax5.legend(handles=legend_elements, loc='upper right', fontsize=10)

# Plot 6: Erosion + Water
erosion_final = initial_elev - elevation_final
erosion_rgb = plt.cm.hot_r(erosion_final / (erosion_final.max() + 1e-9))[:, :, :3]

ax6.imshow(erosion_rgb, origin='lower')
ax6.imshow(water_overlay, origin='lower')
ax6.set_title("Erosion Depth + Water", fontsize=14, fontweight='bold')
ax6.axis('off')

plt.suptitle("CLASSICAL EROSION SYSTEM - WATER FLOW SIMULATION", 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

print("✓ Visualizations complete")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\nSimulation:")
print(f"  Duration: {total_years} sim years = {total_years * TIME_ACCELERATION:.0f} real years")
print(f"  Epochs: {num_epochs}")

print(f"\nErosion:")
final_erosion = initial_elev - epoch_elevations[-1]
print(f"  Mean: {final_erosion.mean():.2f} m")
print(f"  Max: {final_erosion.max():.2f} m")

print(f"\nWater Flow:")
print(f"  Max water depth: {water_depth.max():.3f} m")
print(f"  River cells: {num_river_cells} ({100*num_river_cells/(ny*nx):.2f}%)")
print(f"  Lake cells: {num_lake_cells} ({100*num_lake_cells/(ny*nx):.2f}%)")

print("\n" + "="*80)
print("✅ COMPLETE: CLASSICAL EROSION SYSTEM")
print("="*80)
print("\nFeatures:")
print("  ✓ Fully classical (no quantum/Qiskit)")
print("  ✓ 6 realistic geological layers")
print("  ✓ Wind-rain physics (EAST wind)")
print("  ✓ Non-uniform erosion")
print("  ✓ Proper water flow simulation")
print("  ✓ Rivers and lakes detection")
print("  ✓ Epoch-by-epoch visualization")
print("="*80 + "\n")
