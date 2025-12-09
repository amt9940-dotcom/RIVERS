"""
COMPLETE TERRAIN + WEATHER SYSTEM WITH REALISTIC LAYERS

Features:
- Wind: EAST (90°) with proper barrier/channel physics
- Layers: Topsoil, Subsoil, Colluvium, Saprolite, Weathered Bedrock, Basement
- Different erodibility for each layer → non-uniform erosion
- Quantum random rain within storms
- Single terrain map for erosion simulator
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List
import time

# ==============================================================================
# QUANTUM RNG
# ==============================================================================

try:
    import qiskit
    try:
        import qiskit_aer
        HAVE_QISKIT = True
    except Exception:
        HAVE_QISKIT = False
except Exception:
    HAVE_QISKIT = False

print(f"Quantum RNG available: {HAVE_QISKIT}")


def qrng_uint32(n, nbits=32):
    if not HAVE_QISKIT:
        return np.random.default_rng().integers(0, 2**32, size=n, dtype=np.uint32)
    
    from qiskit import QuantumCircuit
    try:
        from qiskit_aer import Aer
    except Exception:
        try:
            from qiskit import Aer
        except Exception:
            return np.random.default_rng().integers(0, 2**32, size=n, dtype=np.uint32)
    
    qc = QuantumCircuit(nbits, nbits)
    qc.h(range(nbits))
    qc.measure(range(nbits), range(nbits))
    backend = Aer.get_backend("qasm_simulator")
    
    import os
    seed_sim = int.from_bytes(os.urandom(4), "little")
    job = backend.run(qc, shots=n, memory=True, seed_simulator=seed_sim)
    mem = job.result().get_memory(qc)
    
    return np.array([np.uint32(int(bits[::-1], 2)) for bits in mem], dtype=np.uint32)


def rng_from_qrng(n_seeds=4, random_seed=None):
    if random_seed is not None:
        return np.random.default_rng(int(random_seed))
    
    import os, time, hashlib
    seeds = qrng_uint32(n_seeds).tobytes()
    mix = seeds + os.urandom(16) + int(time.time_ns()).to_bytes(8, "little")
    h = hashlib.blake2b(mix, digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(h, "little"))


def quantum_uniforms(n, backend=None, seed_sim=None):
    if not HAVE_QISKIT:
        return np.random.default_rng().uniform(0, 1, size=n)
    
    nbits = 16
    bits = qrng_uint32(n, nbits=nbits)
    return bits.astype(np.float64) / (2.0**nbits)


# ==============================================================================
# TERRAIN GENERATION
# ==============================================================================

def fractional_surface(N, beta=3.0, rng=None):
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
    N = z.shape[0]
    u = fractional_surface(N, beta=beta, rng=rng)*2 - 1
    v = fractional_surface(N, beta=beta, rng=rng)*2 - 1
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    Xw = (ii + amp*N*u) % N
    Yw = (jj + amp*N*v) % N
    return bilinear_sample(z, Xw, Yw)


def ridged_mix(z, alpha=0.15):
    ridged = 1.0 - np.abs(2.0*z - 1.0)
    out = (1-alpha)*z + alpha*ridged
    lo, hi = np.percentile(out, [2, 98])
    return np.clip((out - lo)/(hi - lo + 1e-12), 0, 1)


def quantum_seeded_topography(N=256, beta=3.0, warp_amp=0.10, 
                               ridged_alpha=0.15, random_seed=None):
    rng = rng_from_qrng(n_seeds=4, random_seed=random_seed)
    
    base_low = fractional_surface(N, beta=beta, rng=rng)
    base_high = fractional_surface(N, beta=beta-0.4, rng=rng)
    z = 0.65*base_low + 0.35*base_high
    
    z = domain_warp(z, rng=rng, amp=warp_amp, beta=beta)
    z = ridged_mix(z, alpha=ridged_alpha)
    
    return z, rng


# ==============================================================================
# REALISTIC LAYER GENERATION (From original Project.ipynb)
# ==============================================================================

def generate_stratigraphy_with_layers(z_norm, rng, pixel_scale_m=20.0, elev_range_m=500.0):
    """
    Generate realistic stratigraphy with varied surface materials.
    
    Layers (top to bottom):
    1. Topsoil - thin, erodible, on gentle slopes
    2. Subsoil - thicker, moderately erodible
    3. Colluvium - gravity-deposited, in valleys
    4. Saprolite - weathered bedrock, on interfluves
    5. Weathered Bedrock - partially weathered, resistant
    6. Basement - unweathered, very resistant
    
    Each layer has different thickness based on:
    - Elevation
    - Slope
    - Topographic position
    """
    ny, nx = z_norm.shape
    surface_elev = z_norm * elev_range_m
    
    # Define layer order
    layer_order = ['Topsoil', 'Subsoil', 'Colluvium', 'Saprolite', 'WeatheredBR', 'Basement']
    
    thickness = {}
    
    # Compute slope
    grad_y, grad_x = np.gradient(z_norm, pixel_scale_m / elev_range_m)
    slope_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Compute curvature (for detecting valleys vs ridges)
    grad2_y, grad2_x = np.gradient(slope_mag)
    curvature = grad2_y + grad2_x  # Negative = valleys, Positive = ridges
    
    # ==== LAYER 1: TOPSOIL ====
    # Thicker on gentle slopes, thinner on steep slopes
    gentle_mask = slope_mag < 0.1
    steep_mask = slope_mag > 0.3
    
    topsoil = np.ones((ny, nx)) * 1.5  # Base thickness
    topsoil[gentle_mask] += 1.0  # Extra on gentle slopes
    topsoil[steep_mask] -= 0.8  # Thin on steep slopes
    topsoil += rng.uniform(-0.3, 0.3, size=(ny, nx))  # Variability
    thickness['Topsoil'] = np.maximum(0.2, topsoil)
    
    # ==== LAYER 2: SUBSOIL ====
    # Thicker in mid-elevation areas
    mid_elev_mask = (z_norm > 0.3) & (z_norm < 0.7)
    
    subsoil = np.ones((ny, nx)) * 3.0
    subsoil[mid_elev_mask] += 2.0
    subsoil[gentle_mask] += 1.0
    subsoil += rng.uniform(-0.5, 0.5, size=(ny, nx))
    thickness['Subsoil'] = np.maximum(0.5, subsoil)
    
    # ==== LAYER 3: COLLUVIUM ====
    # Gravity-driven deposits in valleys and at slope bases
    # Thick where curvature is negative (concave = valleys)
    valley_mask = curvature < -0.0001
    low_elev_mask = z_norm < 0.4
    
    colluvium = np.zeros((ny, nx))
    colluvium[valley_mask] = 5.0 + rng.uniform(0, 5, size=np.sum(valley_mask))
    colluvium[valley_mask & low_elev_mask] += 3.0  # Extra thick in low valleys
    colluvium[steep_mask & low_elev_mask] += 2.0  # Accumulated at slope base
    thickness['Colluvium'] = np.maximum(0.0, colluvium)
    
    # ==== LAYER 4: SAPROLITE ====
    # Weathered bedrock, thick on stable interfluves (ridges)
    # Minimal in valleys where it's been stripped
    ridge_mask = curvature > 0.0001
    high_elev_mask = z_norm > 0.5
    
    saprolite = np.ones((ny, nx)) * 8.0
    saprolite[ridge_mask & gentle_mask] += 10.0  # Thick on stable ridges
    saprolite[high_elev_mask] += 5.0  # Thicker at high elevation
    saprolite[valley_mask] *= 0.3  # Thin in valleys (stripped)
    saprolite[steep_mask] *= 0.5  # Thinner on steep slopes
    saprolite += rng.uniform(-2, 2, size=(ny, nx))
    thickness['Saprolite'] = np.maximum(0.5, saprolite)
    
    # ==== LAYER 5: WEATHERED BEDROCK ====
    # Partially weathered, patchy distribution
    # Use fractal pattern for realistic patchiness
    patch_pattern = fractional_surface(ny, beta=3.0, rng=rng)
    
    weathered_br = 3.0 + 4.0 * patch_pattern
    weathered_br[high_elev_mask] += 2.0  # Thicker at high elevation
    weathered_br[valley_mask] *= 0.5  # Thinner in valleys
    thickness['WeatheredBR'] = np.maximum(0.5, weathered_br)
    
    # ==== LAYER 6: BASEMENT ====
    # Unweathered bedrock, infinite thickness
    thickness['Basement'] = np.ones((ny, nx)) * 1000.0
    
    return {
        'surface_elev': surface_elev,
        'thickness': thickness,
        'layer_order': layer_order,
        'pixel_scale_m': pixel_scale_m,
        'slope_mag': slope_mag,
        'curvature': curvature
    }


def compute_top_layer_map(thickness: Dict[str, np.ndarray], 
                          layer_order: List[str]) -> Tuple[np.ndarray, np.ndarray]:
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
# WIND-TOPOGRAPHY INTERACTION (Same as before)
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


def generate_storm_with_quantum_rain(surface_elev, wind_features, storm_center_ij,
                                      storm_radius_cells, base_intensity_m_per_hour,
                                      duration_hours, pixel_scale_m, rng):
    """Generate storm with quantum random rain and wind physics."""
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
        try:
            quantum_mults = quantum_uniforms(n_storm_cells)
            quantum_mults = np.exp((quantum_mults - 0.5) * 1.5)
        except:
            quantum_mults = np.exp((rng.uniform(0, 1, size=n_storm_cells) - 0.5) * 1.5)
        
        base_storm_quantum = base_storm.copy()
        base_storm_quantum[storm_cells] *= quantum_mults
        base_storm = base_storm_quantum
    
    base_rain = base_intensity_m_per_hour * duration_hours * base_storm
    rain = apply_wind_rain_physics(base_rain, wind_features, k_windward=0.8, k_lee=0.6, k_channel=0.5)
    
    return rain


def run_weather_simulation(surface_elev, pixel_scale_m, num_years=10, wind_dir_deg=90.0,
                           mean_annual_rain_m=1.0, random_seed=None):
    """Run multi-year weather simulation."""
    print(f"\nGenerating {num_years} years of weather...")
    print(f"  Wind direction: {wind_dir_deg}° (EAST → to the right)")
    
    ny, nx = surface_elev.shape
    rng = rng_from_qrng(random_seed=random_seed)
    
    wind_features = classify_wind_features(surface_elev, pixel_scale_m, wind_dir_deg)
    
    barrier_cells = np.sum(wind_features['barrier_score'] > 0.5)
    channel_cells = np.sum(wind_features['channel_score'] > 0.5)
    print(f"  Wind barriers: {barrier_cells} cells")
    print(f"  Wind channels: {channel_cells} cells")
    
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
            
            storm_rain = generate_storm_with_quantum_rain(
                surface_elev, wind_features,
                (ci, cj), radius, intensity, duration,
                pixel_scale_m, rng
            )
            
            year_rain += storm_rain
        
        scale_factor = mean_annual_rain_m / (year_rain.mean() + 1e-9)
        year_rain *= scale_factor
        
        annual_rain_maps.append(year_rain)
        
        if (year + 1) % max(1, num_years // 5) == 0:
            print(f"  Year {year+1}/{num_years}: {year_rain.mean():.3f} m/yr (range: {year_rain.min():.3f} - {year_rain.max():.3f})")
    
    total_rain = np.sum(annual_rain_maps, axis=0)
    
    print(f"✓ Weather simulation complete")
    print(f"  Total rainfall: {total_rain.mean():.2f} m over {num_years} years")
    
    return {
        'annual_rain_maps': annual_rain_maps,
        'wind_features': wind_features,
        'total_rain': total_rain,
        'num_years': num_years
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

print("\nGenerating terrain...")
start_time = time.time()

z_norm, rng = quantum_seeded_topography(N=N, beta=3.0, warp_amp=0.10, ridged_alpha=0.15, random_seed=None)

# Use realistic layer generation
GLOBAL_STRATA = generate_stratigraphy_with_layers(z_norm, rng, pixel_scale_m, elev_range_m)

print(f"✓ Terrain generated in {time.time() - start_time:.1f} s")
print(f"  Elevation: {GLOBAL_STRATA['surface_elev'].min():.1f} - {GLOBAL_STRATA['surface_elev'].max():.1f} m")
print(f"  Layers: {GLOBAL_STRATA['layer_order']}")

# Show layer thickness statistics
print(f"\n  Layer thickness summary:")
for layer in GLOBAL_STRATA['layer_order']:
    th = GLOBAL_STRATA['thickness'][layer]
    print(f"    {layer:15s}: {th.min():.2f} - {th.max():.2f} m (mean: {th.mean():.2f} m)")

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
print("✅ DATA READY - VARIED SURFACE MATERIALS FOR NON-UNIFORM EROSION")
print("="*80)
print("\nGLOBAL VARIABLES:")
print("  GLOBAL_STRATA - terrain with 6 different layers")
print("  GLOBAL_WEATHER_DATA - wind-affected rain")
print("  GLOBAL_RAIN_TIMESERIES - annual rain maps")
print("\n✓ Different materials → Different erodibility → Non-uniform erosion")
print("="*80 + "\n")

# Visualize layers
top_idx, top_name = compute_top_layer_map(GLOBAL_STRATA['thickness'], GLOBAL_STRATA['layer_order'])

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

ax = axes[0, 0]
im = ax.imshow(GLOBAL_STRATA['surface_elev'], cmap='terrain', origin='lower')
ax.set_title("Terrain Elevation")
plt.colorbar(im, ax=ax, label="Elevation (m)")

ax = axes[0, 1]
# Map layer names to colors
layer_colors = {
    'Topsoil': 0, 'Subsoil': 1, 'Colluvium': 2,
    'Saprolite': 3, 'WeatheredBR': 4, 'Basement': 5
}
color_map = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        color_map[i, j] = layer_colors.get(top_name[i, j], 5)
im = ax.imshow(color_map, cmap='tab10', origin='lower')
ax.set_title("Surface Material (Varied Erodibility)")
cbar = plt.colorbar(im, ax=ax, ticks=[0,1,2,3,4,5])
cbar.set_ticklabels(['Topsoil', 'Subsoil', 'Colluvium', 'Saprolite', 'W.Bedrock', 'Basement'])

ax = axes[0, 2]
im = ax.imshow(GLOBAL_WEATHER_DATA['total_rain'], cmap='YlGnBu', origin='lower')
ax.set_title(f"Total Rain ({num_weather_years} yrs)")
plt.colorbar(im, ax=ax, label="Rain (m)")

ax = axes[1, 0]
im = ax.imshow(GLOBAL_STRATA['thickness']['Topsoil'], cmap='YlOrBr', origin='lower')
ax.set_title("Topsoil Thickness (Most Erodible)")
plt.colorbar(im, ax=ax, label="Thickness (m)")

ax = axes[1, 1]
im = ax.imshow(GLOBAL_STRATA['thickness']['Colluvium'], cmap='Greens', origin='lower')
ax.set_title("Colluvium Thickness (In Valleys)")
plt.colorbar(im, ax=ax, label="Thickness (m)")

ax = axes[1, 2]
im = ax.imshow(GLOBAL_STRATA['thickness']['Saprolite'], cmap='Purples', origin='lower')
ax.set_title("Saprolite Thickness (On Ridges)")
plt.colorbar(im, ax=ax, label="Thickness (m)")

plt.suptitle("Realistic Layer Distribution → Non-Uniform Erosion", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("✓ Visualization complete - Notice varied surface materials!")
