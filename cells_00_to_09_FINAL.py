"""
STREAMLINED TERRAIN + WEATHER SYSTEM - FINAL VERSION

Wind Physics:
- Wind goes EAST (to the right)
- Barriers (ridges): Wet windward side, dry leeward (rain shadow)
- Channels (valleys): Rain funneled along valleys, concentrated at junctions

This generates:
1. ONE terrain map (stored in global variables)
2. Weather simulation with proper wind-rain physics
3. All data ready for erosion simulator
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List
import time

# ==============================================================================
# QUANTUM RNG (Qiskit with fallback)
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
    """Generate n uint32 values using quantum RNG (or classical fallback)."""
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
    """Create NumPy RNG seeded from quantum source (or specified seed)."""
    if random_seed is not None:
        return np.random.default_rng(int(random_seed))
    
    import os, time, hashlib
    seeds = qrng_uint32(n_seeds).tobytes()
    mix = seeds + os.urandom(16) + int(time.time_ns()).to_bytes(8, "little")
    h = hashlib.blake2b(mix, digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(h, "little"))


def quantum_uniforms(n, backend=None, seed_sim=None):
    """Generate n uniform random numbers in [0, 1) using quantum RNG."""
    if not HAVE_QISKIT:
        return np.random.default_rng().uniform(0, 1, size=n)
    
    nbits = 16
    bits = qrng_uint32(n, nbits=nbits)
    return bits.astype(np.float64) / (2.0**nbits)


# ==============================================================================
# TERRAIN GENERATION (Project33 style)
# ==============================================================================

def fractional_surface(N, beta=3.0, rng=None):
    """Generate fractal terrain using power-law spectrum."""
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
    """Bilinear interpolation for domain warping."""
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
    """Apply domain warping for realistic terrain features."""
    N = z.shape[0]
    u = fractional_surface(N, beta=beta, rng=rng)*2 - 1
    v = fractional_surface(N, beta=beta, rng=rng)*2 - 1
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    Xw = (ii + amp*N*u) % N
    Yw = (jj + amp*N*v) % N
    return bilinear_sample(z, Xw, Yw)


def ridged_mix(z, alpha=0.15):
    """Add ridge/valley features."""
    ridged = 1.0 - np.abs(2.0*z - 1.0)
    out = (1-alpha)*z + alpha*ridged
    lo, hi = np.percentile(out, [2, 98])
    return np.clip((out - lo)/(hi - lo + 1e-12), 0, 1)


def quantum_seeded_topography(N=256, beta=3.0, warp_amp=0.10, 
                               ridged_alpha=0.15, random_seed=None):
    """Generate terrain using quantum-seeded RNG."""
    rng = rng_from_qrng(n_seeds=4, random_seed=random_seed)
    
    base_low = fractional_surface(N, beta=beta, rng=rng)
    base_high = fractional_surface(N, beta=beta-0.4, rng=rng)
    z = 0.65*base_low + 0.35*base_high
    
    z = domain_warp(z, rng=rng, amp=warp_amp, beta=beta)
    z = ridged_mix(z, alpha=ridged_alpha)
    
    return z, rng


# ==============================================================================
# SIMPLIFIED STRATIGRAPHY (4 layers)
# ==============================================================================

def generate_stratigraphy(z_norm, rng, pixel_scale_m=20.0, elev_range_m=500.0):
    """Generate 4-layer stratigraphy."""
    ny, nx = z_norm.shape
    surface_elev = z_norm * elev_range_m
    
    layer_order = ['Topsoil', 'Subsoil', 'Saprolite', 'Basement']
    
    thickness = {}
    thickness['Topsoil'] = 2.0 + 3.0 * (1 - z_norm) + rng.uniform(-0.5, 0.5, size=(ny, nx))
    thickness['Topsoil'] = np.maximum(0.5, thickness['Topsoil'])
    thickness['Subsoil'] = 5.0 + 5.0 * rng.uniform(0, 1, size=(ny, nx))
    thickness['Saprolite'] = 10.0 + 15.0 * rng.uniform(0, 1, size=(ny, nx))
    thickness['Basement'] = np.ones((ny, nx)) * 1000.0
    
    return {
        'surface_elev': surface_elev,
        'thickness': thickness,
        'layer_order': layer_order,
        'pixel_scale_m': pixel_scale_m
    }


def compute_top_layer_map(thickness: Dict[str, np.ndarray], 
                          layer_order: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Determine which layer is exposed at each location."""
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
# WIND-TOPOGRAPHY INTERACTION (CORRECTED PHYSICS)
# ==============================================================================

def classify_wind_features(surface_elev, pixel_scale_m, wind_dir_deg=90.0):
    """
    Classify terrain features for wind interaction.
    
    Wind direction: 90° = EAST (to the right)
    
    Creates:
    - barrier_score: Ridge strength (0-1)
    - channel_score: Valley strength (0-1)
    - slope_vectors: 2D slope direction at each cell
    
    Parameters
    ----------
    surface_elev : np.ndarray (ny, nx)
        Surface elevation [m]
    pixel_scale_m : float
        Grid cell size [m]
    wind_dir_deg : float
        Wind direction [degrees: 0=N, 90=E, 180=S, 270=W]
        Default 90 = EAST wind
    
    Returns
    -------
    wind_features : dict
        - barrier_score: float array [0, 1] (ridge strength)
        - channel_score: float array [0, 1] (valley strength)
        - slope_vectors: tuple (slope_x, slope_y) [m/m]
        - wind_vector: tuple (wind_x, wind_y) [unit vector]
        - wind_dir_deg: float
    """
    ny, nx = surface_elev.shape
    
    # Compute gradients (slope)
    # dy points NORTH (up), dx points EAST (right)
    grad_y, grad_x = np.gradient(surface_elev, pixel_scale_m)
    
    # Wind direction as unit vector
    # 90° = East → (1, 0) in (x, y) coordinates
    wind_rad = np.radians(wind_dir_deg)
    wind_x = np.cos(wind_rad)  # East component
    wind_y = np.sin(wind_rad)  # North component
    wind_vector = (wind_x, wind_y)
    
    # Slope magnitude
    slope_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # BARRIER SCORE: High elevation ridges (peaks)
    # Use second derivatives to find ridges
    grad2_y, grad2_x = np.gradient(slope_mag, pixel_scale_m)
    curvature = np.sqrt(grad2_x**2 + grad2_y**2)
    
    # High curvature + high elevation = barrier
    elev_norm = (surface_elev - surface_elev.min()) / (surface_elev.max() - surface_elev.min() + 1e-9)
    curv_norm = (curvature - curvature.min()) / (curvature.max() - curvature.min() + 1e-9)
    
    # Barrier score: ridges, peaks, high curvature
    barrier_score = 0.5 * curv_norm + 0.5 * elev_norm
    barrier_score = barrier_score ** 2  # Sharpen
    barrier_score = np.clip(barrier_score, 0, 1)
    
    # CHANNEL SCORE: Low elevation valleys
    # Valleys = low elevation + low curvature + converging flow
    channel_score = (1 - elev_norm) * (1 - curv_norm)
    channel_score = channel_score ** 2  # Sharpen
    channel_score = np.clip(channel_score, 0, 1)
    
    # Normalize barrier and channel to be mutually exclusive
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


# ==============================================================================
# STORM RAIN WITH BARRIER/CHANNEL PHYSICS
# ==============================================================================

def apply_wind_rain_physics(base_rain, wind_features, k_windward=0.8, k_lee=0.6, k_channel=0.5):
    """
    Apply wind-terrain physics to rain map.
    
    Physics:
    - Barriers: wet windward, dry leeward (rain shadow)
    - Channels: rain funneled along valleys
    
    Parameters
    ----------
    base_rain : np.ndarray (ny, nx)
        Base storm rain [m]
    wind_features : dict
        From classify_wind_features
    k_windward : float
        Windward boost strength (default 0.8)
    k_lee : float
        Leeward reduction strength (default 0.6)
    k_channel : float
        Channel funneling strength (default 0.5)
    
    Returns
    -------
    rain : np.ndarray (ny, nx)
        Modified rain [m]
    """
    ny, nx = base_rain.shape
    
    # Extract wind data
    slope_x, slope_y = wind_features['slope_vectors']
    wind_x, wind_y = wind_features['wind_vector']
    barrier_score = wind_features['barrier_score']
    channel_score = wind_features['channel_score']
    
    # BARRIER FACTOR: windward vs leeward
    # Compute slope-wind alignment
    # cos_theta = dot(slope, wind) / |slope|
    slope_mag = np.sqrt(slope_x**2 + slope_y**2) + 1e-9
    cos_theta = (slope_x * wind_x + slope_y * wind_y) / slope_mag
    
    # Initialize barrier factor
    barrier_factor = np.ones((ny, nx), dtype=np.float32)
    
    # Windward (cos_theta > 0): slope faces wind → MORE rain
    windward_mask = cos_theta > 0
    barrier_factor[windward_mask] = 1.0 + k_windward * cos_theta[windward_mask] * barrier_score[windward_mask]
    
    # Leeward (cos_theta < 0): rain shadow → LESS rain
    leeward_mask = cos_theta < 0
    barrier_factor[leeward_mask] = 1.0 - k_lee * (-cos_theta[leeward_mask]) * barrier_score[leeward_mask]
    
    # Clamp to reasonable range
    barrier_factor = np.clip(barrier_factor, 0.2, 2.5)
    
    # CHANNEL FACTOR: valleys funnel rain
    channel_factor = 1.0 + k_channel * channel_score
    
    # FINAL RAIN
    rain = base_rain * barrier_factor * channel_factor
    
    return rain


def generate_storm_with_quantum_rain(
    surface_elev,
    wind_features,
    storm_center_ij,
    storm_radius_cells,
    base_intensity_m_per_hour,
    duration_hours,
    pixel_scale_m,
    rng
):
    """
    Generate storm with:
    - Quantum random rain distribution
    - Barrier physics (windward wet, leeward dry)
    - Channel physics (valleys funnel rain)
    """
    ny, nx = surface_elev.shape
    ci, cj = storm_center_ij
    
    # Create base storm shape (Gaussian)
    ii, jj = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    di = ii - ci
    dj = jj - cj
    
    # Periodic boundaries
    di = np.where(di > ny/2, di - ny, di)
    di = np.where(di < -ny/2, di + ny, di)
    dj = np.where(dj > nx/2, dj - nx, dj)
    dj = np.where(dj < -nx/2, dj + nx, dj)
    
    dist = np.sqrt(di**2 + dj**2)
    base_storm = np.exp(-(dist / storm_radius_cells)**2)
    
    # QUANTUM RANDOM RAIN WITHIN STORM
    storm_cells = base_storm > 0.1
    n_storm_cells = np.sum(storm_cells)
    
    if n_storm_cells > 0:
        try:
            quantum_mults = quantum_uniforms(n_storm_cells)
            quantum_mults = np.exp((quantum_mults - 0.5) * 1.5)  # Lognormal
        except:
            quantum_mults = np.exp((rng.uniform(0, 1, size=n_storm_cells) - 0.5) * 1.5)
        
        base_storm_quantum = base_storm.copy()
        base_storm_quantum[storm_cells] *= quantum_mults
        base_storm = base_storm_quantum
    
    # Base rain amount
    base_rain = base_intensity_m_per_hour * duration_hours * base_storm
    
    # APPLY WIND-TERRAIN PHYSICS
    rain = apply_wind_rain_physics(base_rain, wind_features, 
                                   k_windward=0.8, k_lee=0.6, k_channel=0.5)
    
    return rain


# ==============================================================================
# MULTI-YEAR WEATHER SIMULATION
# ==============================================================================

def run_weather_simulation(
    surface_elev,
    pixel_scale_m,
    num_years=10,
    wind_dir_deg=90.0,  # EAST wind (to the right)
    mean_annual_rain_m=1.0,
    random_seed=None
):
    """
    Run multi-year weather simulation with corrected wind physics.
    
    Wind: 90° = EAST (to the right)
    Barriers: wet windward, dry leeward
    Channels: rain funneled along valleys
    """
    print(f"\nGenerating {num_years} years of weather...")
    print(f"  Wind direction: {wind_dir_deg}° (EAST = to the right)")
    
    ny, nx = surface_elev.shape
    rng = rng_from_qrng(random_seed=random_seed)
    
    # Classify wind features
    wind_features = classify_wind_features(surface_elev, pixel_scale_m, wind_dir_deg)
    
    # Count features
    barrier_cells = np.sum(wind_features['barrier_score'] > 0.5)
    channel_cells = np.sum(wind_features['channel_score'] > 0.5)
    print(f"  Wind barriers (ridges): {barrier_cells} cells")
    print(f"  Wind channels (valleys): {channel_cells} cells")
    
    # Generate storms for each year
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
        
        # Scale to target
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
print("GENERATING INITIAL TERRAIN AND WEATHER")
print("="*80)

# Configuration
N = 256
pixel_scale_m = 20.0
elev_range_m = 500.0
num_weather_years = 100
wind_dir_deg = 90.0  # EAST wind (to the right)
mean_annual_rain_m = 1.0

print(f"\nConfiguration:")
print(f"  Grid size: {N}×{N}")
print(f"  Domain size: {N*pixel_scale_m/1000:.2f} × {N*pixel_scale_m/1000:.2f} km")
print(f"  Pixel scale: {pixel_scale_m} m")
print(f"  Elevation range: {elev_range_m} m")
print(f"  Weather years: {num_weather_years}")
print(f"  Wind direction: {wind_dir_deg}° (EAST → to the right)")

# Generate terrain
print("\nGenerating terrain...")
start_time = time.time()

z_norm, rng = quantum_seeded_topography(
    N=N,
    beta=3.0,
    warp_amp=0.10,
    ridged_alpha=0.15,
    random_seed=None
)

GLOBAL_STRATA = generate_stratigraphy(z_norm, rng, pixel_scale_m, elev_range_m)

print(f"✓ Terrain generated in {time.time() - start_time:.1f} s")
print(f"  Elevation range: {GLOBAL_STRATA['surface_elev'].min():.1f} - {GLOBAL_STRATA['surface_elev'].max():.1f} m")
print(f"  Layers: {GLOBAL_STRATA['layer_order']}")

# Generate weather
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
print("✅ INITIAL DATA READY")
print("="*80)
print("\nGLOBAL VARIABLES CREATED:")
print("  GLOBAL_STRATA - terrain and stratigraphy")
print("  GLOBAL_WEATHER_DATA - weather with wind physics")
print("  GLOBAL_RAIN_TIMESERIES - rain time series")
print(f"\n  Wind: EAST (90°) → affects rain distribution")
print(f"  Barriers: Wet windward, dry leeward")
print(f"  Channels: Rain funneled in valleys")
print("="*80 + "\n")

# Visualize
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

ax = axes[0, 0]
im = ax.imshow(GLOBAL_STRATA['surface_elev'], cmap='terrain', origin='lower')
ax.set_title("Initial Terrain")
ax.set_xlabel("X (East →)")
ax.set_ylabel("Y (North ↑)")
plt.colorbar(im, ax=ax, label="Elevation (m)")

ax = axes[0, 1]
im = ax.imshow(GLOBAL_WEATHER_DATA['wind_features']['barrier_score'], cmap='Reds', origin='lower')
ax.set_title("Barrier Score (Ridges)")
ax.set_xlabel("X (East →)")
ax.set_ylabel("Y (North ↑)")
# Add wind arrow
ax.arrow(N*0.1, N*0.9, N*0.15, 0, head_width=N*0.05, head_length=N*0.05, 
         fc='black', ec='black', linewidth=2)
ax.text(N*0.2, N*0.95, 'WIND →', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, label="Score [0-1]")

ax = axes[0, 2]
im = ax.imshow(GLOBAL_WEATHER_DATA['wind_features']['channel_score'], cmap='Blues', origin='lower')
ax.set_title("Channel Score (Valleys)")
ax.set_xlabel("X (East →)")
ax.set_ylabel("Y (North ↑)")
plt.colorbar(im, ax=ax, label="Score [0-1]")

ax = axes[1, 0]
im = ax.imshow(GLOBAL_WEATHER_DATA['total_rain'], cmap='YlGnBu', origin='lower')
ax.set_title(f"Total Rain ({num_weather_years} years)")
ax.set_xlabel("X (East →)")
ax.set_ylabel("Y (North ↑)")
plt.colorbar(im, ax=ax, label="Rain (m)")

ax = axes[1, 1]
# Show windward vs leeward pattern
slope_x, slope_y = GLOBAL_WEATHER_DATA['wind_features']['slope_vectors']
wind_x, wind_y = GLOBAL_WEATHER_DATA['wind_features']['wind_vector']
slope_mag = np.sqrt(slope_x**2 + slope_y**2) + 1e-9
cos_theta = (slope_x * wind_x + slope_y * wind_y) / slope_mag
im = ax.imshow(cos_theta, cmap='RdBu_r', vmin=-1, vmax=1, origin='lower')
ax.set_title("Windward (Red) vs Leeward (Blue)")
ax.set_xlabel("X (East →)")
ax.set_ylabel("Y (North ↑)")
plt.colorbar(im, ax=ax, label="Alignment")

ax = axes[1, 2]
im = ax.imshow(GLOBAL_RAIN_TIMESERIES[0], cmap='YlGnBu', origin='lower')
ax.set_title("Year 1 Rain (with wind effects)")
ax.set_xlabel("X (East →)")
ax.set_ylabel("Y (North ↑)")
plt.colorbar(im, ax=ax, label="Rain (m/yr)")

plt.suptitle("Wind Physics: EAST wind → Wet windward slopes, Dry leeward (rain shadow)", 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("✓ Visualization complete")
print("\nPhysics check:")
print(f"  Wind direction: EAST ({wind_dir_deg}°)")
print(f"  Windward slopes (facing east): More rain")
print(f"  Leeward slopes (facing west): Less rain (rain shadow)")
print(f"  Channels (valleys): Rain funneled and concentrated")
