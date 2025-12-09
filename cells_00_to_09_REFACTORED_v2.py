"""
STREAMLINED TERRAIN + WEATHER SYSTEM (Cells 0-9 Combined & Refactored)

This generates:
1. ONE terrain map (stored in global variables)
2. Weather simulation for that terrain
3. All data ready for erosion simulator to use

The erosion simulator (cells 10-19) will use these global variables.
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
    
    # Generate quantum bits and convert to floats
    nbits = 16  # Use 16 bits per float for precision
    bits = qrng_uint32(n, nbits=nbits)
    return bits.astype(np.float64) / (2.0**nbits)


# ==============================================================================
# TERRAIN GENERATION (Project33 style - simplified)
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
    """
    Generate terrain using quantum-seeded RNG.
    
    Parameters
    ----------
    N : int
        Grid size (default 256 for reasonable speed)
    beta : float
        Power-law exponent (higher = smoother)
    warp_amp : float
        Domain warp amplitude
    ridged_alpha : float
        Ridge sharpening factor
    random_seed : int or None
        For reproducibility (None = quantum random)
    
    Returns
    -------
    z_norm : np.ndarray (N, N)
        Normalized elevation [0, 1]
    rng : np.random.Generator
        Random number generator used
    """
    rng = rng_from_qrng(n_seeds=4, random_seed=random_seed)
    
    # Generate base terrain
    base_low = fractional_surface(N, beta=beta, rng=rng)
    base_high = fractional_surface(N, beta=beta-0.4, rng=rng)
    z = 0.65*base_low + 0.35*base_high
    
    # Apply warping and ridging
    z = domain_warp(z, rng=rng, amp=warp_amp, beta=beta)
    z = ridged_mix(z, alpha=ridged_alpha)
    
    return z, rng


# ==============================================================================
# SIMPLIFIED STRATIGRAPHY (Only 4 layers)
# ==============================================================================

def generate_stratigraphy(z_norm, rng, pixel_scale_m=20.0, elev_range_m=500.0):
    """
    Generate 4-layer stratigraphy (Topsoil, Subsoil, Saprolite, Basement).
    
    Parameters
    ----------
    z_norm : np.ndarray (ny, nx)
        Normalized elevation [0, 1]
    rng : np.random.Generator
        Random number generator
    pixel_scale_m : float
        Grid cell size [m]
    elev_range_m : float
        Elevation range [m]
    
    Returns
    -------
    strata : dict
        Contains:
        - surface_elev: surface elevation [m]
        - thickness: dict of layer thicknesses [m]
        - layer_order: list of layer names (top to bottom)
    """
    ny, nx = z_norm.shape
    
    # Convert to actual elevation
    surface_elev = z_norm * elev_range_m
    
    # Define 4 layers (ONLY)
    layer_order = ['Topsoil', 'Subsoil', 'Saprolite', 'Basement']
    
    thickness = {}
    
    # Layer 1: Topsoil (thin, varies with elevation)
    thickness['Topsoil'] = 2.0 + 3.0 * (1 - z_norm) + rng.uniform(-0.5, 0.5, size=(ny, nx))
    thickness['Topsoil'] = np.maximum(0.5, thickness['Topsoil'])
    
    # Layer 2: Subsoil (thicker, weathered material)
    thickness['Subsoil'] = 5.0 + 5.0 * rng.uniform(0, 1, size=(ny, nx))
    
    # Layer 3: Saprolite (weathered bedrock)
    thickness['Saprolite'] = 10.0 + 15.0 * rng.uniform(0, 1, size=(ny, nx))
    
    # Layer 4: Basement (infinite thickness for now)
    thickness['Basement'] = np.ones((ny, nx)) * 1000.0  # Very thick
    
    strata = {
        'surface_elev': surface_elev,
        'thickness': thickness,
        'layer_order': layer_order,
        'pixel_scale_m': pixel_scale_m
    }
    
    return strata


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
    
    # Fallback to Basement if nothing
    top_name[top_idx == -1] = "Basement"
    
    return top_idx, top_name


# ==============================================================================
# WIND-TOPOGRAPHY INTERACTION
# ==============================================================================

def classify_wind_features(surface_elev, pixel_scale_m, base_wind_dir_deg=270.0):
    """
    Classify terrain features for wind interaction:
    - Wind barriers: High elevation perpendicular to wind
    - Wind channels: Valleys aligned with wind direction
    
    Parameters
    ----------
    surface_elev : np.ndarray (ny, nx)
        Surface elevation [m]
    pixel_scale_m : float
        Grid cell size [m]
    base_wind_dir_deg : float
        Base wind direction [degrees from N, 0=N, 90=E, 180=S, 270=W]
    
    Returns
    -------
    wind_features : dict
        - barrier_mask: bool array (True = wind barrier)
        - channel_mask: bool array (True = wind channel)
        - speed_multiplier: float array (wind speed adjustment)
        - direction_deflection: float array (direction change in degrees)
    """
    ny, nx = surface_elev.shape
    
    # Compute gradients
    dy, dx = np.gradient(surface_elev, pixel_scale_m)
    
    # Convert wind direction to radians (meteorological convention)
    wind_rad = np.radians(base_wind_dir_deg)
    wind_vec = np.array([np.sin(wind_rad), np.cos(wind_rad)])  # [x, y] direction wind GOES
    
    # Project gradient onto wind direction
    grad_along_wind = dx * wind_vec[0] + dy * wind_vec[1]
    grad_across_wind = -dx * wind_vec[1] + dy * wind_vec[0]
    
    # Compute slope magnitude
    slope_mag = np.sqrt(dx**2 + dy**2)
    
    # WIND BARRIERS: High gradient perpendicular to wind (mountains blocking)
    # Strong positive gradient across wind = barrier
    barrier_threshold = np.percentile(np.abs(grad_across_wind), 75)
    barrier_mask = (np.abs(grad_across_wind) > barrier_threshold) & (slope_mag > 0.05)
    
    # WIND CHANNELS: Valleys aligned with wind direction
    # Low cross-wind gradient + low elevation = channel
    channel_threshold = np.percentile(np.abs(grad_across_wind), 25)
    elev_threshold = np.percentile(surface_elev, 40)
    channel_mask = (np.abs(grad_across_wind) < channel_threshold) & (surface_elev < elev_threshold)
    
    # SPEED MULTIPLIER: Wind speeds up in channels, slows at barriers
    speed_multiplier = np.ones((ny, nx), dtype=np.float32)
    speed_multiplier[barrier_mask] = 0.3  # 70% slowdown at barriers
    speed_multiplier[channel_mask] = 1.5  # 50% speedup in channels
    
    # DIRECTION DEFLECTION: Wind deflects around barriers
    direction_deflection = np.zeros((ny, nx), dtype=np.float32)
    
    # Deflect wind perpendicular to local gradient at barriers
    deflection_angle = 30.0  # degrees
    direction_deflection[barrier_mask & (grad_across_wind > 0)] = deflection_angle
    direction_deflection[barrier_mask & (grad_across_wind < 0)] = -deflection_angle
    
    return {
        'barrier_mask': barrier_mask,
        'channel_mask': channel_mask,
        'speed_multiplier': speed_multiplier,
        'direction_deflection': direction_deflection,
        'base_wind_dir_deg': base_wind_dir_deg
    }


# ==============================================================================
# STORM FIELD GENERATION WITH QUANTUM RANDOM RAIN
# ==============================================================================

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
    Generate a storm field with:
    - Wind-affected location and shape
    - Quantum random rain distribution within storm
    
    Parameters
    ----------
    surface_elev : np.ndarray (ny, nx)
        Surface elevation [m]
    wind_features : dict
        Output from classify_wind_features
    storm_center_ij : tuple (i, j)
        Storm center coordinates
    storm_radius_cells : int
        Storm radius in grid cells
    base_intensity_m_per_hour : float
        Base rainfall intensity
    duration_hours : float
        Storm duration
    pixel_scale_m : float
        Grid cell size
    rng : np.random.Generator
        Random number generator
    
    Returns
    -------
    rain_map : np.ndarray (ny, nx)
        Total rainfall for this storm [m]
    """
    ny, nx = surface_elev.shape
    ci, cj = storm_center_ij
    
    # Create base storm shape (circular)
    ii, jj = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    
    # Apply wind deflection to storm center
    wind_dir = wind_features['base_wind_dir_deg']
    deflection = wind_features['direction_deflection']
    
    # Shift storm center based on wind
    wind_rad = np.radians(wind_dir)
    wind_shift_i = int(10 * np.sin(wind_rad))
    wind_shift_j = int(10 * np.cos(wind_rad))
    
    ci_shifted = (ci + wind_shift_i) % ny
    cj_shifted = (cj + wind_shift_j) % nx
    
    # Distance from storm center
    di = ii - ci_shifted
    dj = jj - cj_shifted
    
    # Handle periodic boundaries
    di = np.where(di > ny/2, di - ny, di)
    di = np.where(di < -ny/2, di + ny, di)
    dj = np.where(dj > nx/2, dj - nx, dj)
    dj = np.where(dj < -nx/2, dj + nx, dj)
    
    dist = np.sqrt(di**2 + dj**2)
    
    # Storm intensity profile (Gaussian)
    storm_profile = np.exp(-(dist / storm_radius_cells)**2)
    
    # Apply wind effects
    speed_mult = wind_features['speed_multiplier']
    
    # Channels: Storm spreads out (wider, less intense)
    channel_mask = wind_features['channel_mask']
    storm_profile[channel_mask] *= 0.8  # Less intense but wider
    
    # Barriers: Storm blocked or redirected
    barrier_mask = wind_features['barrier_mask']
    storm_profile[barrier_mask] *= 0.3  # Much less rain on windward side
    
    # QUANTUM RANDOM RAIN WITHIN STORM
    # Generate quantum random field to modulate rain
    storm_cells = storm_profile > 0.1
    n_storm_cells = np.sum(storm_cells)
    
    if n_storm_cells > 0:
        # Get quantum random multipliers for each cell in storm
        try:
            quantum_mults = quantum_uniforms(n_storm_cells)
            # Convert to lognormal distribution (realistic for rain)
            quantum_mults = np.exp((quantum_mults - 0.5) * 1.5)
        except:
            # Fallback to classical
            quantum_mults = np.exp((rng.uniform(0, 1, size=n_storm_cells) - 0.5) * 1.5)
        
        # Apply quantum randomness
        storm_profile_quantum = storm_profile.copy()
        storm_profile_quantum[storm_cells] *= quantum_mults
        storm_profile = storm_profile_quantum
    
    # Total rainfall = intensity × duration × profile
    rain_map = base_intensity_m_per_hour * duration_hours * storm_profile
    
    return rain_map


# ==============================================================================
# MULTI-YEAR WEATHER SIMULATION
# ==============================================================================

def run_weather_simulation(
    surface_elev,
    pixel_scale_m,
    num_years=10,
    base_wind_dir_deg=270.0,
    mean_annual_rain_m=1.0,
    random_seed=None
):
    """
    Run multi-year weather simulation with:
    - Wind-topography interaction
    - Quantum random storm generation
    - Realistic rainfall patterns
    
    Parameters
    ----------
    surface_elev : np.ndarray (ny, nx)
        Surface elevation [m]
    pixel_scale_m : float
        Grid cell size [m]
    num_years : int
        Number of years to simulate
    base_wind_dir_deg : float
        Prevailing wind direction [degrees]
    mean_annual_rain_m : float
        Mean annual rainfall [m/yr]
    random_seed : int or None
        For reproducibility
    
    Returns
    -------
    weather_data : dict
        - annual_rain_maps: list of np.ndarray (ny, nx) [m/yr]
        - wind_features: wind-topography classification
        - total_rain: sum of all rain [m]
    """
    print(f"\nGenerating {num_years} years of weather...")
    
    ny, nx = surface_elev.shape
    rng = rng_from_qrng(random_seed=random_seed)
    
    # Classify wind features
    wind_features = classify_wind_features(surface_elev, pixel_scale_m, base_wind_dir_deg)
    
    print(f"  Wind barriers: {np.sum(wind_features['barrier_mask'])} cells")
    print(f"  Wind channels: {np.sum(wind_features['channel_mask'])} cells")
    
    # Generate storms for each year
    annual_rain_maps = []
    
    for year in range(num_years):
        year_rain = np.zeros((ny, nx), dtype=np.float32)
        
        # Generate 5-15 storms per year
        n_storms = rng.integers(5, 16)
        
        for storm_idx in range(n_storms):
            # Random storm parameters
            ci = rng.integers(0, ny)
            cj = rng.integers(0, nx)
            radius = rng.integers(10, 30)
            intensity = rng.uniform(0.001, 0.005)  # m/hr
            duration = rng.uniform(4, 24)  # hours
            
            # Generate storm with quantum random rain
            storm_rain = generate_storm_with_quantum_rain(
                surface_elev, wind_features,
                (ci, cj), radius, intensity, duration,
                pixel_scale_m, rng
            )
            
            year_rain += storm_rain
        
        # Scale to target annual rainfall
        scale_factor = mean_annual_rain_m / year_rain.mean()
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
# GENERATE INITIAL TERRAIN AND WEATHER (ONCE!)
# ==============================================================================

print("\n" + "="*80)
print("GENERATING INITIAL TERRAIN AND WEATHER")
print("="*80)

# Configuration
N = 256  # Grid size
pixel_scale_m = 20.0  # 20m per pixel
elev_range_m = 500.0  # 500m elevation range
num_weather_years = 100  # Generate 100 years of weather
base_wind_dir_deg = 270.0  # West wind
mean_annual_rain_m = 1.0  # 1 m/yr average

print(f"\nConfiguration:")
print(f"  Grid size: {N}×{N}")
print(f"  Domain size: {N*pixel_scale_m/1000:.2f} × {N*pixel_scale_m/1000:.2f} km")
print(f"  Pixel scale: {pixel_scale_m} m")
print(f"  Elevation range: {elev_range_m} m")
print(f"  Weather years: {num_weather_years}")
print(f"  Wind direction: {base_wind_dir_deg}° (West)")

# Generate terrain
print("\nGenerating terrain...")
start_time = time.time()

z_norm, rng = quantum_seeded_topography(
    N=N,
    beta=3.0,
    warp_amp=0.10,
    ridged_alpha=0.15,
    random_seed=None  # Quantum random
)

GLOBAL_STRATA = generate_stratigraphy(z_norm, rng, pixel_scale_m, elev_range_m)

print(f"✓ Terrain generated in {time.time() - start_time:.1f} s")
print(f"  Elevation range: {GLOBAL_STRATA['surface_elev'].min():.1f} - {GLOBAL_STRATA['surface_elev'].max():.1f} m")
print(f"  Layers: {GLOBAL_STRATA['layer_order']}")

# Generate weather for this terrain
start_time = time.time()

GLOBAL_WEATHER_DATA = run_weather_simulation(
    surface_elev=GLOBAL_STRATA['surface_elev'],
    pixel_scale_m=pixel_scale_m,
    num_years=num_weather_years,
    base_wind_dir_deg=base_wind_dir_deg,
    mean_annual_rain_m=mean_annual_rain_m,
    random_seed=None  # Quantum random
)

print(f"✓ Weather generated in {time.time() - start_time:.1f} s")

# Store rain timeseries as numpy array for easy access
GLOBAL_RAIN_TIMESERIES = np.array(GLOBAL_WEATHER_DATA['annual_rain_maps'], dtype=np.float32)

print("\n" + "="*80)
print("✅ INITIAL DATA READY")
print("="*80)
print("\nGLOBAL VARIABLES CREATED:")
print("  GLOBAL_STRATA - terrain and stratigraphy data")
print("    .surface_elev - initial elevation map [m]")
print("    .thickness - layer thicknesses [m]")
print("    .layer_order - list of 4 layers")
print("    .pixel_scale_m - grid cell size")
print("\n  GLOBAL_WEATHER_DATA - weather simulation results")
print("    .annual_rain_maps - list of yearly rain maps")
print("    .wind_features - wind-topography classification")
print("    .total_rain - cumulative rain over all years")
print("\n  GLOBAL_RAIN_TIMESERIES - numpy array (num_years, ny, nx)")
print("    Shape: ", GLOBAL_RAIN_TIMESERIES.shape)
print("    Mean: ", GLOBAL_RAIN_TIMESERIES.mean(), " m/yr")
print("\n✓ These variables are ready for the erosion simulator (cells 10-19)")
print("="*80 + "\n")

# Visualize initial terrain and weather
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ax = axes[0, 0]
im = ax.imshow(GLOBAL_STRATA['surface_elev'], cmap='terrain', origin='lower')
ax.set_title("Initial Terrain Elevation")
ax.set_xlabel("X (cells)")
ax.set_ylabel("Y (cells)")
plt.colorbar(im, ax=ax, label="Elevation (m)")

ax = axes[0, 1]
viz = np.zeros((N, N, 3))
viz[GLOBAL_WEATHER_DATA['wind_features']['barrier_mask']] = [1, 0, 0]  # Red
viz[GLOBAL_WEATHER_DATA['wind_features']['channel_mask']] = [0, 0, 1]  # Blue
ax.imshow(viz, origin='lower')
ax.set_title("Wind Features (Red=Barriers, Blue=Channels)")
ax.set_xlabel("X (cells)")
ax.set_ylabel("Y (cells)")

ax = axes[1, 0]
im = ax.imshow(GLOBAL_WEATHER_DATA['total_rain'], cmap='Blues', origin='lower')
ax.set_title(f"Total Rain ({num_weather_years} years)")
ax.set_xlabel("X (cells)")
ax.set_ylabel("Y (cells)")
plt.colorbar(im, ax=ax, label="Rain (m)")

ax = axes[1, 1]
im = ax.imshow(GLOBAL_RAIN_TIMESERIES[0], cmap='Blues', origin='lower')
ax.set_title("Year 1 Rain")
ax.set_xlabel("X (cells)")
ax.set_ylabel("Y (cells)")
plt.colorbar(im, ax=ax, label="Rain (m/yr)")

plt.tight_layout()
plt.show()

print("✓ Initial terrain and weather visualization complete")
