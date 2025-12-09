"""
CELL 1: TERRAIN GENERATOR (YOUR STYLE from Project.ipynb)

Uses YOUR code for:
- N=512 high-resolution terrain
- pixel_scale_m=10.0 (5.12 km × 5.12 km domain)
- Quantum-seeded terrain generation
- Power-law spectrum (fractional_surface)
- Domain warp + ridged features
- Your wind structure classification
- Your discrete colormap visualization

This is adapted from your Project.ipynb but organized for the erosion system.
"""
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# QUANTUM RNG (from your Project.ipynb)
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


def qrng_uint32(n, nbits=32):
    """Return n uint32 from Qiskit Aer if available; else PRNG fallback."""
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
    """Random per run if random_seed=None; reproducible if you pass an int."""
    if random_seed is not None:
        return np.random.default_rng(int(random_seed))
    import os, time, hashlib
    seeds = qrng_uint32(n_seeds).tobytes()
    mix = seeds + os.urandom(16) + int(time.time_ns()).to_bytes(8, "little")
    h = hashlib.blake2b(mix, digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(h, "little"))


# ==============================================================================
# TERRAIN GENERATION (from your Project.ipynb)
# ==============================================================================

def fractional_surface(N, beta=3.1, rng=None):
    """Power-law spectrum; higher beta => smoother large-scale terrain."""
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


def domain_warp(z, rng, amp=0.12, beta=3.0):
    """Coordinate distortion; amp↑ => gnarlier micro-relief."""
    N = z.shape[0]
    u = fractional_surface(N, beta=beta, rng=rng)*2 - 1
    v = fractional_surface(N, beta=beta, rng=rng)*2 - 1
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    Xw = (ii + amp*N*u) % N
    Yw = (jj + amp*N*v) % N
    return bilinear_sample(z, Xw, Yw)


def ridged_mix(z, alpha=0.18):
    """Ridge/valley sharpening; alpha↑ => craggier."""
    ridged = 1.0 - np.abs(2.0*z - 1.0)
    out = (1-alpha)*z + alpha*ridged
    lo, hi = np.percentile(out, [2, 98])
    return np.clip((out - lo)/(hi - lo + 1e-12), 0, 1)


def quantum_seeded_topography(
    N=512, beta=3.1, warp_amp=0.12, ridged_alpha=0.18,
    random_seed=None
):
    """
    Generate terrain using YOUR method from Project.ipynb.
    
    Args:
        N: grid size (your default: 512)
        beta: power-law exponent (your default: 3.1-3.2)
        warp_amp: domain warp amplitude (your default: 0.10-0.12)
        ridged_alpha: ridge sharpening (your default: 0.15-0.18)
        random_seed: for reproducibility
    
    Returns:
        z_norm: normalized elevation (0-1)
        rng: random number generator
    """
    rng = rng_from_qrng(n_seeds=4, random_seed=random_seed)
    base_low = fractional_surface(N, beta=beta, rng=rng)
    base_high = fractional_surface(N, beta=beta-0.4, rng=rng)
    z = 0.65*base_low + 0.35*base_high
    z = domain_warp(z, rng=rng, amp=warp_amp, beta=beta)
    z = ridged_mix(z, alpha=ridged_alpha)
    return z, rng


def generate_stratigraphy(z_norm, pixel_scale_m=10.0, elev_range_m=700.0):
    """
    Generate stratigraphy from normalized elevation.
    
    Args:
        z_norm: normalized elevation (0-1)
        pixel_scale_m: cell size (YOUR default: 10.0 m)
        elev_range_m: elevation range (YOUR default: 700.0 m)
    
    Returns:
        strata: dict with surface_elev, interfaces, thickness, properties
    """
    ny, nx = z_norm.shape
    
    # Scale to actual elevation
    surface_elev = z_norm * elev_range_m
    
    # Simple layered stratigraphy
    strata = {
        "surface_elev": surface_elev.copy(),
        "interfaces": {},
        "thickness": {},
        "properties": {},
        "meta": {"pixel_scale_m": pixel_scale_m},
    }
    
    # Define layers (top to bottom)
    layers = {
        "Topsoil": {"thickness": 1.0, "erodibility": 1.0},
        "Saprolite": {"thickness": 8.0, "erodibility": 0.8},
        "Sandstone": {"thickness": 25.0, "erodibility": 0.5},
        "Basement": {"thickness": 50.0, "erodibility": 0.1},
    }
    
    current_interface = surface_elev.copy()
    
    for layer_name, props in layers.items():
        thickness = np.ones((ny, nx)) * props["thickness"]
        interface = current_interface - thickness
        
        strata["thickness"][layer_name] = thickness
        strata["interfaces"][layer_name] = interface
        strata["properties"][layer_name] = {"erodibility": props["erodibility"]}
        
        current_interface = interface
    
    # BasementFloor (deep below everything)
    strata["interfaces"]["BasementFloor"] = current_interface - 500.0
    
    return strata


# ==============================================================================
# TOPOGRAPHIC ANALYSIS (from your Project.ipynb)
# ==============================================================================

def _normalize(x, eps=1e-12):
    lo, hi = np.percentile(x, [2, 98])
    return np.clip((x - lo)/(hi - lo + eps), 0.0, 1.0)


def compute_topo_fields(surface_elev, pixel_scale_m):
    """
    Basic topographic fields from elevation only (from your Project.ipynb).
    
    Returns dict with:
      E, E_norm          : elevation (m) and normalized (0..1)
      dEx, dEy           : gradients in x (cols) and y (rows) (m/m)
      slope_mag, slope_norm
      aspect             : downslope direction (radians, 0 = +x)
      laplacian          : convex/concave indicator
    """
    E = surface_elev
    E_norm = _normalize(E)
    
    # gradient: np.gradient returns [d/drow, d/dcol] = [y, x]
    dEy, dEx = np.gradient(E, pixel_scale_m, pixel_scale_m)
    slope_mag = np.hypot(dEx, dEy) + 1e-12
    slope_norm = _normalize(slope_mag)
    
    # downslope aspect
    aspect = np.arctan2(-dEy, -dEx)
    
    # simple 4-neighbor Laplacian: <0 convex (ridge), >0 concave (valley)
    up = np.roll(E, -1, axis=0)
    down = np.roll(E, 1, axis=0)
    left = np.roll(E, 1, axis=1)
    right = np.roll(E, -1, axis=1)
    lap = (up + down + left + right - 4.0 * E) / (pixel_scale_m**2)
    
    return {
        "E": E,
        "E_norm": E_norm,
        "dEx": dEx,
        "dEy": dEy,
        "slope_mag": slope_mag,
        "slope_norm": slope_norm,
        "aspect": aspect,
        "laplacian": lap,
    }


def classify_windward_leeward(dEx, dEy, slope_norm,
                               base_wind_dir_deg,
                               slope_min=0.15):
    """Per-cell windward / leeward classification (from your Project.ipynb)."""
    theta = np.deg2rad(base_wind_dir_deg)
    wx, wy = np.cos(theta), np.sin(theta)   # wind-from unit vector
    
    # component of gradient along wind-from direction
    up_component = dEx * wx + dEy * wy
    
    slope_enough = slope_norm >= slope_min
    windward_mask = slope_enough & (up_component > 0.0)
    leeward_mask = slope_enough & (up_component < 0.0)
    
    return windward_mask, leeward_mask, up_component


def classify_wind_barriers(E_norm, slope_norm, laplacian, up_component,
                            elev_thresh=0.5,
                            slope_thresh=0.4,
                            convex_frac=0.4,
                            up_quantile=0.4):
    """Wind barriers: mountain walls (from your Project.ipynb)."""
    # convex threshold
    lap_convex_thr = np.quantile(laplacian, convex_frac)
    
    # upslope threshold
    mask_pos = up_component > 0.0
    if np.any(mask_pos):
        up_thr = np.quantile(up_component[mask_pos], up_quantile)
    else:
        up_thr = 0.0
    
    barrier_mask = (
        (E_norm >= elev_thresh) &
        (slope_norm >= slope_thresh) &
        (laplacian <= lap_convex_thr) &
        (up_component >= up_thr)
    )
    return barrier_mask


def classify_wind_channels(E_norm, slope_norm, laplacian,
                            dEx, dEy,
                            base_wind_dir_deg,
                            elev_max=0.7,
                            concave_frac=0.6,
                            slope_min=0.03,
                            slope_max=0.7,
                            align_thresh_deg=45.0):
    """Wind channels: valley axes (from your Project.ipynb)."""
    # concave threshold
    lap_concave_thr = np.quantile(laplacian, concave_frac)
    
    # slope range
    slope_ok = (slope_norm >= slope_min) & (slope_norm <= slope_max)
    
    # elevation (prefer lower areas)
    elev_ok = E_norm <= elev_max
    
    # concave
    concave = laplacian >= lap_concave_thr
    
    # alignment with wind
    theta = np.deg2rad(base_wind_dir_deg)
    wx, wy = np.cos(theta), np.sin(theta)
    
    # gradient perpendicular to wind => valley parallel to wind
    dot_grad_wind = np.abs(dEx * wx + dEy * wy)
    grad_mag = np.hypot(dEx, dEy) + 1e-12
    cos_angle = dot_grad_wind / grad_mag
    
    align_thresh = np.cos(np.deg2rad(align_thresh_deg))
    aligned = cos_angle < align_thresh  # gradient NOT parallel to wind
    
    channel_mask = elev_ok & slope_ok & concave & aligned
    return channel_mask


def classify_basins(E_norm, slope_norm, laplacian,
                     elev_max=0.5,
                     slope_max=0.3,
                     concave_frac=0.7):
    """Basins: low, flat, concave areas (from your Project.ipynb)."""
    lap_concave_thr = np.quantile(laplacian, concave_frac)
    
    basin_mask = (
        (E_norm <= elev_max) &
        (slope_norm <= slope_max) &
        (laplacian >= lap_concave_thr)
    )
    return basin_mask


def build_wind_structures(surface_elev, pixel_scale_m, base_wind_dir_deg):
    """
    Given a topography map, classify geological structures that change wind.
    (from your Project.ipynb)
    
    Returns a dict with per-cell masks.
    """
    topo = compute_topo_fields(surface_elev, pixel_scale_m)
    E = topo["E"]
    E_norm = topo["E_norm"]
    dEx = topo["dEx"]
    dEy = topo["dEy"]
    slope_n = topo["slope_norm"]
    lap = topo["laplacian"]
    
    windward_mask, leeward_mask, up_component = classify_windward_leeward(
        dEx, dEy, slope_n, base_wind_dir_deg
    )
    
    barrier_mask = classify_wind_barriers(
        E_norm, slope_n, lap, up_component
    )
    
    channel_mask = classify_wind_channels(
        E_norm, slope_n, lap, dEx, dEy, base_wind_dir_deg
    )
    
    basin_mask = classify_basins(
        E_norm, slope_n, lap
    )
    
    return {
        "E": E,
        "E_norm": E_norm,
        "slope_norm": slope_n,
        "laplacian": lap,
        "windward_mask": windward_mask,
        "leeward_mask": leeward_mask,
        "up_component": up_component,
        "barrier_mask": barrier_mask,
        "channel_mask": channel_mask,
        "basin_mask": basin_mask,
        "meta": {
            "pixel_scale_m": pixel_scale_m,
        },
    }


# ==============================================================================
# VISUALIZATION (YOUR STYLE from Project.ipynb)
# ==============================================================================

def plot_wind_structures_categorical(wind_structs):
    """
    Visualize geological features using YOUR discrete colormap style.
    (from your Project.ipynb)
    """
    E = wind_structs["E"]
    barrier_mask = wind_structs["barrier_mask"]
    channel_mask = wind_structs["channel_mask"]
    basin_mask = wind_structs["basin_mask"]
    
    # integer feature codes (0 = none, 1 = barrier, 2 = channel, 3 = basin)
    features = np.zeros_like(E, dtype=int)
    features[barrier_mask] = 1
    features[channel_mask] = 2
    features[basin_mask] = 3
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Use discrete tab10 colormap with 4 entries
    cmap = plt.cm.get_cmap("tab10", 4)
    
    im = ax.imshow(features,
                   origin="lower",
                   interpolation="nearest",
                   cmap=cmap,
                   vmin=-0.5, vmax=3.5)
    
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(color=cmap(1), label="Wind barriers (ridges)"),
        Patch(color=cmap(2), label="Wind channels (valleys)"),
        Patch(color=cmap(3), label="Basins / bowls"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", framealpha=0.9)
    
    ax.set_title("Wind-relevant geological features", fontweight='bold', fontsize=14)
    ax.set_xlabel("x (columns)")
    ax.set_ylabel("y (rows)")
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    n_barriers = np.sum(barrier_mask)
    n_channels = np.sum(channel_mask)
    n_basins = np.sum(basin_mask)
    print(f"\nWind feature statistics:")
    print(f"  Barriers: {n_barriers} cells ({100*n_barriers/barrier_mask.size:.1f}%)")
    print(f"  Channels: {n_channels} cells ({100*n_channels/channel_mask.size:.1f}%)")
    print(f"  Basins: {n_basins} cells ({100*n_basins/basin_mask.size:.1f}%)")


print("✓ Terrain generator (YOUR STYLE) loaded successfully!")
print("  Using YOUR code from Project.ipynb:")
print("    - N=512 high-resolution terrain")
print("    - pixel_scale_m=10.0 (5.12 km × 5.12 km domain)")
print("    - Quantum-seeded terrain generation")
print("    - Your wind structure classification")
print("    - Your discrete colormap visualization")
