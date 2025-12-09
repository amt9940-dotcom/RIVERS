#!/usr/bin/env python3
"""
Terrain + Stratigraphy (with optional smoothness controls)

USAGE QUICK HINTS (all optional; defaults keep original behavior):
- Make terrain smoother globally:
    quantum_seeded_topography(..., smooth_cutoff=0.12, smooth_rolloff=0.06)
- Use fractal fBm instead of the 2-surface blend (richer multiscale control):
    quantum_seeded_topography(..., use_fbm=True, H=0.75, octaves=4)
- Soften pixel-to-pixel transitions at the very end (image blur):
    quantum_seeded_topography(..., post_blur_sigma=1.2)
- Smooth subsurface layer contacts slightly (keeps the surface unchanged):
    generate_stratigraphy(..., interface_blur_sigma=0.8)

Everything above is OPTIONAL. If you don’t pass those args, nothing changes.
"""

# -------------------------------------------------------------------
# 0) Self-bootstrapping: ensure numpy/matplotlib (safe to keep)
# -------------------------------------------------------------------
import sys, subprocess, importlib

def ensure(pkg, extra=None):
    """Install `pkg` into THIS interpreter if missing."""
    try:
        return importlib.import_module(pkg)
    except Exception:
        spec = extra if extra else pkg
        print(f"[setup] Installing {spec} into {sys.executable} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", spec])
        return importlib.import_module(pkg)

np       = ensure("numpy", extra="numpy")
plt_mod  = ensure("matplotlib")
plt      = ensure("matplotlib.pyplot")

# Optional quantum seeding (will fall back if not available)
try:
    import qiskit  # noqa
    try:
        import qiskit_aer  # noqa
    except Exception:
        ensure("qiskit_aer")
    HAVE_QISKIT = True
except Exception:
    HAVE_QISKIT = False

print("[info] Python:", sys.version.split()[0])
print("[info] Using interpreter:", sys.executable)
print("[info] Qiskit available?", HAVE_QISKIT)

# -------------------------------------------------------------------
# 1) RNG: Qiskit (if present) or secure PRNG fallback
# -------------------------------------------------------------------
def qrng_uint32(n, nbits=32):
    if not HAVE_QISKIT:
        rng = np.random.default_rng()
        return rng.integers(0, 2**32, size=n, dtype=np.uint32)

    from qiskit import QuantumCircuit
    try:
        from qiskit_aer import Aer
    except Exception:
        try:
            from qiskit import Aer
        except Exception:
            rng = np.random.default_rng()
            return rng.integers(0, 2**32, size=n, dtype=np.uint32)

    qc = QuantumCircuit(nbits, nbits)
    qc.h(range(nbits))
    qc.measure(range(nbits), range(nbits))
    backend = Aer.get_backend('qasm_simulator')
    job = backend.run(qc, shots=n, memory=True)
    mem = job.result().get_memory(qc)
    return np.array([np.uint32(int(bits[::-1], 2)) for bits in mem], dtype=np.uint32)

def rng_from_qrng(n_seeds=4):
    seeds = qrng_uint32(n_seeds)
    seed = 0
    for s in seeds:
        seed = (seed * 4294967291 + int(s)) % (1<<63)
    return np.random.default_rng(seed)

# -------------------------------------------------------------------
# 2) Terrain primitives
# -------------------------------------------------------------------
def fractional_surface(N, beta=3.1, rng=None):
    """Power-law spectrum surface; higher beta => smoother large-scale terrain."""
    if rng is None:
        rng = np.random.default_rng()
    kx = np.fft.fftfreq(N)
    ky = np.fft.rfftfreq(N)
    K = np.sqrt(kx[:, None]**2 + ky[None, :]**2)
    K[0, 0] = np.inf
    amp = 1.0 / (K ** (beta / 2.0))
    phase = rng.uniform(0.0, 2.0*np.pi, size=(N, ky.size))
    spec = amp * (np.cos(phase) + 1j*np.sin(phase))
    spec[0, 0] = 0.0
    z = np.fft.irfftn(spec, s=(N, N))
    lo, hi = np.percentile(z, [2, 98])
    z = np.clip((z - lo) / (hi - lo + 1e-12), 0.0, 1.0)
    return z

def bilinear_sample(img, X, Y):
    N = img.shape[0]
    x0 = np.floor(X).astype(int) % N; y0 = np.floor(Y).astype(int) % N
    x1 = (x0 + 1) % N;              y1 = (y0 + 1) % N
    dx = X - np.floor(X);           dy = Y - np.floor(Y)
    return ((1-dx)*(1-dy)*img[x0, y0] + dx*(1-dy)*img[x1, y0] +
            (1-dx)*dy*img[x0, y1]   + dx*dy*img[x1, y1])

def domain_warp(z, rng, amp=0.12, beta=3.0):
    """Distorts coordinates; higher amp => rougher micro-relief."""
    N = z.shape[0]
    u = fractional_surface(N, beta=beta, rng=rng)*2 - 1
    v = fractional_surface(N, beta=beta, rng=rng)*2 - 1
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    Xw = (ii + amp*N*u) % N
    Yw = (jj + amp*N*v) % N
    return bilinear_sample(z, Xw, Yw)

def ridged_mix(z, alpha=0.18):
    """Adds ridge/valley sharpness; higher alpha => craggier terrain."""
    ridged = 1.0 - np.abs(2.0*z - 1.0)
    out = (1 - alpha)*z + alpha*ridged
    lo, hi = np.percentile(out, [2, 98])
    return np.clip((out - lo) / (hi - lo + 1e-12), 0, 1)

# ---------------- NEW: global low-pass control (opt-in) ----------------
def lowpass2d(z, cutoff=0.20, rolloff=0.08):
    """
    OPTIONAL smoothness knob (no effect unless you call it):
    - smaller `cutoff` => smoother terrain (filters out more high-frequency)
    - `rolloff` softens the filter edge to avoid ringing
    """
    Z = np.fft.rfft2(z)
    Nx, Ny = z.shape
    kx = np.fft.fftfreq(Nx)[:, None]
    ky = np.fft.rfftfreq(Ny)[None, :]
    r = np.sqrt(kx**2 + ky**2)
    m = np.ones_like(r)
    r0, w = cutoff, rolloff
    in_roll = (r > r0) & (r < r0 + w)
    m[r >= r0 + w] = 0.0
    m[in_roll] = 0.5 * (1 + np.cos(np.pi * (r[in_roll] - r0) / w))
    zf = np.fft.irfft2(np.fft.rfft2(z) * m, s=z.shape)
    return zf

# -------------- NEW: fBm option (multioctave roughness, opt-in) --------------
def fBm_surface(N, rng, H=0.65, octaves=5, lacunarity=2.0):
    """
    OPTIONAL alternative base generator:
    - Higher H (e.g., 0.8) => smoother; lower H (e.g., 0.5) => rougher.
    - More octaves => richer detail spectrum.
    """
    z = np.zeros((N, N), dtype=float)
    amp = 1.0
    beta0 = 2.0 + 2.0*H  # relate H to spectral slope
    for _ in range(octaves):
        z += amp * fractional_surface(N, beta=beta0, rng=rng)
        amp *= 0.5
        beta0 += 0.0  # keep slope per octave (simple fBm)
    lo, hi = np.percentile(z, [2, 98])
    return np.clip((z - lo) / (hi - lo + 1e-12), 0, 1)

# -------------- NEW: gentle image-space blur (opt-in) --------------
def gaussian_blur(z, sigma=1.2):
    """
    OPTIONAL last-mile smoother on the heightmap or interfaces:
    - sigma in pixels; try 0.6 (subtle) to 2.5 (very smooth).
    """
    rad = int(np.ceil(3*sigma))
    if rad <= 0:
        return z
    x = np.arange(-rad, rad+1)
    g = np.exp(-0.5*(x/sigma)**2); g /= g.sum()

    tmp = np.zeros_like(z)
    for i, w in enumerate(g):
        tmp += w*np.roll(z, i - rad, axis=1)  # horizontal
    out = np.zeros_like(z)
    for i, w in enumerate(g):
        out += w*np.roll(tmp, i - rad, axis=0)  # vertical
    return out

# -------------------------------------------------------------------
# 3) Topography generator (now with optional smoothness knobs)
# -------------------------------------------------------------------
def quantum_seeded_topography(
    N=512,
    beta=3.1,
    warp_amp=0.12,
    ridged_alpha=0.18,
    *,
    # --- NEW knobs (all default to no-op) ---
    smooth_cutoff=None,      # None => skip low-pass; else 0..0.5 (smaller = smoother)
    smooth_rolloff=0.08,     # width of low-pass transition
    use_fbm=False,           # False => original method; True => fBm base
    H=0.65,                  # fBm roughness exponent (higher = smoother)
    octaves=5,               # fBm number of layers
    post_blur_sigma=None,    # None => skip blur; else e.g. 0.8..2.0 pixels
):
    rng = rng_from_qrng(n_seeds=4)

    if use_fbm:  # --- NEW (opt-in) ---
        z = fBm_surface(N, rng=rng, H=H, octaves=octaves, lacunarity=2.0)
    else:
        base_low  = fractional_surface(N, beta=beta,     rng=rng)
        base_high = fractional_surface(N, beta=beta-0.4, rng=rng)
        z = 0.65*base_low + 0.35*base_high

    z = domain_warp(z, rng=rng, amp=warp_amp, beta=beta)
    z = ridged_mix(z, alpha=ridged_alpha)

    if smooth_cutoff is not None:   # --- NEW (opt-in) ---
        z = lowpass2d(z, cutoff=smooth_cutoff, rolloff=smooth_rolloff)
        lo, hi = np.percentile(z, [2, 98])
        z = np.clip((z - lo) / (hi - lo + 1e-12), 0, 1)

    if post_blur_sigma is not None:  # --- NEW (opt-in) ---
        z = gaussian_blur(z, sigma=post_blur_sigma)
        lo, hi = np.percentile(z, [2, 98])
        z = np.clip((z - lo) / (hi - lo + 1e-12), 0, 1)

    return z, rng

# -------------------------------------------------------------------
# 4) Stratigraphy (layer names simplified; optional interface smoothing)
# -------------------------------------------------------------------
def _box_blur(a, k=5):
    if k <= 1:
        return a
    out = a.copy()
    for axis in (0, 1):
        tmp = out
        s = np.zeros_like(tmp)
        for i in range(-(k//2), k//2 + 1):
            s += np.roll(tmp, i, axis=axis)
        out = s / float(k)
    return out

def _normalize(x, eps=1e-12):
    lo, hi = np.percentile(x, [2, 98])
    return np.clip((x - lo) / (hi - lo + eps), 0.0, 1.0)

def generate_stratigraphy(
    z_norm,
    rng,
    elev_range_m=700.0,
    pixel_scale_m=10.0,
    soil_range_m=(0.2, 2.0),
    colluvium_max_m=12.0,
    unit_thickness_m=(80.0, 100.0, 90.0),  # sandstone, shale, limestone
    undulation_amp_m=12.0,
    undulation_beta=3.0,
    dip_deg=6.0,
    dip_dir_deg=45.0,
    burial_depth_m=120.0,
    *,
    # --- NEW: smooth layer interfaces slightly (opt-in) ---
    interface_blur_sigma=None,  # e.g., 0.6..1.2 to soften contacts a bit
):
    N = z_norm.shape[0]
    E = z_norm * elev_range_m

    dzdx, dzdy = np.gradient(z_norm)
    slope_mag = np.sqrt(dzdx**2 + dzdy**2)
    slope_norm = _normalize(slope_mag)
    soil_thick = soil_range_m[1] - (soil_range_m[1]-soil_range_m[0]) * slope_norm
    soil_thick = _box_blur(soil_thick, k=5)

    lowlands = _normalize(1.0 - z_norm)
    flats    = _normalize(1.0 - slope_norm)
    colluv_field = _normalize(0.6*lowlands + 0.4*flats)
    colluvium_thick = colluvium_max_m * _box_blur(colluv_field, k=9)

    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    X = ii * pixel_scale_m
    Y = jj * pixel_scale_m
    az = np.deg2rad(dip_dir_deg)
    dip = np.deg2rad(dip_deg)
    nx, ny = np.cos(az), np.sin(az)
    plane = np.tan(dip) * (nx*X + ny*Y)
    undul = (fractional_surface(N, beta=undulation_beta, rng=rng)*2 - 1) * undulation_amp_m
    bed_struct = plane + undul
    bed_struct_zm = bed_struct - np.mean(bed_struct)

    T_sand, T_shale, T_lime = unit_thickness_m

    top_topsoil   = E
    top_colluvium = top_topsoil - soil_thick

    # Deeper under highs; guaranteed burial below colluvium
    top_sandstone = top_colluvium - (burial_depth_m + colluvium_thick) - bed_struct_zm
    top_shale     = top_sandstone - T_sand
    top_limestone = top_shale     - T_shale
    top_bedrock   = top_limestone - T_lime

    eps = 0.01
    top_colluvium = np.minimum(top_colluvium, top_topsoil - eps)
    top_sandstone = np.minimum(top_sandstone, top_colluvium - eps)
    top_shale     = np.minimum(top_shale,     top_sandstone - eps)
    top_limestone = np.minimum(top_limestone, top_shale - eps)
    top_bedrock   = np.minimum(top_bedrock,   top_limestone - eps)

    # --- NEW (opt-in): blur interfaces a touch for smooth contacts ---
    if interface_blur_sigma is not None:
        for name, arr in [("top_sandstone", top_sandstone),
                          ("top_shale", top_shale),
                          ("top_limestone", top_limestone)]:
            arr_blur = gaussian_blur(arr, sigma=interface_blur_sigma)
            if name == "top_sandstone": top_sandstone = arr_blur
            if name == "top_shale":     top_shale     = arr_blur
            if name == "top_limestone": top_limestone = arr_blur
        # Respect ordering after blur
        top_shale     = np.minimum(top_shale,     top_sandstone - eps)
        top_limestone = np.minimum(top_limestone, top_shale - eps)
        top_bedrock   = np.minimum(top_bedrock,   top_limestone - eps)

    thickness = {
        "Topsoil":   np.maximum(top_topsoil - top_colluvium, 0.0),
        "Colluvium": np.maximum(top_colluvium - top_sandstone, 0.0),
        "Sandstone": np.maximum(top_sandstone - top_shale, 0.0),
        "Shale":     np.maximum(top_shale - top_limestone, 0.0),
        "Limestone": np.maximum(top_limestone - top_bedrock, 0.0),
    }

    interfaces = {
        "Topsoil":   top_topsoil,
        "Colluvium": top_colluvium,
        "Sandstone": top_sandstone,
        "Shale":     top_shale,
        "Limestone": top_limestone,
        "Bedrock":   top_bedrock,
    }

    properties = {
        "Topsoil":   {"erodibility": 1.00, "density": 1600, "porosity": 0.45},
        "Colluvium": {"erodibility": 0.85, "density": 1750, "porosity": 0.35},
        "Sandstone": {"erodibility": 0.25, "density": 2200, "porosity": 0.20},
        "Shale":     {"erodibility": 0.35, "density": 2300, "porosity": 0.10},
        "Limestone": {"erodibility": 0.20, "density": 2400, "porosity": 0.05},
        "Bedrock":   {"erodibility": 0.10, "density": 2700, "porosity": 0.01},
    }

    return {
        "surface_elev": E,
        "interfaces": interfaces,
        "thickness": thickness,
        "properties": properties,
        "meta": {
            "elev_range_m": elev_range_m,
            "pixel_scale_m": pixel_scale_m,
            "dip_deg": dip_deg,
            "dip_dir_deg": dip_dir_deg,
            "unit_thickness_m": unit_thickness_m,
            "burial_depth_m": burial_depth_m,
            "interface_blur_sigma": interface_blur_sigma,
        }
    }

# -------------------------------------------------------------------
# 5) Cross-section plot (layers visible)
# -------------------------------------------------------------------
def plot_cross_section(strata, row=None, col=None):
    E = strata["surface_elev"]
    N = E.shape[0]
    if (row is None) == (col is None):
        row = N // 2

    if row is not None:
        x = np.arange(N)
        surf = E[row, :]
        tops = {k: v[row, :] for k, v in strata["interfaces"].items()}
        axis_label = "columns (x)"
    else:
        x = np.arange(N)
        surf = E[:, col]
        tops = {k: v[:, col] for k, v in strata["interfaces"].items()}
        axis_label = "rows (y)"

    order = ["Topsoil","Colluvium","Sandstone","Shale","Limestone","Bedrock"]

    plt.figure(figsize=(10,4))
    for i in range(len(order)-1, 0, -1):  # draw from bottom upward
        above = order[i-1]; here = order[i]
        y_top = tops[above]; y_bot = tops[here]
        plt.fill_between(x, y_bot, y_top, alpha=0.85, linewidth=0.6, zorder=5+i, label=here)

    plt.plot(x, surf, linewidth=2.6, zorder=50, label="Surface")
    plt.title("Stratigraphic cross-section")
    plt.xlabel(axis_label); plt.ylabel("Elevation (m)")
    plt.legend(ncol=6, fontsize=8, framealpha=0.9, loc="lower left")
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------
# 6) Demo run (defaults keep original look; see comments to customize)
# -------------------------------------------------------------------
if __name__ == "__main__":
    # --- Terrain (DEFAULTS: original behavior) ---
    z, rng = quantum_seeded_topography(
        N=512,
        beta=3.2,
        warp_amp=0.10,
        ridged_alpha=0.15,
        # >>> OPTIONAL knobs (uncomment to try) <<<
        # smooth_cutoff=0.14,      # smaller -> smoother (e.g., 0.10..0.18)
        # smooth_rolloff=0.06,
        # use_fbm=True, H=0.75, octaves=4,
        # post_blur_sigma=1.0,
    )

    # --- Stratigraphy ---
    strata = generate_stratigraphy(
        z_norm=z,
        rng=rng,
        elev_range_m=700.0,
        pixel_scale_m=10.0,
        soil_range_m=(0.3, 2.0),
        colluvium_max_m=15.0,
        unit_thickness_m=(90.0, 110.0, 100.0),
        undulation_amp_m=12.0,
        undulation_beta=3.2,
        dip_deg=6.0,
        dip_dir_deg=45.0,
        burial_depth_m=140.0,
        # >>> OPTIONAL interface smoother <<<
        # interface_blur_sigma=0.8,
    )

    # --- Plots ---
    plt.figure(figsize=(6,6))
    plt.imshow(strata["surface_elev"], origin="lower")
    plt.colorbar(label="Surface elevation (m)")
    plt.title("Quantum-seeded terrain (absolute elevation)")
    plt.tight_layout()
    plt.show()

    plot_cross_section(strata, row=256)  # try `col=256` for a north–south cut
