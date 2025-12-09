#!/usr/bin/env python3
"""
FIXED TERRAIN + STRATIGRAPHY GENERATOR

Implements proper geological architecture:
- Major structural surfaces (S0-S3) define vertical intervals
- Each formation has its own independent thickness field
- True pinch-outs and absence zones (not just thinning)
- Single upward stacking pass (no duplication)
- Formations assigned to specific intervals

Result: Smooth basin geometry + varied, realistic layers (no "striped cake")
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

# Optional quantum seeding
try:
    import qiskit
    from qiskit import QuantumCircuit
    try:
        import qiskit_aer
        HAVE_QISKIT = True
    except Exception:
        HAVE_QISKIT = False
except Exception:
    HAVE_QISKIT = False
    QuantumCircuit = None


def qrng_uint32(n, nbits=32):
    if not HAVE_QISKIT:
        return np.random.default_rng().integers(0, 2**32, size=n, dtype=np.uint32)
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


# ========================================================================================
# TOPOGRAPHY GENERATOR (same as before)
# ========================================================================================

def fractional_surface(N, beta=3.1, rng=None):
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
    N = z.shape[0]
    u = fractional_surface(N, beta=beta, rng=rng)*2 - 1
    v = fractional_surface(N, beta=beta, rng=rng)*2 - 1
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    Xw = (ii + amp*N*u) % N
    Yw = (jj + amp*N*v) % N
    return bilinear_sample(z, Xw, Yw)


def ridged_mix(z, alpha=0.18):
    ridged = 1.0 - np.abs(2.0*z - 1.0)
    out = (1-alpha)*z + alpha*ridged
    lo, hi = np.percentile(out, [2, 98])
    return np.clip((out - lo)/(hi - lo + 1e-12), 0, 1)


def lowpass2d(z, cutoff=None, rolloff=0.08):
    if cutoff is None:
        return z
    Nx, Ny = z.shape
    Z = np.fft.rfft2(z)
    kx = np.fft.fftfreq(Nx)[:, None]
    ky = np.fft.rfftfreq(Ny)[None, :]
    r = np.sqrt(kx**2 + ky**2)
    m = np.ones_like(r)
    r0, w = float(cutoff), float(rolloff)
    in_roll = (r > r0) & (r < r0 + w)
    m[r >= r0 + w] = 0.0
    m[in_roll] = 0.5 * (1 + np.cos(np.pi * (r[in_roll] - r0) / w))
    zf = np.fft.irfft2(Z * m, s=z.shape)
    lo, hi = np.percentile(zf, [2, 98])
    return np.clip((zf - lo) / (hi - lo + 1e-12), 0, 1)


def gaussian_blur(z, sigma=None):
    if sigma is None or sigma <= 0:
        return z
    rad = int(np.ceil(3*sigma))
    x = np.arange(-rad, rad+1)
    g = np.exp(-0.5*(x/sigma)**2)
    g /= g.sum()
    tmp = np.zeros_like(z)
    for i,w in enumerate(g):
        tmp += w*np.roll(z, i-rad, axis=1)
    out = np.zeros_like(z)
    for i,w in enumerate(g):
        out += w*np.roll(tmp, i-rad, axis=0)
    lo, hi = np.percentile(out, [2,98])
    return np.clip((out - lo)/(hi - lo + 1e-12), 0, 1)


def quantum_seeded_topography(N=512, beta=3.1, warp_amp=0.12, ridged_alpha=0.18,
                                *, random_seed=None, smooth_cutoff=None, 
                                smooth_rolloff=0.08, post_blur_sigma=None):
    rng = rng_from_qrng(n_seeds=4, random_seed=random_seed)
    base_low = fractional_surface(N, beta=beta, rng=rng)
    base_high = fractional_surface(N, beta=beta-0.4, rng=rng)
    z = 0.65*base_low + 0.35*base_high
    z = domain_warp(z, rng=rng, amp=warp_amp, beta=beta)
    z = ridged_mix(z, alpha=ridged_alpha)
    z = lowpass2d(z, cutoff=smooth_cutoff, rolloff=smooth_rolloff)
    z = gaussian_blur(z, sigma=post_blur_sigma)
    return z, rng


# ========================================================================================
# UTILITIES
# ========================================================================================

def _box_blur(a, k=5):
    if k <= 1:
        return a
    out = a.copy()
    for axis in (0,1):
        tmp = out
        s = np.zeros_like(tmp)
        for i in range(-(k//2), k//2+1):
            s += np.roll(tmp, i, axis=axis)
        out = s/float(k)
    return out


def _normalize(x, eps=1e-12):
    lo, hi = np.percentile(x, [2,98])
    return np.clip((x - lo)/(hi - lo + eps), 0.0, 1.0)


def smooth_random_field(shape, rng, k_smooth=15, beta=3.5):
    """Generate a smooth random field with long-wavelength correlation."""
    N = shape[0]
    noise = fractional_surface(N, beta=beta, rng=rng) * 2 - 1
    smoothed = _box_blur(noise, k=max(5, int(k_smooth) | 1))
    return smoothed


# ========================================================================================
# MAJOR STRUCTURAL SURFACES (the key to proper geology)
# ========================================================================================

def generate_structural_surfaces(z_norm, E, rng, pixel_scale_m, elev_span):
    """
    Generate major structural surfaces that define vertical intervals.
    
    Returns S0 (basement), S1 (old sediments), S2 (lower group), S3 (upper group).
    Each is a smooth 2D field following basin-scale geometry.
    """
    N = z_norm.shape[0]
    
    # Compute basin field (very smooth, long-wavelength)
    k_structural = max(63, int(0.35 * N) | 1)
    structural_noise = fractional_surface(N, beta=4.5, rng=rng)  # Very smooth
    structural_field = _box_blur(structural_noise, k=k_structural)
    structural_field = _box_blur(structural_field, k=max(31, int(0.12 * N) | 1))
    basins = _normalize(1.0 - structural_field)
    
    # Blend with current topography
    z_smooth = _box_blur(z_norm, k=k_structural)
    basins_topo = _normalize(1.0 - z_smooth)
    basins_combined = 0.70 * basins + 0.30 * basins_topo
    basins_combined = _box_blur(basins_combined, k=max(15, int(0.08 * N) | 1))
    basins_combined = _normalize(basins_combined)
    
    # Regional dip + undulation for structural trend
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    X = ii * pixel_scale_m
    Y = jj * pixel_scale_m
    az = np.deg2rad(45.0)  # dip direction
    dip = np.deg2rad(5.0)   # dip angle
    ux, uy = np.cos(az), np.sin(az)
    
    plane = np.tan(dip) * (ux * X + uy * Y)
    undul_raw = (fractional_surface(N, beta=3.8, rng=rng)*2 - 1) * 8.0
    undul = _box_blur(undul_raw, k=max(31, int(0.12 * N) | 1))
    bed_struct = plane + undul
    bed_struct_zm = bed_struct - np.mean(bed_struct)
    
    # Reference depth below surface
    Emean = float(E.mean())
    
    # S3: Top of upper sedimentary group (~80-150m below surface)
    burial_S3 = 80.0 + 70.0 * basins_combined
    S3 = Emean - burial_S3 + 0.3 * bed_struct_zm
    S3 = _box_blur(S3, k=max(21, int(0.10 * N) | 1))
    
    # S2: Top of lower sedimentary group (~200-400m below S3)
    gap_S2_S3 = 200.0 + 200.0 * basins_combined
    S2 = S3 - gap_S2_S3
    S2 = S2 + 0.2 * bed_struct_zm
    S2 = _box_blur(S2, k=max(31, int(0.12 * N) | 1))
    
    # S1: Top of old sediments / volcanics (~100-300m below S2)
    gap_S1_S2 = 100.0 + 200.0 * basins_combined
    S1 = S2 - gap_S1_S2
    S1 = S1 + 0.15 * bed_struct_zm
    S1 = _box_blur(S1, k=max(31, int(0.15 * N) | 1))
    
    # S0: Top of basement (deep under basins, shallow under highs)
    basement_depth = 100.0 + 450.0 * basins_combined
    S0 = S1 - basement_depth
    S0 = S0 + 0.1 * bed_struct_zm
    S0 = _box_blur(S0, k=max(41, int(0.18 * N) | 1))  # Very smooth
    
    return {
        'S0_basement': S0,
        'S1_old_sed': S1,
        'S2_lower_group': S2,
        'S3_upper_group': S3,
        'basins': basins_combined,
        'bed_struct': bed_struct_zm,
    }


# ========================================================================================
# FORMATION-SPECIFIC THICKNESS FIELDS (each formation gets its own geology)
# ========================================================================================

def thickness_field_sandstone(shape, rng, basins, slope_norm, z_norm):
    """Sandstone: mid-basin deltaic/fluvial systems."""
    # Environment: favors mid-basin and margins
    mid_basin = (basins > 0.35) & (basins < 0.70)
    margin = (basins > 0.20) & (basins <= 0.35)
    
    env = (1.0 * mid_basin.astype(float) + 
           0.7 * margin.astype(float) +
           0.2 * (basins >= 0.70).astype(float))
    env = _box_blur(env, k=11)
    env = _normalize(env)
    
    # Smooth thickness variation
    noise = smooth_random_field(shape, rng, k_smooth=20, beta=3.5)
    thick = 30.0 + 140.0 * env * _normalize(noise + 1.0)
    
    # Suppress on steep slopes and high elevations
    thick *= (1.0 - slope_norm**1.5)
    thick *= np.clip(1.5 - z_norm, 0.0, 1.0)
    
    # Absence mask: not on very high peaks
    absence = (z_norm > 0.85) | (slope_norm > 0.75)
    thick[absence] = 0.0
    
    return np.maximum(thick, 0.0)


def thickness_field_shale(shape, rng, basins, slope_norm, z_norm):
    """Shale/mudstone: deep basin offshore deposits."""
    # Environment: strongly favors deep basins
    deep = basins > 0.55
    mid = (basins > 0.30) & (basins <= 0.55)
    
    env = (1.0 * deep.astype(float) + 
           0.6 * mid.astype(float) +
           0.15 * (basins <= 0.30).astype(float))
    env = _box_blur(env, k=13)
    env = _normalize(env)
    
    # Smooth thickness variation
    noise = smooth_random_field(shape, rng, k_smooth=25, beta=4.0)
    thick = 60.0 + 280.0 * env * _normalize(noise + 1.0)
    
    # Less slope-sensitive than sandstone
    thick *= (1.0 - slope_norm**1.2)
    thick *= np.clip(1.3 - z_norm, 0.0, 1.0)
    
    # Absence: only on extreme highs
    absence = (z_norm > 0.88) & (basins < 0.15)
    thick[absence] = 0.0
    
    return np.maximum(thick, 0.0)


def thickness_field_limestone(shape, rng, basins, slope_norm, z_norm):
    """Limestone: carbonate platforms on gentle mid-basin areas."""
    # Environment: mid-basin platforms, gentle slopes
    platform = (basins > 0.30) & (basins < 0.65) & (slope_norm < 0.35)
    
    env = platform.astype(float)
    env = _box_blur(env, k=15)
    env = _normalize(env)
    
    # Smooth thickness variation
    noise = smooth_random_field(shape, rng, k_smooth=22, beta=3.8)
    thick = 20.0 + 140.0 * env * _normalize(noise + 1.0)
    
    # Very slope-sensitive (platforms are flat)
    thick *= (1.0 - slope_norm**2.0)
    thick *= np.clip(1.2 - z_norm, 0.0, 1.0)
    
    # Absence: steep or very high/low areas
    absence = (slope_norm > 0.60) | (z_norm > 0.85) | (basins > 0.75)
    thick[absence] = 0.0
    
    return np.maximum(thick, 0.0)


def thickness_field_conglomerate(shape, rng, basins, slope_norm, z_norm):
    """Conglomerate: mountain-front alluvial fans."""
    # Environment: margins of uplifts, moderate slopes
    margin = (basins < 0.35) & (slope_norm > 0.15) & (slope_norm < 0.60)
    
    env = margin.astype(float)
    env = _box_blur(env, k=9)
    env = _normalize(env)
    
    # Patchy thickness (fans are localized)
    noise = smooth_random_field(shape, rng, k_smooth=12, beta=3.0)
    thick = 15.0 + 90.0 * env * _normalize(np.maximum(noise, 0.0))
    
    # Requires moderate elevation (foothill zones)
    thick *= np.clip((z_norm - 0.3) * 2.0, 0.0, 1.0)
    thick *= np.clip((0.75 - z_norm) * 2.0, 0.0, 1.0)
    
    # Absence: deep basins and very high peaks
    absence = (basins > 0.65) | (z_norm > 0.85) | (z_norm < 0.25)
    thick[absence] = 0.0
    
    return np.maximum(thick, 0.0)


def thickness_field_evaporite(shape, rng, basins, slope_norm, z_norm):
    """Evaporite: only in deep, flat basin centers."""
    # VERY restricted environment
    deep_flat = (basins > 0.75) & (slope_norm < 0.12) & (z_norm < 0.30)
    
    env = deep_flat.astype(float)
    env = _box_blur(env, k=11)
    
    # Limited thickness, concentrated at basin center
    noise = smooth_random_field(shape, rng, k_smooth=18, beta=4.0)
    thick = 8.0 + 35.0 * env * _normalize(noise + 1.0)
    
    # Hard absence mask (most of map has NO evaporite)
    thick[~deep_flat] = 0.0
    
    return np.maximum(thick, 0.0)


def thickness_field_dolomite(shape, rng, basins, slope_norm, z_norm):
    """Dolomite: altered limestone, similar distribution but more restricted."""
    # Similar to limestone but slightly more restricted
    platform = (basins > 0.35) & (basins < 0.60) & (slope_norm < 0.30)
    
    env = platform.astype(float)
    env = _box_blur(env, k=13)
    env = _normalize(env)
    
    noise = smooth_random_field(shape, rng, k_smooth=20, beta=3.9)
    thick = 10.0 + 50.0 * env * _normalize(noise + 1.0)
    
    thick *= (1.0 - slope_norm**2.2)
    
    absence = (slope_norm > 0.55) | (z_norm > 0.82) | (basins > 0.70) | (basins < 0.30)
    thick[absence] = 0.0
    
    return np.maximum(thick, 0.0)


def thickness_field_siltstone(shape, rng, basins, slope_norm):
    """Siltstone: intermediate between shale and sandstone."""
    # Moderate basins
    mid = (basins > 0.35) & (basins < 0.65)
    
    env = mid.astype(float) + 0.3 * (basins >= 0.65).astype(float)
    env = _box_blur(env, k=11)
    env = _normalize(env)
    
    noise = smooth_random_field(shape, rng, k_smooth=18, beta=3.7)
    thick = 15.0 + 55.0 * env * _normalize(noise + 1.0)
    
    thick *= (1.0 - slope_norm**1.4)
    
    return np.maximum(thick, 0.0)


def thickness_field_mudstone(shape, rng, basins, slope_norm):
    """Mudstone: like shale but more restricted."""
    deep = basins > 0.60
    
    env = deep.astype(float) + 0.4 * ((basins > 0.40) & (basins <= 0.60)).astype(float)
    env = _box_blur(env, k=13)
    env = _normalize(env)
    
    noise = smooth_random_field(shape, rng, k_smooth=22, beta=3.9)
    thick = 20.0 + 70.0 * env * _normalize(noise + 1.0)
    
    thick *= (1.0 - slope_norm**1.3)
    
    return np.maximum(thick, 0.0)


# ========================================================================================
# MAIN STRATIGRAPHY GENERATOR (proper architecture)
# ========================================================================================

def generate_stratigraphy_fixed(z_norm, rng, elev_range_m=700.0, pixel_scale_m=10.0):
    """
    Generate stratigraphy with proper geological architecture:
    - Major structural surfaces define vertical intervals
    - Each formation has independent thickness field
    - True pinch-outs and absence zones
    - Single upward stacking pass
    """
    N = z_norm.shape[0]
    E = z_norm * elev_range_m
    shape = E.shape
    
    # Topographic derivatives
    dEy, dEx = np.gradient(E, pixel_scale_m, pixel_scale_m)
    slope_mag = np.hypot(dEx, dEy) + 1e-12
    slope_norm = _normalize(slope_mag)
    d2x, _ = np.gradient(dEx)
    _, d2y = np.gradient(dEy)
    laplacian = d2x + d2y
    
    # Generate major structural surfaces
    elev_span = float(E.max() - E.min() + 1e-9)
    surfaces = generate_structural_surfaces(z_norm, E, rng, pixel_scale_m, elev_span)
    
    S0 = surfaces['S0_basement']
    S1 = surfaces['S1_old_sed']
    S2 = surfaces['S2_lower_group']
    S3 = surfaces['S3_upper_group']
    basins = surfaces['basins']
    
    # ========== REGOLITH LAYERS (always from surface downward) ==========
    soil_max = 45.0
    soil_thick = soil_max * (1.0 - slope_norm) * np.clip(1.2 - z_norm, 0.0, 1.0)
    
    topsoil_thick = 0.35 * soil_thick
    subsoil_thick = 0.65 * soil_thick
    
    # Colluvium
    hollows = np.maximum(laplacian, 0.0)
    hollow_strength = _normalize(hollows)
    mid_slope = (z_norm > 0.25) & (z_norm < 0.75)
    colluvium_factor = hollow_strength * mid_slope.astype(float) * (1.0 - slope_norm**2)
    colluvium_factor = _box_blur(colluvium_factor, k=9)
    t_colluvium = 75.0 * _normalize(colluvium_factor)
    
    # Saprolite
    interfluve = (z_norm > 0.35) & (z_norm < 0.80) & (slope_norm < 0.4)
    saprolite_factor = interfluve.astype(float) * (1.0 - slope_norm)
    saprolite_factor = _box_blur(saprolite_factor, k=11)
    t_saprolite = 10.0 + 60.0 * _normalize(saprolite_factor)
    
    # Weathered rind
    rind_texture = fractional_surface(N, beta=3.2, rng=rng)
    t_weathered_rind = 3.0 + 18.0 * rind_texture
    
    # Valley-fill sediments (modern, in current lows)
    k_valley = max(31, int(0.12 * N) | 1)
    z_smooth_valley = _box_blur(z_norm, k=k_valley)
    valley_lows = _normalize(1.0 - z_smooth_valley)
    
    flat_factor = (1.0 - slope_norm)**3
    clay_factor = valley_lows * flat_factor * (z_norm < 0.35)
    clay_factor = _box_blur(clay_factor, k=7)
    t_clay = 50.0 * _normalize(clay_factor)
    
    silt_factor = valley_lows * (1.0 - slope_norm)**2 * (z_norm < 0.45)
    silt_factor = _box_blur(silt_factor, k=7)
    t_silt = 40.0 * _normalize(silt_factor)
    
    sand_factor = valley_lows * (1.0 - slope_norm**1.5) * (z_norm < 0.50)
    sand_noise = rng.lognormal(mean=0.0, sigma=0.35, size=shape)
    sand_factor = sand_factor * sand_noise
    sand_factor = _box_blur(sand_factor, k=7)
    t_sand = 65.0 * _normalize(sand_factor)
    
    # ========== INTERVAL 3: UPPER SEDIMENTARY GROUP [S3 -> S2] ==========
    # Formations in this interval (order matters for stacking)
    upper_group = {}
    
    # Each gets its OWN thickness field
    upper_group['Sandstone'] = thickness_field_sandstone(shape, rng, basins, slope_norm, z_norm)
    upper_group['Conglomerate'] = thickness_field_conglomerate(shape, rng, basins, slope_norm, z_norm)
    upper_group['Siltstone'] = thickness_field_siltstone(shape, rng, basins, slope_norm)
    upper_group['Mudstone'] = thickness_field_mudstone(shape, rng, basins, slope_norm)
    
    # Scale to fit in available space [S3, S2]
    cap_upper = np.maximum(S3 - S2, 1.0)
    total_upper = sum(upper_group.values())
    scale_upper = np.minimum(1.0, cap_upper / np.maximum(total_upper, 1e-3))
    for name in upper_group:
        upper_group[name] *= scale_upper
    
    # ========== INTERVAL 2: LOWER SEDIMENTARY GROUP [S2 -> S1] ==========
    lower_group = {}
    
    lower_group['Shale'] = thickness_field_shale(shape, rng, basins, slope_norm, z_norm)
    lower_group['Limestone'] = thickness_field_limestone(shape, rng, basins, slope_norm, z_norm)
    lower_group['Dolomite'] = thickness_field_dolomite(shape, rng, basins, slope_norm, z_norm)
    lower_group['Evaporite'] = thickness_field_evaporite(shape, rng, basins, slope_norm, z_norm)
    
    # Scale to fit [S2, S1]
    cap_lower = np.maximum(S2 - S1, 1.0)
    total_lower = sum(lower_group.values())
    scale_lower = np.minimum(1.0, cap_lower / np.maximum(total_lower, 1e-3))
    for name in lower_group:
        lower_group[name] *= scale_lower
    
    # ========== INTERVAL 1: OLD SEDIMENTS / VOLCANICS [S1 -> S0] ==========
    # Simple partition: Basalt + minor sediments
    cap_old = np.maximum(S1 - S0, 1.0)
    
    # Basalt: patchy distribution
    basalt_factor = basins * (1.0 - basins) * (slope_norm < 0.3)
    basalt_noise = smooth_random_field(shape, rng, k_smooth=25, beta=3.5)
    t_basalt = 20.0 + 80.0 * _normalize(basalt_factor * _normalize(basalt_noise + 1.0))
    
    # Ancient crust (remainder of interval)
    t_ancient_crust = np.maximum(cap_old - t_basalt, 5.0)
    
    # ========== INTERVAL 0: BASEMENT [S0 -> floor] ==========
    # Basement depth varies with structure
    basement_total = np.maximum(S0 - (S0.min() - 100.0), 10.0)
    
    # Simple partition
    f_granite = 0.45
    f_gneiss = 0.35
    f_basalt_bsmt = 0.20
    
    t_granite = basement_total * f_granite
    t_gneiss = basement_total * f_gneiss
    t_basement = basement_total * f_basalt_bsmt
    
    # ========== BUILD INTERFACES (SINGLE UPWARD PASS) ==========
    interfaces = {}
    thickness = {}
    
    # Start from basement floor
    z_floor = float(S0.min() - 100.0)
    z = np.full_like(E, z_floor)
    
    # Basement interval
    for name, t in [('BasementFloor', np.zeros_like(E)),
                    ('Basement', t_basement),
                    ('Gneiss', t_gneiss),
                    ('Granite', t_granite),
                    ('AncientCrust', t_ancient_crust),
                    ('Basalt', t_basalt)]:
        interfaces[name] = z.copy()
        z = z + t
        thickness[name] = t.copy()
    
    # Lower sedimentary group
    for name in ['Evaporite', 'Dolomite', 'Limestone', 'Shale']:
        t = lower_group[name]
        interfaces[name] = z.copy()
        z = z + t
        thickness[name] = t.copy()
    
    # Upper sedimentary group
    for name in ['Mudstone', 'Siltstone', 'Conglomerate', 'Sandstone']:
        t = upper_group[name]
        interfaces[name] = z.copy()
        z = z + t
        thickness[name] = t.copy()
    
    # Valley-fill sediments (must stay below surface)
    # Clip to ensure they don't exceed surface
    z_valley_base = z.copy()
    available_to_surface = E - z_valley_base
    
    # Scale valley fill if needed
    total_valley = t_sand + t_silt + t_clay
    scale_valley = np.minimum(1.0, available_to_surface / np.maximum(total_valley, 1e-3))
    t_sand *= scale_valley
    t_silt *= scale_valley
    t_clay *= scale_valley
    
    for name, t in [('Sand', t_sand), ('Silt', t_silt), ('Clay', t_clay)]:
        interfaces[name] = z.copy()
        z = z + t
        thickness[name] = t.copy()
    
    # Regolith (work down from surface)
    z_regolith = E.copy()
    
    for name, t in [('Topsoil', topsoil_thick),
                    ('Subsoil', subsoil_thick),
                    ('Colluvium', t_colluvium),
                    ('Saprolite', t_saprolite),
                    ('WeatheredBR', t_weathered_rind)]:
        interfaces[name] = z_regolith.copy()
        z_regolith = z_regolith - t
        thickness[name] = t.copy()
    
    # Ensure regolith doesn't go below sedimentary rocks
    # (In case of thin sediment cover)
    interfaces['WeatheredBR'] = np.maximum(interfaces['WeatheredBR'], z + 0.1)
    
    # Material properties
    properties = {
        "Topsoil": {"erodibility": 1.00, "density": 1600, "porosity": 0.45},
        "Subsoil": {"erodibility": 0.85, "density": 1700, "porosity": 0.40},
        "Colluvium": {"erodibility": 0.90, "density": 1750, "porosity": 0.35},
        "Clay": {"erodibility": 0.80, "density": 1850, "porosity": 0.45},
        "Silt": {"erodibility": 0.90, "density": 1750, "porosity": 0.42},
        "Sand": {"erodibility": 0.85, "density": 1700, "porosity": 0.35},
        "Saprolite": {"erodibility": 0.70, "density": 1900, "porosity": 0.30},
        "WeatheredBR": {"erodibility": 0.55, "density": 2100, "porosity": 0.20},
        "Sandstone": {"erodibility": 0.30, "density": 2200, "porosity": 0.18},
        "Conglomerate": {"erodibility": 0.25, "density": 2300, "porosity": 0.16},
        "Shale": {"erodibility": 0.45, "density": 2300, "porosity": 0.12},
        "Mudstone": {"erodibility": 0.45, "density": 2300, "porosity": 0.12},
        "Siltstone": {"erodibility": 0.35, "density": 2350, "porosity": 0.10},
        "Limestone": {"erodibility": 0.28, "density": 2400, "porosity": 0.08},
        "Dolomite": {"erodibility": 0.24, "density": 2450, "porosity": 0.06},
        "Evaporite": {"erodibility": 0.90, "density": 2200, "porosity": 0.15},
        "Granite": {"erodibility": 0.15, "density": 2700, "porosity": 0.01},
        "Gneiss": {"erodibility": 0.16, "density": 2750, "porosity": 0.01},
        "Basalt": {"erodibility": 0.12, "density": 2950, "porosity": 0.02},
        "AncientCrust": {"erodibility": 0.14, "density": 2800, "porosity": 0.01},
        "Basement": {"erodibility": 0.15, "density": 2700, "porosity": 0.01},
        "BasementFloor": {"erodibility": 0.02, "density": 2850, "porosity": 0.005},
    }
    
    return {
        "surface_elev": E,
        "interfaces": interfaces,
        "thickness": thickness,
        "properties": properties,
        "meta": {
            "elev_range_m": elev_range_m,
            "pixel_scale_m": pixel_scale_m,
            "basins": basins,
            "surfaces": surfaces,
            "z_floor": z_floor,
        }
    }


# ========================================================================================
# VISUALIZATION
# ========================================================================================

MAX_SECTION_DEPTH_M = 800.0

def plot_cross_section(strata, row=None, col=None, min_draw_thickness=0.05, ax=None):
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

    surf_min = float(np.nanmin(surf))
    surf_rel = surf - surf_min
    tops_rel = {k: v - surf_min for k, v in tops.items()}

    order = [
        "Topsoil", "Subsoil", "Colluvium", "Saprolite", "WeatheredBR",
        "Clay", "Silt", "Sand",
        "Sandstone", "Conglomerate", "Mudstone", "Siltstone",
        "Shale", "Limestone", "Dolomite", "Evaporite",
        "Basalt", "AncientCrust", "Granite", "Gneiss", "Basement", "BasementFloor",
    ]

    color_map = {
        "Topsoil": "sienna", "Subsoil": "peru", "Colluvium": "burlywood",
        "Saprolite": "khaki", "WeatheredBR": "darkkhaki",
        "Clay": "lightcoral", "Silt": "thistle", "Sand": "gold",
        "Sandstone": "orange", "Conglomerate": "chocolate",
        "Mudstone": "rosybrown", "Siltstone": "lightsteelblue",
        "Shale": "slategray", "Limestone": "lightgray",
        "Dolomite": "gainsboro", "Evaporite": "plum",
        "Basalt": "royalblue", "AncientCrust": "darkseagreen",
        "Granite": "lightpink", "Gneiss": "violet",
        "Basement": "dimgray", "BasementFloor": "black",
    }

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 5.5))

    handled_labels = set()

    for i in range(len(order) - 1, 0, -1):
        above, here = order[i - 1], order[i]
        if above not in tops_rel or here not in tops_rel:
            continue

        y_top = tops_rel[above]
        y_bot = tops_rel[here]
        y_bot_vis = np.where((y_top - y_bot) < min_draw_thickness, y_top - min_draw_thickness, y_bot)

        color = color_map.get(here, None)
        label = here if here not in handled_labels else None

        ax.fill_between(x, y_bot_vis, y_top, alpha=0.9, linewidth=0.6, zorder=5 + i,
                        color=color, label=label)
        if label is not None:
            handled_labels.add(label)

    ax.plot(x, surf_rel, linewidth=2.4, zorder=50, color="black", label="Surface")

    surf_top_rel = float(np.nanmax(surf_rel))
    margin = 0.05 * (surf_top_rel + MAX_SECTION_DEPTH_M)
    ax.set_ylim(-MAX_SECTION_DEPTH_M, surf_top_rel + margin)

    ax.set_title("Stratigraphic Cross-Section (FIXED: Proper Geological Architecture)")
    ax.set_xlabel(axis_label)
    ax.set_ylabel("Elevation relative to lowest surface (m)")
    ax.legend(ncol=1, fontsize=8, framealpha=0.95, loc="center left", bbox_to_anchor=(1.02, 0.5))

    if ax is None:
        plt.tight_layout()
        plt.show()

    return ax


def plot_cross_sections_xy(strata, row=None, col=None):
    N = strata["surface_elev"].shape[0]
    if row is None: row = N // 2
    if col is None: col = N // 2
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 11.5), constrained_layout=True)
    plot_cross_section(strata, row=row, ax=ax1)
    plot_cross_section(strata, col=col, ax=ax2)
    plt.show()


# ========================================================================================
# MAIN
# ========================================================================================

if __name__ == "__main__":
    print("="*70)
    print("FIXED TERRAIN GENERATOR - Proper Geological Architecture")
    print("="*70)
    
    z, rng = quantum_seeded_topography(N=512, beta=3.2, random_seed=None)
    
    strata = generate_stratigraphy_fixed(z_norm=z, rng=rng)
    
    # Re-zero datum
    E = strata["surface_elev"]
    offset = float(E.min())
    strata["surface_elev"] = E - offset
    for name, arr in strata["interfaces"].items():
        strata["interfaces"][name] = arr - offset
    
    # Visualize
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(z, cmap='terrain', origin='lower', interpolation='bilinear')
    ax.set_title("Quantum-Seeded Topography")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Normalized elevation")
    plt.tight_layout()
    plt.show()
    
    plot_cross_sections_xy(strata)
    
    # Diagnostics
    print("\n" + "="*70)
    print("LAYER STATISTICS")
    print("="*70)
    for layer in ["Sand", "Sandstone", "Conglomerate", "Shale", "Limestone", "Evaporite"]:
        if layer in strata["thickness"]:
            t = strata["thickness"][layer]
            present = (t > 0.1).sum()
            total = t.size
            pct = 100.0 * present / total
            print(f"{layer:15s}: mean={t.mean():6.2f}m  max={t.max():6.2f}m  present={pct:5.1f}%")
    
    print("="*70)
    print("✅ FIXED GENERATOR COMPLETE - No more striped cake!")
    print("="*70)
