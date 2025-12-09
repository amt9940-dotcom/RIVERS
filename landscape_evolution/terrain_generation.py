"""
Terrain Generation Module

This module preserves your existing terrain generation code and makes it
compatible with the new landscape evolution framework.

The main functions:
- quantum_seeded_topography: Generate initial terrain
- fractional_surface: Power-law spectrum terrain
- domain_warp: Add micro-relief
- ridged_mix: Ridge/valley sharpening
- compute_topo_fields: Extract topographic features
"""

import numpy as np
import os
import time
import hashlib
from typing import Tuple, Optional, Dict


# Check for quantum RNG support
try:
    from qiskit import QuantumCircuit
    try:
        from qiskit_aer import Aer
        HAVE_QISKIT = True
    except:
        try:
            from qiskit import Aer
            HAVE_QISKIT = True
        except:
            HAVE_QISKIT = False
except:
    HAVE_QISKIT = False


def qrng_uint32(n: int, nbits: int = 32) -> np.ndarray:
    """
    Return n uint32 from Qiskit Aer if available; else PRNG fallback.
    
    Parameters
    ----------
    n : int
        Number of random integers
    nbits : int
        Number of bits per integer
        
    Returns
    -------
    np.ndarray
        Array of random uint32
    """
    if not HAVE_QISKIT:
        return np.random.default_rng().integers(0, 2**32, size=n, dtype=np.uint32)
    
    from qiskit import QuantumCircuit
    try:
        from qiskit_aer import Aer
    except:
        try:
            from qiskit import Aer
        except:
            return np.random.default_rng().integers(0, 2**32, size=n, dtype=np.uint32)
    
    qc = QuantumCircuit(nbits, nbits)
    qc.h(range(nbits))
    qc.measure(range(nbits), range(nbits))
    backend = Aer.get_backend("qasm_simulator")
    seed_sim = int.from_bytes(os.urandom(4), "little")
    job = backend.run(qc, shots=n, memory=True, seed_simulator=seed_sim)
    mem = job.result().get_memory(qc)
    return np.array([np.uint32(int(bits[::-1], 2)) for bits in mem], dtype=np.uint32)


def rng_from_qrng(n_seeds: int = 4, random_seed: Optional[int] = None) -> np.random.Generator:
    """
    Random per run if random_seed=None; reproducible if you pass an int.
    
    Parameters
    ----------
    n_seeds : int
        Number of quantum seeds to mix
    random_seed : int, optional
        If provided, use as deterministic seed instead of quantum
        
    Returns
    -------
    np.random.Generator
        Seeded random number generator
    """
    if random_seed is not None:
        return np.random.default_rng(int(random_seed))
    
    seeds = qrng_uint32(n_seeds).tobytes()
    mix = seeds + os.urandom(16) + int(time.time_ns()).to_bytes(8, "little")
    h = hashlib.blake2b(mix, digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(h, "little"))


def fractional_surface(N: int, beta: float = 3.1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Power-law spectrum; higher beta => smoother large-scale terrain.
    
    Parameters
    ----------
    N : int
        Grid size (N × N)
    beta : float
        Power law exponent (higher = smoother)
    rng : np.random.Generator, optional
        Random number generator
        
    Returns
    -------
    np.ndarray
        Normalized terrain (0-1)
    """
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
    """
    Bilinear interpolation with periodic boundaries.
    
    Parameters
    ----------
    img : np.ndarray
        2D image to sample
    X, Y : np.ndarray
        Sample coordinates
        
    Returns
    -------
    np.ndarray
        Interpolated values
    """
    N = img.shape[0]
    x0 = np.floor(X).astype(int) % N
    y0 = np.floor(Y).astype(int) % N
    x1 = (x0 + 1) % N
    y1 = (y0 + 1) % N
    dx = X - np.floor(X)
    dy = Y - np.floor(Y)
    return ((1-dx)*(1-dy)*img[x0, y0] + dx*(1-dy)*img[x1, y0] +
            (1-dx)*dy*img[x0, y1] + dx*dy*img[x1, y1])


def domain_warp(z: np.ndarray, rng: np.random.Generator, amp: float = 0.12, beta: float = 3.0) -> np.ndarray:
    """
    Coordinate distortion; amp↑ => gnarlier micro-relief.
    
    Parameters
    ----------
    z : np.ndarray
        Input terrain
    rng : np.random.Generator
        Random number generator
    amp : float
        Warp amplitude
    beta : float
        Warp smoothness
        
    Returns
    -------
    np.ndarray
        Warped terrain
    """
    N = z.shape[0]
    u = fractional_surface(N, beta=beta, rng=rng)*2 - 1
    v = fractional_surface(N, beta=beta, rng=rng)*2 - 1
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    Xw = (ii + amp*N*u) % N
    Yw = (jj + amp*N*v) % N
    return bilinear_sample(z, Xw, Yw)


def ridged_mix(z: np.ndarray, alpha: float = 0.18) -> np.ndarray:
    """
    Ridge/valley sharpening; alpha↑ => craggier.
    
    Parameters
    ----------
    z : np.ndarray
        Input terrain (0-1)
    alpha : float
        Ridge strength
        
    Returns
    -------
    np.ndarray
        Sharpened terrain (0-1)
    """
    ridged = 1.0 - np.abs(2.0*z - 1.0)
    out = (1-alpha)*z + alpha*ridged
    lo, hi = np.percentile(out, [2, 98])
    return np.clip((out - lo)/(hi - lo + 1e-12), 0, 1)


def lowpass2d(z: np.ndarray, cutoff: Optional[float] = None, rolloff: float = 0.08) -> np.ndarray:
    """
    Set cutoff (0..0.5) for smoothing; None disables.
    
    Parameters
    ----------
    z : np.ndarray
        Input terrain
    cutoff : float, optional
        Cutoff frequency (None = no filtering)
    rolloff : float
        Rolloff width
        
    Returns
    -------
    np.ndarray
        Filtered terrain
    """
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


def gaussian_blur(z: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
    """
    Optional small Gaussian blur (last-mile softness; default: off).
    
    Parameters
    ----------
    z : np.ndarray
        Input terrain
    sigma : float, optional
        Blur standard deviation (None = no blur)
        
    Returns
    -------
    np.ndarray
        Blurred terrain
    """
    if sigma is None or sigma <= 0:
        return z
    
    rad = int(np.ceil(3*sigma))
    x = np.arange(-rad, rad+1)
    g = np.exp(-0.5*(x/sigma)**2)
    g /= g.sum()
    
    tmp = np.zeros_like(z)
    for i, w in enumerate(g):
        tmp += w*np.roll(z, i-rad, axis=1)
    
    out = np.zeros_like(z)
    for i, w in enumerate(g):
        out += w*np.roll(tmp, i-rad, axis=0)
    
    lo, hi = np.percentile(out, [2, 98])
    return np.clip((out - lo)/(hi - lo + 1e-12), 0, 1)


def quantum_seeded_topography(
    N: int = 512,
    beta: float = 3.1,
    warp_amp: float = 0.12,
    ridged_alpha: float = 0.18,
    random_seed: Optional[int] = None,
    smooth_cutoff: Optional[float] = None,
    smooth_rolloff: float = 0.08,
    post_blur_sigma: Optional[float] = None
) -> Tuple[np.ndarray, np.random.Generator]:
    """
    Generate terrain using quantum-seeded (or pseudo-random) process.
    
    This is your existing terrain generation pipeline, preserved.
    
    Parameters
    ----------
    N : int
        Grid size (N × N)
    beta : float
        Power law exponent (smoothness)
    warp_amp : float
        Domain warp amplitude
    ridged_alpha : float
        Ridge sharpening strength
    random_seed : int, optional
        Random seed (None = quantum seeded)
    smooth_cutoff : float, optional
        Low-pass filter cutoff
    smooth_rolloff : float
        Low-pass filter rolloff
    post_blur_sigma : float, optional
        Post-processing blur
        
    Returns
    -------
    z : np.ndarray
        Normalized terrain (0-1), shape (N, N)
    rng : np.random.Generator
        The RNG used (for reproducibility)
    """
    rng = rng_from_qrng(n_seeds=4, random_seed=random_seed)
    
    # Multi-scale base
    base_low = fractional_surface(N, beta=beta, rng=rng)
    base_high = fractional_surface(N, beta=beta-0.4, rng=rng)
    z = 0.65*base_low + 0.35*base_high
    
    # Warp for micro-relief
    z = domain_warp(z, rng=rng, amp=warp_amp, beta=beta)
    
    # Sharpen ridges
    z = ridged_mix(z, alpha=ridged_alpha)
    
    # Optional smoothing
    z = lowpass2d(z, cutoff=smooth_cutoff, rolloff=smooth_rolloff)
    z = gaussian_blur(z, sigma=post_blur_sigma)
    
    return z, rng


def _normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize to [0, 1] using 2nd/98th percentiles."""
    lo, hi = np.percentile(x, [2, 98])
    return np.clip((x - lo)/(hi - lo + eps), 0.0, 1.0)


def compute_topo_fields(surface_elev: np.ndarray, pixel_scale_m: float) -> Dict[str, np.ndarray]:
    """
    Basic topographic fields from elevation only.
    
    Returns dict with:
      E, E_norm          : elevation (m) and normalized (0..1)
      dEx, dEy           : gradients in x (cols) and y (rows) (m/m)
      slope_mag, slope_norm
      aspect             : downslope direction (radians, 0 = +x)
      laplacian          : convex/concave indicator
    
    Parameters
    ----------
    surface_elev : np.ndarray
        Surface elevation (m)
    pixel_scale_m : float
        Grid spacing (m)
        
    Returns
    -------
    dict
        Dictionary of topographic fields
    """
    E = surface_elev
    E_norm = _normalize(E)
    
    # Gradient: np.gradient returns [d/drow, d/dcol] = [y, x]
    dEy, dEx = np.gradient(E, pixel_scale_m, pixel_scale_m)
    slope_mag = np.hypot(dEx, dEy) + 1e-12
    slope_norm = _normalize(slope_mag)
    
    # Downslope aspect (for windward/leeward logic)
    aspect = np.arctan2(-dEy, -dEx)
    
    # Simple 4-neighbor Laplacian: <0 convex (ridge), >0 concave (valley)
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


def denormalize_elevation(
    z_norm: np.ndarray,
    elev_range_m: Tuple[float, float] = (0.0, 2000.0)
) -> np.ndarray:
    """
    Convert normalized terrain [0, 1] to actual elevation in meters.
    
    Parameters
    ----------
    z_norm : np.ndarray
        Normalized terrain (0-1)
    elev_range_m : tuple
        (min_elev, max_elev) in meters
        
    Returns
    -------
    np.ndarray
        Elevation in meters
    """
    z_min, z_max = elev_range_m
    return z_min + z_norm * (z_max - z_min)
