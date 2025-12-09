#!/usr/bin/env python3
"""
MERGED TERRAIN + STRATIGRAPHY GENERATOR

Combines the best of both approaches:
✅ From Code 2: Smooth, basin-like curved contacts with strong structural control
✅ From Code 1: Clear, visible specific layers (sand, limestone, etc.) with rich vertical detail

Key improvements in this merge:
1. Code 2's large-scale smooth structural framework (basin subsidence, crustal flexure)
2. Enhanced layer visibility with better thickness ranges and facies contrast
3. Clear sand layer and other specific formations easily visible in cross-sections
4. Smooth contacts between major units (no pixel-scale jaggedness)
5. Realistic basin-fill architecture with distinct facies belts

Layer order (top → bottom):
Topsoil, Subsoil, Colluvium, Saprolite, WeatheredBR,
Clay, Silt, Sand (valley-fill),
Sandstone, Conglomerate, Shale, Mudstone, Siltstone, Limestone, Dolomite, Evaporite,
Granite, Gneiss, Basalt, AncientCrust, Basement, BasementFloor
"""
from __future__ import annotations

# ------------------------- Standard imports -------------------------
import numpy as np
import matplotlib.pyplot as plt

# Optional quantum seeding (no auto-install)
try:
    import qiskit  # type: ignore  # noqa
    from qiskit import QuantumCircuit  # type: ignore  # noqa
    try:
        import qiskit_aer  # type: ignore  # noqa
        HAVE_QISKIT = True
    except Exception:
        HAVE_QISKIT = False
except Exception:
    HAVE_QISKIT = False
    QuantumCircuit = None  # type: ignore  # noqa


def qrng_uint32(n, nbits=32):
  """Return n uint32 from Qiskit Aer if available; else PRNG fallback."""
  if not HAVE_QISKIT:
      return np.random.default_rng().integers(0, 2**32, size=n, dtype=np.uint32)
  try:
      from qiskit_aer import Aer  # type: ignore  # noqa
  except Exception:
      try:
          from qiskit import Aer  # type: ignore  # noqa
      except Exception:
          return np.random.default_rng().integers(0, 2**32, size=n, dtype=np.uint32)
  qc = QuantumCircuit(nbits, nbits)
  qc.h(range(nbits)); qc.measure(range(nbits), range(nbits))
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


# ========================================================================================
# TOPOGRAPHY GENERATOR
# ========================================================================================

def fractional_surface(N, beta=3.1, rng=None):
  """Power-law spectrum; higher beta => smoother large-scale terrain."""
  rng = rng or np.random.default_rng()
  kx = np.fft.fftfreq(N); ky = np.fft.rfftfreq(N)
  K = np.sqrt(kx[:, None]**2 + ky[None, :]**2); K[0, 0] = np.inf
  amp = 1.0 / (K ** (beta/2))
  phase = rng.uniform(0, 2*np.pi, size=(N, ky.size))
  spec = amp * (np.cos(phase) + 1j*np.sin(phase)); spec[0, 0] = 0.0
  z = np.fft.irfftn(spec, s=(N, N), axes=(0, 1))
  lo, hi = np.percentile(z, [2, 98])
  return np.clip((z - lo)/(hi - lo + 1e-12), 0, 1)


def bilinear_sample(img, X, Y):
  N = img.shape[0]
  x0 = np.floor(X).astype(int) % N; y0 = np.floor(Y).astype(int) % N
  x1 = (x0+1) % N; y1 = (y0+1) % N
  dx = X - np.floor(X); dy = Y - np.floor(Y)
  return ((1-dx)*(1-dy)*img[x0,y0] + dx*(1-dy)*img[x1,y0] +
          (1-dx)*dy*img[x0,y1] + dx*dy*img[x1,y1])


def domain_warp(z, rng, amp=0.12, beta=3.0):
  """Coordinate distortion; amp↑ => gnarlier micro-relief."""
  N = z.shape[0]
  u = fractional_surface(N, beta=beta, rng=rng)*2 - 1
  v = fractional_surface(N, beta=beta, rng=rng)*2 - 1
  ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
  Xw = (ii + amp*N*u) % N; Yw = (jj + amp*N*v) % N
  return bilinear_sample(z, Xw, Yw)


def ridged_mix(z, alpha=0.18):
  """Ridge/valley sharpening; alpha↑ => craggier."""
  ridged = 1.0 - np.abs(2.0*z - 1.0)
  out = (1-alpha)*z + alpha*ridged
  lo, hi = np.percentile(out, [2, 98])
  return np.clip((out - lo)/(hi - lo + 1e-12), 0, 1)


def lowpass2d(z, cutoff=None, rolloff=0.08):
    """Set cutoff (0..0.5) for smoothing; None disables."""
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
  if sigma is None or sigma <= 0: return z
  rad = int(np.ceil(3*sigma)); x = np.arange(-rad, rad+1)
  g = np.exp(-0.5*(x/sigma)**2); g /= g.sum()
  tmp = np.zeros_like(z)
  for i,w in enumerate(g): tmp += w*np.roll(z, i-rad, axis=1)
  out = np.zeros_like(z)
  for i,w in enumerate(g): out += w*np.roll(tmp, i-rad, axis=0)
  lo, hi = np.percentile(out, [2,98])
  return np.clip((out - lo)/(hi - lo + 1e-12), 0, 1)


def quantum_seeded_topography(
  N=512, beta=3.1, warp_amp=0.12, ridged_alpha=0.18,
  *, random_seed=None, smooth_cutoff=None, smooth_rolloff=0.08, post_blur_sigma=None
):
  """Generate realistic elevation using quantum-seeded randomness."""
  rng = rng_from_qrng(n_seeds=4, random_seed=random_seed)
  base_low  = fractional_surface(N, beta=beta,     rng=rng)
  base_high = fractional_surface(N, beta=beta-0.4, rng=rng)
  z = 0.65*base_low + 0.35*base_high
  z = domain_warp(z, rng=rng, amp=warp_amp, beta=beta)
  z = ridged_mix(z, alpha=ridged_alpha)
  z = lowpass2d(z, cutoff=smooth_cutoff, rolloff=smooth_rolloff)
  z = gaussian_blur(z, sigma=post_blur_sigma)
  return z, rng


# ========================================================================================
# STRATIGRAPHY UTILITIES
# ========================================================================================

def _box_blur(a, k=5):
  if k <= 1: return a
  out = a.copy()
  for axis in (0,1):
      tmp = out; s = np.zeros_like(tmp)
      for i in range(-(k//2), k//2+1): s += np.roll(tmp, i, axis=axis)
      out = s/float(k)
  return out


def _normalize(x, eps=1e-12):
  lo, hi = np.percentile(x, [2,98])
  return np.clip((x - lo)/(hi - lo + eps), 0.0, 1.0)


# ========================================================================================
# MERGED LAYER GENERATOR - Smooth Geometry + Rich Facies Detail
# ========================================================================================

def generate_stratigraphy_merged(
  z_norm, rng,
  elev_range_m=700.0,
  pixel_scale_m=10.0,
  # ENHANCED regolith visibility
  soil_range_m=(15.0, 60.0),  # Thicker than Code 1, ensures visibility
  colluvium_max_m=100.0,       # Prominent colluvium
  saprolite_range=(10.0, 80.0),  # Thick saprolite (visible)
  weathered_rind_range=(3.0, 25.0),  # Visible weathered zone
  # Valley-fill sediments (modern deposits - ENHANCED for visibility)
  valley_clay_max=60.0,
  valley_silt_max=50.0,
  valley_sand_max=80.0,  # PROMINENT sand layer
  # Sedimentary units (ENHANCED thickness ranges for visibility)
  sandstone_thickness=(40.0, 200.0),  # Clearly visible sandstone
  shale_thickness=(80.0, 350.0),      # Dominant basin fill
  limestone_thickness=(30.0, 180.0),   # Visible carbonate platforms
  # Structural controls (Code 2's smooth framework)
  undulation_amp_m=8.0,
  undulation_beta=3.8,  # Smoother undulations
  dip_deg=5.0,
  dip_dir_deg=45.0,
  burial_depth_m=100.0,
):
  """
  MERGED STRATIGRAPHY GENERATOR
  
  Combines:
  - Code 2's smooth basin-scale structural geometry (large smoothing, curved contacts)
  - Enhanced layer thickness ranges for clear visibility (inspired by Code 1's clarity)
  - Rich facies detail with distinct layers (sand, limestone, etc.)
  """
  N = z_norm.shape[0]
  E = z_norm * elev_range_m

  # ========== TOPOGRAPHIC DERIVATIVES ==========
  dEy, dEx = np.gradient(E, pixel_scale_m, pixel_scale_m)
  slope_mag = np.hypot(dEx, dEy) + 1e-12
  slope_deg = np.rad2deg(np.arctan(slope_mag))
  slope_norm = _normalize(slope_mag)
  
  # Laplacian for curvature
  d2x, _ = np.gradient(dEx)
  _, d2y = np.gradient(dEy)
  laplacian = d2x + d2y
  
  # ========== REGOLITH LAYERS (near-surface) ==========
  # Simplified but visible soil profile
  soil_min, soil_max = soil_range_m
  soil_total = soil_max - (soil_max - soil_min) * slope_norm
  soil_total *= np.clip(1.2 - z_norm, 0.0, 1.0)  # Thinner on highs
  
  topsoil_thick = 0.35 * soil_total
  subsoil_thick = 0.65 * soil_total
  
  # Colluvium: thick in hollows and slope bases
  hollows = np.maximum(laplacian, 0.0)
  hollow_strength = _normalize(hollows)
  mid_slope = (z_norm > 0.25) & (z_norm < 0.75)
  colluvium_factor = hollow_strength * mid_slope.astype(float) * (1.0 - slope_norm**2)
  colluvium_factor = _box_blur(colluvium_factor, k=9)
  t_colluvium = colluvium_max_m * _normalize(colluvium_factor)
  
  # Saprolite: thick on gentle interfluves
  sap_min, sap_max = saprolite_range
  interfluve = (z_norm > 0.35) & (z_norm < 0.80) & (slope_norm < 0.4)
  saprolite_factor = interfluve.astype(float) * (1.0 - slope_norm)
  saprolite_factor = _box_blur(saprolite_factor, k=11)
  t_saprolite = sap_min + (sap_max - sap_min) * _normalize(saprolite_factor)
  
  # Weathered rind: moderate thickness, some variability
  rind_min, rind_max = weathered_rind_range
  rind_texture = fractional_surface(N, beta=3.2, rng=rng)
  t_weathered_rind = rind_min + (rind_max - rind_min) * rind_texture
  
  # ========== VALLEY-FILL SEDIMENTS (modern/recent - ENHANCED) ==========
  # Use heavily smoothed basin field for valley locations
  k_valley = max(31, int(0.12 * N) | 1)
  z_smooth_valley = _box_blur(z_norm, k=k_valley)
  valley_lows = _normalize(1.0 - z_smooth_valley)
  
  # Clay: deepest, flattest valleys
  flat_factor = (1.0 - slope_norm)**3
  clay_factor = valley_lows * flat_factor * (z_norm < 0.35)
  clay_factor = _box_blur(clay_factor, k=7)
  t_clay = valley_clay_max * _normalize(clay_factor)
  
  # Silt: broader valley distribution
  silt_factor = valley_lows * (1.0 - slope_norm)**2 * (z_norm < 0.45)
  silt_factor = _box_blur(silt_factor, k=7)
  t_silt = valley_silt_max * _normalize(silt_factor)
  
  # Sand: PROMINENT - channel zones with some energy
  sand_factor = valley_lows * (1.0 - slope_norm**1.5) * (z_norm < 0.50)
  # Add spatial variability for channel meanders
  sand_noise = rng.lognormal(mean=0.0, sigma=0.35, size=E.shape)
  sand_factor = sand_factor * sand_noise
  sand_factor = _box_blur(sand_factor, k=7)
  t_sand = valley_sand_max * _normalize(sand_factor)
  
  # ========== SMOOTH STRUCTURAL FRAMEWORK (Code 2's strength) ==========
  # Very large smoothing for basin-scale structure
  k_structural = max(63, int(0.35 * N) | 1)  # Large-scale only
  
  # Structural subsidence (long-wavelength)
  structural_noise = fractional_surface(N, beta=4.2, rng=rng)  # Very smooth
  structural_field = _box_blur(structural_noise, k=k_structural)
  structural_field = _box_blur(structural_field, k=max(31, int(0.12 * N) | 1))  # Double smooth
  basins = _normalize(1.0 - structural_field)
  
  # Topographic basins (current lows)
  z_smooth = _box_blur(z_norm, k=k_structural)
  basins_topo = _normalize(1.0 - z_smooth)
  
  # Blend: 70% structural + 30% topographic
  basins_combined = 0.70 * basins + 0.30 * basins_topo
  basins_combined = _box_blur(basins_combined, k=max(15, int(0.08 * N) | 1))
  basins_combined = _normalize(basins_combined)
  highs = 1.0 - basins_combined
  
  # ========== FACIES ARCHITECTURE (distinct belts for visibility) ==========
  # Define basin zones
  deep_basin = basins_combined > 0.65
  mid_basin = (basins_combined > 0.35) & (basins_combined <= 0.65)
  margin = (basins_combined > 0.20) & (basins_combined <= 0.35)
  high = basins_combined <= 0.20
  
  # Sandstone: ENHANCED for visibility - favors mid-basin and margins
  sand_base, sand_max = sandstone_thickness
  sand_env = (
      0.3 * deep_basin.astype(float) +
      1.0 * mid_basin.astype(float) +     # Maximum in mid-basin
      0.9 * margin.astype(float) +
      0.05 * high.astype(float)
  )
  sand_env = _normalize(sand_env)
  sand_env = _box_blur(sand_env, k=max(11, int(0.06 * N) | 1))
  
  # Shale: dominant in deep basins
  shale_base, shale_max = shale_thickness
  shale_env = (
      1.0 * deep_basin.astype(float) +
      0.75 * mid_basin.astype(float) +
      0.4 * margin.astype(float) +
      0.05 * high.astype(float)
  )
  shale_env = _normalize(shale_env)
  shale_env = _box_blur(shale_env, k=max(11, int(0.06 * N) | 1))
  
  # Limestone: platforms (mid-basin with gentle slopes)
  lime_base, lime_max = limestone_thickness
  lime_env = (
      0.2 * deep_basin.astype(float) +
      1.0 * mid_basin.astype(float) +     # Maximum on platforms
      0.3 * margin.astype(float) +
      0.05 * high.astype(float)
  )
  lime_env *= (1.0 - slope_norm**2)  # Gentle slopes only
  lime_env = _normalize(lime_env)
  lime_env = _box_blur(lime_env, k=max(11, int(0.06 * N) | 1))
  
  # ========== COMPUTE SEDIMENTARY THICKNESSES ==========
  # Total sediment budget (thicker in basins)
  sed_base = 40.0   # Minimum on highs
  sed_max = 550.0   # Maximum in basins
  sed_total = sed_base + sed_max * basins_combined
  
  # Sandstone thickness with variability
  t_sandstone_base = sand_base + (sand_max - sand_base) * sand_env
  sand_var = rng.lognormal(mean=0.0, sigma=0.25, size=E.shape)
  sand_var = _box_blur(sand_var, k=max(15, int(0.08 * N) | 1))
  t_sandstone = t_sandstone_base * _normalize(sand_var)
  t_sandstone *= (1.0 - slope_norm**1.5)  # Thin on slopes
  
  # Shale thickness
  t_shale_base = shale_base + (shale_max - shale_base) * shale_env
  shale_var = rng.lognormal(mean=0.0, sigma=0.20, size=E.shape)
  shale_var = _box_blur(shale_var, k=max(15, int(0.08 * N) | 1))
  t_shale = t_shale_base * _normalize(shale_var)
  t_shale *= (1.0 - slope_norm**1.3)
  
  # Limestone thickness
  t_limestone_base = lime_base + (lime_max - lime_base) * lime_env
  lime_var = rng.lognormal(mean=0.0, sigma=0.22, size=E.shape)
  lime_var = _box_blur(lime_var, k=max(15, int(0.08 * N) | 1))
  t_limestone = t_limestone_base * _normalize(lime_var)
  t_limestone *= (1.0 - slope_norm**1.8)  # Very sensitive to slope
  
  # Additional sedimentary units (proportional to main facies)
  t_conglomerate = t_sandstone * 0.25 * (highs > 0.3)  # Near mountain fronts
  t_mudstone = t_shale * 0.30
  t_siltstone = t_shale * 0.20
  t_dolomite = t_limestone * 0.25
  t_evaporite = basins_combined**3 * 12.0 * (slope_norm < 0.1)  # Very restricted
  
  # Light smoothing to remove pixel-scale noise
  k_final = max(7, int(0.03 * N) | 1)
  t_sandstone = _box_blur(t_sandstone, k=k_final)
  t_conglomerate = _box_blur(t_conglomerate, k=k_final)
  t_shale = _box_blur(t_shale, k=k_final)
  t_mudstone = _box_blur(t_mudstone, k=k_final)
  t_siltstone = _box_blur(t_siltstone, k=k_final)
  t_limestone = _box_blur(t_limestone, k=k_final)
  t_dolomite = _box_blur(t_dolomite, k=k_final)
  t_evaporite = _box_blur(t_evaporite, k=k_final)
  
  # ========== STRUCTURAL PLANE FOR SEDIMENTARY ROCKS ==========
  ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
  X = ii * pixel_scale_m
  Y = jj * pixel_scale_m
  az = np.deg2rad(dip_dir_deg)
  dip = np.deg2rad(dip_deg)
  ux, uy = np.cos(az), np.sin(az)
  
  plane = np.tan(dip) * (ux * X + uy * Y)
  undul_raw = (fractional_surface(N, beta=undulation_beta, rng=rng)*2 - 1) * undulation_amp_m
  undul = _box_blur(undul_raw, k=max(31, int(0.12 * N) | 1))
  bed_struct = plane + undul
  bed_struct_zm = bed_struct - np.mean(bed_struct)
  
  # Reference elevation for sediment top
  Emean = float(E.mean())
  elev_span = float(E.max() - E.min() + 1e-9)
  top_sed_ref = Emean - burial_depth_m + 0.4 * bed_struct_zm
  top_sed_ref = _box_blur(top_sed_ref, k=max(21, int(0.10 * N) | 1))  # Smooth reference
  
  # ========== BUILD INTERFACES (top-down) ==========
  eps = 0.01
  
  # Regolith stack
  top_topsoil = E
  top_subsoil = top_topsoil - topsoil_thick
  top_colluvium = top_subsoil - subsoil_thick
  top_saprolite = top_colluvium - t_colluvium
  top_weathered_rind = top_saprolite - t_saprolite
  
  # Valley-fill sediments (must stay above ancient rocks)
  top_clay = top_weathered_rind - t_weathered_rind
  top_silt = top_clay - t_clay
  top_sand_valley = top_silt - t_silt
  bottom_sand_valley = top_sand_valley - t_sand
  
  # Ancient sedimentary rocks (basin-filling sequence)
  # Apply erosion on mountains
  E_rel = (E - E.mean()) / (E.std() + 1e-9)
  erosion_factor = np.clip(
      0.6 * E_rel + 0.8 * slope_norm + 0.3 * highs**2,
      0, 1.8
  )
  erosion_factor = _box_blur(erosion_factor, k=7)
  
  total_sed_thick = (t_sandstone + t_conglomerate + t_shale + 
                     t_mudstone + t_siltstone + t_limestone + t_dolomite + t_evaporite)
  erosion_depth = erosion_factor * total_sed_thick
  
  top_sandstone_raw = top_sed_ref - erosion_depth
  top_sandstone = np.minimum(top_sandstone_raw, bottom_sand_valley - eps)
  
  top_conglomerate = top_sandstone - t_sandstone
  top_shale = top_conglomerate - t_conglomerate
  top_mudstone = top_shale - t_shale
  top_siltstone = top_mudstone - t_mudstone
  top_limestone = top_siltstone - t_siltstone
  top_dolomite = top_limestone - t_limestone
  top_evaporite = top_dolomite - t_dolomite
  
  # Basement (crystalline)
  # Simple partition: Granite + Gneiss + Basalt
  basement_total = np.maximum(top_evaporite - t_evaporite, 5.0)
  basement_total -= (sed_base + sed_max * basins_combined) * 0.4  # Deeper under basins
  
  f_granite = 0.40 + 0.15 * (1.0 - basins_combined)
  f_gneiss = 0.35 + 0.20 * z_norm
  f_basalt = 0.03 + 0.05 * basins_combined
  
  f_total = f_granite + f_gneiss + f_basalt
  f_granite /= f_total
  f_gneiss /= f_total
  f_basalt /= f_total
  
  t_granite = basement_total * f_granite * 0.5
  t_gneiss = basement_total * f_gneiss * 0.5
  t_basalt = basement_total * f_basalt * 0.5
  t_ancient_crust = basement_total * 0.5
  
  top_granite = top_evaporite - t_evaporite
  top_gneiss = top_granite - t_granite
  top_basalt = top_gneiss - t_gneiss
  top_ancient_crust = top_basalt - t_basalt
  top_basement = top_ancient_crust - t_ancient_crust
  
  z_floor = float(top_basement.min() - 0.15 * elev_span)
  top_basement_floor = np.full_like(top_basement, z_floor)
  
  # ========== THICKNESS & INTERFACE DICTIONARIES ==========
  thickness = {
      "Topsoil": np.maximum(top_topsoil - top_subsoil, 0.0),
      "Subsoil": np.maximum(top_subsoil - top_colluvium, 0.0),
      "Colluvium": np.maximum(top_colluvium - top_saprolite, 0.0),
      "Saprolite": np.maximum(top_saprolite - top_weathered_rind, 0.0),
      "WeatheredBR": np.maximum(top_weathered_rind - top_clay, 0.0),
      
      "Clay": np.maximum(top_clay - top_silt, 0.0),
      "Silt": np.maximum(top_silt - top_sand_valley, 0.0),
      "Sand": np.maximum(top_sand_valley - bottom_sand_valley, 0.0),
      
      "Sandstone": np.maximum(top_sandstone - top_conglomerate, 0.0),
      "Conglomerate": np.maximum(top_conglomerate - top_shale, 0.0),
      "Shale": np.maximum(top_shale - top_mudstone, 0.0),
      "Mudstone": np.maximum(top_mudstone - top_siltstone, 0.0),
      "Siltstone": np.maximum(top_siltstone - top_limestone, 0.0),
      "Limestone": np.maximum(top_limestone - top_dolomite, 0.0),
      "Dolomite": np.maximum(top_dolomite - top_evaporite, 0.0),
      "Evaporite": np.maximum(top_evaporite - top_granite, 0.0),
      
      "Granite": np.maximum(top_granite - top_gneiss, 0.0),
      "Gneiss": np.maximum(top_gneiss - top_basalt, 0.0),
      "Basalt": np.maximum(top_basalt - top_ancient_crust, 0.0),
      "AncientCrust": np.maximum(top_ancient_crust - top_basement, 0.0),
      "Basement": np.maximum(top_basement - top_basement_floor, 0.0),
      "BasementFloor": np.maximum(top_basement_floor - (top_basement_floor - 0.0), 0.0),
  }
  
  interfaces = {
      "Topsoil": top_topsoil,
      "Subsoil": top_subsoil,
      "Colluvium": top_colluvium,
      "Saprolite": top_saprolite,
      "WeatheredBR": top_weathered_rind,
      
      "Clay": top_clay,
      "Silt": top_silt,
      "Sand": top_sand_valley,
      
      "Sandstone": top_sandstone,
      "Conglomerate": top_conglomerate,
      "Shale": top_shale,
      "Mudstone": top_mudstone,
      "Siltstone": top_siltstone,
      "Limestone": top_limestone,
      "Dolomite": top_dolomite,
      "Evaporite": top_evaporite,
      
      "Granite": top_granite,
      "Gneiss": top_gneiss,
      "Basalt": top_basalt,
      "AncientCrust": top_ancient_crust,
      "Basement": top_basement,
      "BasementFloor": top_basement_floor,
  }
  
  # Material properties
  properties = {
      "Topsoil": {"erodibility": 1.00, "density": 1600, "porosity": 0.45, "K_rel": 1.00},
      "Subsoil": {"erodibility": 0.85, "density": 1700, "porosity": 0.40, "K_rel": 0.85},
      "Colluvium": {"erodibility": 0.90, "density": 1750, "porosity": 0.35, "K_rel": 0.90},
      "Clay": {"erodibility": 0.80, "density": 1850, "porosity": 0.45, "K_rel": 0.80},
      "Silt": {"erodibility": 0.90, "density": 1750, "porosity": 0.42, "K_rel": 0.90},
      "Sand": {"erodibility": 0.85, "density": 1700, "porosity": 0.35, "K_rel": 0.85},
      "Saprolite": {"erodibility": 0.70, "density": 1900, "porosity": 0.30, "K_rel": 0.70},
      "WeatheredBR": {"erodibility": 0.55, "density": 2100, "porosity": 0.20, "K_rel": 0.55},
      "Sandstone": {"erodibility": 0.30, "density": 2200, "porosity": 0.18, "K_rel": 0.30},
      "Conglomerate": {"erodibility": 0.25, "density": 2300, "porosity": 0.16, "K_rel": 0.25},
      "Shale": {"erodibility": 0.45, "density": 2300, "porosity": 0.12, "K_rel": 0.45},
      "Mudstone": {"erodibility": 0.45, "density": 2300, "porosity": 0.12, "K_rel": 0.45},
      "Siltstone": {"erodibility": 0.35, "density": 2350, "porosity": 0.10, "K_rel": 0.35},
      "Limestone": {"erodibility": 0.28, "density": 2400, "porosity": 0.08, "K_rel": 0.28},
      "Dolomite": {"erodibility": 0.24, "density": 2450, "porosity": 0.06, "K_rel": 0.24},
      "Evaporite": {"erodibility": 0.90, "density": 2200, "porosity": 0.15, "K_rel": 0.90},
      "Granite": {"erodibility": 0.15, "density": 2700, "porosity": 0.01, "K_rel": 0.15},
      "Gneiss": {"erodibility": 0.16, "density": 2750, "porosity": 0.01, "K_rel": 0.16},
      "Basalt": {"erodibility": 0.12, "density": 2950, "porosity": 0.02, "K_rel": 0.12},
      "AncientCrust": {"erodibility": 0.14, "density": 2800, "porosity": 0.01, "K_rel": 0.14},
      "Basement": {"erodibility": 0.15, "density": 2700, "porosity": 0.01, "K_rel": 0.15},
      "BasementFloor": {"erodibility": 0.02, "density": 2850, "porosity": 0.005, "K_rel": 0.02},
  }
  
  return {
      "surface_elev": E,
      "interfaces": interfaces,
      "thickness": thickness,
      "properties": properties,
      "meta": {
          "elev_range_m": elev_range_m,
          "pixel_scale_m": pixel_scale_m,
          "basins": basins_combined,
          "highs": highs,
          "z_floor": z_floor,
      }
  }


# ========================================================================================
# VISUALIZATION FUNCTIONS
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
        "Sandstone", "Conglomerate", "Shale", "Mudstone", "Siltstone", 
        "Limestone", "Dolomite", "Evaporite",
        "Granite", "Gneiss", "Basalt", "AncientCrust",
        "Basement", "BasementFloor",
    ]

    color_map = {
        "Topsoil": "sienna", "Subsoil": "peru", "Colluvium": "burlywood",
        "Saprolite": "khaki", "WeatheredBR": "darkkhaki",
        "Clay": "lightcoral", "Silt": "thistle", "Sand": "gold",
        "Sandstone": "orange", "Conglomerate": "chocolate", "Shale": "slategray",
        "Mudstone": "rosybrown", "Siltstone": "lightsteelblue", 
        "Limestone": "lightgray", "Dolomite": "gainsboro", "Evaporite": "plum",
        "Granite": "lightpink", "Gneiss": "violet", "Basalt": "royalblue",
        "AncientCrust": "darkseagreen", "Basement": "dimgray", "BasementFloor": "black",
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

    ax.set_title("Stratigraphic Cross-Section (MERGED: Smooth Curves + Rich Facies)")
    ax.set_xlabel(axis_label)
    ax.set_ylabel("Elevation relative to lowest surface (m)")
    ax.legend(ncol=1, fontsize=8, framealpha=0.95, loc="center left", bbox_to_anchor=(1.02, 0.5))

    if ax is None:
        plt.tight_layout()
        plt.show()

    return ax


def plot_cross_sections_xy(strata, row=None, col=None, min_draw_thickness=0.05):
    N = strata["surface_elev"].shape[0]
    if row is None: row = N // 2
    if col is None: col = N // 2
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 11.5), constrained_layout=True)
    plot_cross_section(strata, row=row, min_draw_thickness=min_draw_thickness, ax=ax1)
    plot_cross_section(strata, col=col, min_draw_thickness=min_draw_thickness, ax=ax2)
    plt.show()


# ========================================================================================
# MAIN EXECUTION
# ========================================================================================

if __name__ == "__main__":
    print("="*70)
    print("MERGED TERRAIN GENERATOR")
    print("Combining smooth basin geometry + rich facies detail")
    print("="*70)
    
    # Generate topography
    z, rng = quantum_seeded_topography(
        N=512, beta=3.2, warp_amp=0.10, ridged_alpha=0.15, random_seed=None
    )
    
    # Generate merged stratigraphy
    strata = generate_stratigraphy_merged(z_norm=z, rng=rng)
    
    # Re-zero vertical datum
    E = strata["surface_elev"]
    offset = float(E.min())
    strata["surface_elev"] = E - offset
    for name, arr in strata["interfaces"].items():
        strata["interfaces"][name] = arr - offset
    
    # Plot topography
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(z, cmap='terrain', origin='lower', interpolation='bilinear')
    ax.set_title("Quantum-Seeded Topography")
    ax.set_xlabel("X (columns)")
    ax.set_ylabel("Y (rows)")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Normalized elevation")
    plt.tight_layout()
    plt.show()
    
    # Plot cross-sections
    plot_cross_sections_xy(strata)
    
    # Print diagnostics
    print("\n" + "="*70)
    print("LAYER THICKNESS STATISTICS (meters)")
    print("="*70)
    for layer_name in ["Topsoil", "Subsoil", "Clay", "Silt", "Sand", "Colluvium",
                       "Saprolite", "WeatheredBR", "Sandstone", "Conglomerate", 
                       "Shale", "Mudstone", "Siltstone", "Limestone", "Dolomite", 
                       "Evaporite", "Granite", "Gneiss", "Basalt"]:
        if layer_name in strata["thickness"]:
            t = strata["thickness"][layer_name]
            print(f"{layer_name:15s}: min={t.min():6.2f}  mean={t.mean():6.2f}  max={t.max():6.2f}")
    
    # Verify basin/ridge variation
    print("\n" + "="*70)
    print("BASIN vs RIDGE VERIFICATION")
    print("="*70)
    basins = strata["meta"]["basins"]
    basin_mask = basins > 0.7
    ridge_mask = basins < 0.3
    
    for layer in ["Sandstone", "Shale", "Limestone"]:
        if layer in strata["thickness"]:
            t = strata["thickness"][layer]
            basin_mean = t[basin_mask].mean() if np.any(basin_mask) else 0
            ridge_mean = t[ridge_mask].mean() if np.any(ridge_mask) else 0
            ratio = basin_mean / (ridge_mean + 0.1)
            status = "✅ GOOD" if ratio > 1.3 else "⚠ CHECK"
            print(f"{layer:15s}: Basin={basin_mean:5.1f}m  Ridge={ridge_mean:5.1f}m  Ratio={ratio:.2f}x  {status}")
    
    print("="*70)
    print("✅ MERGED GENERATOR COMPLETE")
    print("="*70)
