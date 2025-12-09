#!/usr/bin/env python3
"""
Realistic terrain + stratigraphy (now three plots):
- Surface elevation map
- One stratigraphic cross-section along X (constant row)
- One stratigraphic cross-section along Y (constant column)




Layer order (top -> bottom):
Topsoil, Subsoil, Colluvium, Saprolite, WeatheredBR (rind/grus),
Sandstone, Shale, Limestone, Basement, BasementFloor
(+ Alluvium & other deposits initialized but not plotted)
"""
from __future__ import annotations





 
# ------------------------- Standard imports -------------------------
import numpy as np
import matplotlib.pyplot as plt





# Optional quantum seeding (no auto-install)
try:
    import qiskit  # noqa
    try:
        import qiskit_aer  # noqa
        HAVE_QISKIT = True
    except Exception:
        HAVE_QISKIT = False
except Exception:
    HAVE_QISKIT = False





# --------------------------- RNG utilities ---------------------------
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

# ------------------------ Terrain primitives ------------------------
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


# Optional global low-pass (smoothness control; default: off)
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


# Optional small Gaussian blur (last-mile softness; default: off)
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
  rng = rng_from_qrng(n_seeds=4, random_seed=random_seed)
  base_low  = fractional_surface(N, beta=beta,     rng=rng)
  base_high = fractional_surface(N, beta=beta-0.4, rng=rng)
  z = 0.65*base_low + 0.35*base_high
  z = domain_warp(z, rng=rng, amp=warp_amp, beta=beta)
  z = ridged_mix(z, alpha=ridged_alpha)
  z = lowpass2d(z, cutoff=smooth_cutoff, rolloff=smooth_rolloff)
  z = gaussian_blur(z, sigma=post_blur_sigma)
  return z, rng


# ---------------------- Stratigraphy utilities ----------------------
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

# --------------- Wind-relevant geological features -------------------
def compute_topo_fields(surface_elev, pixel_scale_m):
    """
    Basic topographic fields from elevation only.

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

    # downslope aspect (for windward/leeward logic later)
    aspect = np.arctan2(-dEy, -dEx)

    # simple 4-neighbor Laplacian: <0 convex (ridge), >0 concave (valley)
    up    = np.roll(E, -1, axis=0)
    down  = np.roll(E,  1, axis=0)
    left  = np.roll(E,  1, axis=1)
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
    """
    Per-cell windward / leeward classification.

    base_wind_dir_deg : direction FROM WHICH the wind blows (0° = +x, 90° = +y)
    slope_min         : ignore very flat cells
    """
    theta = np.deg2rad(base_wind_dir_deg)
    wx, wy = np.cos(theta), np.sin(theta)   # wind-from unit vector

    # component of gradient along wind-from direction
    # >0: terrain rises into the wind (windward); <0: drops away (leeward)
    up_component = dEx * wx + dEy * wy

    slope_enough = slope_norm >= slope_min
    windward_mask = slope_enough & (up_component > 0.0)
    leeward_mask  = slope_enough & (up_component < 0.0)

    return windward_mask, leeward_mask, up_component

def classify_wind_barriers(E_norm, slope_norm, laplacian, up_component,
                           elev_thresh=0.5,
                           slope_thresh=0.4,
                           convex_frac=0.4,
                           up_quantile=0.4):
    """
    Wind barriers: mountain walls that strongly lift/deflect flow.

    Conditions (now a bit looser):
      - moderately high elevation (E_norm >= elev_thresh)
      - moderately steep slopes (slope_norm >= slope_thresh)
      - convex curvature (ridge-like)
      - reasonably strong upslope component along wind
    """
    # convex threshold (more negative laplacian = more ridge-like)
    lap_convex_thr = np.quantile(laplacian, convex_frac)

    # only consider positive upslope; choose upper quantile as "strong" barrier
    mask_pos = up_component > 0.0
    if np.any(mask_pos):
        up_thr = np.quantile(up_component[mask_pos], up_quantile)
    else:
        up_thr = 0.0  # fallback: any positive upslope can count

    barrier_mask = (
        (E_norm      >= elev_thresh) &
        (slope_norm  >= slope_thresh) &
        (laplacian   <= lap_convex_thr) &
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
    """
    Wind channels: valley axes that guide flow.

    Looser conditions:
      - low to mid elevation (E_norm <= elev_max)
      - gentle to moderately steep slopes
      - concave curvature
      - downslope direction roughly ALIGNED with wind direction
    """
    theta = np.deg2rad(base_wind_dir_deg)
    wx, wy = np.cos(theta), np.sin(theta)

    # downslope direction vector
    fx, fy = -dEx, -dEy
    magf = np.hypot(fx, fy) + 1e-12
    fxu, fyu = fx / magf, fy / magf

    # cosine of angle between downslope and wind direction
    cos_ang = fxu * wx + fyu * wy
    cos_align = np.cos(np.deg2rad(align_thresh_deg))

    # concave valleys
    lap_concave_thr = np.quantile(laplacian, concave_frac)

    channel_mask = (
        (E_norm      <= elev_max) &
        (slope_norm  >= slope_min) &
        (slope_norm  <= slope_max) &
        (laplacian   >= lap_concave_thr) &
        (cos_ang     >= cos_align)
    )
    return channel_mask


def classify_basins(E_norm, slope_norm, laplacian,
                    elev_max=0.5,
                    slope_max=0.3,
                    concave_frac=0.6):
    """
    Basins / bowls:
      - relatively low elevation
      - gentle slopes
      - concave (bowls)
    (Looser thresholds so we actually catch some.)
    """
    lap_concave_thr = np.quantile(laplacian, concave_frac)
    basin_mask = (
        (E_norm      <= elev_max) &
        (slope_norm  <= slope_max) &
        (laplacian   >= lap_concave_thr)
    )
    return basin_mask


def extract_region_summaries(mask, surface_elev, pixel_scale_m, min_cells=3):
    """
    Connected-component labeling for a boolean mask.

    Uses 8-neighbor connectivity so long skinny ridges/valleys are treated
    as single structures instead of many tiny diagonal fragments.

    Each region becomes a 'structure' with:
      - indices        : (N_i, 2) array of (row, col)
      - centroid_rc    : (row, col) center
      - size_cells     : area in cells
      - mean/max/min elevation, relief
      - orientation_rad: main axis orientation (0 = +x)
      - length_scale_m : rough length along main axis (m)
    """
    ny, nx = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    regions = []

    for r0 in range(ny):
        for c0 in range(nx):
            if not mask[r0, c0] or visited[r0, c0]:
                continue

            # flood-fill with 8-neighbor connectivity
            stack = [(r0, c0)]
            visited[r0, c0] = True
            cells = []

            while stack:
                r, c = stack.pop()
                cells.append((r, c))

                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        rr, cc = r + dr, c + dc
                        if (
                            0 <= rr < ny and 0 <= cc < nx and
                            mask[rr, cc] and not visited[rr, cc]
                        ):
                            visited[rr, cc] = True
                            stack.append((rr, cc))

            idxs = np.array(cells, dtype=int)
            if idxs.shape[0] < min_cells:
                continue

            rows = idxs[:, 0].astype(float)
            cols = idxs[:, 1].astype(float)
            centroid_r = rows.mean()
            centroid_c = cols.mean()

            vals = surface_elev[idxs[:, 0], idxs[:, 1]]
            mean_e = float(vals.mean())
            max_e  = float(vals.max())
            min_e  = float(vals.min())
            relief = max_e - min_e

            # PCA for main axis orientation / length
            x = cols - centroid_c
            y = rows - centroid_r
            C = np.cov(np.vstack([x, y]))
            eigvals, eigvecs = np.linalg.eigh(C)
            i_max = int(np.argmax(eigvals))
            v = eigvecs[:, i_max]
            orientation = float(np.arctan2(v[1], v[0]))
            length_scale = float(2.0 * np.sqrt(max(eigvals[i_max], 0.0)) * pixel_scale_m)

            regions.append({
                "indices": idxs,
                "centroid_rc": (float(centroid_r), float(centroid_c)),
                "size_cells": int(idxs.shape[0]),
                "mean_elev_m": mean_e,
                "max_elev_m": max_e,
                "min_elev_m": min_e,
                "relief_m": relief,
                "orientation_rad": orientation,
                "length_scale_m": length_scale,
            })

    return regions


def build_wind_structures(surface_elev, pixel_scale_m, base_wind_dir_deg):
    """
    Given a topography map, classify only geological structures that change wind:
      - windward / leeward slopes
      - wind barriers (mountain walls)
      - wind channels (valley corridors)
      - basins / bowls (air pooling zones)

    Returns a dict with per-cell masks and grouped regions.
    Does NOT modify terrain or simulate weather.
    """
    topo = compute_topo_fields(surface_elev, pixel_scale_m)
    E        = topo["E"]
    E_norm   = topo["E_norm"]
    dEx      = topo["dEx"]
    dEy      = topo["dEy"]
    slope_n  = topo["slope_norm"]
    aspect   = topo["aspect"]
    lap      = topo["laplacian"]

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

    barrier_regions = extract_region_summaries(barrier_mask, E, pixel_scale_m, min_cells=10)
    channel_regions = extract_region_summaries(channel_mask, E, pixel_scale_m, min_cells=10)
    basin_regions   = extract_region_summaries(basin_mask,   E, pixel_scale_m, min_cells=10)


    return {
        "E": E,
        "E_norm": E_norm,
        "slope_norm": slope_n,
        "aspect": aspect,
        "laplacian": lap,

        "windward_mask": windward_mask,
        "leeward_mask": leeward_mask,
        "up_component": up_component,

        "barrier_mask": barrier_mask,
        "channel_mask": channel_mask,
        "basin_mask": basin_mask,

        "barrier_regions": barrier_regions,   # mountain walls that block/deflect flow
        "channel_regions": channel_regions,   # valley corridors that funnel flow
        "basin_regions": basin_regions,       # bowls where air/storms pool

        "meta": {
            "pixel_scale_m": pixel_scale_m,
        },
    }


# -------------------- Wind / pseudo-low-pressure helpers --------------------
def compute_orographic_low_pressure(
    surface_elev,
    rng,
    pixel_scale_m,
    base_wind_dir_deg=45.0,
    mode="day",               # "day" ~ valley breeze, "night" ~ mountain breeze, "mixed"
    smooth_scale_rel=0.20,    # fraction of domain for large-scale smoothing
):
    """
    Build a 0..1 'low-pressure likelihood' map over mountains WITHOUT
    explicitly storing pressure, using only terrain geometry + wind direction.

    Concept:
    - Wind is driven from higher-pressure to lower-pressure areas.
      We don't model pressure; instead we mark where low-pressure *would* occur
      (ridges on windward side, heated slopes by day, cold pooled valleys at night).
    - Mountains force air to rise on windward slopes => effective low-P there.
    - Daytime: valley breeze -> upslope flow, low-P near heated slopes & ridges.
    - Nighttime: mountain breeze -> downslope flow, low-P in cold pooled valleys.

    Inputs
    ------
    surface_elev : 2D array of absolute elevation (m)
    rng          : np.random.Generator (Qiskit-seeded in your code)
    pixel_scale_m: grid spacing (m)
    base_wind_dir_deg : mean large-scale wind direction *from which* air comes
                        (0° = +x, 90° = +y)
    mode         : "day", "night", or "mixed"
    smooth_scale_rel : how coarse the large-scale basin/high field is (0..1)
    """
    z = surface_elev
    ny, nx = z.shape

    # --- gradient & slope ---
    dzdx, dzdy = np.gradient(z, pixel_scale_m, pixel_scale_m)
    slope_mag = np.sqrt(dzdx**2 + dzdy**2) + 1e-12
    slope_n   = _normalize(slope_mag)
    gentle    = 1.0 - slope_n

    # --- basic elevation normalization / basins vs highs ---
    z_smooth_k = max(5, int(smooth_scale_rel * min(nx, ny)) | 1)
    z_smooth   = _box_blur(z, k=z_smooth_k)

    elev_n = _normalize(z)            # 0 low → 1 high
    highs  = _normalize(z_smooth)     # broad highs (ridges, plateaus)
    basins = _normalize(1.0 - z_smooth)  # broad valleys/depressions

    # --- large-scale wind direction (unit vector) ---
    # wind blows from this direction into the domain
    az = np.deg2rad(base_wind_dir_deg)
    wx, wy = np.cos(az), np.sin(az)

    # directional derivative of elevation along wind direction:
    # positive where flow goes upslope (orographic lifting on windward side)
    dzw = dzdx * wx + dzdy * wy
    orographic_raw = np.maximum(dzw, 0.0)   # only upslope component
    orographic = _normalize(orographic_raw)

    # --- valley vs mountain breeze components ---
    # day: slopes heated, air rises -> low-P along sun-facing + valley-slopes
    # night: air cools, drains into valleys -> low-P pooled in basins

    # treat "sun direction" similar to wind_dir for now; you can make it separate later
    sx, sy = wx, wy
    dzs = dzdx * sx + dzdy * sy
    sun_slope_raw = np.maximum(dzs, 0.0)    # slopes facing the "sun"
    sun_slope = _normalize(sun_slope_raw)

    # Daytime valley-breeze low-P:
    # - on heated, sun-facing slopes (sun_slope)
    # - near ridge tops / high terrain (highs)
    lowP_day = _normalize(0.5 * sun_slope + 0.5 * highs)

    # Nighttime mountain-breeze low-P:
    # - in basins/valleys (basins)
    # - where slopes are gentle (cold air accumulates more easily)
    lowP_night = _normalize(0.7 * basins + 0.3 * gentle)

    # --- combine with orographic lifting (always present if mountains + wind) ---
    mode = str(mode).lower()
    if mode == "day":
        lowP = 0.50 * orographic + 0.50 * lowP_day
    elif mode == "night":
        lowP = 0.40 * orographic + 0.60 * lowP_night
    else:  # "mixed"
        lowP = 0.40 * orographic + 0.30 * lowP_day + 0.30 * lowP_night

    # small quantum-random perturbation, smoothed so it doesn't make 'salt & pepper'
    k_noise = max(7, int(0.05 * min(nx, ny)) | 1)
    rnd = rng.standard_normal(size=z.shape)
    rnd_smooth = _box_blur(rnd, k=k_noise)
    rnd_smooth = rnd_smooth / (np.std(rnd_smooth) + 1e-9)

    lowP += 0.15 * rnd_smooth  # gentle perturbation
    lowP = _normalize(lowP)

    return lowP  # 0..1: higher = more likely effective low-pressure zone


# ---------------- Physics-informed near-surface layers ----------------
def soil_thickness_from_slope(z_norm, soil_range_m=(0.3, 1.8)):
  dzdx, dzdy = np.gradient(z_norm)
  slope_mag = np.sqrt(dzdx**2 + dzdy**2)
  slope_n   = _normalize(slope_mag)
  t = soil_range_m[1] - (soil_range_m[1] - soil_range_m[0]) * slope_n
  return _box_blur(t, k=5)


def colluvium_thickness_field(
  z_norm, rng, pixel_scale_m,
  colluvium_max_m=18.0,
  *,
  w_gentle=0.35, w_curv=0.30, w_low=0.20, w_twi=0.15,
  smooth_relief_px=31, twi_k1=7, twi_k2=13,
  lognorm_sigma=0.20, floor_m=0.5, bias=1.0
):
  dzdx, dzdy = np.gradient(z_norm)
  slope_mag = np.sqrt(dzdx**2 + dzdy**2)
  slope_n   = _normalize(slope_mag)
  gentle    = 1.0 - slope_n
  d2x,_ = np.gradient(dzdx); _,d2y = np.gradient(dzdy)
  curv  = d2x + d2y
  hollows = _normalize(np.maximum(curv, 0.0))
  k = max(5, int(smooth_relief_px)|1)
  z_smooth = _box_blur(z_norm, k=k)
  lowlands = _normalize(1.0 - z_smooth)
  catch = _box_blur(_box_blur(1.0 - slope_n, k=7), k=13)
  wet = _normalize(catch - slope_n)
  w = np.array([w_gentle, w_curv, w_low, w_twi], float); w = np.clip(w,0,None); w /= (w.sum()+1e-12)
  index = _normalize(w[0]*gentle + w[1]*hollows + w[2]*lowlands + w[3]*wet)
  noise = rng.lognormal(mean=0.0, sigma=float(lognorm_sigma), size=index.shape)
  index_noisy = _normalize(index * noise)
  return bias * (floor_m + index_noisy * (colluvium_max_m - floor_m))


def saprolite_thickness_field(
  z_norm, rng,
  *,
  median_m=6.0, clamp_min=0.5, clamp_max=30.0,
  w_gentle=0.6, w_interfluve=0.4,
  relief_window_px=61, sigma=0.35
):
  dzdx, dzdy = np.gradient(z_norm)
  slope_mag = np.sqrt(dzdx**2 + dzdy**2)
  gentle = 1.0 - _normalize(slope_mag)
  k = max(5, int(relief_window_px)|1)
  z_smooth = _box_blur(z_norm, k=k)
  interfluves = _normalize(z_smooth)
  idx = _normalize(w_gentle*gentle + w_interfluve*interfluves)
  base = np.exp(np.log(median_m) + sigma * rng.standard_normal(size=idx.shape))
  return np.clip(base * (0.4 + 0.6*idx), clamp_min, clamp_max)


def weathered_rind_thickness_field(
  z_norm, rng,
  *,
  median_m=1.8, clamp_min=0.4, clamp_max=6.0,
  patch_beta=3.0, patch_alpha=0.5
):
  N = z_norm.shape[0]
  tex = fractional_surface(N, beta=patch_beta, rng=rng)
  tex = (1 - np.abs(2*tex - 1))
  base = np.exp(np.log(median_m) + 0.25 * rng.standard_normal(size=tex.shape))
  return np.clip((1 - patch_alpha) * base + patch_alpha * base * tex, clamp_min, clamp_max)

# ---------------- Glacial / aeolian deposit thickness fields ----------------
def till_thickness_field(
    z_norm, rng,
    *,
    max_till_m=25.0,
    floor_m=0.0,
    relief_window_px=61,
    lognorm_sigma=0.35,
):
    dzdx, dzdy = np.gradient(z_norm)
    slope_mag = np.sqrt(dzdx**2 + dzdy**2)
    slope_n   = _normalize(slope_mag)
    gentle    = 1.0 - slope_n

    k = max(5, int(relief_window_px) | 1)
    z_smooth = _box_blur(z_norm, k=k)
    basins   = _normalize(1.0 - z_smooth)

    index = _normalize(0.6 * basins + 0.4 * gentle)
    noise = rng.lognormal(mean=0.0, sigma=float(lognorm_sigma), size=index.shape)
    index_noisy = _normalize(index * noise)
    return floor_m + index_noisy * (max_till_m - floor_m)


def loess_thickness_field(
    z_norm, rng,
    *,
    max_loess_m=6.0,
    floor_m=0.0,
    relief_window_px=81,
    lognorm_sigma=0.30,
):
    dzdx, dzdy = np.gradient(z_norm)
    slope_mag = np.sqrt(dzdx**2 + dzdy**2)
    slope_n   = _normalize(slope_mag)
    gentle    = 1.0 - slope_n

    k = max(5, int(relief_window_px) | 1)
    z_smooth = _box_blur(z_norm, k=k)
    uplands  = _normalize(z_smooth)
    mid_uplands = uplands * (1.0 - uplands)

    index = _normalize(0.6 * gentle + 0.4 * mid_uplands)
    noise = rng.lognormal(mean=0.0, sigma=float(lognorm_sigma), size=index.shape)
    index_noisy = _normalize(index * noise)
    return floor_m + index_noisy * (max_loess_m - floor_m)


def dune_thickness_field(
    z_norm, rng,
    *,
    max_dune_m=10.0,
    floor_m=0.0,
    relief_window_px=41,
    lognorm_sigma=0.40,
):
    dzdx, dzdy = np.gradient(z_norm)
    slope_mag = np.sqrt(dzdx**2 + dzdy**2)
    slope_n   = _normalize(slope_mag)
    gentle    = 1.0 - slope_n

    k = max(5, int(relief_window_px) | 1)
    z_smooth = _box_blur(z_norm, k=k)
    lows     = _normalize(1.0 - z_smooth)

    d2x,_ = np.gradient(dzdx); _,d2y = np.gradient(dzdy)
    curv  = d2x + d2y
    convex = _normalize(np.maximum(-curv, 0.0))

    index = _normalize(0.5 * gentle + 0.3 * lows + 0.2 * convex)
    noise = rng.lognormal(mean=0.0, sigma=float(lognorm_sigma), size=index.shape)
    index_noisy = _normalize(index * noise)
    return floor_m + index_noisy * (max_dune_m - floor_m)


# ---------------- Continental crust trend helper ----------------
def crust_thickness_field(surface_elev, elev_range_m, min_factor=2.0, max_factor=5.0):
    """
    Continental-only 'effective crust thickness' field (same shape as surface_elev).
    Encodes only the pattern: higher topography → thicker crustal column.
    """
    span = max(float(elev_range_m), 1e-3)
    zmin = float(surface_elev.min())
    zmax = float(surface_elev.max())
    znorm = (surface_elev - zmin) / (zmax - zmin + 1e-12)  # 0..1
    factor = min_factor + (max_factor - min_factor) * znorm
    return span * factor


# ------------------------ Layered stratigraphy -----------------------
def generate_stratigraphy(
  z_norm, rng,
  elev_range_m=700.0,
  pixel_scale_m=10.0,
  soil_range_m=(0.3, 1.8),
  # Colluvium controls
  colluvium_max_m=18.0,
  wC_gentle=0.35, wC_curv=0.30, wC_low=0.20, wC_twi=0.15,
  C_relief_px=31, C_twi_k1=7, C_twi_k2=13, C_sigma=0.20, C_floor=0.5, C_bias=1.0,
  # Saprolite controls
  sap_median=6.0, sap_min=0.5, sap_max=30.0, sap_w_gentle=0.6, sap_w_inter=0.4, sap_relief_px=61, sap_sigma=0.35,
  # Weathered rind controls
  rind_median=1.8, rind_min=0.4, rind_max=6.0, rind_patch_beta=3.0, rind_patch_alpha=0.5,
  # Competent rock package (relative proportions)
  unit_thickness_m=(90.0,110.0,100.0),  # sandstone, shale, limestone
  undulation_amp_m=10.0, undulation_beta=3.2,
  dip_deg=6.0, dip_dir_deg=45.0,
  burial_depth_m=120.0,
  bed_struct_weight=0.45,
  interface_blur_sigma=None
):
  """
  Build stratigraphy where:
  - Topsoil / Colluvium / Saprolite / WeatheredBR follow hillslope & weathering rules.
  - Sandstone / Shale / Limestone form broad, gently dipping sheets that thicken
    in basins and thin on highs (marine transgression style).
  - Basement provides a smooth crystalline foundation underneath.
  - All randomness uses Qiskit-seeded rng, but is heavily smoothed to avoid 'spiky columns'.
  """
  N = z_norm.shape[0]
  # Absolute elevation (m)
  E = z_norm * elev_range_m

    # ---------- 1) Near-surface regolith ----------
  # Total soil thickness: thicker on low, gentle slopes
  soil_total = soil_thickness_from_slope(z_norm, soil_range_m)

  # Split into Topsoil (A) and Subsoil (B horizon).
  # You can tune top_frac (0.3–0.5) if you want a thicker A horizon.
  top_frac = 0.4
  topsoil_thick  = top_frac * soil_total
  subsoil_thick  = (1.0 - top_frac) * soil_total

  # Colluvium: gravity-driven, thickest at slope bases/hollows/lowlands.
  tC = colluvium_thickness_field(
      z_norm, rng, pixel_scale_m, colluvium_max_m,
      w_gentle=wC_gentle, w_curv=wC_curv, w_low=wC_low, w_twi=wC_twi,
      smooth_relief_px=C_relief_px, twi_k1=C_twi_k1, twi_k2=C_twi_k2,
      lognorm_sigma=C_sigma, floor_m=C_floor, bias=C_bias
  )

  # Saprolite: thick in stable, moderately elevated interfluves.
  tS = saprolite_thickness_field(
      z_norm, rng,
      median_m=sap_median, clamp_min=sap_min, clamp_max=sap_max,
      w_gentle=sap_w_gentle, w_interfluve=sap_w_inter, relief_window_px=sap_relief_px, sigma=sap_sigma
  )

  # Weathered bedrock rind / grus: patchy, overlying basement.
  tR = weathered_rind_thickness_field(
      z_norm, rng,
      median_m=rind_median, clamp_min=rind_min, clamp_max=rind_max,
      patch_beta=rind_patch_beta, patch_alpha=rind_patch_alpha
  )

  # Glacial / aeolian mantles (thickness fields; returned via 'deposits')
  t_till  = till_thickness_field(z_norm, rng)
  t_loess = loess_thickness_field(z_norm, rng)
  t_dune  = dune_thickness_field(z_norm, rng)



  # ---------- 2) Structural plane for sedimentary cover ----------
  ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
  X = ii * pixel_scale_m
  Y = jj * pixel_scale_m
  az  = np.deg2rad(dip_dir_deg)   # map-view azimuth of dip direction
  dip = np.deg2rad(dip_deg)       # dip angle
  ux, uy = np.cos(az), np.sin(az) # unit vector along dip direction

  plane = np.tan(dip) * (ux * X + uy * Y)   # regional dip
  undul = (fractional_surface(N, beta=undulation_beta, rng=rng)*2 - 1) * undulation_amp_m
  bed_struct = plane + undul
  bed_struct_zm = bed_struct - np.mean(bed_struct)


  # ---------- 3) Continental crust & sedimentary thickness trend ----------
  elev_span = float(E.max() - E.min() + 1e-9)
  crust_thick = crust_thickness_field(E, elev_span)  # thicker crust under high topo

  # Assume ~35% of crust_thick is sedimentary cover on continental crust. :contentReference[oaicite:10]{index=10}
  sed_frac  = 0.35
  sed_total = sed_frac * crust_thick

  # Very coarse topographic smooth to define basins vs highs
  k_coarse = max(31, int(0.15 * N) | 1)
  z_smooth = _box_blur(z_norm, k=k_coarse)
  basins   = _normalize(1.0 - z_smooth)   # 1 in lows, 0 on highs
  highs    = _normalize(z_smooth)         # 1 on highs

  # Present slope – steeper = more erosion, so thinner preserved cover.
  dEx, dEy = np.gradient(E)
  slope_mag = np.sqrt(dEx**2 + dEy**2)
  slope_n   = _normalize(slope_mag)
  gentle    = 1.0 - slope_n

  # Facies environments (all 0–1 after _normalize):
  # - Sandstone: shorelines / basin margins → intermediate basinness, some slope. :contentReference[oaicite:11]{index=11}
  sand_env_raw  = basins * (1.0 - basins) * (0.5 + 0.5 * slope_n)
  shale_env_raw = basins                        # deepening into basins (offshore muds).
  lime_env_raw  = highs * gentle                # carbonate platforms on gentle highs. :contentReference[oaicite:12]{index=12}

  sand_env  = _normalize(sand_env_raw)
  shale_env = _normalize(shale_env_raw)
  lime_env  = _normalize(lime_env_raw)

  # Relative vertical proportions from unit_thickness_m
  T_sand, T_shale, T_lime = unit_thickness_m
  total_units = float(T_sand + T_shale + T_lime + 1e-12)
  f_sand  = T_sand  / total_units
  f_shale = T_shale / total_units
  f_lime  = T_lime  / total_units

  # Helper: smoothed random field (quantum-seeded, but laterally coherent)
  def smooth_random_field(k):
      noise = rng.standard_normal(size=E.shape)
      return _box_blur(noise, k=max(5, int(k) | 1))

  k_thick = max(15, int(0.06 * N) | 1)

  # Base thickness fields (trend) for each unit
  base_sand  = sed_total * f_sand  * (0.4 + 0.6 * sand_env)
  base_shale = sed_total * f_shale * (0.3 + 0.7 * shale_env)
  base_lime  = sed_total * f_lime  * (0.3 + 0.7 * lime_env)

  # Quantum-smoothed variability (keeps sheets coherent, avoids "icicles")
  rnd_sand  = smooth_random_field(k_thick)
  rnd_shale = smooth_random_field(k_thick)
  rnd_lime  = smooth_random_field(k_thick)

  def apply_variation(base, rnd, amp=0.3):
      std = float(np.std(rnd) + 1e-9)
      f = 1.0 + amp * (rnd / std)
      f = np.clip(f, 0.5, 1.5)
      return np.clip(base * f, 0.0, None)

  t_sand_trend  = apply_variation(base_sand,  rnd_sand)
  t_shale_trend = apply_variation(base_shale, rnd_shale)
  t_lime_trend  = apply_variation(base_lime,  rnd_lime)

  # Thin units modestly on steep slopes, but never turn them off entirely
  thin_factor = 0.4 + 0.6 * gentle  # 1 on gentle, 0.4 on steep
  t_sand  = t_sand_trend  * thin_factor
  t_shale = t_shale_trend * thin_factor
  t_lime  = t_lime_trend  * thin_factor

  # ---------- 4) Structural tops for Sandstone / Shale / Limestone ----------
  Emean       = float(E.mean())
  crust_mean  = float(crust_thick.mean())
  crust_anom  = (crust_thick - crust_mean) / (crust_mean + 1e-9)

  # Deeper sediment stack under thicker crust (mountain belts), shallower under thinner crust.
  top_sed_ref = (
      (Emean - burial_depth_m)
      - 0.3 * crust_anom * elev_span
      + bed_struct_weight * bed_struct_zm
  )

  top_sandstone_raw = top_sed_ref
  top_shale_raw     = top_sandstone_raw - t_sand
  top_limestone_raw = top_shale_raw     - t_shale
  top_basement_raw  = top_limestone_raw - t_lime  # top of crystalline basement

    # ---------- 5) Regolith stack above rock ----------
  eps = 0.01

  # A and B horizons
  top_topsoil   = E
  top_subsoil   = top_topsoil   - topsoil_thick
  top_colluvium = top_subsoil   - subsoil_thick

  # Colluvium, saprolite, weathered rind
  top_saprolite = top_colluvium - tC
  top_rind      = top_saprolite - tS

  # Ensure rock units stay below weathered rind and respect ordering
  top_sandstone = np.minimum(top_sandstone_raw, top_rind - eps)
  top_shale     = np.minimum(top_shale_raw,     top_sandstone - eps)
  top_limestone = np.minimum(top_limestone_raw, top_shale - eps)
  top_basement  = np.minimum(top_basement_raw,  top_limestone - eps)

  # ---------- 6) Basement floor ----------
  z_floor = float(top_basement.min() - 0.2 * elev_span)
  top_basement_floor = np.full_like(top_basement, z_floor)

  # ---------- 7) Optional smoothing of rock interfaces ----------
  if interface_blur_sigma is not None and interface_blur_sigma > 0:
      def blur(a): return gaussian_blur(a, sigma=interface_blur_sigma)
      top_sandstone = blur(top_sandstone)
      top_shale     = blur(top_shale)
      top_limestone = blur(top_limestone)
      top_basement  = blur(top_basement)
      # Re-enforce ordering after blur
      top_sandstone = np.minimum(top_sandstone, top_rind - eps)
      top_shale     = np.minimum(top_shale,     top_sandstone - eps)
      top_limestone = np.minimum(top_limestone, top_shale - eps)
      top_basement  = np.minimum(top_basement,  top_limestone - eps)
      z_floor = float(top_basement.min() - 0.2 * elev_span)
      top_basement_floor = np.full_like(top_basement, z_floor)

  # ---------- 8) Thickness rasters ----------
  thickness = {
      "Topsoil":       np.maximum(top_topsoil     - top_subsoil,         0.0),
      "Subsoil":       np.maximum(top_subsoil     - top_colluvium,       0.0),
      "Colluvium":     np.maximum(top_colluvium   - top_saprolite,       0.0),
      "Saprolite":     np.maximum(top_saprolite   - top_rind,            0.0),
      "WeatheredBR":   np.maximum(top_rind        - top_sandstone,       0.0),
      "Sandstone":     np.maximum(top_sandstone   - top_shale,           0.0),
      "Shale":         np.maximum(top_shale       - top_limestone,       0.0),
      "Limestone":     np.maximum(top_limestone   - top_basement,        0.0),
      "Basement":      np.maximum(top_basement    - top_basement_floor,  0.0),
      "BasementFloor": np.maximum(top_basement_floor - (top_basement_floor - 0.0), 0.0),
  }

  interfaces = {
      "Topsoil":       top_topsoil,
      "Subsoil":       top_subsoil,
      "Colluvium":     top_colluvium,
      "Saprolite":     top_saprolite,
      "WeatheredBR":   top_rind,
      "Sandstone":     top_sandstone,
      "Shale":         top_shale,
      "Limestone":     top_limestone,
      "Basement":      top_basement,
      "BasementFloor": top_basement_floor,
  }


    # ---------- 9) Alluvium (channels / floodplains) ----------
  dzdx, dzdy = np.gradient(z_norm)
  slope_mag2 = np.sqrt(dzdx**2 + dzdy**2)
  slope_n2   = _normalize(slope_mag2)
  catch = _box_blur(_box_blur(1.0 - slope_n2, k=7), k=13)
  wet = _normalize(catch - slope_n2)
  alluvium = np.where(
      wet > 0.7,
      np.minimum(2.0 * rng.random(size=wet.shape), 2.0),
      0.0,
  )

  # Glacial / aeolian + fluvial deposit rasters (thickness in m)
  deposits = {
      "Till":     t_till,
      "Loess":    t_loess,
      "DuneSand": t_dune,
      "Alluvium": alluvium,
  }


  # ---------- 10) Material properties ----------
  properties = {
      # REGOLITH / SOIL
      "Topsoil": {
          "erodibility": 1.00,  # high: silty/loamy, organic-rich
          "density":     1600,
          "porosity":    0.45,
          "K_rel":       1.00,
      },
      "Subsoil": {
          # B horizon: more clay/oxides, less organic, somewhat less erodible
          "erodibility": 0.85,
          "density":     1700,
          "porosity":    0.40,
          "K_rel":       0.85,
      },
      "Colluvium": {
          "erodibility": 0.90,  # very erodible slope wash
          "density":     1750,
          "porosity":    0.35,
          "K_rel":       0.90,
      },
      "Alluvium": {
          # river & floodplain deposits
          "erodibility": 0.95,
          "density":     1700,
          "porosity":    0.40,
          "K_rel":       0.95,
      },
      "Till": {
          # glacial rubble; can be quite erodible or resistant depending on compaction
          "erodibility": 0.75,
          "density":     1900,
          "porosity":    0.25,
          "K_rel":       0.75,
      },
      "Loess": {
          # wind-blown silt; extremely erodible once exposed
          "erodibility": 1.05,
          "density":     1550,
          "porosity":    0.50,
          "K_rel":       1.05,
      },
      "DuneSand": {
          # loose, well-sorted sand; very low cohesion
          "erodibility": 0.95,
          "density":     1650,
          "porosity":    0.40,
          "K_rel":       0.95,
      },

      # WEATHERED ROCK
      "Saprolite": {
          "erodibility": 0.70,
          "density":     1900,
          "porosity":    0.30,
          "K_rel":       0.70,
      },
      "WeatheredBR": {
          # weathered bedrock / rind
          "erodibility": 0.55,
          "density":     2100,
          "porosity":    0.20,
          "K_rel":       0.55,
      },

      # WEAK CLASTIC (MUDSTONE / SHALE)
      "Shale": {
          "erodibility": 0.45,
          "density":     2300,
          "porosity":    0.12,
          "K_rel":       0.45,
      },
      "Mudstone": {
          # treated as an alias of Shale for now
          "erodibility": 0.45,
          "density":     2300,
          "porosity":    0.12,
          "K_rel":       0.45,
      },

      # MEDIUM CLASTIC (SILTSTONE)
      "Siltstone": {
          # intermediate between shale and sandstone
          "erodibility": 0.35,
          "density":     2350,
          "porosity":    0.10,
          "K_rel":       0.35,
      },

      # STRONGER CLASTIC (SANDSTONE / CONGLOMERATE)
      "Sandstone": {
          "erodibility": 0.30,
          "density":     2200,
          "porosity":    0.18,
          "K_rel":       0.30,
      },
      "Conglomerate": {
          # often slightly more resistant than sandstone (coarse clasts + cement)
          "erodibility": 0.25,
          "density":     2300,
          "porosity":    0.16,
          "K_rel":       0.25,
      },

      # CARBONATES
      "Limestone": {
          "erodibility": 0.28,  # mechanically strong, but dissolves chemically
          "density":     2400,
          "porosity":    0.08,
          "K_rel":       0.28,
      },
      "Dolomite": {
          # often slightly more resistant than limestone
          "erodibility": 0.24,
          "density":     2450,
          "porosity":    0.06,
          "K_rel":       0.24,
      },

      # EVAPORITES (GYPSUM / HALITE)
      "Evaporite": {
          # mechanically moderate but chemically super erodible
          "erodibility": 0.90,
          "density":     2200,
          "porosity":    0.15,
          "K_rel":       0.90,
      },

      # CRYSTALLINE BASEMENT (GRANITE / GNEISS / BASALT) + AGGREGATE
      "Basement": {
          # generic crystalline basement
          "erodibility": 0.15,
          "density":     2700,
          "porosity":    0.01,
          "K_rel":       0.15,
      },
      "Granite": {
          "erodibility": 0.15,
          "density":     2700,
          "porosity":    0.01,
          "K_rel":       0.15,
      },
      "Gneiss": {
          "erodibility": 0.16,
          "density":     2750,
          "porosity":    0.01,
          "K_rel":       0.16,
      },
      "Basalt": {
          # dense mafic volcanic; often forms very resistant flows
          "erodibility": 0.12,
          "density":     2950,
          "porosity":    0.02,
          "K_rel":       0.12,
      },

      # BASEMENT FLOOR (numerical bottom)
      "BasementFloor": {
          "erodibility": 0.02,
          "density":     2850,
          "porosity":    0.005,
          "K_rel":       0.02,
      },
  }

  facies_controls = {
      "basins":       basins,    # 0 (high areas) → 1 (deep basins)
      "highs":        highs,     # 0 (basins)    → 1 (high areas)
      "paleo_slope":  slope_n,   # 0 (gentle)   → 1 (steep)
  }


  return {
      "surface_elev": E,
      "interfaces": interfaces,
      "thickness": thickness,
      "properties": properties,
      "alluvium_init": alluvium,
      "deposits": deposits,
      "meta": {
          "elev_range_m": elev_range_m,
          "pixel_scale_m": pixel_scale_m,
          "dip_deg": dip_deg,
          "dip_dir_deg": dip_dir_deg,
          "unit_thickness_m": unit_thickness_m,
          "burial_depth_m": burial_depth_m,
          "bed_struct_weight": bed_struct_weight,
          "z_floor": z_floor
      }
  }

def compute_top_material_map(strata, min_thick=0.05):
    """
    Return a 2D array of material names representing
    the 'topmost' layer at each (row, col) cell.

    Priority:
      1. Glacial/aeolian/fluvial deposits (if local thickness > 0)
      2. Stratigraphic interfaces, from Topsoil down to Basement
      3. Never returns 'BasementFloor' by design.
    """
    interfaces = strata["interfaces"]
    thickness  = strata["thickness"]
    deposits   = strata.get("deposits", {})
    E          = strata["surface_elev"]

    ny, nx = E.shape
    top_mat = np.empty((ny, nx), dtype=object)

    # 1) Deposits priority (if you want a different order, change this list)
    deposit_order = ["Loess", "DuneSand", "Till", "Alluvium"]
    deposit_order = [d for d in deposit_order if d in deposits]

    # 2) Stratigraphic order, top -> bottom (excluding BasementFloor)
    strat_order = [
        "Topsoil",
        "Subsoil",
        "Colluvium",
        "Saprolite",
        "WeatheredBR",
        "Sandstone",
        "Shale",
        "Limestone",
        "Basement",
        # NO BasementFloor here
    ]
    strat_order = [k for k in strat_order if k in interfaces]

    # Initialize with a safe default (Basement, not BasementFloor)
    top_mat[:] = "Basement"

    # a) Deposits where they actually exist
    for name in deposit_order:
        field = deposits[name]
        mask = field > min_thick
        top_mat[mask] = name

    # b) Stratigraphy – from top layer downward
    for i, name in enumerate(strat_order[:-1]):
        below = strat_order[i+1]
        top_here   = interfaces[name]
        top_below  = interfaces[below]
        thick_here = np.maximum(top_here - top_below, 0.0)
        mask = thick_here > min_thick

        # Only overwrite where no deposit already sits
        no_deposit = ~np.isin(top_mat, deposit_order)
        top_mat[mask & no_deposit] = name

    # c) Any cells still not in the strat_order get Basement as fallback.
    # (BasementFloor is intentionally never used as a top material.)
    not_assigned = ~np.isin(top_mat, deposit_order + strat_order)
    top_mat[not_assigned] = "Basement"

    return top_mat

def compute_top_facies_map(strata, min_thick=0.05):
    """
    Use:
      - top layer map from compute_top_material_map(...)
      - facies_controls (basins, highs, paleo_slope)
    to assign a *facies-level* lithology per cell, e.g.:
      Sandstone package -> Sandstone vs Conglomerate
      Shale package     -> Shale vs Mudstone vs Siltstone
      Limestone package -> Limestone vs Dolomite vs Evaporite
    """
    top_layer = compute_top_material_map(strata, min_thick=min_thick)
    controls  = strata.get("facies_controls", {})

    basins      = controls.get("basins", None)
    highs       = controls.get("highs", None)
    paleo_slope = controls.get("paleo_slope", None)

    # If controls are missing for some reason, just return the layer map.
    if basins is None or highs is None or paleo_slope is None:
        return top_layer

    ny, nx = top_layer.shape
    facies = np.empty_like(top_layer, dtype=object)

    # Precompute masks for environment
    deep_basin   = basins > 0.7
    mid_basin    = (basins > 0.4) & (basins <= 0.7)
    shallow_basin= (basins > 0.2) & (basins <= 0.4)
    high_zone    = highs > 0.6

    steep        = paleo_slope > 0.5
    moderate_slp = (paleo_slope > 0.2) & (paleo_slope <= 0.5)
    gentle       = paleo_slope <= 0.2

    # Start with a copy: default facies = top layer name
    facies[:] = top_layer

    # --- SANDSTONE PACKAGE: Sandstone vs Conglomerate ---
    mask_sand = top_layer == "Sandstone"
    # Conglomerates near steep paleo-slopes (coarse clastics near steep source)
    mask_cong = mask_sand & (steep | (mid_basin & moderate_slp))
    facies[mask_cong] = "Conglomerate"
    # Remaining sandstone cells keep "Sandstone"

    # --- SHALE PACKAGE: Shale vs Mudstone vs Siltstone ---
    mask_shale = top_layer == "Shale"
    # Deepest basins => Mudstone (very fine, quiet water)
    mask_mud = mask_shale & deep_basin
    facies[mask_mud] = "Mudstone"
    # Highs and gentle slopes => Siltstone (slightly coarser, near shore/shallows)
    mask_silt = mask_shale & high_zone & gentle
    facies[mask_silt] = "Siltstone"
    # Everything else in shale package stays "Shale"

    # --- LIMESTONE PACKAGE: Limestone vs Dolomite vs Evaporite ---
    mask_lime = top_layer == "Limestone"
    # Evaporites only in deepest basins (restricted marine / sabkha-style)
    mask_evap = mask_lime & deep_basin
    facies[mask_evap] = "Evaporite"
    # Dolomite on persistent highs (carbonate platforms)
    mask_dolo = mask_lime & high_zone & (gentle | moderate_slp)
    facies[mask_dolo] = "Dolomite"
    # Remaining limestone stays "Limestone"

    # Everything else (Topsoil, Subsoil, Colluvium, Saprolite, Till, etc.)
    # already has a meaningful name and remains unchanged.
    return facies


# ----------------------- Cross-section plotting ----------------------
def plot_cross_section(strata, row=None, col=None, min_draw_thickness=0.05, ax=None):
  E = strata["surface_elev"]; N = E.shape[0]
  if (row is None) == (col is None): row = N//2
  if row is not None:
      x = np.arange(N); surf = E[row,:]
      tops = {k: v[row,:] for k,v in strata["interfaces"].items()}
      axis_label = "columns (x)"
  else:
      x = np.arange(N); surf = E[:,col]
      tops = {k: v[:,col] for k,v in strata["interfaces"].items()}
      axis_label = "rows (y)"


  order = ["Topsoil","Subsoil","Colluvium","Saprolite","WeatheredBR",
           "Sandstone","Shale","Limestone","Basement","BasementFloor"]

  if ax is None:
      fig, ax = plt.subplots(figsize=(14, 5.5))
  for i in range(len(order)-1, 0, -1):  # bottom-up drawing
      above, here = order[i-1], order[i]
      y_top = tops[above]; y_bot = tops[here]
      y_bot_vis = np.where((y_top - y_bot) < min_draw_thickness, y_top - min_draw_thickness, y_bot)
      ax.fill_between(x, y_bot_vis, y_top, alpha=0.9, linewidth=0.6, zorder=5+i, label=here)
  ax.plot(x, surf, linewidth=2.4, zorder=50, label="Surface")
  ax.set_title("Stratigraphic cross-section (top→bottom)")
  ax.set_xlabel(axis_label); ax.set_ylabel("Elevation (m)")
  ax.legend(ncol=1, fontsize=8, framealpha=0.95, loc="center left", bbox_to_anchor=(1.02, 0.5))
  if ax is None:
      plt.tight_layout(); plt.show()
  return ax


# Convenience: plot both orthogonal sections
def plot_cross_sections_xy(strata, row=None, col=None, min_draw_thickness=0.05):
  N = strata["surface_elev"].shape[0]
  if row is None: row = N // 2
  if col is None: col = N // 2
  fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 11.5), constrained_layout=True)
  # Along X (constant row)
  plot_cross_section(strata, row=row, min_draw_thickness=min_draw_thickness, ax=ax1)
  # Along Y (constant column)
  plot_cross_section(strata, col=col, min_draw_thickness=min_draw_thickness, ax=ax2)
  plt.show()

# ------------- Wind-relevant geological feature plot (separate block) -------------
def plot_wind_structures_debug(wind_structs):
    """
    Visualize where different wind-relevant geological features occur.

    Uses the masks from build_wind_structures(...) and makes a categorical map
    the same size as the terrain.
    """
    E = wind_structs["E"]
    barrier_mask = wind_structs["barrier_mask"]
    channel_mask = wind_structs["channel_mask"]
    basin_mask   = wind_structs["basin_mask"]

    # integer feature codes (0 = none, 1 = barrier, 2 = channel, 3 = basin)
    features = np.zeros_like(E, dtype=int)
    features[barrier_mask] = 1
    features[channel_mask] = 2
    features[basin_mask]   = 3

    fig, ax = plt.subplots(figsize=(6, 6))

    # Use a discrete tab10 colormap with 4 entries: indices 0,1,2,3
    cmap = plt.cm.get_cmap("tab10", 4)

    # vmin/vmax make sure ints 0,1,2,3 map cleanly to those 4 colors
    im = ax.imshow(features,
                   origin="lower",
                   interpolation="nearest",
                   cmap=cmap,
                   vmin=-0.5, vmax=3.5)

    from matplotlib.patches import Patch

    # IMPORTANT: use the SAME indices the image uses: 1, 2, 3
    legend_patches = [
        Patch(color=cmap(1), label="Wind barriers (ridges)"),   # code 1
        Patch(color=cmap(2), label="Wind channels (valleys)"),  # code 2
        Patch(color=cmap(3), label="Basins / bowls"),           # code 3
    ]
    ax.legend(handles=legend_patches, loc="upper right", framealpha=0.9)

    ax.set_title("Wind-relevant geological features")
    ax.set_xlabel("x (columns)")
    ax.set_ylabel("y (rows)")
    plt.tight_layout()
    plt.show()



# ------------------------------ Master Terrain --------------------------------
if __name__ == "__main__":
  z, rng = quantum_seeded_topography(
      N=512, beta=3.2, warp_amp=0.10, ridged_alpha=0.15, random_seed=None
  )




  strata = generate_stratigraphy(
      z_norm=z, rng=rng,
      elev_range_m=700.0, pixel_scale_m=10.0,
      soil_range_m=(0.3, 1.8),
      colluvium_max_m=18.0, wC_gentle=0.35, wC_curv=0.30, wC_low=0.20, wC_twi=0.15,
      C_relief_px=31, C_twi_k1=7, C_twi_k2=13, C_sigma=0.20, C_floor=0.5, C_bias=1.0,
      sap_median=6.0, sap_min=0.5, sap_max=30.0, sap_w_gentle=0.6, sap_w_inter=0.4, sap_relief_px=61, sap_sigma=0.35,
      rind_median=1.8, rind_min=0.4, rind_max=6.0, rind_patch_beta=3.0, rind_patch_alpha=0.5,
      unit_thickness_m=(90.0,110.0,100.0),
      undulation_amp_m=10.0, undulation_beta=3.2,
      dip_deg=6.0, dip_dir_deg=45.0,
      burial_depth_m=120.0, bed_struct_weight=0.45
  )

  # ---------------- Record wind-relevant geological features (read-only) ----------------
  surface_elev  = strata["surface_elev"]
  pixel_scale_m = strata["meta"]["pixel_scale_m"]

  # choose a prevailing wind direction (FROM this angle, in degrees)
  base_wind_dir_deg = 45.0  # NE → SW as example

  wind_structs = build_wind_structures(surface_elev, pixel_scale_m, base_wind_dir_deg)

    # Debug: how many cells are in each mask?
  for key in ["barrier_mask", "channel_mask", "basin_mask"]:
      m = wind_structs[key]
      print(key, "cells:", int(m.sum()))

  # (optional debug)
  print("Wind barriers (ranges):", len(wind_structs["barrier_regions"]))
  print("Wind channels (valleys):", len(wind_structs["channel_regions"]))
  print("Basins / bowls:", len(wind_structs["basin_regions"]))

  # NEW: plot wind-relevant features as a separate map
  plot_wind_structures_debug(wind_structs)




  # Plot 1: surface map
  plt.figure(figsize=(6,6))
  im = plt.imshow(strata["surface_elev"], origin="lower")
  plt.colorbar(im, label="Surface elevation (m)")
  plt.title("Quantum-seeded terrain (absolute elevation)")
  plt.tight_layout(); plt.show()

  # Plots 2 & 3: orthogonal stratigraphic sections (now in one figure)
  plot_cross_sections_xy(strata, row=256, col=256)

# ---------- Wrappers so the weather sim reuses the DEMO terrain ----------

def terrain_topography_from_demo(N=512, **kwargs):
    """
    Ignore N and kwargs; reuse the quantum-seeded terrain from the demo.
    Returns the same (z_norm, rng) that were used to build `strata` above.
    """
    return z, rng  # defined in the demo block


def stratigraphy_from_demo(z_norm, rng_obj, **kwargs):
    """
    Ignore z_norm and rng_obj; reuse the precomputed `strata` from the demo.
    """
    return strata
