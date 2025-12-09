#!/usr/bin/env python3
"""
PASTE THIS INTO NOTEBOOK CELL 1: Terrain Generator (NO SCIPY VERSION)

This version doesn't require scipy - it's simplified but still works!
"""

import numpy as np
import matplotlib.pyplot as plt

print("Loading terrain generation functions (scipy-free version)...")

# ======================= Fractional Brownian Motion =======================
def fractional_surface(N, beta=3.0, rng=None):
    """Generate fractal terrain using FFT."""
    if rng is None:
        rng = np.random.default_rng()
    
    freq = np.fft.fftfreq(N)
    kx, ky = np.meshgrid(freq, freq, indexing='ij')
    k = np.sqrt(kx**2 + ky**2)
    k[0, 0] = 1.0
    
    power_spectrum = k ** (-beta)
    power_spectrum[0, 0] = 0.0
    
    phase = rng.uniform(0, 2*np.pi, size=(N, N))
    fourier_coeff = np.sqrt(power_spectrum) * np.exp(1j * phase)
    
    z = np.fft.ifft2(fourier_coeff).real
    z = (z - z.min()) / (z.max() - z.min() + 1e-12)
    
    return z


def ridged_mix(z, alpha=0.18):
    """Add ridges to terrain."""
    ridged = 1.0 - np.abs(2.0*z - 1.0)
    return (1.0 - alpha)*z + alpha*ridged


def simple_blur(z, iterations=2):
    """Simple blur without scipy."""
    result = z.copy()
    for _ in range(iterations):
        up = np.roll(result, -1, axis=0)
        down = np.roll(result, 1, axis=0)
        left = np.roll(result, 1, axis=1)
        right = np.roll(result, -1, axis=1)
        result = (result + up + down + left + right) / 5.0
    return result


def quantum_seeded_topography(N=512, beta=3.1, random_seed=None):
    """
    Generate quantum-seeded terrain (simplified, no scipy needed).
    
    Args:
        N: Grid size (e.g., 128, 256, 512)
        beta: Fractal dimension parameter (2.5-3.5)
        random_seed: Random seed for reproducibility
    
    Returns:
        z_norm: Normalized terrain (0-1)
        rng: Random number generator
    """
    rng = np.random.default_rng(random_seed)
    
    # Generate base terrain
    base_low = fractional_surface(N, beta=beta, rng=rng)
    base_high = fractional_surface(N, beta=beta-0.4, rng=rng)
    z = 0.65*base_low + 0.35*base_high
    
    # Add ridges
    z = ridged_mix(z, alpha=0.18)
    
    # Simple smoothing
    z = simple_blur(z, iterations=2)
    
    # Normalize
    z = (z - z.min()) / (z.max() - z.min() + 1e-12)
    
    return z, rng


def _normalize(x):
    """Normalize array to 0-1."""
    xmin = float(x.min())
    xmax = float(x.max())
    if xmax - xmin < 1e-12:
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)


def generate_stratigraphy(
    z_norm,
    elev_range_m,
    pixel_scale_m,
    rng,
    dip_deg=5.0,
    dip_dir_deg=45.0,
    **kwargs
):
    """
    Generate layered stratigraphy from normalized terrain.
    
    Args:
        z_norm: Normalized terrain (0-1)
        elev_range_m: Elevation range in meters
        pixel_scale_m: Cell size in meters
        rng: Random number generator
        dip_deg: Dip angle of layers
        dip_dir_deg: Dip direction
    
    Returns:
        Dict with:
            - surface_elev: 2D elevation array (m)
            - interfaces: dict of layer top elevations
            - thickness: dict of layer thicknesses
            - properties: dict of material properties
            - deposits: dict of surficial deposits
            - meta: metadata
    """
    N = z_norm.shape[0]
    
    # Scale to actual elevations
    E = z_norm * elev_range_m
    
    # Create layer structure
    depth_factor = _normalize(1.0 - z_norm)  # deeper in valleys
    
    # Layer thicknesses
    t_topsoil = 0.5 + 1.5 * rng.random(size=(N, N))
    t_subsoil = 1.0 + 2.0 * rng.random(size=(N, N))
    t_colluvium = 0.5 + 1.5 * rng.random(size=(N, N))
    t_saprolite = 5.0 + 10.0 * depth_factor
    t_weathered = 10.0 + 20.0 * depth_factor
    t_sandstone = 50.0 + 50.0 * depth_factor
    t_shale = 40.0 + 40.0 * depth_factor
    t_limestone = 30.0 + 30.0 * depth_factor
    t_basement = 100.0 + 100.0 * depth_factor
    
    # Build interfaces (top-down)
    top_topsoil = E
    top_subsoil = top_topsoil - t_topsoil
    top_colluvium = top_subsoil - t_subsoil
    top_saprolite = top_colluvium - t_colluvium
    top_weathered = top_saprolite - t_saprolite
    top_sandstone = top_weathered - t_weathered
    top_shale = top_sandstone - t_sandstone
    top_limestone = top_shale - t_shale
    top_basement = top_limestone - t_limestone
    top_basement_floor = top_basement - t_basement
    
    interfaces = {
        "Topsoil": top_topsoil,
        "Subsoil": top_subsoil,
        "Colluvium": top_colluvium,
        "Saprolite": top_saprolite,
        "WeatheredBR": top_weathered,
        "Sandstone": top_sandstone,
        "Shale": top_shale,
        "Limestone": top_limestone,
        "Basement": top_basement,
        "BasementFloor": top_basement_floor,
    }
    
    thickness = {
        "Topsoil": t_topsoil,
        "Subsoil": t_subsoil,
        "Colluvium": t_colluvium,
        "Saprolite": t_saprolite,
        "WeatheredBR": t_weathered,
        "Sandstone": t_sandstone,
        "Shale": t_shale,
        "Limestone": t_limestone,
        "Basement": t_basement,
        "BasementFloor": np.zeros_like(E),
    }
    
    # Material properties (erodibility, density, porosity)
    properties = {
        "Topsoil": {"erodibility": 1.00, "density": 1600, "porosity": 0.45, "K_rel": 1.00},
        "Subsoil": {"erodibility": 0.85, "density": 1700, "porosity": 0.40, "K_rel": 0.85},
        "Colluvium": {"erodibility": 0.90, "density": 1750, "porosity": 0.35, "K_rel": 0.90},
        "Alluvium": {"erodibility": 0.95, "density": 1700, "porosity": 0.40, "K_rel": 0.95},
        "Till": {"erodibility": 0.75, "density": 1900, "porosity": 0.25, "K_rel": 0.75},
        "Loess": {"erodibility": 1.05, "density": 1550, "porosity": 0.50, "K_rel": 1.05},
        "DuneSand": {"erodibility": 0.95, "density": 1650, "porosity": 0.40, "K_rel": 0.95},
        "Saprolite": {"erodibility": 0.70, "density": 1900, "porosity": 0.30, "K_rel": 0.70},
        "WeatheredBR": {"erodibility": 0.55, "density": 2100, "porosity": 0.20, "K_rel": 0.55},
        "Shale": {"erodibility": 0.45, "density": 2300, "porosity": 0.12, "K_rel": 0.45},
        "Sandstone": {"erodibility": 0.30, "density": 2200, "porosity": 0.18, "K_rel": 0.30},
        "Limestone": {"erodibility": 0.28, "density": 2400, "porosity": 0.08, "K_rel": 0.28},
        "Basement": {"erodibility": 0.15, "density": 2700, "porosity": 0.01, "K_rel": 0.15},
        "BasementFloor": {"erodibility": 0.02, "density": 2850, "porosity": 0.005, "K_rel": 0.02},
    }
    
    # Surficial deposits
    alluvium = np.where(_normalize(1.0 - z_norm) > 0.7, 
                        rng.random(size=(N, N)) * 2.0, 
                        0.0)
    
    deposits = {
        "Till": np.zeros_like(E),
        "Loess": np.zeros_like(E),
        "DuneSand": np.zeros_like(E),
        "Alluvium": alluvium,
    }
    
    return {
        "surface_elev": E,
        "interfaces": interfaces,
        "thickness": thickness,
        "properties": properties,
        "deposits": deposits,
        "meta": {
            "elev_range_m": elev_range_m,
            "pixel_scale_m": pixel_scale_m,
            "dip_deg": dip_deg,
            "dip_dir_deg": dip_dir_deg,
        }
    }


print("✓ Terrain generation functions loaded (scipy-free version)!")
print("  Main functions:")
print("    - quantum_seeded_topography()")
print("    - generate_stratigraphy()")
print()
print("NOTE: This is a simplified version that doesn't need scipy.")
print("      The terrain will be slightly less detailed but still works great!")
