"""
EROSION SYSTEM - HILLSLOPE DIFFUSION

Implements hillslope diffusion (creep, soil movement):
- Material slowly slides from higher cells to lower neighbors
- Smooths sharp ridges and cliffs
- Simulates mass wasting and soil creep
- Uses simple diffusion equation
"""

import numpy as np

def apply_hillslope_diffusion(
    elevation: np.ndarray,
    diffusion_k: float,
    pixel_scale_m: float,
    dt: float = 1.0
) -> np.ndarray:
    """
    Apply hillslope diffusion to smooth terrain.
    
    Uses explicit finite difference scheme:
    dz/dt = K * ∇²z
    
    Where ∇²z is the Laplacian (sum of height differences to neighbors).
    
    Parameters
    ----------
    elevation : np.ndarray (ny, nx)
        Current elevation [m].
    diffusion_k : float
        Diffusion coefficient [m²/yr].
    pixel_scale_m : float
        Grid cell size [m].
    dt : float
        Timestep [years].
    
    Returns
    -------
    elevation_new : np.ndarray (ny, nx)
        Smoothed elevation [m].
    """
    ny, nx = elevation.shape
    elevation_new = elevation.copy()
    
    # 4-neighbor diffusion (N, E, S, W)
    # Could also use 8-neighbors for more isotropic diffusion
    di = np.array([-1, 0, 1, 0])
    dj = np.array([0, 1, 0, -1])
    
    # Diffusion coefficient scaled by timestep and grid spacing
    # Stability condition: K_eff * dt < dx² / 4 (for 2D)
    # We'll clamp to ensure stability
    k_eff = diffusion_k / (pixel_scale_m ** 2)
    max_k_dt = 0.2  # Safety factor for stability
    k_dt = min(k_eff * dt, max_k_dt)
    
    # Apply diffusion
    for i in range(ny):
        for j in range(nx):
            z_center = elevation[i, j]
            laplacian = 0.0
            
            # Sum height differences to neighbors
            for k in range(4):
                ni = (i + di[k]) % ny
                nj = (j + dj[k]) % nx
                laplacian += (elevation[ni, nj] - z_center)
            
            # Update elevation
            elevation_new[i, j] = z_center + k_dt * laplacian
    
    return elevation_new


def apply_hillslope_diffusion_8neighbor(
    elevation: np.ndarray,
    diffusion_k: float,
    pixel_scale_m: float,
    dt: float = 1.0
) -> np.ndarray:
    """
    Apply hillslope diffusion with 8-neighbor connectivity.
    
    More isotropic than 4-neighbor version.
    
    Parameters
    ----------
    elevation : np.ndarray (ny, nx)
        Current elevation [m].
    diffusion_k : float
        Diffusion coefficient [m²/yr].
    pixel_scale_m : float
        Grid cell size [m].
    dt : float
        Timestep [years].
    
    Returns
    -------
    elevation_new : np.ndarray (ny, nx)
        Smoothed elevation [m].
    """
    ny, nx = elevation.shape
    elevation_new = elevation.copy()
    
    # 8-neighbor offsets
    di = np.array([-1, -1, 0, 1, 1, 1, 0, -1])
    dj = np.array([0, 1, 1, 1, 0, -1, -1, -1])
    
    # Weights (diagonal neighbors are sqrt(2) away)
    sqrt2 = np.sqrt(2.0)
    weights = np.array([1.0, 1.0/sqrt2, 1.0, 1.0/sqrt2, 
                       1.0, 1.0/sqrt2, 1.0, 1.0/sqrt2])
    weights = weights / weights.sum()  # Normalize
    
    # Diffusion coefficient
    k_eff = diffusion_k / (pixel_scale_m ** 2)
    max_k_dt = 0.15  # More conservative for 8-neighbor
    k_dt = min(k_eff * dt, max_k_dt)
    
    # Apply diffusion
    for i in range(ny):
        for j in range(nx):
            z_center = elevation[i, j]
            weighted_laplacian = 0.0
            
            for k in range(8):
                ni = (i + di[k]) % ny
                nj = (j + dj[k]) % nx
                weighted_laplacian += weights[k] * (elevation[ni, nj] - z_center)
            
            elevation_new[i, j] = z_center + k_dt * weighted_laplacian * 8.0
    
    return elevation_new


def test_hillslope_diffusion():
    """Test diffusion smoothing."""
    print("Hillslope Diffusion Test:")
    
    # Create sharp peak
    ny, nx = 11, 11
    test_elev = np.zeros((ny, nx), dtype=np.float32)
    
    # Put a spike in the center
    cy, cx = ny // 2, nx // 2
    test_elev[cy, cx] = 10.0
    
    initial_max = test_elev.max()
    initial_std = test_elev.std()
    
    # Apply diffusion several times
    elev = test_elev.copy()
    for _ in range(10):
        elev = apply_hillslope_diffusion_8neighbor(
            elev, diffusion_k=0.1, pixel_scale_m=10.0, dt=1.0
        )
    
    final_max = elev.max()
    final_std = elev.std()
    
    print(f"  Initial: max={initial_max:.2f}, std={initial_std:.2f}")
    print(f"  After 10 steps: max={final_max:.2f}, std={final_std:.2f}")
    print(f"  Peak reduced by {(1 - final_max/initial_max)*100:.1f}%")
    print(f"  Peak spread to ~{np.sum(elev > 0.1)} cells")
    print("✅ Hillslope diffusion test complete!")

# Run test
test_hillslope_diffusion()

print("\n✅ Hillslope diffusion module loaded!")
