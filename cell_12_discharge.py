"""
EROSION SYSTEM - DISCHARGE COMPUTATION

Computes discharge (Q) = total water passing through each cell.
- Processes cells from highest to lowest elevation
- Accumulates local runoff + upstream contributions
- Critical for stream power erosion
"""

import numpy as np
from typing import Tuple

def compute_runoff(
    rain: np.ndarray,
    infiltration_fraction: float = 0.3
) -> np.ndarray:
    """
    Compute runoff from rainfall after infiltration.
    
    Parameters
    ----------
    rain : np.ndarray (ny, nx)
        Rainfall depth [meters or mm].
    infiltration_fraction : float
        Fraction of rain that infiltrates (0-1).
    
    Returns
    -------
    runoff : np.ndarray (ny, nx)
        Surface runoff [same units as rain].
    """
    runoff = rain * (1.0 - infiltration_fraction)
    runoff = np.maximum(0.0, runoff)  # Ensure non-negative
    return runoff


def compute_discharge(
    elevation: np.ndarray,
    flow_dir: np.ndarray,
    receivers: np.ndarray,
    runoff: np.ndarray,
    pixel_scale_m: float
) -> np.ndarray:
    """
    Compute discharge Q (total water passing through each cell).
    
    Uses topological sorting: processes cells from high to low elevation
    so upstream contributions are computed before downstream.
    
    Parameters
    ----------
    elevation : np.ndarray (ny, nx)
        Elevation grid [m].
    flow_dir : np.ndarray (ny, nx)
        Flow direction indices (-1 for pits).
    receivers : np.ndarray (ny, nx, 2)
        Downstream neighbor coordinates.
    runoff : np.ndarray (ny, nx)
        Local runoff contribution [m/yr or m].
    pixel_scale_m : float
        Grid cell size [m].
    
    Returns
    -------
    Q : np.ndarray (ny, nx)
        Discharge = local runoff + upstream water [m²/yr or m²].
    """
    ny, nx = elevation.shape
    
    # Initialize discharge with local runoff (converted to flux)
    # Q has units [m³/yr] = [m/yr] * [m²]
    cell_area = pixel_scale_m ** 2
    Q = runoff * cell_area  # Local contribution
    
    # Create list of all cells sorted by elevation (high to low)
    indices = [(i, j) for i in range(ny) for j in range(nx)]
    indices_sorted = sorted(indices, key=lambda idx: elevation[idx], reverse=True)
    
    # Process cells from high to low
    for (i, j) in indices_sorted:
        # If this cell has a downstream neighbor, send water there
        if flow_dir[i, j] >= 0:
            ni, nj = receivers[i, j]
            if ni >= 0 and nj >= 0:  # Valid receiver
                Q[ni, nj] += Q[i, j]
    
    return Q


def test_discharge():
    """Quick test of discharge computation."""
    # Create simple valley: water flows from edges to center
    ny, nx = 7, 7
    test_elev = np.zeros((ny, nx), dtype=np.float32)
    
    # Create bowl shape (low in center)
    for i in range(ny):
        for j in range(nx):
            di = i - ny // 2
            dj = j - nx // 2
            test_elev[i, j] = di**2 + dj**2
    
    # Invert so center is low (valley)
    test_elev = test_elev.max() - test_elev
    
    # Compute flow
    flow_dir, receivers, slopes = compute_flow_direction_d8(test_elev, pixel_scale_m=10.0)
    
    # Uniform rainfall
    rain = np.ones((ny, nx), dtype=np.float32) * 0.1  # 0.1 m/yr
    runoff = compute_runoff(rain, infiltration_fraction=0.3)
    
    # Compute discharge
    Q = compute_discharge(test_elev, flow_dir, receivers, runoff, pixel_scale_m=10.0)
    
    print("Discharge Test:")
    print(f"  Domain: {ny}×{nx}")
    print(f"  Uniform rain: {rain[0,0]} m/yr")
    print(f"  Runoff: {runoff[0,0]} m/yr (after 30% infiltration)")
    print(f"  Q range: {Q.min():.2f} - {Q.max():.2f} m³/yr")
    print(f"  Q at center (valley bottom): {Q[ny//2, nx//2]:.2f} m³/yr")
    print(f"  Q at corner (ridge): {Q[0, 0]:.2f} m³/yr")
    print("✅ Discharge test complete!")

# Run test
test_discharge()

print("\n✅ Discharge computation module loaded!")
