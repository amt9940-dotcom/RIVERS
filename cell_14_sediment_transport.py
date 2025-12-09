"""
EROSION SYSTEM - SEDIMENT TRANSPORT AND DEPOSITION (PASS B)

Implements capacity-based sediment transport:
1. Route sediment from upstream to downstream
2. Compare total sediment vs transport capacity
3. Deposit excess sediment (raise elevation)
4. Carry remaining sediment downstream

This creates realistic alluvial fans, floodplains, and delta deposits.
"""

import numpy as np
from typing import Tuple

def compute_sediment_transport_pass_b(
    elevation: np.ndarray,
    Q: np.ndarray,
    slope: np.ndarray,
    flow_dir: np.ndarray,
    receivers: np.ndarray,
    sediment_out: np.ndarray,
    capacity_k: float,
    capacity_m: float,
    capacity_n: float,
    pixel_scale_m: float
) -> np.ndarray:
    """
    Sediment Transport Pass B: Route sediment and apply deposition.
    
    Parameters
    ----------
    elevation : np.ndarray (ny, nx)
        Current elevation [m] (after erosion).
    Q : np.ndarray (ny, nx)
        Discharge [m³/yr].
    slope : np.ndarray (ny, nx)
        Slope along flow direction.
    flow_dir : np.ndarray (ny, nx)
        Flow direction index (-1 for pits).
    receivers : np.ndarray (ny, nx, 2)
        Downstream neighbor coordinates.
    sediment_out : np.ndarray (ny, nx)
        Sediment generated locally [m] (from Pass A).
    capacity_k : float
        Transport capacity coefficient.
    capacity_m : float
        Discharge exponent for capacity.
    capacity_n : float
        Slope exponent for capacity.
    pixel_scale_m : float
        Grid cell size [m].
    
    Returns
    -------
    elevation_new : np.ndarray (ny, nx)
        Updated elevation after deposition [m].
    """
    ny, nx = elevation.shape
    
    # Copy elevation
    elevation_new = elevation.copy()
    
    # Initialize sediment flux arrays
    sediment_in = np.zeros((ny, nx), dtype=np.float32)  # Incoming from upstream
    
    # Avoid division by zero
    Q_safe = np.maximum(Q, 1e-6)
    slope_safe = np.maximum(slope, 1e-9)
    
    # Sort cells by elevation (high to low) for topological ordering
    indices = [(i, j) for i in range(ny) for j in range(nx)]
    indices_sorted = sorted(indices, key=lambda idx: elevation[idx], reverse=True)
    
    # Process each cell from high to low
    for (i, j) in indices_sorted:
        # Total sediment at this cell = incoming + locally produced
        total_sediment = sediment_in[i, j] + sediment_out[i, j]
        
        # Compute transport capacity
        # Capacity = K * Q^m * S^n [dimensionless or m]
        capacity = (
            capacity_k
            * (Q_safe[i, j] ** capacity_m)
            * (slope_safe[i, j] ** capacity_n)
        )
        
        # Flat cells have very low capacity → tend to deposit
        if slope[i, j] < 1e-6:
            capacity = capacity * 0.1  # Reduce capacity in flats
        
        # Compare sediment vs capacity
        if total_sediment > capacity:
            # Too much sediment: deposit the excess
            deposit = total_sediment - capacity
            elevation_new[i, j] += deposit
            sediment_to_downstream = capacity
        else:
            # Can carry everything
            deposit = 0.0
            sediment_to_downstream = total_sediment
        
        # Send sediment downstream
        if flow_dir[i, j] >= 0:
            ni, nj = receivers[i, j]
            if ni >= 0 and nj >= 0:  # Valid receiver
                sediment_in[ni, nj] += sediment_to_downstream
        # else: sediment lost off edge (or in pit)
    
    return elevation_new


def test_sediment_transport():
    """Quick test of sediment transport and deposition."""
    print("Sediment Transport Pass B Test:")
    
    # Create slope with flat area at bottom
    ny, nx = 7, 7
    test_elev = np.zeros((ny, nx), dtype=np.float32)
    
    # Steep slope at top, flat at bottom
    for i in range(ny):
        if i < 4:
            test_elev[i, :] = (ny - i) * 5.0  # Steep
        else:
            test_elev[i, :] = 10.0  # Flat (depositional area)
    
    # Flow and discharge
    flow_dir, receivers, slopes = compute_flow_direction_d8(test_elev, pixel_scale_m=10.0)
    rain = np.ones((ny, nx), dtype=np.float32) * 0.5
    runoff = compute_runoff(rain, infiltration_fraction=0.3)
    Q = compute_discharge(test_elev, flow_dir, receivers, runoff, pixel_scale_m=10.0)
    
    # Generate some sediment (simulate erosion)
    sediment = np.zeros((ny, nx), dtype=np.float32)
    sediment[:4, :] = 0.1  # Sediment from eroded steep area
    
    # Uniform erodibility
    erodibility = np.ones((ny, nx), dtype=np.float32)
    
    # Apply transport
    elev_new = compute_sediment_transport_pass_b(
        test_elev, Q, slopes, flow_dir, receivers, sediment,
        capacity_k=0.01, capacity_m=0.5, capacity_n=1.0,
        pixel_scale_m=10.0
    )
    
    # Check deposition
    deposition = elev_new - test_elev
    
    print(f"  Initial elevation range: {test_elev.min():.1f} - {test_elev.max():.1f} m")
    print(f"  Final elevation range: {elev_new.min():.1f} - {elev_new.max():.1f} m")
    print(f"  Total deposition: {deposition.sum():.4f} m")
    print(f"  Max deposition location: row {np.unravel_index(deposition.argmax(), deposition.shape)[0]}")
    print(f"  (Should be in flat area, rows 4-6)")
    print("✅ Sediment Transport Pass B test complete!")

# Run test
test_sediment_transport()

print("\n✅ Sediment Transport Pass B module loaded!")
