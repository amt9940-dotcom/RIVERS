"""
EROSION SYSTEM - EROSION WITH HALF-LOSS RULE (PASS A)

Implements stream power erosion with half-loss rule:
1. Compute erosion power from discharge, slope, and erodibility
2. Apply erosion (lower elevation)
3. Convert eroded material to sediment
4. Apply HALF-LOSS RULE: only 50% of eroded material can move downstream
5. The other 50% is deleted from the system (washed away)

This allows valleys, channels, and lake basins to deepen over time.
"""

import numpy as np
from typing import Tuple, Dict

def get_erodibility_grid(
    top_layer_name: np.ndarray,
    erodibility_map: Dict[str, float]
) -> np.ndarray:
    """
    Create erodibility grid from top layer names.
    
    Parameters
    ----------
    top_layer_name : np.ndarray (ny, nx), dtype=object
        Name of top layer at each cell.
    erodibility_map : dict
        Map from layer name to erodibility multiplier.
    
    Returns
    -------
    erodibility : np.ndarray (ny, nx)
        Erodibility multiplier at each cell.
    """
    ny, nx = top_layer_name.shape
    erodibility = np.ones((ny, nx), dtype=np.float32)
    
    for layer_name, erod_val in erodibility_map.items():
        mask = (top_layer_name == layer_name)
        erodibility[mask] = erod_val
    
    return erodibility


def compute_erosion_pass_a(
    elevation: np.ndarray,
    Q: np.ndarray,
    slope: np.ndarray,
    flow_dir: np.ndarray,
    erodibility: np.ndarray,
    base_k: float,
    flat_k: float,
    max_erode_per_step: float,
    slope_threshold: float,
    half_loss_fraction: float,
    m_discharge: float = 0.5,
    n_slope: float = 1.0,
    dt: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Erosion Pass A: Compute erosion and generate sediment (with half-loss).
    
    Parameters
    ----------
    elevation : np.ndarray (ny, nx)
        Current elevation [m].
    Q : np.ndarray (ny, nx)
        Discharge [m³/yr].
    slope : np.ndarray (ny, nx)
        Slope along flow direction [dimensionless].
    flow_dir : np.ndarray (ny, nx)
        Flow direction index (-1 for pits).
    erodibility : np.ndarray (ny, nx)
        Erodibility multiplier.
    base_k : float
        Base erosion coefficient.
    flat_k : float
        Erosion coefficient for flat areas with high discharge.
    max_erode_per_step : float
        Maximum erosion per timestep [m].
    slope_threshold : float
        Threshold to distinguish flat vs sloping cells.
    half_loss_fraction : float
        Fraction of eroded material deleted (typically 0.5).
    m_discharge : float
        Discharge exponent in stream power law.
    n_slope : float
        Slope exponent in stream power law.
    dt : float
        Timestep [years].
    
    Returns
    -------
    elevation_new : np.ndarray (ny, nx)
        Updated elevation after erosion [m].
    sediment_out : np.ndarray (ny, nx)
        Sediment generated at each cell (after half-loss) [m³].
    """
    ny, nx = elevation.shape
    
    # Copy elevation (will modify)
    elevation_new = elevation.copy()
    
    # Initialize sediment output (volume of movable sediment per cell)
    sediment_out = np.zeros((ny, nx), dtype=np.float32)
    
    # Avoid division by zero in Q
    Q_safe = np.maximum(Q, 1e-6)
    
    # Loop over all cells
    for i in range(ny):
        for j in range(nx):
            # Determine if this is a downslope or flat cell
            is_downslope = (slope[i, j] > slope_threshold) and (flow_dir[i, j] >= 0)
            
            if is_downslope:
                # DOWNSLOPE CELL: normal stream power erosion
                # Erosion power = K * Q^m * S^n * erodibility
                erosion_power = (
                    base_k 
                    * (Q_safe[i, j] ** m_discharge)
                    * (slope[i, j] ** n_slope)
                    * erodibility[i, j]
                    * dt
                )
                
                # Clamp to max erosion
                dz_erosion = -min(max_erode_per_step * dt, erosion_power)
                
            else:
                # FLAT / PIT CELL
                # Check if high discharge (ponding/lake)
                if Q[i, j] > 1000.0:  # Threshold for "high water"
                    # Allow scouring in lakes/ponds
                    erosion_power = (
                        flat_k
                        * (Q_safe[i, j] ** m_discharge)
                        * erodibility[i, j]
                        * dt
                    )
                    dz_erosion = -min(max_erode_per_step * dt, erosion_power)
                else:
                    # Low-energy flat: no erosion
                    dz_erosion = 0.0
            
            # Apply erosion to elevation
            elevation_new[i, j] += dz_erosion  # dz_erosion is negative
            
            # Convert to positive sediment volume
            eroded_material = -dz_erosion  # Now positive
            
            # HALF-LOSS RULE: only half can move downstream
            sediment_to_move = (1.0 - half_loss_fraction) * eroded_material
            sediment_lost = half_loss_fraction * eroded_material
            # sediment_lost is deleted (not stored anywhere)
            
            # Store movable sediment (per unit area → multiply by cell area later)
            sediment_out[i, j] = sediment_to_move
    
    return elevation_new, sediment_out


def test_erosion_pass_a():
    """Quick test of erosion computation."""
    print("Erosion Pass A Test:")
    
    # Create simple slope
    ny, nx = 5, 5
    test_elev = np.arange(ny)[:, None] * np.ones(nx)[None, :]  # Slope from N to S
    test_elev = test_elev.astype(np.float32) * 10.0  # 10m per row
    
    # Flow and discharge
    flow_dir, receivers, slopes = compute_flow_direction_d8(test_elev, pixel_scale_m=10.0)
    rain = np.ones((ny, nx), dtype=np.float32) * 0.5  # 0.5 m/yr
    runoff = compute_runoff(rain, infiltration_fraction=0.3)
    Q = compute_discharge(test_elev, flow_dir, receivers, runoff, pixel_scale_m=10.0)
    
    # Uniform erodibility
    erodibility = np.ones((ny, nx), dtype=np.float32)
    
    # Apply erosion
    elev_new, sediment = compute_erosion_pass_a(
        test_elev, Q, slopes, flow_dir, erodibility,
        base_k=0.001, flat_k=0.0005, max_erode_per_step=0.5,
        slope_threshold=0.001, half_loss_fraction=0.5,
        dt=1.0
    )
    
    total_erosion = test_elev - elev_new
    total_sediment = sediment.sum()
    
    print(f"  Initial elevation range: {test_elev.min():.1f} - {test_elev.max():.1f} m")
    print(f"  Final elevation range: {elev_new.min():.1f} - {elev_new.max():.1f} m")
    print(f"  Total erosion: {total_erosion.sum():.4f} m")
    print(f"  Total sediment (50% of erosion): {total_sediment:.4f} m")
    print(f"  Ratio: {total_sediment / (total_erosion.sum() + 1e-9):.2f} (should be ~0.5)")
    print("✅ Erosion Pass A test complete!")

# Run test
test_erosion_pass_a()

print("\n✅ Erosion Pass A module loaded!")
