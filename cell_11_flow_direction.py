"""
EROSION SYSTEM - FLOW DIRECTION (D8 Algorithm)

Implements D8 flow direction:
- Each cell flows to its steepest downhill neighbor (8-connectivity)
- Handles flat areas and local pits (potential lakes)
- Returns flow direction indices and receiver coordinates
"""

import numpy as np
from typing import Tuple, Optional

def compute_flow_direction_d8(
    elevation: np.ndarray,
    pixel_scale_m: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute D8 flow direction for each cell.
    
    Parameters
    ----------
    elevation : np.ndarray (ny, nx)
        Elevation grid in meters.
    pixel_scale_m : float
        Grid cell size in meters.
    
    Returns
    -------
    flow_dir : np.ndarray (ny, nx), dtype=int8
        Flow direction index (0-7 for neighbors, -1 for no flow/pit).
        Directions: 0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW
    receivers : np.ndarray (ny, nx, 2), dtype=int32
        Coordinates [i, j] of downstream neighbor. [-1, -1] for pits.
    slope : np.ndarray (ny, nx), dtype=float32
        Slope to downstream neighbor (elevation drop / distance).
    """
    ny, nx = elevation.shape
    
    # Initialize outputs
    flow_dir = np.full((ny, nx), -1, dtype=np.int8)
    receivers = np.full((ny, nx, 2), -1, dtype=np.int32)
    slopes = np.zeros((ny, nx), dtype=np.float32)
    
    # 8 neighbor offsets (N, NE, E, SE, S, SW, W, NW)
    di = np.array([-1, -1, 0, 1, 1, 1, 0, -1], dtype=np.int32)
    dj = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int32)
    
    # Distances to neighbors
    sqrt2 = np.sqrt(2.0)
    distances = np.array([
        pixel_scale_m, pixel_scale_m * sqrt2, pixel_scale_m, pixel_scale_m * sqrt2,
        pixel_scale_m, pixel_scale_m * sqrt2, pixel_scale_m, pixel_scale_m * sqrt2
    ], dtype=np.float32)
    
    # Find steepest downhill neighbor for each cell
    for i in range(ny):
        for j in range(nx):
            z_center = elevation[i, j]
            steepest_slope = 0.0
            steepest_dir = -1
            
            # Check all 8 neighbors
            for k in range(8):
                # Use periodic boundary conditions (wrapping)
                ni = (i + di[k]) % ny
                nj = (j + dj[k]) % nx
                
                # Compute slope (positive if downhill)
                dz = z_center - elevation[ni, nj]
                slope = dz / distances[k]
                
                # Track steepest downhill direction
                if slope > steepest_slope:
                    steepest_slope = slope
                    steepest_dir = k
            
            # Store results
            if steepest_dir >= 0 and steepest_slope > 0:
                flow_dir[i, j] = steepest_dir
                receivers[i, j, 0] = (i + di[steepest_dir]) % ny
                receivers[i, j, 1] = (j + dj[steepest_dir]) % nx
                slopes[i, j] = steepest_slope
            # else: flow_dir[i,j] remains -1 (pit/flat), receivers remain [-1,-1]
    
    return flow_dir, receivers, slopes


def test_flow_direction():
    """Quick test of flow direction computation."""
    # Create simple test terrain: 5×5 grid with peak in center
    test_elev = np.array([
        [10, 11, 12, 11, 10],
        [11, 13, 15, 13, 11],
        [12, 15, 20, 15, 12],  # Peak at center
        [11, 13, 15, 13, 11],
        [10, 11, 12, 11, 10]
    ], dtype=np.float32)
    
    flow_dir, receivers, slopes = compute_flow_direction_d8(test_elev, pixel_scale_m=10.0)
    
    print("Flow Direction Test:")
    print(f"  Peak at center (2,2): elev={test_elev[2,2]}")
    print(f"  Center flows to: dir={flow_dir[2,2]}, receiver={receivers[2,2]}")
    print(f"  Corner (0,0) flows to: dir={flow_dir[0,0]}, receiver={receivers[0,0]}")
    print(f"  Number of pits: {np.sum(flow_dir == -1)}")
    print("✅ Flow direction test complete!")

# Run test
test_flow_direction()

print("\n✅ Flow direction module loaded!")
