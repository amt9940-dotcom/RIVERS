"""
EROSION SYSTEM - RIVER AND LAKE DETECTION

Implements detection of hydrological features:
- Rivers: high-discharge channels following flow paths
- Lakes: local minima (pits) where water accumulates
- Drainage basins: areas contributing to each outlet

Uses discharge thresholds and flow accumulation analysis.
"""

import numpy as np
from typing import Tuple, List

# Import scipy if available (for connected components)
try:
    from scipy import ndimage
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

def detect_rivers(
    Q: np.ndarray,
    flow_dir: np.ndarray,
    receivers: np.ndarray,
    discharge_threshold: float = 5000.0
) -> np.ndarray:
    """
    Detect river cells based on discharge threshold.
    
    Parameters
    ----------
    Q : np.ndarray (ny, nx)
        Discharge [m³/yr].
    flow_dir : np.ndarray (ny, nx)
        Flow direction index.
    receivers : np.ndarray (ny, nx, 2)
        Downstream neighbors.
    discharge_threshold : float
        Minimum discharge to be classified as river [m³/yr].
    
    Returns
    -------
    river_mask : np.ndarray (ny, nx), dtype=bool
        True for river cells.
    """
    river_mask = Q >= discharge_threshold
    
    # Also require that the cell has a downstream neighbor
    # (not a pit, unless it's a very high-discharge pit)
    has_outlet = flow_dir >= 0
    river_mask = river_mask & (has_outlet | (Q > discharge_threshold * 2))
    
    return river_mask


def detect_lakes(
    elevation: np.ndarray,
    flow_dir: np.ndarray,
    Q: np.ndarray,
    min_discharge_threshold: float = 1000.0,
    min_area_cells: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect lakes as local minima (pits) with significant discharge.
    
    Parameters
    ----------
    elevation : np.ndarray (ny, nx)
        Elevation [m].
    flow_dir : np.ndarray (ny, nx)
        Flow direction (-1 for pits).
    Q : np.ndarray (ny, nx)
        Discharge [m³/yr].
    min_discharge_threshold : float
        Minimum discharge for a pit to be a lake [m³/yr].
    min_area_cells : int
        Minimum contiguous area for a lake [cells].
    
    Returns
    -------
    lake_mask : np.ndarray (ny, nx), dtype=bool
        True for lake cells.
    lake_labels : np.ndarray (ny, nx), dtype=int
        Lake ID for each cell (0 for non-lake).
    """
    ny, nx = elevation.shape
    
    # Find pits (cells with no downstream neighbor)
    pit_mask = (flow_dir == -1)
    
    # Lakes must have sufficient water accumulation
    lake_seed_mask = pit_mask & (Q >= min_discharge_threshold)
    
    # Lakes may extend beyond just the lowest cell
    # Expand to include nearby low-lying cells with high discharge
    lake_mask = lake_seed_mask.copy()
    
    # Simple expansion: include neighbors of seed cells if they're similar elevation
    # and have high discharge
    di = np.array([-1, -1, 0, 1, 1, 1, 0, -1])
    dj = np.array([0, 1, 1, 1, 0, -1, -1, -1])
    
    for i in range(ny):
        for j in range(nx):
            if lake_seed_mask[i, j]:
                z_seed = elevation[i, j]
                # Check neighbors
                for k in range(8):
                    ni = (i + di[k]) % ny
                    nj = (j + dj[k]) % nx
                    # Include if within 1m elevation and has discharge
                    if (abs(elevation[ni, nj] - z_seed) < 1.0 and 
                        Q[ni, nj] > min_discharge_threshold * 0.5):
                        lake_mask[ni, nj] = True
    
    # Label connected components
    if HAVE_SCIPY:
        try:
            lake_labels, num_lakes = ndimage.label(lake_mask)
            
            # Filter by size
            for lake_id in range(1, num_lakes + 1):
                lake_size = np.sum(lake_labels == lake_id)
                if lake_size < min_area_cells:
                    lake_labels[lake_labels == lake_id] = 0
            
            # Update mask
            lake_mask = lake_labels > 0
        except Exception:
            # If scipy fails, just use the expanded mask
            lake_labels = lake_mask.astype(np.int32)
    else:
        # Without scipy, just use the expanded mask
        lake_labels = lake_mask.astype(np.int32)
    
    return lake_mask, lake_labels


def detect_drainage_basins(
    flow_dir: np.ndarray,
    receivers: np.ndarray,
    min_basin_size: int = 100
) -> np.ndarray:
    """
    Identify drainage basins (watersheds).
    
    Each basin is a set of cells that flow to the same outlet.
    
    Parameters
    ----------
    flow_dir : np.ndarray (ny, nx)
        Flow direction index.
    receivers : np.ndarray (ny, nx, 2)
        Downstream neighbors.
    min_basin_size : int
        Minimum basin size [cells].
    
    Returns
    -------
    basin_labels : np.ndarray (ny, nx), dtype=int
        Basin ID for each cell (0 for small basins).
    """
    ny, nx = flow_dir.shape
    basin_labels = np.zeros((ny, nx), dtype=np.int32)
    
    # Find outlets (pits or edge cells)
    outlet_mask = (flow_dir == -1)
    
    # Assign each outlet a basin ID
    outlet_cells = np.argwhere(outlet_mask)
    basin_id = 1
    
    for outlet_pos in outlet_cells:
        oi, oj = outlet_pos
        
        # Trace backwards to find all cells that drain to this outlet
        basin_mask = np.zeros((ny, nx), dtype=bool)
        basin_mask[oi, oj] = True
        
        # Simple backward tracing (not efficient but clear)
        changed = True
        iterations = 0
        while changed and iterations < 100:
            changed = False
            iterations += 1
            for i in range(ny):
                for j in range(nx):
                    if not basin_mask[i, j] and flow_dir[i, j] >= 0:
                        # Check if this cell flows to a cell in the basin
                        ni, nj = receivers[i, j]
                        if ni >= 0 and nj >= 0 and basin_mask[ni, nj]:
                            basin_mask[i, j] = True
                            changed = True
        
        # Assign basin ID if large enough
        basin_size = np.sum(basin_mask)
        if basin_size >= min_basin_size:
            basin_labels[basin_mask] = basin_id
            basin_id += 1
    
    return basin_labels


def test_river_lake_detection():
    """Test river and lake detection."""
    print("River and Lake Detection Test:")
    
    # Create valley with central depression (lake)
    ny, nx = 21, 21
    test_elev = np.zeros((ny, nx), dtype=np.float32)
    
    # Create bowl shape
    cy, cx = ny // 2, nx // 2
    for i in range(ny):
        for j in range(nx):
            r = np.sqrt((i - cy)**2 + (j - cx)**2)
            test_elev[i, j] = r * 2.0
    
    # Add depression in center (lake)
    test_elev[cy-1:cy+2, cx-1:cx+2] = 5.0
    test_elev[cy, cx] = 4.0  # Lowest point
    
    # Compute flow
    flow_dir, receivers, slopes = compute_flow_direction_d8(test_elev, pixel_scale_m=10.0)
    
    # Uniform rainfall
    rain = np.ones((ny, nx), dtype=np.float32) * 1.0
    runoff = compute_runoff(rain, infiltration_fraction=0.3)
    Q = compute_discharge(test_elev, flow_dir, receivers, runoff, pixel_scale_m=10.0)
    
    # Detect features
    rivers = detect_rivers(Q, flow_dir, receivers, discharge_threshold=500.0)
    lakes, lake_labels = detect_lakes(test_elev, flow_dir, Q, 
                                      min_discharge_threshold=100.0, min_area_cells=1)
    
    print(f"  Domain: {ny}×{nx}")
    print(f"  River cells: {np.sum(rivers)}")
    print(f"  Lake cells: {np.sum(lakes)}")
    print(f"  Number of lakes: {lake_labels.max()}")
    print(f"  Max discharge: {Q.max():.1f} m³/yr")
    print(f"  Lake location: center={lakes[cy, cx]}")
    print("✅ River and lake detection test complete!")

# Run test
test_river_lake_detection()

print("\n✅ River and lake detection module loaded!")
