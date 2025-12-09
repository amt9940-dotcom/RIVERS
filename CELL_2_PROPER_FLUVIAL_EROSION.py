"""
CELL 2: PROPER FLUVIAL EROSION MODEL

Implements the user's exact specifications:
1. Extreme rain boost (100×)
2. Runoff calculation
3. D8 flow direction
4. Discharge accumulation
5. Slope computation
6. TWO-PASS erosion:
   - Pass A: Erosion with HALF-LOSS RULE (50% removed, 50% transported)
   - Pass B: Transport capacity + deposition
7. Sediment routing downstream
8. Optional hillslope diffusion

KEY DIFFERENCE from particle model:
- Half-loss rule creates NET VOLUME LOSS
- Valleys, channels, and lakes can deepen over time
- No random deposition uphill
"""
import numpy as np

# ==============================================================================
# GLOBAL PARAMETERS
# ==============================================================================

# Extreme rain strength
RAIN_BOOST = 100.0  # Each unit of rain = 100× erosive power

# Erosion parameters
BASE_K = 1e-4  # Base erosion coefficient (small because rain is boosted)
FLAT_K = 1e-5  # Erosion in flat/pit cells (weaker)
MAX_ERODE_PER_STEP = 5.0  # Maximum elevation change per step (meters)

# Transport capacity
CAPACITY_K = 0.1  # Transport capacity coefficient

# Infiltration
INFILTRATION_RATE = 0.3  # Fraction of rain that infiltrates (doesn't run off)

# Hillslope diffusion
DIFFUSION_K = 0.01  # Diffusion coefficient (optional smoothing)

# Slope threshold for "flat" cells
FLAT_SLOPE_THRESHOLD = 0.001  # Cells flatter than this are "flat"

print(f"⚡ PROPER FLUVIAL EROSION MODEL LOADED")
print(f"   Rain boost: {RAIN_BOOST}×")
print(f"   Half-loss rule: 50% eroded material removed, 50% transported")
print(f"   Result: NET VOLUME LOSS → valleys can deepen!")

# ==============================================================================
# 1. EXTREME RAIN STRENGTH
# ==============================================================================

def apply_rain_boost(rain_raw):
    """
    Apply extreme rain multiplier.
    
    Args:
        rain_raw: Base rainfall (from weather system)
    
    Returns:
        rain: Boosted rainfall
    """
    return rain_raw * RAIN_BOOST


# ==============================================================================
# 2. COMPUTE RUNOFF
# ==============================================================================

def compute_runoff(rain, infiltration_rate=INFILTRATION_RATE):
    """
    Compute surface runoff (water that flows and causes erosion).
    
    Args:
        rain: Rainfall array
        infiltration_rate: Fraction of rain that infiltrates
    
    Returns:
        runoff: Water that becomes surface flow
    """
    infiltration = rain * infiltration_rate
    runoff = np.maximum(0.0, rain - infiltration)
    return runoff


# ==============================================================================
# 3. DETERMINE FLOW DIRECTION (D8)
# ==============================================================================

def compute_flow_direction_d8(elevation, pixel_scale_m):
    """
    Compute D8 flow direction (steepest descent).
    
    Args:
        elevation: Terrain elevation
        pixel_scale_m: Cell size
    
    Returns:
        flow_dir: Direction index (0-7) or -1 for no flow
        receivers: (i, j) of downstream neighbor or (-1, -1)
    """
    ny, nx = elevation.shape
    flow_dir = np.full((ny, nx), -1, dtype=np.int8)
    receivers = np.full((ny, nx, 2), -1, dtype=np.int32)
    
    # 8 neighbors (N, NE, E, SE, S, SW, W, NW)
    di = np.array([-1, -1, 0, 1, 1, 1, 0, -1])
    dj = np.array([0, 1, 1, 1, 0, -1, -1, -1])
    distances = np.array([pixel_scale_m, pixel_scale_m * np.sqrt(2), pixel_scale_m,
                          pixel_scale_m * np.sqrt(2), pixel_scale_m, pixel_scale_m * np.sqrt(2),
                          pixel_scale_m, pixel_scale_m * np.sqrt(2)])
    
    for i in range(ny):
        for j in range(nx):
            z_center = elevation[i, j]
            steepest_slope = 0.0
            steepest_dir = -1
            
            for k in range(8):
                ni = (i + di[k]) % ny  # Periodic boundaries
                nj = (j + dj[k]) % nx
                dz = z_center - elevation[ni, nj]
                slope = dz / distances[k]
                
                if slope > steepest_slope:
                    steepest_slope = slope
                    steepest_dir = k
            
            if steepest_dir >= 0:
                flow_dir[i, j] = steepest_dir
                receivers[i, j, 0] = (i + di[steepest_dir]) % ny
                receivers[i, j, 1] = (j + dj[steepest_dir]) % nx
    
    return flow_dir, receivers


# ==============================================================================
# 4. COMPUTE DISCHARGE Q (water passing through each cell)
# ==============================================================================

def compute_discharge(elevation, runoff, flow_dir, receivers, pixel_scale_m):
    """
    Compute discharge Q using topological sort.
    
    Args:
        elevation: Terrain elevation
        runoff: Surface runoff
        flow_dir: Flow direction
        receivers: Downstream neighbors
        pixel_scale_m: Cell size
    
    Returns:
        Q: Discharge (total water passing through each cell)
    """
    ny, nx = elevation.shape
    cell_area = pixel_scale_m ** 2
    
    # Initialize discharge with local runoff
    Q = runoff * cell_area
    
    # Sort cells by elevation (high to low)
    indices = [(i, j) for i in range(ny) for j in range(nx)]
    indices_sorted = sorted(indices, key=lambda idx: elevation[idx], reverse=True)
    
    # Accumulate discharge from high to low
    for (i, j) in indices_sorted:
        if flow_dir[i, j] >= 0:
            ni, nj = receivers[i, j]
            Q[ni, nj] += Q[i, j]
    
    return Q


# ==============================================================================
# 5. COMPUTE SLOPE ALONG FLOW DIRECTION
# ==============================================================================

def compute_slope(elevation, flow_dir, receivers, pixel_scale_m):
    """
    Compute slope along flow direction.
    
    Args:
        elevation: Terrain elevation
        flow_dir: Flow direction
        receivers: Downstream neighbors
        pixel_scale_m: Cell size
    
    Returns:
        slope: Slope along flow direction
    """
    ny, nx = elevation.shape
    slope = np.zeros((ny, nx))
    
    # 8 neighbor distances
    di = np.array([-1, -1, 0, 1, 1, 1, 0, -1])
    dj = np.array([0, 1, 1, 1, 0, -1, -1, -1])
    distances = np.array([pixel_scale_m, pixel_scale_m * np.sqrt(2), pixel_scale_m,
                          pixel_scale_m * np.sqrt(2), pixel_scale_m, pixel_scale_m * np.sqrt(2),
                          pixel_scale_m, pixel_scale_m * np.sqrt(2)])
    
    for i in range(ny):
        for j in range(nx):
            if flow_dir[i, j] >= 0:
                k = flow_dir[i, j]
                ni, nj = receivers[i, j]
                dz = elevation[i, j] - elevation[ni, nj]
                slope[i, j] = max(0.0, dz / distances[k])
    
    return slope


# ==============================================================================
# 6. EROSION - TWO-PASS WITH HALF-LOSS RULE
# ==============================================================================

def erosion_pass_a(elevation, Q, slope, erodibility, flow_dir, 
                   base_k=BASE_K, flat_k=FLAT_K, max_erode=MAX_ERODE_PER_STEP,
                   flat_threshold=FLAT_SLOPE_THRESHOLD):
    """
    PASS A: Erosion with HALF-LOSS RULE.
    
    - Erode cells based on discharge and slope
    - Apply half-loss rule: 50% of eroded material is removed, 50% transported
    - Return sediment_out (material available for transport)
    
    Args:
        elevation: Terrain elevation (MODIFIED IN PLACE)
        Q: Discharge
        slope: Slope
        erodibility: Erodibility of each cell
        flow_dir: Flow direction
        base_k: Base erosion coefficient
        flat_k: Flat cell erosion coefficient
        max_erode: Maximum erosion per step
        flat_threshold: Slope threshold for "flat" cells
    
    Returns:
        sediment_out: Sediment produced at each cell (50% of erosion)
    """
    ny, nx = elevation.shape
    sediment_out = np.zeros((ny, nx))
    
    for i in range(ny):
        for j in range(nx):
            # Separate behavior: downslope vs flat cells
            if slope[i, j] > flat_threshold and flow_dir[i, j] >= 0:
                # DOWNSLOPE CELL: main erosion zone
                erosion_power = base_k * Q[i, j] * slope[i, j] * erodibility[i, j]
                dz_erosion = -min(max_erode, erosion_power)
                
                # Apply erosion (lower the cell)
                elevation[i, j] += dz_erosion  # dz_erosion is negative
                
                # Convert to positive sediment amount
                eroded_material = -dz_erosion
                
                # HALF-LOSS RULE: only 50% enters sediment flux
                sediment_to_move = 0.5 * eroded_material
                # sediment_lost = 0.5 * eroded_material  # forever removed
                
                sediment_out[i, j] = sediment_to_move
                
            elif slope[i, j] <= flat_threshold or flow_dir[i, j] < 0:
                # FLAT / PIT CELL
                if Q[i, j] > 1.0:  # High water: allow scouring
                    erosion_power_flat = flat_k * Q[i, j] * erodibility[i, j]
                    dz_erosion = -min(max_erode, erosion_power_flat)
                    
                    elevation[i, j] += dz_erosion
                    
                    eroded_material = -dz_erosion
                    
                    # HALF-LOSS RULE still applies
                    sediment_to_move = 0.5 * eroded_material
                    
                    sediment_out[i, j] = sediment_to_move
                # else: low water flat → no erosion
    
    return sediment_out


def deposition_pass_b(elevation, Q, slope, sediment_out, flow_dir, receivers,
                      capacity_k=CAPACITY_K):
    """
    PASS B: Transport capacity + deposition + sediment routing.
    
    - Process cells from high to low elevation
    - Compute transport capacity at each cell
    - Deposit excess sediment (capacity exceeded)
    - Route remaining sediment downstream
    
    Args:
        elevation: Terrain elevation (MODIFIED IN PLACE)
        Q: Discharge
        slope: Slope
        sediment_out: Sediment produced at each cell (from Pass A)
        flow_dir: Flow direction
        receivers: Downstream neighbors
        capacity_k: Transport capacity coefficient
    
    Returns:
        deposited: Deposition at each cell
    """
    ny, nx = elevation.shape
    sediment_in = np.zeros((ny, nx))
    deposited = np.zeros((ny, nx))
    
    # Sort cells by elevation (high to low)
    indices = [(i, j) for i in range(ny) for j in range(nx)]
    indices_sorted = sorted(indices, key=lambda idx: elevation[idx], reverse=True)
    
    for (i, j) in indices_sorted:
        # Total sediment at this cell
        total_sediment = sediment_in[i, j] + sediment_out[i, j]
        
        # Transport capacity
        capacity = capacity_k * Q[i, j] * slope[i, j]
        
        if total_sediment > capacity:
            # Too much sediment → deposit excess
            deposit = total_sediment - capacity
            elevation[i, j] += deposit  # Raise the cell
            deposited[i, j] = deposit
            sediment_to_downstream = capacity
        else:
            # Can carry everything
            sediment_to_downstream = total_sediment
        
        # Route sediment downstream
        if flow_dir[i, j] >= 0:
            ni, nj = receivers[i, j]
            sediment_in[ni, nj] += sediment_to_downstream
        # else: sediment lost off map
    
    return deposited


# ==============================================================================
# 7. HILLSLOPE DIFFUSION (optional smoothing)
# ==============================================================================

def hillslope_diffusion(elevation, diffusion_k=DIFFUSION_K, pixel_scale_m=10.0):
    """
    Apply hillslope diffusion (slope-dependent smoothing).
    
    Material slides from higher cells to lower neighbors.
    
    Args:
        elevation: Terrain elevation (MODIFIED IN PLACE)
        diffusion_k: Diffusion coefficient
        pixel_scale_m: Cell size
    
    Returns:
        None (modifies elevation in place)
    """
    ny, nx = elevation.shape
    changes = np.zeros((ny, nx))
    
    # 4-neighbor diffusion (simpler, more stable)
    di = np.array([-1, 0, 1, 0])
    dj = np.array([0, 1, 0, -1])
    
    for i in range(ny):
        for j in range(nx):
            for k in range(4):
                ni = (i + di[k]) % ny
                nj = (j + dj[k]) % nx
                
                height_diff = elevation[i, j] - elevation[ni, nj]
                if height_diff > 0:
                    slide = diffusion_k * height_diff
                    changes[i, j] -= slide
                    changes[ni, nj] += slide
    
    elevation += changes


# ==============================================================================
# 8. COMPLETE EROSION STEP
# ==============================================================================

def run_fluvial_erosion_step(strata, rain_raw, pixel_scale_m, dt,
                              apply_diffusion=True):
    """
    Run one complete fluvial erosion step.
    
    Pipeline:
    1. Apply extreme rain boost
    2. Compute runoff
    3. Compute flow direction
    4. Compute discharge Q
    5. Compute slope
    6. PASS A: Erosion with half-loss rule
    7. PASS B: Transport capacity + deposition
    8. Optional: hillslope diffusion
    
    Args:
        strata: Terrain dict with 'surface_elev'
        rain_raw: Base rainfall
        pixel_scale_m: Cell size
        dt: Time step (years)
        apply_diffusion: Whether to apply hillslope diffusion
    
    Returns:
        diagnostics: Dict with erosion/deposition info
    """
    elevation = strata["surface_elev"]
    ny, nx = elevation.shape
    
    # Uniform erodibility for now (can be layer-aware later)
    erodibility = np.ones((ny, nx))
    
    # 1. Apply extreme rain boost
    rain = apply_rain_boost(rain_raw)
    
    # 2. Compute runoff
    runoff = compute_runoff(rain)
    
    # 3. Compute flow direction
    flow_dir, receivers = compute_flow_direction_d8(elevation, pixel_scale_m)
    
    # 4. Compute discharge Q
    Q = compute_discharge(elevation, runoff, flow_dir, receivers, pixel_scale_m)
    
    # 5. Compute slope
    slope = compute_slope(elevation, flow_dir, receivers, pixel_scale_m)
    
    # 6. PASS A: Erosion with half-loss rule
    sediment_out = erosion_pass_a(elevation, Q, slope, erodibility, flow_dir)
    
    # 7. PASS B: Transport capacity + deposition
    deposited = deposition_pass_b(elevation, Q, slope, sediment_out, flow_dir, receivers)
    
    # 8. Optional: hillslope diffusion
    if apply_diffusion:
        hillslope_diffusion(elevation, DIFFUSION_K, pixel_scale_m)
    
    # Compute net change (for diagnostics)
    # Note: erosion is already applied in Pass A (elevation lowered)
    # deposition is already applied in Pass B (elevation raised)
    # We need to track what happened for visualization
    
    # Erosion = amount removed (positive values)
    erosion_amount = sediment_out * 2.0  # sediment_out is 50%, so erosion is 2×
    
    # Deposition = deposited (already computed)
    deposition_amount = deposited
    
    return {
        "erosion": erosion_amount,
        "deposition": deposition_amount,
        "Q": Q,
        "slope": slope,
        "flow_dir": flow_dir,
    }


# ==============================================================================
# 9. MULTI-EPOCH SIMULATION
# ==============================================================================

def run_fluvial_simulation(strata, rain_func, pixel_scale_m, num_epochs, dt,
                            apply_diffusion=True, verbose=True):
    """
    Run multi-epoch fluvial erosion simulation.
    
    Args:
        strata: Terrain dict
        rain_func: Function that returns rain array (can vary by epoch)
        pixel_scale_m: Cell size
        num_epochs: Number of simulation epochs
        dt: Time step per epoch (years)
        apply_diffusion: Whether to apply hillslope diffusion
        verbose: Print progress
    
    Returns:
        history: List of diagnostics per epoch
    """
    history = []
    
    if verbose:
        print(f"\n🌊 STARTING FLUVIAL EROSION SIMULATION")
        print(f"   Epochs: {num_epochs}")
        print(f"   Time step: {dt} years per epoch")
        print(f"   Rain boost: {RAIN_BOOST}×")
        print(f"   Half-loss rule: ACTIVE (net volume loss!)")
    
    for epoch in range(num_epochs):
        if verbose:
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Surface range: {strata['surface_elev'].min():.1f} - {strata['surface_elev'].max():.1f} m")
        
        # Get rainfall for this epoch (can be spatially varying)
        rain_raw = rain_func(strata, epoch)
        
        # Run erosion step
        diagnostics = run_fluvial_erosion_step(
            strata, rain_raw, pixel_scale_m, dt,
            apply_diffusion=apply_diffusion
        )
        
        history.append(diagnostics)
        
        if verbose:
            print(f"  ✓ Epoch complete")
            print(f"     Erosion: {diagnostics['erosion'].mean():.3f} m avg, {diagnostics['erosion'].max():.3f} m max")
            print(f"     Deposition: {diagnostics['deposition'].mean():.3f} m avg, {diagnostics['deposition'].max():.3f} m max")
            print(f"     Net change: {(diagnostics['deposition'].sum() - diagnostics['erosion'].sum()):.1f} m³ (negative = volume loss!)")
    
    if verbose:
        print(f"\n✓ SIMULATION COMPLETE!")
    
    return history


print("✓ Proper fluvial erosion model loaded!")
print("  Key features:")
print("  • Extreme rain boost (100×)")
print("  • Half-loss rule (50% eroded material removed)")
print("  • Two-pass erosion (erosion → transport → deposition)")
print("  • Transport capacity based on Q and slope")
print("  • Result: NET VOLUME LOSS → valleys deepen!")
