"""
CELL 2: EROSION MODEL (PHYSICS FIXED)

Fixes the major physics problems:
1. Proper sediment routing (supply vs capacity, route downstream)
2. Realistic erosion magnitudes
3. Runoff-based erosion (not just local rain divots)

Key concepts implemented:
- Rainfall → runoff (after infiltration)
- Flow routing with discharge accumulation (already had D8)
- Stream power: E = K * Q^m * S^n (already had)
- Sediment transport: route sediment downstream, compare supply vs capacity
- Hillslope diffusion: material slides down slopes
- Layer-aware erodibility
"""
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 1. FLOW ROUTING (same as before - this part was correct)
# ==============================================================================

def compute_flow_direction_d8(elevation, pixel_scale_m):
    """Compute D8 flow direction for each cell."""
    ny, nx = elevation.shape
    flow_dir = np.full((ny, nx), -1, dtype=np.int8)
    receivers = np.full((ny, nx, 2), -1, dtype=np.int32)
    
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
                ni = (i + di[k]) % ny
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


def compute_flow_accumulation_proper(elevation, flow_dir, receivers, pixel_scale_m, rainfall=None):
    """
    PROPER flow accumulation.
    
    Returns discharge Q (m²/s or m³/year depending on units).
    This represents UPSLOPE CONTRIBUTING AREA × runoff rate.
    """
    ny, nx = elevation.shape
    cell_area = pixel_scale_m ** 2
    
    # Runoff: some fraction of rainfall becomes surface flow
    # For now, assume 50% infiltrates, 50% runs off
    if rainfall is not None:
        runoff = rainfall * 0.5  # m/year
        water = runoff * cell_area  # m³/year
    else:
        # Default: assume 1 m/year runoff
        water = np.ones((ny, nx)) * cell_area
    
    accumulation = water.copy()
    
    # Topological sort
    indices = [(i, j) for i in range(ny) for j in range(nx)]
    indices_sorted = sorted(indices, key=lambda idx: elevation[idx], reverse=True)
    
    # Accumulate flow
    for (i, j) in indices_sorted:
        if flow_dir[i, j] >= 0:
            ni, nj = receivers[i, j]
            accumulation[ni, nj] += accumulation[i, j]
    
    return accumulation


def route_flow_proper(elevation, pixel_scale_m, rainfall=None):
    """Complete flow routing."""
    flow_dir, receivers = compute_flow_direction_d8(elevation, pixel_scale_m)
    discharge = compute_flow_accumulation_proper(elevation, flow_dir, receivers, 
                                                  pixel_scale_m, rainfall)
    
    dy, dx = np.gradient(elevation, pixel_scale_m)
    slope = np.sqrt(dx**2 + dy**2)
    slope = np.maximum(slope, 1e-6)
    
    return {
        "flow_dir": flow_dir,
        "receivers": receivers,
        "discharge": discharge,
        "slope": slope,
    }


# ==============================================================================
# 2. EROSION LAWS (same, but with LARGER coefficients)
# ==============================================================================

def get_effective_erodibility(strata, K_base):
    """Get erodibility of top layer at each cell."""
    ny, nx = strata["surface_elev"].shape
    K_eff = np.ones((ny, nx)) * K_base
    
    if "deposits" in strata:
        for dep_name, dep_thickness in strata["deposits"].items():
            if dep_name in strata["properties"]:
                erodibility = strata["properties"][dep_name].get("erodibility", 1.0)
                has_deposit = dep_thickness > 0
                K_eff[has_deposit] = K_base * erodibility
    
    for layer_name in ["Topsoil", "Saprolite", "Sandstone", "Basement"]:
        if layer_name in strata["thickness"] and layer_name in strata["properties"]:
            thickness = strata["thickness"][layer_name]
            erodibility = strata["properties"][layer_name].get("erodibility", 1.0)
            is_exposed = thickness > 0
            K_eff[is_exposed] = K_base * erodibility
    
    return K_eff


def compute_potential_erosion(strata, flow_data, pixel_scale_m, dt,
                               K_base=1e-4, m=0.5, n=1.0, Q_threshold=100.0):
    """
    Compute POTENTIAL erosion from stream power.
    
    This is how much COULD erode if there's no sediment supply limit.
    
    E_potential = K_eff * Q^m * S^n * dt
    
    Note: Using K_base=1e-4 (was 1e-6) for realistic erosion rates.
    """
    A = flow_data["discharge"]
    S = flow_data["slope"]
    
    K_eff = get_effective_erodibility(strata, K_base)
    
    # Only apply to channels (Q > threshold)
    is_channel = A > Q_threshold
    
    erosion_potential = np.zeros_like(A)
    erosion_potential[is_channel] = K_eff[is_channel] * (A[is_channel] ** m) * (S[is_channel] ** n) * dt
    
    # Bound to prevent blow-up
    max_erosion_per_step = 2.0  # meters (increased from 1.0)
    erosion_potential = np.minimum(erosion_potential, max_erosion_per_step)
    
    return erosion_potential


def hillslope_diffusion(surface_elev, pixel_scale_m, dt, D=0.01):
    """Hillslope diffusion (material slides down slopes)."""
    from scipy.ndimage import laplace
    
    laplacian = laplace(surface_elev, mode='wrap') / (pixel_scale_m ** 2)
    dz = D * laplacian * dt
    erosion = -dz
    
    max_change = 0.5  # meters
    erosion = np.clip(erosion, -max_change, max_change)
    
    return erosion


# ==============================================================================
# 3. SEDIMENT TRANSPORT (FIXED - proper downstream routing)
# ==============================================================================

def compute_transport_capacity(discharge, slope, coeff=0.5):
    """
    Transport capacity: how much sediment water can carry.
    
    C = coeff * Q^0.5 * S
    
    where:
    - Q = discharge (m³/year or m²)
    - S = slope (m/m)
    - coeff = transport coefficient (tunable)
    
    Returns capacity in m³ per cell per time step.
    """
    capacity = coeff * (discharge ** 0.5) * slope
    return capacity


def route_sediment_downstream(strata, flow_data, erosion_potential, erosion_hillslope,
                               pixel_scale_m, dt, transport_coeff=0.5):
    """
    FIXED: Properly route sediment downstream.
    
    Process (following your description):
    1. Start from highest cells (topologically sorted)
    2. For each cell:
       a. Compute sediment supply = local_erosion + sediment_from_upstream
       b. Compute transport capacity
       c. If supply > capacity: deposit excess, pass capacity downstream
       d. If supply < capacity: erode more to fill capacity, pass all downstream
    3. Track actual erosion and deposition at each cell
    
    Returns:
        erosion_actual: actual erosion that occurred (m)
        deposition: deposition that occurred (m)
    """
    ny, nx = strata["surface_elev"].shape
    cell_area = pixel_scale_m ** 2
    
    flow_dir = flow_data["flow_dir"]
    receivers = flow_data["receivers"]
    discharge = flow_data["discharge"]
    slope = flow_data["slope"]
    
    # Transport capacity at each cell
    capacity = compute_transport_capacity(discharge, slope, transport_coeff)
    
    # Sediment supply arriving at each cell (starts with local erosion potential)
    # Convert erosion depth (m) to volume (m³)
    local_erosion_volume = (erosion_potential + np.maximum(erosion_hillslope, 0)) * cell_area
    sediment_supply = local_erosion_volume.copy()
    
    # Actual erosion and deposition (depth in meters)
    erosion_actual = np.zeros((ny, nx))
    deposition = np.zeros((ny, nx))
    
    # Sediment flux passing downstream (volume in m³)
    sediment_flux_out = np.zeros((ny, nx))
    
    # Process cells from high to low elevation
    elevation = strata["surface_elev"]
    indices = [(i, j) for i in range(ny) for j in range(nx)]
    indices_sorted = sorted(indices, key=lambda idx: elevation[idx], reverse=True)
    
    for (i, j) in indices_sorted:
        # Sediment supply at this cell (already includes upstream contributions)
        supply = sediment_supply[i, j]
        
        # Transport capacity at this cell (volume)
        cap = capacity[i, j] * cell_area * dt
        
        if supply > cap:
            # More sediment than can be carried
            # Deposit the excess
            excess_volume = supply - cap
            deposition[i, j] = excess_volume / cell_area
            
            # Pass capacity downstream
            sediment_flux_out[i, j] = cap
            
            # Actual erosion is the local potential (no additional erosion needed)
            erosion_actual[i, j] = (erosion_potential[i, j] + 
                                     np.maximum(erosion_hillslope[i, j], 0))
        
        else:
            # Capacity exceeds supply
            # Try to erode more to fill capacity
            deficit_volume = cap - supply
            additional_erosion_depth = deficit_volume / cell_area
            
            # Limit additional erosion
            max_additional = 1.0  # m
            additional_erosion_depth = min(additional_erosion_depth, max_additional)
            
            # Actual erosion = potential + additional
            erosion_actual[i, j] = (erosion_potential[i, j] + 
                                     np.maximum(erosion_hillslope[i, j], 0) +
                                     additional_erosion_depth)
            
            # No deposition
            deposition[i, j] = 0.0
            
            # Pass all sediment (supply + additional erosion) downstream
            sediment_flux_out[i, j] = supply + additional_erosion_depth * cell_area
        
        # Route sediment to downstream cell
        if flow_dir[i, j] >= 0:
            ni, nj = receivers[i, j]
            sediment_supply[ni, nj] += sediment_flux_out[i, j]
    
    # Bound deposition
    deposition = np.minimum(deposition, 1.0)  # Max 1m deposition per step
    
    return erosion_actual, deposition


# ==============================================================================
# 4. STRATIGRAPHY UPDATE (same as before)
# ==============================================================================

def update_stratigraphy_with_erosion_BOUNDED(strata, erosion, pixel_scale_m):
    """Apply erosion with depth limits."""
    thickness = strata["thickness"]
    interfaces = strata["interfaces"]
    deposits = strata.get("deposits", {})
    
    ny, nx = erosion.shape
    
    if "BasementFloor" in interfaces:
        basement_floor = interfaces["BasementFloor"]
    else:
        basement_floor = strata["surface_elev"] - 100.0
    
    deposit_order = ["Alluvium", "Till", "Loess", "DuneSand"]
    strat_order = ["Topsoil", "Subsoil", "Colluvium", "Saprolite", "WeatheredBR",
                   "Sandstone", "Shale", "Limestone", "Basement"]
    
    for i in range(ny):
        for j in range(nx):
            remaining_erosion = erosion[i, j]
            
            if remaining_erosion <= 0:
                continue
            
            current_elev = strata["surface_elev"][i, j]
            min_elev = basement_floor[i, j] + 1.0
            
            if current_elev - remaining_erosion < min_elev:
                remaining_erosion = max(0, current_elev - min_elev)
            
            if remaining_erosion <= 0:
                continue
            
            for dep in deposit_order:
                if dep not in deposits:
                    continue
                
                if deposits[dep][i, j] > 0 and remaining_erosion > 0:
                    removed = min(deposits[dep][i, j], remaining_erosion)
                    deposits[dep][i, j] -= removed
                    remaining_erosion -= removed
            
            for layer in strat_order:
                if layer not in thickness:
                    continue
                
                if thickness[layer][i, j] > 0 and remaining_erosion > 0:
                    removed = min(thickness[layer][i, j], remaining_erosion)
                    thickness[layer][i, j] -= removed
                    remaining_erosion -= removed
    
    strata["surface_elev"] -= erosion
    strata["surface_elev"] = np.maximum(strata["surface_elev"], basement_floor + 1.0)
    
    surface_elev = strata["surface_elev"]
    for layer in strat_order:
        if layer in interfaces and layer in thickness:
            cumulative_above = np.zeros_like(surface_elev)
            
            for dep in deposit_order:
                if dep in deposits:
                    cumulative_above += deposits[dep]
            
            for upper_layer in strat_order:
                if upper_layer == layer:
                    break
                if upper_layer in thickness:
                    cumulative_above += thickness[upper_layer]
            
            interfaces[layer][:, :] = surface_elev - cumulative_above


def update_stratigraphy_with_deposition(strata, deposition, pixel_scale_m):
    """Add deposition to Alluvium layer."""
    deposits = strata.get("deposits", {})
    
    if "Alluvium" not in deposits:
        deposits["Alluvium"] = np.zeros_like(deposition)
        strata["deposits"] = deposits
    
    deposits["Alluvium"] += deposition
    strata["surface_elev"] += deposition


# ==============================================================================
# 5. UPLIFT (same as before)
# ==============================================================================

def apply_uplift(strata, uplift_rate, dt):
    """Apply uplift."""
    interfaces = strata["interfaces"]
    uplift = uplift_rate * dt
    strata["surface_elev"] += uplift
    for layer in interfaces:
        interfaces[layer] += uplift


# ==============================================================================
# 6. TIME-STEPPING (PHYSICS FIXED)
# ==============================================================================

def run_erosion_epoch_PHYSICS_FIXED(
    strata,
    pixel_scale_m,
    dt,
    rainfall=None,
    uplift_rate=0.0,
    K_channel=1e-4,  # INCREASED from 1e-6
    D_hillslope=0.01,
    stream_power_m=0.5,
    stream_power_n=1.0,
    Q_threshold=100.0,
    transport_coeff=0.5
):
    """
    Run one epoch with FIXED physics.
    
    Key changes:
    1. Realistic erosion magnitudes (K_channel=1e-4, not 1e-6)
    2. Proper sediment routing (supply vs capacity, route downstream)
    3. Runoff-based discharge (50% infiltration)
    """
    # Uplift
    if np.isscalar(uplift_rate):
        if uplift_rate != 0:
            apply_uplift(strata, uplift_rate, dt)
    else:
        if np.any(uplift_rate != 0):
            apply_uplift(strata, uplift_rate, dt)
    
    # Flow routing (with runoff calculation)
    flow_data = route_flow_proper(
        strata["surface_elev"], 
        pixel_scale_m, 
        rainfall  # This gets converted to runoff inside route_flow_proper
    )
    
    # Compute potential erosion
    erosion_potential = compute_potential_erosion(
        strata,
        flow_data,
        pixel_scale_m,
        dt,
        K_base=K_channel,
        m=stream_power_m,
        n=stream_power_n,
        Q_threshold=Q_threshold
    )
    
    # Hillslope diffusion
    erosion_hillslope = hillslope_diffusion(
        strata["surface_elev"],
        pixel_scale_m,
        dt,
        D=D_hillslope
    )
    
    # FIXED: Properly route sediment downstream
    erosion_actual, deposition = route_sediment_downstream(
        strata,
        flow_data,
        erosion_potential,
        erosion_hillslope,
        pixel_scale_m,
        dt,
        transport_coeff=transport_coeff
    )
    
    # Apply erosion
    update_stratigraphy_with_erosion_BOUNDED(strata, erosion_actual, pixel_scale_m)
    
    # Apply deposition
    update_stratigraphy_with_deposition(strata, deposition, pixel_scale_m)
    
    return {
        "erosion_potential": erosion_potential,
        "erosion_hillslope": erosion_hillslope,
        "erosion_actual": erosion_actual,
        "deposition": deposition,
        "flow_data": flow_data,
        "total_erosion": erosion_actual,  # For compatibility
    }


def run_erosion_simulation_PHYSICS_FIXED(
    strata,
    pixel_scale_m,
    num_epochs,
    dt,
    rainfall_func=None,
    uplift_rate=0.0,
    K_channel=1e-4,  # INCREASED
    D_hillslope=0.01,
    verbose=True
):
    """Run multiple epochs with FIXED physics."""
    history = []
    
    for epoch in range(num_epochs):
        if verbose and epoch % max(1, num_epochs // 10) == 0:
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"  Surface range: {strata['surface_elev'].min():.1f} - {strata['surface_elev'].max():.1f} m")
        
        if rainfall_func is not None:
            rainfall = rainfall_func(epoch)
        else:
            rainfall = None
        
        if callable(uplift_rate):
            uplift = uplift_rate(epoch)
        else:
            uplift = uplift_rate
        
        diagnostics = run_erosion_epoch_PHYSICS_FIXED(
            strata,
            pixel_scale_m,
            dt,
            rainfall=rainfall,
            uplift_rate=uplift,
            K_channel=K_channel,
            D_hillslope=D_hillslope
        )
        
        history.append(diagnostics)
    
    return history


print("✓ Erosion model (PHYSICS FIXED) loaded successfully!")
print("  Key fixes:")
print("    1. Realistic erosion: K_channel=1e-4 (was 1e-6) → 100× more erosion")
print("    2. Proper sediment routing: supply vs capacity, route downstream")
print("    3. Runoff-based: 50% infiltration, 50% runoff")
print("    4. No more local divots: water flows and erodes along its path")
print("  Expected: Continuous channels, not isolated pits!")
