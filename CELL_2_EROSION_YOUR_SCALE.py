"""
CELL 2: EROSION MODEL (Scaled for YOUR resolution)

FIXED erosion model with proper flow routing, but scaled for YOUR parameters:
- N=512 (not N=50)
- pixel_scale_m=10.0 (not 1000.0)  
- Domain: 5.12 km × 5.12 km (not 50 km × 50 km)

Key scaling adjustments:
- dt: smaller time steps for smaller cells
- Q_threshold: adjusted for 10m cells vs 1000m cells
- Max erosion: scaled appropriately

The core algorithms (D8 flow routing, stream power law) are the same FIXED versions.
"""
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 1. FLOW ROUTING (PROPER D8 + UPSLOPE AREA - same as before)
# ==============================================================================

def compute_flow_direction_d8(elevation, pixel_scale_m):
    """Compute D8 flow direction for each cell."""
    ny, nx = elevation.shape
    flow_dir = np.full((ny, nx), -1, dtype=np.int8)
    receivers = np.full((ny, nx, 2), -1, dtype=np.int32)
    
    # 8 neighbors
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
    """PROPER flow accumulation using topological sort."""
    ny, nx = elevation.shape
    cell_area = pixel_scale_m ** 2
    
    if rainfall is not None:
        water = rainfall * cell_area
    else:
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
# 2. EROSION LAWS (Stream power with upslope area)
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


def channel_incision_stream_power_SCALED(strata, flow_data, pixel_scale_m, dt,
                                          K_base=1e-5, m=0.5, n=1.0, Q_threshold=100.0):
    """
    Stream power erosion SCALED for YOUR resolution.
    
    Q_threshold is adjusted for pixel_scale_m=10.0:
    - At pixel_scale=1000m, Q_threshold=1e4 meant ~10 cells contributing
    - At pixel_scale=10m, Q_threshold=100 means ~1 cell contributing
    - This scales correctly: 100 * (10m)^2 = 10,000 m² ~ 1e4 * (1m)^2
    """
    A = flow_data["discharge"]
    S = flow_data["slope"]
    
    K_eff = get_effective_erodibility(strata, K_base)
    
    is_channel = A > Q_threshold
    
    erosion = np.zeros_like(A)
    erosion[is_channel] = K_eff[is_channel] * (A[is_channel] ** m) * (S[is_channel] ** n) * dt
    
    # BOUND erosion (scaled for 10m cells)
    # At 10m resolution, 1m erosion per step is still quite a lot
    max_erosion_per_step = 1.0  # meters (was 10m for 1000m cells)
    erosion = np.minimum(erosion, max_erosion_per_step)
    
    return erosion


def hillslope_diffusion(surface_elev, pixel_scale_m, dt, D=0.01):
    """Hillslope diffusion."""
    from scipy.ndimage import laplace
    
    laplacian = laplace(surface_elev, mode='wrap') / (pixel_scale_m ** 2)
    dz = D * laplacian * dt
    erosion = -dz
    
    # Bound (scaled for 10m cells)
    max_change = 0.5  # meters (was 5m for 1000m cells)
    erosion = np.clip(erosion, -max_change, max_change)
    
    return erosion


# ==============================================================================
# 3. SEDIMENT TRANSPORT
# ==============================================================================

def compute_sediment_transport(flow_data, erosion_channel, erosion_hillslope, pixel_scale_m,
                                transport_capacity_coeff=0.1):
    """Simple sediment transport."""
    A = flow_data["discharge"]
    S = flow_data["slope"]
    
    erosion_total = np.maximum(erosion_channel, 0) + np.maximum(erosion_hillslope, 0)
    
    capacity = transport_capacity_coeff * (A ** 0.5) * S
    
    excess = erosion_total - capacity
    deposition = np.maximum(excess, 0)
    
    # Limit (scaled for 10m cells)
    deposition = np.minimum(deposition, 0.5)  # Max 0.5m per step
    
    return deposition, capacity


# ==============================================================================
# 4. STRATIGRAPHY UPDATE
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
        basement_floor = strata["surface_elev"] - 100.0  # 100m basement
    
    deposit_order = ["Alluvium", "Till", "Loess", "DuneSand"]
    strat_order = ["Topsoil", "Subsoil", "Colluvium", "Saprolite", "WeatheredBR",
                   "Sandstone", "Shale", "Limestone", "Basement"]
    
    for i in range(ny):
        for j in range(nx):
            remaining_erosion = erosion[i, j]
            
            if remaining_erosion <= 0:
                continue
            
            # Check depth limit
            current_elev = strata["surface_elev"][i, j]
            min_elev = basement_floor[i, j] + 1.0  # Keep 1m above floor
            
            if current_elev - remaining_erosion < min_elev:
                remaining_erosion = max(0, current_elev - min_elev)
            
            if remaining_erosion <= 0:
                continue
            
            # Remove from deposits
            for dep in deposit_order:
                if dep not in deposits:
                    continue
                
                if deposits[dep][i, j] > 0 and remaining_erosion > 0:
                    removed = min(deposits[dep][i, j], remaining_erosion)
                    deposits[dep][i, j] -= removed
                    remaining_erosion -= removed
            
            # Remove from layers
            for layer in strat_order:
                if layer not in thickness:
                    continue
                
                if thickness[layer][i, j] > 0 and remaining_erosion > 0:
                    removed = min(thickness[layer][i, j], remaining_erosion)
                    thickness[layer][i, j] -= removed
                    remaining_erosion -= removed
    
    # Update surface
    strata["surface_elev"] -= erosion
    strata["surface_elev"] = np.maximum(strata["surface_elev"], basement_floor + 1.0)
    
    # Recompute interfaces
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
# 5. UPLIFT
# ==============================================================================

def apply_uplift(strata, uplift_rate, dt):
    """Apply uplift."""
    interfaces = strata["interfaces"]
    uplift = uplift_rate * dt
    strata["surface_elev"] += uplift
    for layer in interfaces:
        interfaces[layer] += uplift


# ==============================================================================
# 6. TIME-STEPPING (SCALED for YOUR resolution)
# ==============================================================================

def run_erosion_epoch_SCALED(
    strata,
    pixel_scale_m,
    dt,
    rainfall=None,
    uplift_rate=0.0,
    K_channel=1e-5,
    D_hillslope=0.01,
    stream_power_m=0.5,
    stream_power_n=1.0,
    Q_threshold=100.0,  # SCALED for 10m cells
    transport_coeff=0.1
):
    """Run one epoch SCALED for YOUR resolution (pixel_scale_m=10.0)."""
    # Uplift
    if np.isscalar(uplift_rate):
        if uplift_rate != 0:
            apply_uplift(strata, uplift_rate, dt)
    else:
        if np.any(uplift_rate != 0):
            apply_uplift(strata, uplift_rate, dt)
    
    # Flow routing (proper D8 + upslope area)
    flow_data = route_flow_proper(
        strata["surface_elev"], 
        pixel_scale_m, 
        rainfall
    )
    
    # Channel incision (SCALED)
    erosion_channel = channel_incision_stream_power_SCALED(
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
    
    # Sediment transport
    deposition, sediment_flux = compute_sediment_transport(
        flow_data,
        erosion_channel,
        erosion_hillslope,
        pixel_scale_m,
        transport_capacity_coeff=transport_coeff
    )
    
    # Apply erosion (with bounds)
    total_erosion = np.maximum(erosion_channel, 0) + np.maximum(erosion_hillslope, 0)
    update_stratigraphy_with_erosion_BOUNDED(strata, total_erosion, pixel_scale_m)
    
    # Apply deposition
    update_stratigraphy_with_deposition(strata, deposition, pixel_scale_m)
    
    return {
        "erosion_channel": erosion_channel,
        "erosion_hillslope": erosion_hillslope,
        "deposition": deposition,
        "flow_data": flow_data,
        "total_erosion": total_erosion,
    }


def run_erosion_simulation_SCALED(
    strata,
    pixel_scale_m,
    num_epochs,
    dt,
    rainfall_func=None,
    uplift_rate=0.0,
    K_channel=1e-5,
    D_hillslope=0.01,
    verbose=True
):
    """Run multiple epochs SCALED for YOUR resolution."""
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
        
        diagnostics = run_erosion_epoch_SCALED(
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


print("✓ Erosion model (SCALED for YOUR resolution) loaded successfully!")
print("  Scaling adjustments:")
print("    - Q_threshold: 100.0 m² (was 10000.0 for 1000m cells)")
print("    - Max erosion: 1.0 m/step channel, 0.5 m/step hillslope")
print("    - Deposition limit: 0.5 m/step")
print("  Core algorithms unchanged:")
print("    - D8 flow routing + topological sort")
print("    - Stream power: E = K * A^m * S^n")
print("    - Depth limits and bounds")
