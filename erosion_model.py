# ==============================================================================
# EROSION MODEL ENGINE
# ==============================================================================
# This module adds landscape evolution on top of the existing quantum-seeded
# terrain and stratigraphy from generate_stratigraphy().
#
# Key components:
# 1. Water routing (flow direction, discharge, slope)
# 2. Channel incision (stream-power erosion)
# 3. Hillslope diffusion (mass wasting)
# 4. Sediment transport & deposition
# 5. Layer-aware stratigraphy updates
# 6. Tectonic uplift
# ==============================================================================

import numpy as np


# ==============================================================================
# 1. WATER ROUTING MODULE
# ==============================================================================

def compute_flow_direction_d8(surface_elev):
    """
    Compute D8 flow direction (steepest descent) for each cell.
    
    Returns:
        flow_dir: 2D array of integers (0-7) indicating direction to neighbor,
                  or -1 for pits/sinks
        slope: 2D array of slopes to the downstream neighbor
    """
    ny, nx = surface_elev.shape
    flow_dir = np.full((ny, nx), -1, dtype=np.int32)
    slope = np.zeros((ny, nx), dtype=np.float64)
    
    # D8 neighbor offsets: [N, NE, E, SE, S, SW, W, NW]
    di = np.array([-1, -1, 0, 1, 1, 1, 0, -1], dtype=np.int32)
    dj = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int32)
    
    # Distance to each neighbor (diagonals are sqrt(2) longer)
    distances = np.array([1.0, np.sqrt(2), 1.0, np.sqrt(2), 
                          1.0, np.sqrt(2), 1.0, np.sqrt(2)])
    
    for i in range(ny):
        for j in range(nx):
            z_here = surface_elev[i, j]
            max_slope = 0.0
            best_dir = -1
            
            for d in range(8):
                ni, nj = i + di[d], j + dj[d]
                if 0 <= ni < ny and 0 <= nj < nx:
                    z_neighbor = surface_elev[ni, nj]
                    drop = z_here - z_neighbor
                    if drop > 0:
                        s = drop / distances[d]
                        if s > max_slope:
                            max_slope = s
                            best_dir = d
            
            flow_dir[i, j] = best_dir
            slope[i, j] = max_slope
    
    return flow_dir, slope


def compute_flow_accumulation(flow_dir, rainfall=None, surface_elev=None):
    """
    Compute flow accumulation (discharge) by routing water downslope.
    
    Uses proper topological sorting to ensure each cell is processed only after
    all of its upstream donors have been processed.
    
    Args:
        flow_dir: 2D array from compute_flow_direction_d8
        rainfall: 2D array of rainfall intensity (same shape), or None for uniform
        surface_elev: 2D elevation array (needed for topological sort), or None
    
    Returns:
        discharge: 2D array of accumulated flow (in units of cell area)
    """
    ny, nx = flow_dir.shape
    
    # Initialize with local rainfall contribution
    if rainfall is None:
        discharge = np.ones((ny, nx), dtype=np.float64)
    else:
        discharge = np.array(rainfall, dtype=np.float64).copy()
    
    # Build donor-receiver graph
    # For each cell, find all cells that flow into it
    donors = [[[] for _ in range(nx)] for _ in range(ny)]
    
    di = np.array([-1, -1, 0, 1, 1, 1, 0, -1], dtype=np.int32)
    dj = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int32)
    
    for i in range(ny):
        for j in range(nx):
            d = flow_dir[i, j]
            if d >= 0:
                ni, nj = i + di[d], j + dj[d]
                if 0 <= ni < ny and 0 <= nj < nx:
                    donors[ni][nj].append((i, j))
    
    # Topological sort: order cells by elevation (high to low)
    # This ensures we process upstream cells before downstream
    if surface_elev is not None:
        # Create list of (elevation, i, j) tuples
        cells = []
        for i in range(ny):
            for j in range(nx):
                cells.append((surface_elev[i, j], i, j))
        
        # Sort by elevation (descending)
        cells.sort(reverse=True, key=lambda x: x[0])
        
        # Process cells in topological order
        for elev, i, j in cells:
            # Accumulate from all donors (upstream cells that flow into this cell)
            for di_src, dj_src in donors[i][j]:
                discharge[i, j] += discharge[di_src, dj_src]
    else:
        # Fallback: use multiple passes (less efficient but works)
        # Each pass propagates flow one step downstream
        for _ in range(min(ny + nx, 100)):
            discharge_old = discharge.copy()
            for i in range(ny):
                for j in range(nx):
                    # Accumulate from donors
                    contrib = 0.0
                    for di_src, dj_src in donors[i][j]:
                        contrib += discharge_old[di_src, dj_src]
                    discharge[i, j] = discharge_old[i, j] + contrib
            
            # Check convergence
            if np.allclose(discharge, discharge_old):
                break
    
    return discharge


def route_flow_simple(surface_elev, pixel_scale_m, rainfall=None):
    """
    Simple flow routing combining direction, slope, and accumulation.
    
    Args:
        surface_elev: 2D elevation array (m)
        pixel_scale_m: cell size (m)
        rainfall: 2D rainfall field (m/time), or None for uniform
    
    Returns:
        dict with:
            - discharge: accumulated flow (m²)
            - slope: local slope (dimensionless)
            - flow_dir: flow direction codes
    """
    flow_dir, slope_raw = compute_flow_direction_d8(surface_elev)
    discharge = compute_flow_accumulation(flow_dir, rainfall, surface_elev=surface_elev)
    
    # Convert discharge from cell counts to area (m²)
    discharge_area = discharge * (pixel_scale_m ** 2)
    
    # Smooth slope slightly to avoid numerical issues
    slope = np.maximum(slope_raw, 1e-6)
    
    return {
        "discharge": discharge_area,
        "slope": slope,
        "flow_dir": flow_dir,
    }


# ==============================================================================
# 2. LAYER-AWARE EROSION MODULE
# ==============================================================================

def get_top_layer_at_surface(strata, i, j):
    """
    Determine which layer is currently at the surface at cell (i, j).
    
    Returns:
        layer_name: str, the topmost layer with thickness > 0
    """
    thickness = strata["thickness"]
    deposits = strata.get("deposits", {})
    
    # Check deposits first (they sit on top)
    for dep_name in ["Alluvium", "Till", "Loess", "DuneSand"]:
        if dep_name in deposits and deposits[dep_name][i, j] > 0.01:
            return dep_name
    
    # Check stratigraphic layers in order (top to bottom)
    layer_order = [
        "Topsoil", "Subsoil", "Colluvium", "Saprolite", "WeatheredBR",
        "Sandstone", "Shale", "Limestone", "Basement", "BasementFloor"
    ]
    
    for layer in layer_order:
        if layer in thickness and thickness[layer][i, j] > 0.01:
            return layer
    
    return "Basement"  # fallback


def get_effective_erodibility(strata, i, j):
    """
    Get the erodibility (K_rel) of the topmost layer at (i, j).
    """
    layer_name = get_top_layer_at_surface(strata, i, j)
    properties = strata["properties"]
    
    if layer_name in properties:
        return properties[layer_name].get("erodibility", 0.5)
    else:
        return 0.5  # default


def channel_incision_stream_power(
    strata, 
    flow_data, 
    pixel_scale_m, 
    dt, 
    K_base=1e-5,
    m=0.5, 
    n=1.0,
    Q_threshold=1e4
):
    """
    Apply stream-power channel incision to the surface.
    
    E = K_eff * Q^m * S^n * dt
    
    Only applies to cells with high discharge (Q > Q_threshold).
    Updates surface_elev and removes material from top layers.
    
    Args:
        strata: the stratigraphy dict
        flow_data: output from route_flow_simple
        pixel_scale_m: cell size
        dt: time step (years)
        K_base: base erosion coefficient
        m, n: stream power exponents
        Q_threshold: minimum discharge to be considered a channel
    
    Returns:
        erosion: 2D array of eroded thickness (m)
    """
    surface_elev = strata["surface_elev"]
    discharge = flow_data["discharge"]
    slope = flow_data["slope"]
    
    ny, nx = surface_elev.shape
    erosion = np.zeros((ny, nx), dtype=np.float64)
    
    # Identify channel cells
    is_channel = discharge > Q_threshold
    
    for i in range(ny):
        for j in range(nx):
            if not is_channel[i, j]:
                continue
            
            # Get layer-specific erodibility
            K_layer = get_effective_erodibility(strata, i, j)
            K_eff = K_base * K_layer
            
            Q = discharge[i, j]
            S = slope[i, j]
            
            # Stream power erosion
            E = K_eff * (Q ** m) * (S ** n) * dt
            E = max(0, E)
            
            erosion[i, j] = E
    
    return erosion


def hillslope_diffusion(surface_elev, pixel_scale_m, dt, D=0.01):
    """
    Apply hillslope diffusion (mass wasting) using Laplacian smoothing.
    
    ∂z/∂t = D * ∇²z
    
    Args:
        surface_elev: 2D elevation array
        pixel_scale_m: cell size
        dt: time step (years)
        D: diffusivity coefficient (m²/year)
    
    Returns:
        erosion: 2D array of erosion (positive) / deposition (negative) in meters
    """
    # Compute Laplacian
    up    = np.roll(surface_elev, -1, axis=0)
    down  = np.roll(surface_elev,  1, axis=0)
    left  = np.roll(surface_elev,  1, axis=1)
    right = np.roll(surface_elev, -1, axis=1)
    
    laplacian = (up + down + left + right - 4.0 * surface_elev) / (pixel_scale_m ** 2)
    
    # Change in elevation
    dz = D * laplacian * dt
    
    # Negative dz means deposition, positive means erosion
    # We return as "erosion" where positive = material removed
    erosion = -dz
    
    return erosion


# ==============================================================================
# 3. SEDIMENT TRANSPORT & DEPOSITION MODULE
# ==============================================================================

def compute_sediment_transport(
    flow_data, 
    erosion_channel, 
    erosion_hillslope,
    pixel_scale_m,
    transport_capacity_coeff=0.1
):
    """
    Compute sediment transport and deposition.
    
    Simple model:
    - Sediment is generated by erosion
    - Transport capacity ~ Q * S
    - If sediment > capacity, deposit excess
    
    Returns:
        deposition: 2D array of deposited thickness (m)
        sediment: 2D array of mobile sediment thickness (m)
    """
    discharge = flow_data["discharge"]
    slope = flow_data["slope"]
    flow_dir = flow_data["flow_dir"]
    
    ny, nx = erosion_channel.shape
    
    # Total local erosion
    erosion_total = erosion_channel + np.maximum(erosion_hillslope, 0)
    
    # Transport capacity (simple: proportional to Q * S)
    capacity = transport_capacity_coeff * discharge * slope
    
    # Initialize sediment flux
    sediment_in = np.zeros((ny, nx), dtype=np.float64)
    sediment_out = np.zeros((ny, nx), dtype=np.float64)
    deposition = np.zeros((ny, nx), dtype=np.float64)
    
    # Route sediment downslope (simplified)
    di = np.array([-1, -1, 0, 1, 1, 1, 0, -1], dtype=np.int32)
    dj = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int32)
    
    for i in range(ny):
        for j in range(nx):
            # Available sediment: eroded locally + incoming
            sed_available = erosion_total[i, j] + sediment_in[i, j]
            cap = capacity[i, j]
            
            if sed_available > cap:
                # Deposit excess
                deposition[i, j] = sed_available - cap
                sediment_out[i, j] = cap
            else:
                # Transport all
                sediment_out[i, j] = sed_available
                deposition[i, j] = 0.0
            
            # Route sediment to downstream cell
            d = flow_dir[i, j]
            if d >= 0:
                ni, nj = i + di[d], j + dj[d]
                if 0 <= ni < ny and 0 <= nj < nx:
                    sediment_in[ni, nj] += sediment_out[i, j]
    
    return deposition, sediment_out


# ==============================================================================
# 4. STRATIGRAPHY UPDATE LOGIC
# ==============================================================================

def update_stratigraphy_with_erosion(strata, erosion, pixel_scale_m):
    """
    Remove material from the top of the stratigraphy based on erosion field.
    
    Maintains layer ordering and ensures no layer goes negative.
    
    Args:
        strata: the stratigraphy dict (modified in place)
        erosion: 2D array of material to remove (m)
        pixel_scale_m: cell size
    """
    thickness = strata["thickness"]
    interfaces = strata["interfaces"]
    deposits = strata.get("deposits", {})
    
    ny, nx = erosion.shape
    
    # Layer order: top to bottom
    deposit_order = ["Alluvium", "Till", "Loess", "DuneSand"]
    strat_order = [
        "Topsoil", "Subsoil", "Colluvium", "Saprolite", "WeatheredBR",
        "Sandstone", "Shale", "Limestone", "Basement"
    ]
    
    for i in range(ny):
        for j in range(nx):
            remaining_erosion = erosion[i, j]
            
            if remaining_erosion <= 0:
                continue
            
            # Remove from deposits first
            for dep in deposit_order:
                if dep not in deposits:
                    continue
                
                if deposits[dep][i, j] > 0:
                    removed = min(deposits[dep][i, j], remaining_erosion)
                    deposits[dep][i, j] -= removed
                    remaining_erosion -= removed
                    
                    if remaining_erosion <= 0:
                        break
            
            # Remove from stratigraphic layers
            for layer in strat_order:
                if layer not in thickness:
                    continue
                
                if thickness[layer][i, j] > 0 and remaining_erosion > 0:
                    removed = min(thickness[layer][i, j], remaining_erosion)
                    thickness[layer][i, j] -= removed
                    remaining_erosion -= removed
                    
                    if remaining_erosion <= 0:
                        break
    
    # Recompute interfaces from thickness (top-down)
    surface_elev = strata["surface_elev"]
    
    # Update interfaces to match new thickness
    for layer in strat_order:
        if layer in interfaces and layer in thickness:
            # Interface is surface minus cumulative thickness above
            cumulative_above = np.zeros_like(surface_elev)
            
            # Add deposits
            for dep in deposit_order:
                if dep in deposits:
                    cumulative_above += deposits[dep]
            
            # Add stratigraphic layers above this one
            for upper_layer in strat_order:
                if upper_layer == layer:
                    break
                if upper_layer in thickness:
                    cumulative_above += thickness[upper_layer]
            
            # Interface is below surface by the cumulative thickness above it
            interfaces[layer][:, :] = surface_elev - cumulative_above
    
    # Update surface elevation (remove erosion)
    strata["surface_elev"] -= erosion


def update_stratigraphy_with_deposition(strata, deposition, pixel_scale_m):
    """
    Add material to the Alluvium layer (depositional unit).
    
    Args:
        strata: the stratigraphy dict (modified in place)
        deposition: 2D array of material to add (m)
        pixel_scale_m: cell size
    """
    deposits = strata.get("deposits", {})
    
    if "Alluvium" not in deposits:
        # Initialize if needed
        deposits["Alluvium"] = np.zeros_like(deposition)
        strata["deposits"] = deposits
    
    # Add to Alluvium thickness
    deposits["Alluvium"] += deposition
    
    # Update surface elevation (add deposition)
    strata["surface_elev"] += deposition


# ==============================================================================
# 5. TECTONIC UPLIFT MODULE
# ==============================================================================

def apply_uplift(strata, uplift_rate, dt):
    """
    Apply uniform or spatially variable tectonic uplift.
    
    Args:
        strata: the stratigraphy dict (modified in place)
        uplift_rate: scalar or 2D array (m/year)
        dt: time step (years)
    """
    interfaces = strata["interfaces"]
    
    # Uplift amount
    uplift = uplift_rate * dt
    
    # Raise surface
    strata["surface_elev"] += uplift
    
    # Raise all interfaces
    for layer in interfaces:
        interfaces[layer] += uplift


# ==============================================================================
# 6. TIME-STEPPING LOOP
# ==============================================================================

def run_erosion_epoch(
    strata,
    pixel_scale_m,
    dt,
    rainfall=None,
    uplift_rate=0.0,
    K_channel=1e-5,
    D_hillslope=0.01,
    stream_power_m=0.5,
    stream_power_n=1.0,
    Q_threshold=1e4,
    transport_coeff=0.1
):
    """
    Run one epoch (time step) of landscape evolution.
    
    Steps:
    1. Apply uplift
    2. Route water over current surface
    3. Compute channel incision
    4. Compute hillslope diffusion
    5. Compute sediment transport & deposition
    6. Update stratigraphy (erosion and deposition)
    
    Args:
        strata: stratigraphy dict from generate_stratigraphy (modified in place)
        pixel_scale_m: cell size (m)
        dt: time step duration (years)
        rainfall: 2D rainfall field (m/year), or None for uniform
        uplift_rate: scalar or 2D uplift rate (m/year)
        K_channel: channel erosion coefficient
        D_hillslope: hillslope diffusivity (m²/year)
        stream_power_m, stream_power_n: stream power exponents
        Q_threshold: minimum discharge to be a channel (m²)
        transport_coeff: sediment transport capacity coefficient
    
    Returns:
        dict with diagnostics:
            - erosion_channel: 2D array
            - erosion_hillslope: 2D array
            - deposition: 2D array
            - flow_data: dict
    """
    # Step 1: Uplift
    if uplift_rate != 0:
        apply_uplift(strata, uplift_rate, dt)
    
    # Step 2: Route flow
    flow_data = route_flow_simple(
        strata["surface_elev"], 
        pixel_scale_m, 
        rainfall
    )
    
    # Step 3: Channel incision
    erosion_channel = channel_incision_stream_power(
        strata,
        flow_data,
        pixel_scale_m,
        dt,
        K_base=K_channel,
        m=stream_power_m,
        n=stream_power_n,
        Q_threshold=Q_threshold
    )
    
    # Step 4: Hillslope diffusion
    erosion_hillslope = hillslope_diffusion(
        strata["surface_elev"],
        pixel_scale_m,
        dt,
        D=D_hillslope
    )
    
    # Step 5: Sediment transport & deposition
    deposition, sediment_flux = compute_sediment_transport(
        flow_data,
        erosion_channel,
        erosion_hillslope,
        pixel_scale_m,
        transport_capacity_coeff=transport_coeff
    )
    
    # Step 6a: Apply erosion
    total_erosion = np.maximum(erosion_channel, 0) + np.maximum(erosion_hillslope, 0)
    update_stratigraphy_with_erosion(strata, total_erosion, pixel_scale_m)
    
    # Step 6b: Apply deposition
    update_stratigraphy_with_deposition(strata, deposition, pixel_scale_m)
    
    return {
        "erosion_channel": erosion_channel,
        "erosion_hillslope": erosion_hillslope,
        "deposition": deposition,
        "flow_data": flow_data,
        "total_erosion": total_erosion,
    }


def run_erosion_simulation(
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
    """
    Run multiple epochs of landscape evolution.
    
    Args:
        strata: initial stratigraphy dict (will be modified)
        pixel_scale_m: cell size (m)
        num_epochs: number of time steps
        dt: time step duration (years)
        rainfall_func: callable(epoch) -> 2D rainfall array, or None
        uplift_rate: constant or callable(epoch) -> 2D uplift array
        K_channel: channel erosion coefficient
        D_hillslope: hillslope diffusivity
        verbose: print progress
    
    Returns:
        history: list of diagnostic dicts from each epoch
    """
    history = []
    
    for epoch in range(num_epochs):
        if verbose and epoch % max(1, num_epochs // 10) == 0:
            print(f"Epoch {epoch}/{num_epochs}")
        
        # Get rainfall for this epoch
        if rainfall_func is not None:
            rainfall = rainfall_func(epoch)
        else:
            rainfall = None
        
        # Get uplift for this epoch
        if callable(uplift_rate):
            uplift = uplift_rate(epoch)
        else:
            uplift = uplift_rate
        
        # Run one epoch
        diagnostics = run_erosion_epoch(
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


# ==============================================================================
# 7. VISUALIZATION FUNCTIONS
# ==============================================================================

def plot_erosion_evolution(strata_before, strata_after, diagnostics, pixel_scale_m):
    """
    Visualize the before/after state of an erosion simulation.
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Before surface
    ax = axes[0, 0]
    im = ax.imshow(strata_before["surface_elev"], origin="lower", cmap="terrain")
    ax.set_title("Surface Elevation (Before)")
    plt.colorbar(im, ax=ax, label="Elevation (m)")
    
    # After surface
    ax = axes[0, 1]
    im = ax.imshow(strata_after["surface_elev"], origin="lower", cmap="terrain")
    ax.set_title("Surface Elevation (After)")
    plt.colorbar(im, ax=ax, label="Elevation (m)")
    
    # Change in elevation
    ax = axes[0, 2]
    delta = strata_after["surface_elev"] - strata_before["surface_elev"]
    im = ax.imshow(delta, origin="lower", cmap="RdBu_r", vmin=-np.abs(delta).max(), vmax=np.abs(delta).max())
    ax.set_title("Elevation Change")
    plt.colorbar(im, ax=ax, label="Δz (m)")
    
    # Channel erosion
    ax = axes[1, 0]
    im = ax.imshow(diagnostics["erosion_channel"], origin="lower", cmap="hot_r")
    ax.set_title("Channel Erosion")
    plt.colorbar(im, ax=ax, label="Erosion (m)")
    
    # Deposition
    ax = axes[1, 1]
    im = ax.imshow(diagnostics["deposition"], origin="lower", cmap="Blues")
    ax.set_title("Deposition")
    plt.colorbar(im, ax=ax, label="Deposition (m)")
    
    # Discharge (log scale)
    ax = axes[1, 2]
    Q = diagnostics["flow_data"]["discharge"]
    im = ax.imshow(np.log10(Q + 1), origin="lower", cmap="viridis")
    ax.set_title("Flow Accumulation (log₁₀)")
    plt.colorbar(im, ax=ax, label="log₁₀(Q + 1)")
    
    plt.tight_layout()
    return fig


def plot_cross_section_evolution(strata_before, strata_after, row_idx, pixel_scale_m):
    """
    Plot a cross-section showing layer evolution.
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    N = strata_before["surface_elev"].shape[1]
    x = np.arange(N) * pixel_scale_m / 1000.0  # km
    
    # Before
    ax = axes[0]
    ax.plot(x, strata_before["surface_elev"][row_idx, :], 'k-', linewidth=2, label="Surface")
    
    # Plot some interfaces
    for layer in ["Topsoil", "Saprolite", "Sandstone", "Basement"]:
        if layer in strata_before["interfaces"]:
            ax.plot(x, strata_before["interfaces"][layer][row_idx, :], '--', alpha=0.6, label=layer)
    
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Elevation (m)")
    ax.set_title(f"Cross-Section Before (row {row_idx})")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    
    # After
    ax = axes[1]
    ax.plot(x, strata_after["surface_elev"][row_idx, :], 'k-', linewidth=2, label="Surface")
    
    for layer in ["Topsoil", "Saprolite", "Sandstone", "Basement"]:
        if layer in strata_after["interfaces"]:
            ax.plot(x, strata_after["interfaces"][layer][row_idx, :], '--', alpha=0.6, label=layer)
    
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Elevation (m)")
    ax.set_title(f"Cross-Section After (row {row_idx})")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


print("✓ Erosion model engine loaded successfully!")
print("  Main functions:")
print("    - run_erosion_epoch(): run one time step")
print("    - run_erosion_simulation(): run multiple time steps")
print("    - plot_erosion_evolution(): visualize results")
print("    - plot_cross_section_evolution(): show layer changes")
