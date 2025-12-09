"""
EROSION SYSTEM - MAIN SIMULATION FUNCTION

Integrates all erosion components into a complete simulation:
1. Apply extreme rain boost (100×)
2. Compute runoff from rainfall
3. Determine flow directions (D8)
4. Compute discharge Q
5. Compute slopes
6. PASS A: Erode terrain (with half-loss rule)
7. PASS B: Transport and deposit sediment
8. Apply hillslope diffusion
9. Update layer information

Includes time acceleration: 1 sim year = 10 real years
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
import copy

def run_erosion_timestep(
    elevation: np.ndarray,
    thickness: Dict[str, np.ndarray],
    layer_order: List[str],
    rain_map: np.ndarray,
    pixel_scale_m: float,
    dt: float = 1.0,
    apply_diffusion: bool = True,
    verbose: bool = False
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Run one erosion timestep.
    
    Parameters
    ----------
    elevation : np.ndarray (ny, nx)
        Surface elevation [m].
    thickness : dict of np.ndarray
        Layer thicknesses [m] for each layer name.
    layer_order : list of str
        Layer names from top to bottom.
    rain_map : np.ndarray (ny, nx)
        Rainfall for this timestep [m/yr or m].
    pixel_scale_m : float
        Grid cell size [m].
    dt : float
        Timestep duration [years].
    apply_diffusion : bool
        Whether to apply hillslope diffusion.
    verbose : bool
        Print progress messages.
    
    Returns
    -------
    elevation_new : np.ndarray (ny, nx)
        Updated elevation [m].
    thickness_new : dict of np.ndarray
        Updated layer thicknesses [m].
    diagnostics : dict
        Diagnostic information (Q, rivers, lakes, etc.).
    """
    ny, nx = elevation.shape
    
    # STEP 1: Apply extreme rain boost
    rain_boosted = rain_map * RAIN_BOOST
    
    if verbose:
        print(f"  Rain: {rain_map.mean():.3f} m/yr → {rain_boosted.mean():.1f} m/yr (boosted)")
    
    # STEP 2: Compute runoff
    runoff = compute_runoff(rain_boosted, infiltration_fraction=INFILTRATION_FRACTION)
    
    # STEP 3: Compute flow direction
    flow_dir, receivers, slopes = compute_flow_direction_d8(elevation, pixel_scale_m)
    
    # STEP 4: Compute discharge Q
    Q = compute_discharge(elevation, flow_dir, receivers, runoff, pixel_scale_m)
    
    if verbose:
        print(f"  Discharge: {Q.min():.1f} - {Q.max():.1f} m³/yr")
    
    # STEP 5: Get erodibility from top layer
    top_idx, top_name = compute_top_layer_map(thickness, layer_order)
    erodibility = get_erodibility_grid(top_name, ERODIBILITY_MAP)
    
    # STEP 6: PASS A - Erosion with half-loss
    elevation_eroded, sediment_out = compute_erosion_pass_a(
        elevation, Q, slopes, flow_dir, erodibility,
        base_k=BASE_K,
        flat_k=FLAT_K,
        max_erode_per_step=MAX_ERODE_PER_STEP,
        slope_threshold=SLOPE_THRESHOLD,
        half_loss_fraction=HALF_LOSS_FRACTION,
        m_discharge=M_DISCHARGE,
        n_slope=N_SLOPE,
        dt=dt
    )
    
    # Compute erosion depth
    erosion_depth = elevation - elevation_eroded
    
    if verbose:
        total_erosion = erosion_depth.sum() * pixel_scale_m ** 2 / 1e6
        print(f"  Erosion: {total_erosion:.2f} km³")
    
    # STEP 7: PASS B - Sediment transport and deposition
    elevation_final = compute_sediment_transport_pass_b(
        elevation_eroded, Q, slopes, flow_dir, receivers, sediment_out,
        capacity_k=CAPACITY_K,
        capacity_m=CAPACITY_M,
        capacity_n=CAPACITY_N,
        pixel_scale_m=pixel_scale_m
    )
    
    # Compute deposition
    deposition = elevation_final - elevation_eroded
    
    if verbose:
        total_deposition = deposition.sum() * pixel_scale_m ** 2 / 1e6
        print(f"  Deposition: {total_deposition:.2f} km³")
    
    # STEP 8: Apply hillslope diffusion
    if apply_diffusion:
        elevation_final = apply_hillslope_diffusion_8neighbor(
            elevation_final, 
            diffusion_k=DIFFUSION_K,
            pixel_scale_m=pixel_scale_m,
            dt=dt
        )
    
    # STEP 9: Update layer thicknesses
    thickness_new = copy.deepcopy(thickness)
    
    # Total elevation change
    dz_total = elevation_final - elevation
    
    # Update layers based on erosion/deposition
    for i in range(ny):
        for j in range(nx):
            dz = dz_total[i, j]
            
            if dz < 0:
                # EROSION: remove material from top layers
                remaining_erosion = -dz
                for layer in layer_order:
                    if remaining_erosion <= 0:
                        break
                    available = thickness_new[layer][i, j]
                    removed = min(available, remaining_erosion)
                    thickness_new[layer][i, j] -= removed
                    remaining_erosion -= removed
            
            elif dz > 0:
                # DEPOSITION: add to topsoil or alluvium
                # Use "Alluvium" if it exists, else "Topsoil"
                if "Alluvium" in thickness_new:
                    thickness_new["Alluvium"][i, j] += dz
                elif "Topsoil" in thickness_new:
                    thickness_new["Topsoil"][i, j] += dz
                else:
                    # Create a new top layer if needed
                    thickness_new[layer_order[0]][i, j] += dz
    
    # Prepare diagnostics
    diagnostics = {
        "Q": Q,
        "flow_dir": flow_dir,
        "receivers": receivers,
        "slopes": slopes,
        "erosion": erosion_depth,
        "deposition": deposition,
        "dz_total": dz_total,
        "runoff": runoff,
        "top_layer": top_name,
        "erodibility": erodibility
    }
    
    return elevation_final, thickness_new, diagnostics


def run_erosion_simulation(
    elevation_initial: np.ndarray,
    thickness_initial: Dict[str, np.ndarray],
    layer_order: List[str],
    rain_timeseries: np.ndarray,
    pixel_scale_m: float,
    dt: float = 1.0,
    num_timesteps: int = 100,
    save_interval: int = 10,
    apply_diffusion: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Run full erosion simulation over multiple timesteps.
    
    Parameters
    ----------
    elevation_initial : np.ndarray (ny, nx)
        Initial surface elevation [m].
    thickness_initial : dict
        Initial layer thicknesses [m].
    layer_order : list of str
        Layer names from top to bottom.
    rain_timeseries : np.ndarray (num_timesteps, ny, nx)
        Rainfall for each timestep [m/yr].
    pixel_scale_m : float
        Grid cell size [m].
    dt : float
        Timestep duration [years].
    num_timesteps : int
        Number of timesteps to simulate.
    save_interval : int
        Save snapshots every N timesteps.
    apply_diffusion : bool
        Apply hillslope diffusion.
    verbose : bool
        Print progress.
    
    Returns
    -------
    results : dict
        Contains:
        - elevation_history: list of elevation snapshots
        - thickness_history: list of thickness dict snapshots
        - diagnostics_history: list of diagnostic dicts
        - time_points: list of time values [years]
    """
    print("\n" + "="*80)
    print("EROSION SIMULATION START")
    print("="*80)
    print(f"Time acceleration: {TIME_ACCELERATION}×")
    print(f"Timesteps: {num_timesteps} × {dt} years = {num_timesteps * dt} sim years")
    print(f"Real time equivalent: {num_timesteps * dt * TIME_ACCELERATION} years")
    print(f"Rain boost: {RAIN_BOOST}×")
    print("="*80)
    
    # Initialize
    elevation = elevation_initial.copy()
    thickness = copy.deepcopy(thickness_initial)
    
    # Storage
    elevation_history = [elevation.copy()]
    thickness_history = [copy.deepcopy(thickness)]
    diagnostics_history = []
    time_points = [0.0]
    
    # Main loop
    for step in range(num_timesteps):
        t = step * dt
        
        if verbose and step % max(1, num_timesteps // 10) == 0:
            print(f"\nStep {step}/{num_timesteps} (t={t:.1f} yr, real={t*TIME_ACCELERATION:.1f} yr)")
        
        # Get rain for this timestep
        if rain_timeseries.ndim == 3:
            rain_map = rain_timeseries[step]
        else:
            rain_map = rain_timeseries  # Single map for all timesteps
        
        # Run one timestep
        elevation, thickness, diagnostics = run_erosion_timestep(
            elevation, thickness, layer_order, rain_map, pixel_scale_m,
            dt=dt, apply_diffusion=apply_diffusion, 
            verbose=(verbose and step % max(1, num_timesteps // 10) == 0)
        )
        
        # Save snapshots
        if (step + 1) % save_interval == 0 or step == num_timesteps - 1:
            elevation_history.append(elevation.copy())
            thickness_history.append(copy.deepcopy(thickness))
            diagnostics_history.append(diagnostics)
            time_points.append(t + dt)
            
            if verbose:
                print(f"  → Snapshot saved (t={t+dt:.1f} yr)")
    
    print("\n" + "="*80)
    print("EROSION SIMULATION COMPLETE")
    print("="*80)
    print(f"Initial elevation: {elevation_initial.min():.1f} - {elevation_initial.max():.1f} m")
    print(f"Final elevation: {elevation.min():.1f} - {elevation.max():.1f} m")
    print(f"Total elevation change: {(elevation - elevation_initial).sum():.1f} m")
    print("="*80 + "\n")
    
    results = {
        "elevation_history": elevation_history,
        "thickness_history": thickness_history,
        "diagnostics_history": diagnostics_history,
        "time_points": time_points,
        "elevation_initial": elevation_initial,
        "elevation_final": elevation,
        "thickness_initial": thickness_initial,
        "thickness_final": thickness
    }
    
    return results

print("\n✅ Main erosion simulation function loaded!")
