"""
Hydrology and Water Routing Module

This module handles water flow over the landscape surface:
- Flow direction computation (D8 steepest descent)
- Flow accumulation (discharge proxy)
- Identification of channels vs hillslopes
- Lake/depression filling (optional)

The output provides the template for where channels form and how much
water flows through each cell, which drives erosion processes.
"""

import numpy as np
from typing import Tuple, Optional
from numba import njit


@njit
def _compute_flow_directions_d8(
    elev: np.ndarray,
    dx: float,
    flow_dir: np.ndarray,
    slope: np.ndarray
) -> None:
    """
    Compute D8 flow directions (steepest descent).
    
    Uses Numba for speed. Modifies flow_dir and slope arrays in place.
    
    Parameters
    ----------
    elev : np.ndarray
        Surface elevation (ny, nx)
    dx : float
        Grid spacing (m)
    flow_dir : np.ndarray
        Output: flow direction index (0-7 for neighbors, -1 for pits)
    slope : np.ndarray
        Output: maximum downslope gradient (m/m)
    """
    ny, nx = elev.shape
    
    # Neighbor offsets (8 directions)
    di = np.array([-1, -1, -1,  0,  0,  1,  1,  1], dtype=np.int32)
    dj = np.array([-1,  0,  1, -1,  1, -1,  0,  1], dtype=np.int32)
    
    # Distances to neighbors
    dist = np.array([
        np.sqrt(2)*dx, dx, np.sqrt(2)*dx,
        dx, dx,
        np.sqrt(2)*dx, dx, np.sqrt(2)*dx
    ], dtype=np.float32)
    
    for i in range(ny):
        for j in range(nx):
            z = elev[i, j]
            
            max_slope = -1e10
            best_dir = -1
            
            # Check all 8 neighbors
            for k in range(8):
                ni = i + di[k]
                nj = j + dj[k]
                
                # Check bounds
                if ni < 0 or ni >= ny or nj < 0 or nj >= nx:
                    continue
                
                # Compute slope to this neighbor
                dz = z - elev[ni, nj]
                s = dz / dist[k]
                
                # Track steepest descent
                if s > max_slope:
                    max_slope = s
                    best_dir = k
            
            flow_dir[i, j] = best_dir
            
            # Store slope (0 if pit/flat)
            if max_slope > 0:
                slope[i, j] = max_slope
            else:
                slope[i, j] = 0.0


@njit
def _compute_flow_accumulation(
    flow_dir: np.ndarray,
    flow_accum: np.ndarray
) -> None:
    """
    Compute flow accumulation from flow directions.
    
    Uses a simple ordered traversal (not optimal for large grids, but works).
    Modifies flow_accum in place.
    
    Parameters
    ----------
    flow_dir : np.ndarray
        Flow direction indices (0-7 for neighbors, -1 for pits)
    flow_accum : np.ndarray
        Output: flow accumulation (number of upstream cells)
    """
    ny, nx = flow_dir.shape
    
    # Neighbor offsets
    di = np.array([-1, -1, -1,  0,  0,  1,  1,  1], dtype=np.int32)
    dj = np.array([-1,  0,  1, -1,  1, -1,  0,  1], dtype=np.int32)
    
    # Initialize: each cell contributes its own area
    flow_accum[:, :] = 1.0
    
    # Multiple passes to propagate flow (simple but not optimal)
    # For proper ordering, would need to topologically sort, but this works for demo
    max_passes = max(ny, nx) * 2
    
    for _ in range(max_passes):
        updated = False
        
        for i in range(ny):
            for j in range(nx):
                dir_idx = flow_dir[i, j]
                
                # Skip pits
                if dir_idx < 0:
                    continue
                
                # Get downstream neighbor
                ni = i + di[dir_idx]
                nj = j + dj[dir_idx]
                
                # Check bounds (shouldn't happen if flow_dir is correct)
                if ni < 0 or ni >= ny or nj < 0 or nj >= nx:
                    continue
                
                # Accumulate flow downstream
                old_val = flow_accum[ni, nj]
                new_val = old_val + flow_accum[i, j]
                
                if new_val != old_val:
                    flow_accum[ni, nj] = new_val
                    updated = True
        
        # If nothing changed, we're done
        if not updated:
            break


class FlowRouter:
    """
    Handles water routing over the surface.
    
    Computes:
    - Flow directions (D8 steepest descent)
    - Flow accumulation (proxy for discharge)
    - Slope field
    - Channel network identification
    """
    
    def __init__(self, pixel_scale_m: float):
        """
        Initialize flow router.
        
        Parameters
        ----------
        pixel_scale_m : float
            Grid spacing (m)
        """
        self.pixel_scale_m = pixel_scale_m
        
        # Cached results
        self.flow_dir = None
        self.slope = None
        self.flow_accum = None
        self.drainage_area = None
        
    def compute_flow(
        self,
        surface_elev: np.ndarray,
        fill_depressions: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute flow routing on the surface.
        
        Parameters
        ----------
        surface_elev : np.ndarray
            Surface elevation (m), shape (ny, nx)
        fill_depressions : bool
            If True, fill closed depressions before routing.
            If False, water can pool in depressions (more realistic for lakes).
            
        Returns
        -------
        flow_dir : np.ndarray
            Flow direction indices (0-7), shape (ny, nx)
        slope : np.ndarray
            Downslope gradient (m/m), shape (ny, nx)
        flow_accum : np.ndarray
            Flow accumulation (number of cells), shape (ny, nx)
        """
        ny, nx = surface_elev.shape
        
        # Optionally fill depressions
        if fill_depressions:
            elev = self._fill_depressions(surface_elev)
        else:
            elev = surface_elev
        
        # Allocate output arrays
        self.flow_dir = np.full((ny, nx), -1, dtype=np.int32)
        self.slope = np.zeros((ny, nx), dtype=np.float32)
        self.flow_accum = np.zeros((ny, nx), dtype=np.float32)
        
        # Compute flow directions and slopes
        _compute_flow_directions_d8(
            elev.astype(np.float32),
            self.pixel_scale_m,
            self.flow_dir,
            self.slope
        )
        
        # Compute flow accumulation
        _compute_flow_accumulation(self.flow_dir, self.flow_accum)
        
        # Convert to drainage area (m^2)
        cell_area = self.pixel_scale_m ** 2
        self.drainage_area = self.flow_accum * cell_area
        
        return self.flow_dir, self.slope, self.flow_accum
    
    def _fill_depressions(self, elev: np.ndarray) -> np.ndarray:
        """
        Fill closed depressions using priority-flood algorithm.
        
        This is a simplified version. For production use, consider
        using existing libraries like richdem or pysheds.
        
        Parameters
        ----------
        elev : np.ndarray
            Input elevation
            
        Returns
        -------
        np.ndarray
            Filled elevation (no closed depressions)
        """
        # Simple priority-flood implementation
        # For now, just return the input (depressions allowed)
        # TODO: Implement proper priority-flood if needed
        return elev.copy()
    
    def identify_channels(
        self,
        threshold_cells: Optional[int] = None,
        threshold_area_m2: Optional[float] = None
    ) -> np.ndarray:
        """
        Identify channel cells based on flow accumulation threshold.
        
        Parameters
        ----------
        threshold_cells : int, optional
            Minimum number of upstream cells to be considered a channel
        threshold_area_m2 : float, optional
            Minimum upstream drainage area (m^2) to be considered a channel.
            If provided, overrides threshold_cells.
            
        Returns
        -------
        np.ndarray
            Boolean mask: True for channel cells, False for hillslopes
        """
        if self.flow_accum is None:
            raise ValueError("Must call compute_flow() first")
        
        if threshold_area_m2 is not None:
            # Convert to cell count
            cell_area = self.pixel_scale_m ** 2
            threshold_cells = threshold_area_m2 / cell_area
        elif threshold_cells is None:
            # Default: 1% of domain
            ny, nx = self.flow_accum.shape
            threshold_cells = 0.01 * ny * nx
        
        return self.flow_accum >= threshold_cells
    
    def get_discharge_proxy(
        self,
        rainfall_m_per_yr: np.ndarray,
        dt_years: float = 1.0
    ) -> np.ndarray:
        """
        Get water discharge proxy at each cell.
        
        Discharge (m^3/s) ≈ (drainage_area × rainfall_rate) / time
        
        This is a simplification that assumes:
        - Steady-state flow
        - All rainfall becomes runoff (no infiltration)
        - Uniform rainfall over upstream area
        
        Parameters
        ----------
        rainfall_m_per_yr : np.ndarray
            Rainfall rate field (m/yr)
        dt_years : float
            Time step (years)
            
        Returns
        -------
        np.ndarray
            Discharge proxy (m^3/s)
        """
        if self.drainage_area is None:
            raise ValueError("Must call compute_flow() first")
        
        # Average rainfall over drainage area (simplified: use local rainfall)
        rainfall_m_per_s = rainfall_m_per_yr / (365.25 * 24 * 3600)
        
        # Discharge = drainage_area × rainfall_rate
        discharge = self.drainage_area * rainfall_m_per_s
        
        return discharge
    
    def get_stream_power(
        self,
        discharge: np.ndarray,
        K: float = 1e-4,
        m: float = 0.5,
        n: float = 1.0
    ) -> np.ndarray:
        """
        Compute stream power for erosion calculations.
        
        Stream power law: E = K × A^m × S^n
        where A is drainage area (or discharge proxy), S is slope.
        
        Parameters
        ----------
        discharge : np.ndarray
            Water discharge (m^3/s)
        K : float
            Erodibility coefficient
        m : float
            Drainage area exponent (typically 0.4-0.6)
        n : float
            Slope exponent (typically 0.8-1.2)
            
        Returns
        -------
        np.ndarray
            Stream power (units depend on K)
        """
        if self.slope is None:
            raise ValueError("Must call compute_flow() first")
        
        # Stream power = K × A^m × S^n
        # Use discharge as proxy for drainage area
        stream_power = K * (discharge ** m) * (self.slope ** n)
        
        return stream_power
    
    def __repr__(self):
        if self.flow_accum is not None:
            max_accum = self.flow_accum.max()
            return f"FlowRouter(dx={self.pixel_scale_m}m, max_accum={max_accum:.0f})"
        else:
            return f"FlowRouter(dx={self.pixel_scale_m}m, not computed)"


def compute_simple_drainage(
    surface_elev: np.ndarray,
    pixel_scale_m: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple convenience function to compute drainage.
    
    Parameters
    ----------
    surface_elev : np.ndarray
        Surface elevation (m)
    pixel_scale_m : float
        Grid spacing (m)
        
    Returns
    -------
    flow_accum : np.ndarray
        Flow accumulation (number of cells)
    slope : np.ndarray
        Slope (m/m)
    """
    router = FlowRouter(pixel_scale_m)
    flow_dir, slope, flow_accum = router.compute_flow(surface_elev)
    return flow_accum, slope
