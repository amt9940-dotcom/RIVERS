#!/usr/bin/env python3
"""
Realistic Erosion Simulation Model

This model simulates realistic erosion of layered earth over time with:
- Multi-layer geological stratigraphy with different erodibility
- Rainfall-driven erosion from weather systems
- Water flow accumulation and pathfinding
- River and lake formation
- Realistic sediment transport and deposition
- Time-stepped simulation with adjustable parameters

Integrates with the quantum-seeded terrain and weather generation from Rivers new.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# LAYER PROPERTIES AND ERODIBILITY
# ============================================================================

# Erodibility coefficient (K) for different geological materials
# Higher K = more easily eroded. Units: (m^2 * s) / kg
ERODIBILITY = {
    # Surface materials (most erodible)
    "Topsoil": 0.0050,
    "Subsoil": 0.0040,
    "Clay": 0.0045,
    "Silt": 0.0055,
    "Sand": 0.0070,
    
    # Colluvial and weathered materials
    "Colluvium": 0.0035,
    "Saprolite": 0.0030,
    "WeatheredBR": 0.0025,
    "Till": 0.0035,
    "Loess": 0.0060,
    
    # Sedimentary rocks (moderate)
    "Sandstone": 0.0020,
    "Conglomerate": 0.0015,
    "Shale": 0.0025,
    "Mudstone": 0.0028,
    "Siltstone": 0.0022,
    "Limestone": 0.0018,
    "Dolomite": 0.0016,
    "Evaporite": 0.0030,
    
    # Volcanic rocks
    "Basalt": 0.0012,
    "Volcanic_rock": 0.0014,
    
    # Metamorphic and crystalline (resistant)
    "Granite": 0.0008,
    "Gneiss": 0.0010,
    "Schist": 0.0012,
    "AncientCrust": 0.0005,
    
    # Basement (very resistant)
    "Basement": 0.0003,
    "BasementFloor": 0.0001,
    
    # Default for unknown materials
    "Unknown": 0.0020,
}

# Soil cohesion (resistance to detachment) in Pa
COHESION = {
    "Topsoil": 1000,
    "Subsoil": 1500,
    "Clay": 2500,
    "Silt": 800,
    "Sand": 500,
    "Colluvium": 2000,
    "Saprolite": 3000,
    "WeatheredBR": 4000,
    "Sandstone": 10000,
    "Shale": 8000,
    "Limestone": 15000,
    "Granite": 50000,
    "Basement": 100000,
}


# ============================================================================
# EROSION SIMULATION CLASS
# ============================================================================

class ErosionSimulation:
    """
    Main erosion simulation class that handles:
    - Terrain with multiple geological layers
    - Rainfall-driven erosion
    - Water flow and accumulation
    - Sediment transport and deposition
    - River and lake formation
    """
    
    def __init__(
        self,
        surface_elevation: np.ndarray,
        layer_interfaces: Dict[str, np.ndarray],
        layer_order: List[str],
        pixel_scale_m: float = 100.0,
        uplift_rate: float = 0.0,
    ):
        """
        Initialize erosion simulation.
        
        Parameters:
        -----------
        surface_elevation : ndarray
            Initial surface elevation map (m)
        layer_interfaces : dict
            Dictionary mapping layer name to elevation of layer top (m)
        layer_order : list
            Ordered list of layer names from top to bottom
        pixel_scale_m : float
            Spatial resolution in meters per pixel
        uplift_rate : float
            Tectonic uplift rate in m/year (optional)
        """
        self.elevation = np.array(surface_elevation, dtype=np.float64)
        self.original_elevation = self.elevation.copy()
        self.ny, self.nx = self.elevation.shape
        
        # Layer information
        self.layer_interfaces = {k: np.array(v, dtype=np.float64) 
                                for k, v in layer_interfaces.items()}
        self.layer_order = layer_order
        
        # Spatial parameters
        self.pixel_scale_m = pixel_scale_m
        self.cell_area_m2 = pixel_scale_m ** 2
        
        # Tectonic parameters
        self.uplift_rate = uplift_rate
        
        # State variables
        self.water_depth = np.zeros_like(self.elevation)
        self.sediment_depth = np.zeros_like(self.elevation)
        self.flow_accumulation = np.zeros_like(self.elevation)
        self.cumulative_erosion = np.zeros_like(self.elevation)
        self.cumulative_deposition = np.zeros_like(self.elevation)
        
        # River and lake tracking
        self.river_mask = np.zeros(self.elevation.shape, dtype=bool)
        self.lake_mask = np.zeros(self.elevation.shape, dtype=bool)
        
        # Time tracking
        self.current_time = 0.0  # in years
        
        # Compute initial properties
        self._update_derived_properties()
    
    def _update_derived_properties(self):
        """Update slope, aspect, and other derived topographic properties."""
        # Compute gradients (dy, dx from np.gradient)
        dy, dx = np.gradient(self.elevation, self.pixel_scale_m, self.pixel_scale_m)
        
        self.slope_x = dx
        self.slope_y = dy
        self.slope_magnitude = np.sqrt(dx**2 + dy**2)
        
        # Aspect (direction of steepest descent)
        self.aspect = np.arctan2(-dy, -dx)
        
        # Curvature (simple Laplacian for now)
        self.curvature = self._compute_curvature()
    
    def _compute_curvature(self) -> np.ndarray:
        """Compute surface curvature (2nd derivative)."""
        # Simple 5-point Laplacian
        up = np.roll(self.elevation, -1, axis=0)
        down = np.roll(self.elevation, 1, axis=0)
        left = np.roll(self.elevation, 1, axis=1)
        right = np.roll(self.elevation, -1, axis=1)
        
        laplacian = (up + down + left + right - 4.0 * self.elevation) / (self.pixel_scale_m ** 2)
        return laplacian
    
    def get_surface_material(self) -> np.ndarray:
        """
        Determine the surface material at each cell based on current elevation
        and layer interfaces.
        
        Returns:
        --------
        material : ndarray of str
            2D array of material names at surface
        """
        material = np.full(self.elevation.shape, "Unknown", dtype=object)
        
        # Check layers from top to bottom
        for layer_name in self.layer_order:
            if layer_name not in self.layer_interfaces:
                continue
            
            layer_top = self.layer_interfaces[layer_name]
            
            # Cells where current elevation is at or above this layer top
            mask = self.elevation >= layer_top
            material[mask] = layer_name
        
        return material
    
    def get_erodibility_map(self) -> np.ndarray:
        """
        Get the erodibility coefficient for each surface cell.
        
        Returns:
        --------
        K : ndarray
            2D array of erodibility coefficients
        """
        material = self.get_surface_material()
        K = np.zeros(self.elevation.shape, dtype=np.float64)
        
        for material_name, k_value in ERODIBILITY.items():
            mask = material == material_name
            K[mask] = k_value
        
        # Set default for any Unknown materials
        K[K == 0] = ERODIBILITY["Unknown"]
        
        return K
    
    def compute_flow_accumulation(self) -> np.ndarray:
        """
        Compute flow accumulation using D8 flow routing algorithm.
        
        Returns:
        --------
        flow_acc : ndarray
            Flow accumulation (upstream contributing area in cells)
        """
        # Initialize flow accumulation (each cell contributes 1)
        flow_acc = np.ones(self.elevation.shape, dtype=np.float64)
        
        # Get flow directions (D8)
        flow_dir = self._compute_flow_directions()
        
        # Sort cells by elevation (process from high to low)
        sorted_indices = np.argsort(self.elevation.ravel())[::-1]
        rows, cols = np.unravel_index(sorted_indices, self.elevation.shape)
        
        # Accumulate flow
        for r, c in zip(rows, cols):
            # Get downstream cell
            dr, dc = flow_dir[r, c]
            nr, nc = r + dr, c + dc
            
            # Check bounds
            if 0 <= nr < self.ny and 0 <= nc < self.nx:
                # Add this cell's flow to downstream cell
                flow_acc[nr, nc] += flow_acc[r, c]
        
        self.flow_accumulation = flow_acc
        return flow_acc
    
    def _compute_flow_directions(self) -> np.ndarray:
        """
        Compute D8 flow directions (steepest descent).
        
        Returns:
        --------
        flow_dir : ndarray of shape (ny, nx, 2)
            Direction to downstream cell as (dr, dc)
        """
        flow_dir = np.zeros((self.ny, self.nx, 2), dtype=np.int32)
        
        # 8 neighbors
        neighbors = [(-1, -1), (-1, 0), (-1, 1),
                     (0, -1),           (0, 1),
                     (1, -1),  (1, 0),  (1, 1)]
        
        for r in range(self.ny):
            for c in range(self.nx):
                max_slope = -1e9
                best_dir = (0, 0)
                
                for dr, dc in neighbors:
                    nr, nc = r + dr, c + dc
                    
                    # Check bounds
                    if not (0 <= nr < self.ny and 0 <= nc < self.nx):
                        continue
                    
                    # Compute slope to neighbor
                    distance = np.sqrt((dr * self.pixel_scale_m)**2 + 
                                     (dc * self.pixel_scale_m)**2)
                    slope = (self.elevation[r, c] - self.elevation[nr, nc]) / distance
                    
                    if slope > max_slope:
                        max_slope = slope
                        best_dir = (dr, dc)
                
                flow_dir[r, c] = best_dir
        
        return flow_dir
    
    def apply_rainfall(
        self, 
        rainfall_mm: float,
        duration_hours: float = 1.0,
        rainfall_map: Optional[np.ndarray] = None
    ):
        """
        Apply rainfall to the terrain, adding water depth.
        
        Parameters:
        -----------
        rainfall_mm : float
            Rainfall amount in millimeters (uniform if rainfall_map not provided)
        duration_hours : float
            Duration of rainfall event in hours
        rainfall_map : ndarray, optional
            Spatially-varying rainfall intensity (mm). If None, uniform rainfall.
        """
        if rainfall_map is None:
            rainfall_map = np.full(self.elevation.shape, rainfall_mm)
        
        # Convert mm to meters and add to water depth
        water_added_m = rainfall_map / 1000.0
        self.water_depth += water_added_m
    
    def simulate_water_flow(self, dt: float = 0.1):
        """
        Simulate water flow across the terrain using shallow water approximation.
        
        Parameters:
        -----------
        dt : float
            Time step in hours
        """
        if self.water_depth.max() < 1e-6:
            return  # No water to flow
        
        # Convert dt to seconds
        dt_s = dt * 3600.0
        
        # Compute water surface elevation
        water_surface = self.elevation + self.water_depth
        
        # Compute flow between cells based on water surface gradient
        dy, dx = np.gradient(water_surface, self.pixel_scale_m, self.pixel_scale_m)
        
        # Manning's equation for flow velocity
        # v = (1/n) * R^(2/3) * S^(1/2)
        # Simplified: use water depth as hydraulic radius, slope from gradient
        n_manning = 0.03  # roughness coefficient
        
        slope = np.sqrt(dx**2 + dy**2) + 1e-9
        velocity = (1.0 / n_manning) * (self.water_depth ** (2.0/3.0)) * (slope ** 0.5)
        
        # Limit velocity for stability
        max_velocity = self.pixel_scale_m / dt_s
        velocity = np.minimum(velocity, max_velocity)
        
        # Compute flux (m^2/s per unit width)
        flux = velocity * self.water_depth
        
        # Update water depth based on flux divergence
        # Simple diffusion-like approach
        water_gradient_mag = np.sqrt(dx**2 + dy**2) + 1e-9
        
        # Water flows from high to low
        water_flux_div = gaussian_filter(flux * water_gradient_mag, sigma=1.0)
        
        # Update water depth
        delta_water = -water_flux_div * dt_s / self.pixel_scale_m
        self.water_depth = np.maximum(self.water_depth + delta_water, 0.0)
        
        # Account for infiltration and evaporation (simple)
        infiltration_rate = 0.01  # m/hour
        evaporation_rate = 0.001  # m/hour
        
        loss_rate = (infiltration_rate + evaporation_rate) * dt
        self.water_depth = np.maximum(self.water_depth - loss_rate, 0.0)
    
    def compute_erosion(
        self,
        dt: float = 0.1,
        transport_coefficient: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute erosion and deposition based on water flow and sediment transport.
        
        Uses the stream power law: E = K * A^m * S^n
        where K is erodibility, A is drainage area, S is slope
        
        Parameters:
        -----------
        dt : float
            Time step in years
        transport_coefficient : float
            Sediment transport capacity coefficient
        
        Returns:
        --------
        erosion : ndarray
            Erosion amount in meters
        deposition : ndarray
            Deposition amount in meters
        """
        # Get current erodibility based on surface material
        K = self.get_erodibility_map()
        
        # Compute flow accumulation if not already done
        if self.flow_accumulation.max() == 0:
            self.compute_flow_accumulation()
        
        # Drainage area (flow accumulation * cell area)
        drainage_area = self.flow_accumulation * self.cell_area_m2
        
        # Stream power erosion law: E = K * A^m * S^n
        m = 0.5  # drainage area exponent
        n = 1.0  # slope exponent
        
        # Compute erosion rate (m/year)
        erosion_rate = K * (drainage_area ** m) * (self.slope_magnitude ** n)
        
        # Apply rainfall intensity modifier (more water = more erosion)
        rainfall_factor = 1.0 + (self.water_depth / 0.1)  # Scale with water depth
        erosion_rate *= rainfall_factor
        
        # Compute actual erosion over timestep
        erosion = erosion_rate * dt
        
        # Limit erosion to available material
        # Don't erode below the lowest layer interface
        if len(self.layer_interfaces) > 0:
            lowest_interface = np.min([v for v in self.layer_interfaces.values()], axis=0)
            max_erosion = np.maximum(self.elevation - lowest_interface, 0.0)
            erosion = np.minimum(erosion, max_erosion)
        
        # Sediment transport capacity (based on flow and slope)
        transport_capacity = transport_coefficient * drainage_area * self.slope_magnitude
        
        # Compute deposition
        # Sediment deposited when transport capacity is insufficient
        deposition = np.zeros_like(erosion)
        
        # Track sediment in transport
        sediment_flux = erosion.copy()
        
        # Route sediment downstream and deposit where capacity is exceeded
        flow_dir = self._compute_flow_directions()
        sorted_indices = np.argsort(self.elevation.ravel())[::-1]
        rows, cols = np.unravel_index(sorted_indices, self.elevation.shape)
        
        for r, c in zip(rows, cols):
            # Get incoming sediment
            incoming_sed = self.sediment_depth[r, c]
            
            # Total sediment at this cell
            total_sed = incoming_sed + sediment_flux[r, c]
            
            # Transport capacity at this cell
            capacity = transport_capacity[r, c]
            
            if total_sed > capacity:
                # Deposit excess
                deposit = total_sed - capacity
                deposition[r, c] = deposit
                sediment_flux[r, c] = capacity
            else:
                sediment_flux[r, c] = total_sed
            
            # Route sediment downstream
            dr, dc = flow_dir[r, c]
            nr, nc = r + dr, c + dc
            
            if 0 <= nr < self.ny and 0 <= nc < self.nx:
                self.sediment_depth[nr, nc] += sediment_flux[r, c]
        
        # Clear sediment depth after routing
        self.sediment_depth = deposition.copy()
        
        return erosion, deposition
    
    def update_elevation(
        self,
        erosion: np.ndarray,
        deposition: np.ndarray,
        dt: float = 1.0
    ):
        """
        Update elevation based on erosion, deposition, and uplift.
        
        Parameters:
        -----------
        erosion : ndarray
            Erosion amount in meters
        deposition : ndarray
            Deposition amount in meters
        dt : float
            Time step in years
        """
        # Apply erosion (lower elevation)
        self.elevation -= erosion
        
        # Apply deposition (raise elevation)
        self.elevation += deposition
        
        # Apply tectonic uplift
        if self.uplift_rate > 0:
            self.elevation += self.uplift_rate * dt
        
        # Update cumulative totals
        self.cumulative_erosion += erosion
        self.cumulative_deposition += deposition
        
        # Update derived properties
        self._update_derived_properties()
    
    def detect_rivers(self, threshold_accumulation: float = 100.0):
        """
        Detect river channels based on flow accumulation.
        
        Parameters:
        -----------
        threshold_accumulation : float
            Minimum flow accumulation (in cells) to be considered a river
        """
        self.river_mask = self.flow_accumulation >= threshold_accumulation
    
    def detect_lakes(self, min_depth: float = 0.1):
        """
        Detect lakes as closed depressions with standing water.
        
        Parameters:
        -----------
        min_depth : float
            Minimum water depth (m) to be considered a lake
        """
        # Lakes are areas with significant water depth
        potential_lakes = self.water_depth >= min_depth
        
        # Additional criterion: low slope (flat areas)
        flat_areas = self.slope_magnitude < 0.01
        
        self.lake_mask = potential_lakes & flat_areas
    
    def step(
        self,
        dt: float = 1.0,
        rainfall_mm: float = 100.0,
        rainfall_map: Optional[np.ndarray] = None
    ):
        """
        Perform one time step of the erosion simulation.
        
        Parameters:
        -----------
        dt : float
            Time step in years
        rainfall_mm : float
            Rainfall amount in millimeters
        rainfall_map : ndarray, optional
            Spatially-varying rainfall (mm)
        """
        # 1. Apply rainfall
        self.apply_rainfall(rainfall_mm, duration_hours=dt*8760, rainfall_map=rainfall_map)
        
        # 2. Simulate water flow (multiple sub-steps for stability)
        n_flow_steps = max(1, int(dt * 10))
        for _ in range(n_flow_steps):
            self.simulate_water_flow(dt=dt * 8760 / n_flow_steps)
        
        # 3. Compute flow accumulation
        self.compute_flow_accumulation()
        
        # 4. Compute erosion and deposition
        erosion, deposition = self.compute_erosion(dt=dt)
        
        # 5. Update elevation
        self.update_elevation(erosion, deposition, dt=dt)
        
        # 6. Detect rivers and lakes
        self.detect_rivers()
        self.detect_lakes()
        
        # 7. Update time
        self.current_time += dt
    
    def get_total_erosion(self) -> float:
        """Get total volume of material eroded (m^3)."""
        return np.sum(self.cumulative_erosion) * self.cell_area_m2
    
    def get_total_deposition(self) -> float:
        """Get total volume of material deposited (m^3)."""
        return np.sum(self.cumulative_deposition) * self.cell_area_m2
    
    def get_elevation_change(self) -> np.ndarray:
        """Get net elevation change from initial state."""
        return self.elevation - self.original_elevation


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_topography(
    sim: ErosionSimulation,
    title: str = "Topography",
    ax: Optional[plt.Axes] = None,
    show_rivers: bool = True,
    show_lakes: bool = True
):
    """
    Plot the current topography with rivers and lakes.
    
    Parameters:
    -----------
    sim : ErosionSimulation
        Simulation object
    title : str
        Plot title
    ax : plt.Axes, optional
        Axes to plot on
    show_rivers : bool
        Whether to overlay rivers
    show_lakes : bool
        Whether to overlay lakes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot elevation
    im = ax.imshow(
        sim.elevation,
        origin='lower',
        cmap='terrain',
        extent=[0, sim.nx * sim.pixel_scale_m / 1000,
                0, sim.ny * sim.pixel_scale_m / 1000]
    )
    
    # Overlay rivers
    if show_rivers and np.any(sim.river_mask):
        river_overlay = np.ma.masked_where(~sim.river_mask, sim.river_mask)
        ax.imshow(
            river_overlay,
            origin='lower',
            cmap='Blues',
            alpha=0.6,
            extent=[0, sim.nx * sim.pixel_scale_m / 1000,
                    0, sim.ny * sim.pixel_scale_m / 1000]
        )
    
    # Overlay lakes
    if show_lakes and np.any(sim.lake_mask):
        lake_overlay = np.ma.masked_where(~sim.lake_mask, sim.lake_mask)
        ax.imshow(
            lake_overlay,
            origin='lower',
            cmap='Blues_r',
            alpha=0.7,
            extent=[0, sim.nx * sim.pixel_scale_m / 1000,
                    0, sim.ny * sim.pixel_scale_m / 1000]
        )
    
    plt.colorbar(im, ax=ax, label='Elevation (m)')
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Distance (km)')
    ax.set_title(title)
    
    return ax


def plot_erosion_deposition(
    sim: ErosionSimulation,
    ax: Optional[plt.Axes] = None
):
    """
    Plot cumulative erosion and deposition.
    
    Parameters:
    -----------
    sim : ErosionSimulation
        Simulation object
    ax : plt.Axes, optional
        Axes to plot on
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Net change (positive = deposition, negative = erosion)
    net_change = sim.cumulative_deposition - sim.cumulative_erosion
    
    # Use diverging colormap
    vmax = np.abs(net_change).max()
    
    im = ax.imshow(
        net_change,
        origin='lower',
        cmap='RdBu',
        vmin=-vmax,
        vmax=vmax,
        extent=[0, sim.nx * sim.pixel_scale_m / 1000,
                0, sim.ny * sim.pixel_scale_m / 1000]
    )
    
    plt.colorbar(im, ax=ax, label='Net Change (m)\nRed=Deposition, Blue=Erosion')
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Distance (km)')
    ax.set_title(f'Cumulative Erosion and Deposition (t={sim.current_time:.1f} years)')
    
    return ax


def plot_water_features(
    sim: ErosionSimulation,
    ax: Optional[plt.Axes] = None
):
    """
    Plot water depth, rivers, and lakes.
    
    Parameters:
    -----------
    sim : ErosionSimulation
        Simulation object
    ax : plt.Axes, optional
        Axes to plot on
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Base: elevation in grayscale
    ax.imshow(
        sim.elevation,
        origin='lower',
        cmap='gray',
        alpha=0.3,
        extent=[0, sim.nx * sim.pixel_scale_m / 1000,
                0, sim.ny * sim.pixel_scale_m / 1000]
    )
    
    # Water depth
    if sim.water_depth.max() > 0:
        water_masked = np.ma.masked_where(sim.water_depth < 0.001, sim.water_depth)
        im = ax.imshow(
            water_masked,
            origin='lower',
            cmap='Blues',
            alpha=0.7,
            extent=[0, sim.nx * sim.pixel_scale_m / 1000,
                    0, sim.ny * sim.pixel_scale_m / 1000]
        )
        plt.colorbar(im, ax=ax, label='Water Depth (m)')
    
    # Rivers
    if np.any(sim.river_mask):
        river_y, river_x = np.where(sim.river_mask)
        ax.scatter(
            river_x * sim.pixel_scale_m / 1000,
            river_y * sim.pixel_scale_m / 1000,
            c='blue',
            s=1,
            alpha=0.5,
            label='Rivers'
        )
    
    # Lakes
    if np.any(sim.lake_mask):
        lake_y, lake_x = np.where(sim.lake_mask)
        ax.scatter(
            lake_x * sim.pixel_scale_m / 1000,
            lake_y * sim.pixel_scale_m / 1000,
            c='cyan',
            s=2,
            alpha=0.6,
            label='Lakes'
        )
    
    if np.any(sim.river_mask) or np.any(sim.lake_mask):
        ax.legend()
    
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Distance (km)')
    ax.set_title(f'Water Features (t={sim.current_time:.1f} years)')
    
    return ax


def plot_simulation_summary(sim: ErosionSimulation, figsize=(15, 10)):
    """
    Create a comprehensive summary plot of the simulation state.
    
    Parameters:
    -----------
    sim : ErosionSimulation
        Simulation object
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Topography with rivers and lakes
    plot_topography(sim, title=f'Topography (t={sim.current_time:.1f} years)', 
                   ax=axes[0, 0])
    
    # Erosion and deposition
    plot_erosion_deposition(sim, ax=axes[0, 1])
    
    # Water features
    plot_water_features(sim, ax=axes[1, 0])
    
    # Flow accumulation
    flow_log = np.log10(sim.flow_accumulation + 1)
    im = axes[1, 1].imshow(
        flow_log,
        origin='lower',
        cmap='viridis',
        extent=[0, sim.nx * sim.pixel_scale_m / 1000,
                0, sim.ny * sim.pixel_scale_m / 1000]
    )
    plt.colorbar(im, ax=axes[1, 1], label='log10(Flow Accumulation)')
    axes[1, 1].set_xlabel('Distance (km)')
    axes[1, 1].set_ylabel('Distance (km)')
    axes[1, 1].set_title('Flow Accumulation (Drainage Network)')
    
    plt.tight_layout()
    return fig


# ============================================================================
# MAIN SIMULATION RUNNER
# ============================================================================

def run_erosion_simulation(
    surface_elevation: np.ndarray,
    layer_interfaces: Dict[str, np.ndarray],
    layer_order: List[str],
    n_years: float = 1000.0,
    dt: float = 1.0,
    rainfall_mm_per_year: float = 1000.0,
    pixel_scale_m: float = 100.0,
    uplift_rate: float = 0.0,
    plot_interval: int = 100,
    rainfall_generator: Optional[Any] = None,
) -> ErosionSimulation:
    """
    Run a complete erosion simulation over specified time period.
    
    Parameters:
    -----------
    surface_elevation : ndarray
        Initial surface elevation (m)
    layer_interfaces : dict
        Layer interface elevations
    layer_order : list
        Ordered list of layer names
    n_years : float
        Total simulation time in years
    dt : float
        Time step in years
    rainfall_mm_per_year : float
        Average annual rainfall (mm/year)
    pixel_scale_m : float
        Spatial resolution (m/pixel)
    uplift_rate : float
        Tectonic uplift rate (m/year)
    plot_interval : int
        Plot results every N steps (0 = no plotting)
    rainfall_generator : callable, optional
        Function that generates rainfall maps: rainfall_map = fn(time_year)
    
    Returns:
    --------
    sim : ErosionSimulation
        Final simulation state
    """
    # Initialize simulation
    sim = ErosionSimulation(
        surface_elevation=surface_elevation,
        layer_interfaces=layer_interfaces,
        layer_order=layer_order,
        pixel_scale_m=pixel_scale_m,
        uplift_rate=uplift_rate
    )
    
    # Calculate number of steps
    n_steps = int(n_years / dt)
    
    print(f"Starting erosion simulation:")
    print(f"  Domain: {sim.nx} x {sim.ny} cells")
    print(f"  Resolution: {pixel_scale_m} m/pixel")
    print(f"  Duration: {n_years} years")
    print(f"  Time step: {dt} years")
    print(f"  Total steps: {n_steps}")
    print()
    
    # Run simulation
    for step in range(n_steps):
        # Get rainfall for this time step
        if rainfall_generator is not None:
            rainfall_map = rainfall_generator(sim.current_time)
        else:
            # Uniform rainfall
            rainfall_map = np.full(
                sim.elevation.shape,
                rainfall_mm_per_year * dt
            )
        
        # Perform one time step
        sim.step(dt=dt, rainfall_map=rainfall_map)
        
        # Progress report
        if (step + 1) % 10 == 0:
            total_erosion = sim.get_total_erosion()
            total_deposition = sim.get_total_deposition()
            n_rivers = np.sum(sim.river_mask)
            n_lakes = np.sum(sim.lake_mask)
            
            print(f"Step {step+1}/{n_steps} (t={sim.current_time:.1f} yr): "
                  f"Erosion={total_erosion/1e6:.2f} km³, "
                  f"Deposition={total_deposition/1e6:.2f} km³, "
                  f"Rivers={n_rivers} cells, Lakes={n_lakes} cells")
        
        # Plot if requested
        if plot_interval > 0 and (step + 1) % plot_interval == 0:
            plot_simulation_summary(sim)
            plt.savefig(f'erosion_t{sim.current_time:.0f}yr.png', dpi=150, bbox_inches='tight')
            plt.show()
    
    print(f"\nSimulation complete!")
    print(f"Final time: {sim.current_time:.1f} years")
    print(f"Total erosion: {sim.get_total_erosion()/1e9:.3f} km³")
    print(f"Total deposition: {sim.get_total_deposition()/1e9:.3f} km³")
    
    return sim


if __name__ == "__main__":
    print("Erosion Simulation Module")
    print("=" * 60)
    print("This module provides realistic erosion simulation with:")
    print("  - Multi-layer geology with different erodibility")
    print("  - Rainfall-driven erosion")
    print("  - Water flow and sediment transport")
    print("  - River and lake formation")
    print("  - Time-stepped simulation")
    print()
    print("See example_erosion_simulation.py for usage examples.")
