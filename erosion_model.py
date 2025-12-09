#!/usr/bin/env python3
"""
Realistic Erosion Model Simulation

This script integrates:
- Random terrain generation (from Rivers new)
- Layer/stratigraphy generation (from Rivers new)  
- Weather/rain generation (from Rivers new)
- Erosion simulation based on water flow, layer erodibility, and time
- River and lake formation and evolution
- Topography visualization over time

Usage:
    python erosion_model.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from typing import Dict, List, Tuple, Optional, Any
import sys
import os

# Import functions from "Rivers new" file
# We'll need to import these functions - for now we'll define wrappers
# that can read from the file if needed

# Import functions from "Rivers new" file
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import importlib.util
    
    rivers_new_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Rivers new")
    if os.path.exists(rivers_new_path):
        spec = importlib.util.spec_from_file_location("rivers_new", rivers_new_path)
        if spec and spec.loader:
            rivers_new = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(rivers_new)
            # Import key functions
            quantum_seeded_topography = rivers_new.quantum_seeded_topography
            generate_stratigraphy = rivers_new.generate_stratigraphy
            compute_top_material_map = rivers_new.compute_top_material_map
            print("Successfully imported functions from 'Rivers new'")
        else:
            raise ImportError("Could not create spec from 'Rivers new'")
    else:
        raise FileNotFoundError(f"'Rivers new' file not found at {rivers_new_path}")
except Exception as e:
    print(f"Warning: Could not import from 'Rivers new': {e}")
    print("Will use simplified fallback functions")
    quantum_seeded_topography = None
    generate_stratigraphy = None
    compute_top_material_map = None


class ErosionModel:
    """
    Main erosion simulation model that integrates terrain, layers, weather, and erosion.
    """
    
    def __init__(
        self,
        grid_size: int = 256,
        pixel_scale_m: float = 10.0,
        elev_range_m: float = 700.0,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the erosion model.
        
        Parameters:
        -----------
        grid_size : int
            Size of the grid (grid_size x grid_size)
        pixel_scale_m : float
            Physical size of one pixel in meters
        elev_range_m : float
            Elevation range in meters
        random_seed : int, optional
            Random seed for reproducibility
        """
        self.grid_size = grid_size
        self.pixel_scale_m = pixel_scale_m
        self.elev_range_m = elev_range_m
        self.random_seed = random_seed
        
        # Initialize terrain and stratigraphy
        self.surface_elev = None
        self.strata = None
        self.layer_properties = None
        self.layer_order = None
        
        # Water and erosion state
        self.water_depth = None  # Current water depth map
        self.flow_accumulation = None  # Flow accumulation map
        self.erosion_rate = None  # Current erosion rate map
        self.total_erosion = None  # Cumulative erosion map
        
        # Rivers and lakes
        self.river_network = None  # Binary map of river channels
        self.lakes = None  # Binary map of lakes
        self.lake_depths = None  # Depth of lakes
        
        # Time tracking
        self.current_time_years = 0.0
        self.time_step_years = 0.1  # Default time step
        
    def generate_initial_terrain(self):
        """Generate initial terrain and stratigraphy."""
        print("Generating initial terrain...")
        
        if quantum_seeded_topography is None:
            # Fallback: simple random terrain
            print("Using fallback terrain generation...")
            rng = np.random.default_rng(self.random_seed)
            z_norm = self._generate_simple_terrain(rng)
            rng_obj = rng
        else:
            # Generate normalized topography using Rivers new functions
            z_norm, rng_obj = quantum_seeded_topography(
                N=self.grid_size,
                random_seed=self.random_seed
            )
        
        # Generate stratigraphy
        if generate_stratigraphy is None:
            print("Using simplified stratigraphy...")
            self.strata = self._generate_simple_stratigraphy(z_norm, rng_obj)
        else:
            self.strata = generate_stratigraphy(
                z_norm=z_norm,
                rng=rng_obj,
                elev_range_m=self.elev_range_m,
                pixel_scale_m=self.pixel_scale_m
            )
        
        self.surface_elev = self.strata["surface_elev"].copy()
        
        # Extract layer properties
        if "properties" in self.strata:
            self.layer_properties = self.strata["properties"]
        else:
            # Default properties if not in strata
            self.layer_properties = self._get_default_layer_properties()
        
        # Get layer order (top to bottom)
        self.layer_order = [
            "Topsoil", "Subsoil", "Colluvium", "Saprolite", "WeatheredBR",
            "Clay", "Silt", "Sand",
            "Sandstone", "Conglomerate", "Shale", "Mudstone", "Siltstone",
            "Limestone", "Dolomite", "Evaporite",
            "Granite", "Gneiss", "Basalt", "Basement", "BasementFloor"
        ]
        
        # Initialize erosion tracking
        self.total_erosion = np.zeros_like(self.surface_elev)
        self.water_depth = np.zeros_like(self.surface_elev)
        self.flow_accumulation = np.zeros_like(self.surface_elev)
        self.erosion_rate = np.zeros_like(self.surface_elev)
        
        print(f"Terrain generated: {self.grid_size}x{self.grid_size} grid")
        print(f"Elevation range: {self.surface_elev.min():.1f}m to {self.surface_elev.max():.1f}m")
    
    def _generate_simple_terrain(self, rng: np.random.Generator) -> np.ndarray:
        """Generate simple terrain as fallback."""
        N = self.grid_size
        # Use FFT-based noise for realistic terrain
        kx = np.fft.fftfreq(N)
        ky = np.fft.rfftfreq(N)
        K = np.sqrt(kx[:, None]**2 + ky[None, :]**2)
        K[0, 0] = np.inf
        amp = 1.0 / (K ** 1.5)
        phase = rng.uniform(0, 2*np.pi, size=(N, ky.size))
        spec = amp * (np.cos(phase) + 1j*np.sin(phase))
        spec[0, 0] = 0.0
        z = np.fft.irfftn(spec, s=(N, N))
        lo, hi = np.percentile(z, [2, 98])
        return np.clip((z - lo)/(hi - lo + 1e-12), 0, 1)
    
    def _generate_simple_stratigraphy(self, z_norm: np.ndarray, rng: np.random.Generator) -> Dict[str, Any]:
        """Generate simplified stratigraphy as fallback."""
        E = z_norm * self.elev_range_m
        
        # Simple layer thicknesses
        thickness = {
            "Topsoil": np.ones_like(E) * 0.5,
            "Subsoil": np.ones_like(E) * 0.8,
            "Colluvium": np.ones_like(E) * 2.0,
            "Saprolite": np.ones_like(E) * 5.0,
            "WeatheredBR": np.ones_like(E) * 2.0,
            "Sandstone": np.ones_like(E) * 50.0,
            "Shale": np.ones_like(E) * 40.0,
            "Limestone": np.ones_like(E) * 30.0,
            "Basement": np.ones_like(E) * 1000.0,
        }
        
        return {
            "surface_elev": E,
            "thickness": thickness,
            "properties": self._get_default_layer_properties()
        }
    
    def _get_default_layer_properties(self) -> Dict[str, Dict[str, float]]:
        """Get default erodibility values for different layers."""
        return {
            "Topsoil": {"erodibility": 1.00},
            "Subsoil": {"erodibility": 0.85},
            "Colluvium": {"erodibility": 0.90},
            "Alluvium": {"erodibility": 0.95},
            "Clay": {"erodibility": 0.80},
            "Silt": {"erodibility": 0.90},
            "Sand": {"erodibility": 0.85},
            "Till": {"erodibility": 0.75},
            "Loess": {"erodibility": 1.05},
            "DuneSand": {"erodibility": 0.95},
            "Saprolite": {"erodibility": 0.70},
            "WeatheredBR": {"erodibility": 0.55},
            "Shale": {"erodibility": 0.45},
            "Mudstone": {"erodibility": 0.45},
            "Siltstone": {"erodibility": 0.35},
            "Sandstone": {"erodibility": 0.30},
            "Conglomerate": {"erodibility": 0.25},
            "Limestone": {"erodibility": 0.28},
            "Dolomite": {"erodibility": 0.24},
            "Evaporite": {"erodibility": 0.90},
            "Basement": {"erodibility": 0.15},
            "Granite": {"erodibility": 0.15},
            "Gneiss": {"erodibility": 0.16},
            "Basalt": {"erodibility": 0.12},
            "BasementFloor": {"erodibility": 0.02},
        }
    
    def compute_flow_direction(self, elevation: np.ndarray) -> np.ndarray:
        """
        Compute flow direction using D8 algorithm (8-direction flow).
        
        Returns:
        --------
        flow_dir : np.ndarray, shape (Ny, Nx, 2)
            Flow direction vectors [dy, dx] for each cell
        """
        Ny, Nx = elevation.shape
        
        # Compute gradients
        dy, dx = np.gradient(elevation, self.pixel_scale_m)
        
        # D8 flow direction: find steepest descent among 8 neighbors
        flow_dir = np.zeros((Ny, Nx, 2), dtype=np.float32)
        
        # Check all 8 neighbors
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                
                # Shift elevation
                elev_shift = np.roll(np.roll(elevation, di, axis=0), dj, axis=1)
                
                # Distance to neighbor (diagonal = sqrt(2))
                dist = np.sqrt(di**2 + dj**2) * self.pixel_scale_m
                if dist == 0:
                    continue
                
                # Slope to neighbor
                slope = (elevation - elev_shift) / dist
                
                # Update flow direction if this is steeper
                mask = slope > 0  # Only downhill
                flow_dir[mask, 0] = np.where(
                    np.abs(slope[mask]) > np.abs(flow_dir[mask, 0] * di + flow_dir[mask, 1] * dj) / max(dist, 1e-9),
                    di,
                    flow_dir[mask, 0]
                )
                flow_dir[mask, 1] = np.where(
                    np.abs(slope[mask]) > np.abs(flow_dir[mask, 0] * di + flow_dir[mask, 1] * dj) / max(dist, 1e-9),
                    dj,
                    flow_dir[mask, 1]
                )
        
        # Normalize
        norm = np.sqrt(flow_dir[:, :, 0]**2 + flow_dir[:, :, 1]**2) + 1e-9
        flow_dir[:, :, 0] /= norm
        flow_dir[:, :, 1] /= norm
        
        return flow_dir
    
    def compute_flow_accumulation(self, elevation: np.ndarray, rainfall: np.ndarray) -> np.ndarray:
        """
        Compute flow accumulation using D8 flow direction algorithm.
        
        Parameters:
        -----------
        elevation : np.ndarray
            Current elevation map
        rainfall : np.ndarray
            Rainfall map (depth per time step)
        
        Returns:
        --------
        flow_accum : np.ndarray
            Flow accumulation map (total water flowing through each cell)
        """
        Ny, Nx = elevation.shape
        flow_accum = rainfall.copy().astype(np.float64)
        
        # Compute flow direction for each cell
        # D8: find steepest descent among 8 neighbors
        flow_dir = np.zeros((Ny, Nx, 2), dtype=np.int32)
        
        for i in range(Ny):
            for j in range(Nx):
                max_slope = -np.inf
                best_dir = [0, 0]
                
                # Check all 8 neighbors
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        
                        ni, nj = i + di, j + dj
                        if ni < 0 or ni >= Ny or nj < 0 or nj >= Nx:
                            continue
                        
                        # Distance (diagonal = sqrt(2))
                        dist = np.sqrt(di**2 + dj**2) * self.pixel_scale_m
                        if dist == 0:
                            continue
                        
                        # Slope to neighbor
                        slope = (elevation[i, j] - elevation[ni, nj]) / dist
                        
                        if slope > max_slope:
                            max_slope = slope
                            best_dir = [di, dj]
                
                flow_dir[i, j] = best_dir
        
        # Accumulate flow: process cells from high to low elevation
        sorted_indices = np.argsort(elevation.ravel())[::-1]
        
        for flat_idx in sorted_indices:
            i, j = np.unravel_index(flat_idx, (Ny, Nx))
            
            # Find cells that flow into this cell
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    
                    ni, nj = i - di, j - dj
                    if ni < 0 or ni >= Ny or nj < 0 or nj >= Nx:
                        continue
                    
                    # Check if neighbor flows into this cell
                    if (flow_dir[ni, nj, 0] == di and flow_dir[ni, nj, 1] == dj):
                        flow_accum[i, j] += flow_accum[ni, nj]
        
        return flow_accum
    
    def get_surface_layer_erodibility(self) -> np.ndarray:
        """
        Get erodibility map for the current surface layer.
        
        Returns:
        --------
        erodibility : np.ndarray
            Erodibility values (0-1) for each surface cell
        """
        if self.strata is None:
            return np.ones_like(self.surface_elev) * 0.5
        
        # Get top material map
        if "top_material" in self.strata:
            top_material = self.strata["top_material"]
        elif compute_top_material_map is not None and "thickness" in self.strata:
            # Use function from Rivers new if available
            try:
                top_material = compute_top_material_map(self.strata, min_thick=0.05)
            except:
                top_material = self._compute_top_material_simple()
        elif "thickness" in self.strata:
            top_material = self._compute_top_material_simple()
        else:
            # Fallback: use default based on elevation
            top_material = np.full_like(self.surface_elev, "Topsoil", dtype=object)
        
        # Map material names to erodibility
        erodibility = np.zeros_like(self.surface_elev, dtype=np.float32)
        
        # Vectorized approach for better performance
        if isinstance(top_material, np.ndarray) and top_material.dtype == object:
            # Handle object array (string array)
            for material_name in self.layer_properties.keys():
                mask = top_material == material_name
                if mask.any():
                    erodibility[mask] = self.layer_properties[material_name].get("erodibility", 0.5)
        else:
            # Fallback: iterate
            for i in range(self.surface_elev.shape[0]):
                for j in range(self.surface_elev.shape[1]):
                    material = top_material[i, j] if isinstance(top_material, np.ndarray) else "Topsoil"
                    if isinstance(material, np.ndarray):
                        material = str(material)
                    
                    if material in self.layer_properties:
                        erodibility[i, j] = self.layer_properties[material].get("erodibility", 0.5)
                    else:
                        erodibility[i, j] = 0.5  # Default
        
        return erodibility
    
    def _compute_top_material_simple(self) -> np.ndarray:
        """Simple method to compute top material from thickness."""
        if "thickness" not in self.strata:
            return np.full_like(self.surface_elev, "Topsoil", dtype=object)
        
        thickness = self.strata["thickness"]
        top_material = np.full_like(self.surface_elev, "Basement", dtype=object)
        
        # Check layers from top to bottom
        for layer_name in self.layer_order:
            if layer_name in thickness:
                layer_thick = thickness[layer_name]
                mask = layer_thick > 0.01  # At least 1cm thick
                top_material[mask] = layer_name
        
        return top_material
    
    def compute_erosion_rate(
        self,
        flow_accum: np.ndarray,
        slope: np.ndarray,
        erodibility: np.ndarray,
        water_depth: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute erosion rate based on flow, slope, and erodibility.
        
        Uses a simplified stream power law: E = K * A^m * S^n
        where:
        - E = erosion rate
        - K = erodibility
        - A = flow accumulation (drainage area)
        - S = slope
        - m, n = exponents (typically m=0.5, n=1.0)
        
        Parameters:
        -----------
        flow_accum : np.ndarray
            Flow accumulation map
        slope : np.ndarray
            Slope magnitude map
        erodibility : np.ndarray
            Erodibility map
        water_depth : np.ndarray, optional
            Water depth map (for sediment transport capacity)
        
        Returns:
        --------
        erosion_rate : np.ndarray
            Erosion rate in meters per year
        """
        # Stream power law parameters
        m = 0.5  # Flow exponent
        n = 1.0  # Slope exponent
        K_base = 1e-4  # Base erosion coefficient (m/year)
        
        # Normalize flow accumulation (0-1 scale)
        flow_norm = flow_accum / (flow_accum.max() + 1e-9)
        
        # Compute erosion rate
        erosion_rate = (
            K_base *
            erodibility *
            (flow_norm ** m) *
            (slope ** n)
        )
        
        # Add water depth effect (deeper water = more transport capacity)
        if water_depth is not None:
            water_factor = 1.0 + 0.5 * (water_depth / (water_depth.max() + 1e-9))
            erosion_rate *= water_factor
        
        # Cap maximum erosion rate (realistic limit)
        max_erosion_rate = 0.1  # meters per year
        erosion_rate = np.clip(erosion_rate, 0.0, max_erosion_rate)
        
        return erosion_rate
    
    def apply_erosion(self, erosion_rate: np.ndarray, dt_years: float):
        """
        Apply erosion to the surface elevation.
        
        Parameters:
        -----------
        erosion_rate : np.ndarray
            Erosion rate map (m/year)
        dt_years : float
            Time step in years
        """
        # Compute erosion depth
        erosion_depth = erosion_rate * dt_years
        
        # Update surface elevation
        self.surface_elev -= erosion_depth
        
        # Update total erosion tracking
        self.total_erosion += erosion_depth
        
        # Update stratigraphy (simplified: just reduce top layer thickness)
        if self.strata and "thickness" in self.strata:
            thickness = self.strata["thickness"]
            for layer_name in self.layer_order:
                if layer_name in thickness:
                    # Erode from top layer first
                    layer_eroded = np.minimum(erosion_depth, thickness[layer_name])
                    thickness[layer_name] -= layer_eroded
                    erosion_depth -= layer_eroded
                    
                    # Remove negative thicknesses
                    thickness[layer_name] = np.maximum(thickness[layer_name], 0.0)
                    
                    if erosion_depth.max() < 1e-6:
                        break
    
    def detect_rivers(self, flow_accum: np.ndarray, threshold_percentile: float = 95.0) -> np.ndarray:
        """
        Detect river network from flow accumulation.
        
        Parameters:
        -----------
        flow_accum : np.ndarray
            Flow accumulation map
        threshold_percentile : float
            Percentile threshold for river detection (e.g., 95 = top 5% are rivers)
        
        Returns:
        --------
        river_network : np.ndarray (bool)
            Binary map of river channels
        """
        threshold = np.percentile(flow_accum, threshold_percentile)
        river_network = flow_accum >= threshold
        
        # Clean up: remove isolated pixels
        river_network = ndimage.binary_opening(river_network, structure=np.ones((3, 3)))
        
        return river_network
    
    def detect_lakes(self, elevation: np.ndarray, water_depth: np.ndarray, min_depth: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect lakes from water depth and elevation.
        
        Parameters:
        -----------
        elevation : np.ndarray
            Surface elevation map
        water_depth : np.ndarray
            Water depth map
        min_depth : float
            Minimum depth to be considered a lake (meters)
        
        Returns:
        --------
        lakes : np.ndarray (bool)
            Binary map of lakes
        lake_depths : np.ndarray
            Depth map of lakes
        """
        # Find depressions (local minima)
        lakes = water_depth >= min_depth
        
        # Find connected components (individual lakes)
        labeled_lakes, num_lakes = ndimage.label(lakes)
        
        # Filter small lakes (less than 4 pixels)
        for label_id in range(1, num_lakes + 1):
            mask = labeled_lakes == label_id
            if mask.sum() < 4:
                lakes[mask] = False
        
        lake_depths = np.where(lakes, water_depth, 0.0)
        
        return lakes, lake_depths
    
    def simulate_water_flow(self, rainfall: np.ndarray, dt_years: float):
        """
        Simulate water flow, accumulation, and lake formation.
        
        Parameters:
        -----------
        rainfall : np.ndarray
            Rainfall map (depth per time step, meters)
        dt_years : float
            Time step in years
        """
        # Compute flow accumulation
        self.flow_accumulation = self.compute_flow_accumulation(self.surface_elev, rainfall)
        
        # Compute slope
        dy, dx = np.gradient(self.surface_elev, self.pixel_scale_m)
        slope = np.sqrt(dx**2 + dy**2)
        
        # Simple water routing: water flows downhill and accumulates
        # Water depth is proportional to flow accumulation and local slope
        flow_factor = self.flow_accumulation / (self.flow_accumulation.max() + 1e-9)
        slope_factor = 1.0 / (slope + 0.01)  # More water in flatter areas
        
        # Base water depth from rainfall
        base_water = rainfall * dt_years
        
        # Accumulated water depth (simplified)
        self.water_depth = base_water + 0.1 * flow_factor * slope_factor
        
        # Detect rivers
        self.river_network = self.detect_rivers(self.flow_accumulation)
        
        # Detect lakes
        self.lakes, self.lake_depths = self.detect_lakes(self.surface_elev, self.water_depth)
    
    def step(self, rainfall: np.ndarray, dt_years: Optional[float] = None):
        """
        Perform one time step of erosion simulation.
        
        Parameters:
        -----------
        rainfall : np.ndarray
            Rainfall map for this time step (meters)
        dt_years : float, optional
            Time step in years (uses self.time_step_years if not provided)
        """
        if dt_years is None:
            dt_years = self.time_step_years
        
        # Simulate water flow
        self.simulate_water_flow(rainfall, dt_years)
        
        # Get surface erodibility
        erodibility = self.get_surface_layer_erodibility()
        
        # Compute slope
        dy, dx = np.gradient(self.surface_elev, self.pixel_scale_m)
        slope = np.sqrt(dx**2 + dy**2)
        
        # Compute erosion rate
        self.erosion_rate = self.compute_erosion_rate(
            self.flow_accumulation,
            slope,
            erodibility,
            self.water_depth
        )
        
        # Apply erosion
        self.apply_erosion(self.erosion_rate, dt_years)
        
        # Update time
        self.current_time_years += dt_years
    
    def simulate(
        self,
        num_years: float,
        annual_rainfall_mm: float = 1000.0,
        time_step_years: float = 0.1,
        save_snapshots: bool = True,
        snapshot_interval_years: float = 1.0
    ) -> Dict[str, Any]:
        """
        Run erosion simulation for specified number of years.
        
        Parameters:
        -----------
        num_years : float
            Number of years to simulate
        annual_rainfall_mm : float
            Annual rainfall in mm (will be distributed spatially)
        time_step_years : float
            Time step for simulation
        save_snapshots : bool
            Whether to save elevation snapshots
        snapshot_interval_years : float
            Interval between snapshots
        
        Returns:
        --------
        results : dict
            Dictionary containing simulation results and snapshots
        """
        self.time_step_years = time_step_years
        
        # Generate initial terrain if not done
        if self.surface_elev is None:
            self.generate_initial_terrain()
        
        # Initialize results
        results = {
            "initial_elevation": self.surface_elev.copy(),
            "final_elevation": None,
            "snapshots": [],
            "snapshot_times": [],
            "total_erosion": None,
            "final_rivers": None,
            "final_lakes": None,
        }
        
        # Create base rainfall pattern (spatially variable)
        # Higher rainfall on higher elevations (orographic effect)
        elev_norm = (self.surface_elev - self.surface_elev.min()) / (self.surface_elev.max() - self.surface_elev.min() + 1e-9)
        base_rainfall_pattern = 0.5 + 0.5 * elev_norm  # More rain on higher ground
        
        # Normalize to get annual total
        base_rainfall_pattern = base_rainfall_pattern / base_rainfall_pattern.mean()
        annual_rainfall_m = annual_rainfall_mm / 1000.0  # Convert mm to m
        
        print(f"\nStarting erosion simulation for {num_years} years...")
        print(f"Time step: {time_step_years} years")
        print(f"Annual rainfall: {annual_rainfall_mm} mm")
        
        next_snapshot_time = snapshot_interval_years
        
        # Simulation loop
        while self.current_time_years < num_years:
            # Compute rainfall for this time step
            # Add some temporal variation
            time_variation = 0.8 + 0.4 * np.sin(2 * np.pi * self.current_time_years / 1.0)
            rainfall = (
                base_rainfall_pattern *
                annual_rainfall_m *
                time_step_years *
                time_variation
            )
            
            # Perform time step
            self.step(rainfall, time_step_years)
            
            # Save snapshot if needed
            if save_snapshots and self.current_time_years >= next_snapshot_time:
                snapshot = {
                    "time_years": self.current_time_years,
                    "elevation": self.surface_elev.copy(),
                    "rivers": self.river_network.copy() if self.river_network is not None else None,
                    "lakes": self.lakes.copy() if self.lakes is not None else None,
                    "erosion_rate": self.erosion_rate.copy(),
                    "flow_accumulation": self.flow_accumulation.copy(),
                }
                results["snapshots"].append(snapshot)
                results["snapshot_times"].append(self.current_time_years)
                print(f"  Snapshot at {self.current_time_years:.1f} years")
                next_snapshot_time += snapshot_interval_years
        
        # Final results
        results["final_elevation"] = self.surface_elev.copy()
        results["total_erosion"] = self.total_erosion.copy()
        results["final_rivers"] = self.river_network.copy() if self.river_network is not None else None
        results["final_lakes"] = self.lakes.copy() if self.lakes is not None else None
        
        print(f"\nSimulation complete!")
        print(f"Total erosion: {self.total_erosion.max():.2f} m maximum")
        print(f"Elevation change: {(results['initial_elevation'].max() - results['final_elevation'].max()):.2f} m")
        
        return results
    
    def plot_topography(
        self,
        elevation: Optional[np.ndarray] = None,
        rivers: Optional[np.ndarray] = None,
        lakes: Optional[np.ndarray] = None,
        title: str = "Topography",
        ax: Optional[plt.Axes] = None
    ):
        """
        Plot topography map with rivers and lakes overlaid.
        
        Parameters:
        -----------
        elevation : np.ndarray, optional
            Elevation map (uses self.surface_elev if not provided)
        rivers : np.ndarray, optional
            River network binary map
        lakes : np.ndarray, optional
            Lakes binary map
        title : str
            Plot title
        ax : plt.Axes, optional
            Matplotlib axes to plot on
        """
        if elevation is None:
            elevation = self.surface_elev
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot elevation
        im = ax.imshow(
            elevation,
            cmap='terrain',
            origin='lower',
            extent=[0, self.grid_size * self.pixel_scale_m / 1000,
                   0, self.grid_size * self.pixel_scale_m / 1000]
        )
        ax.set_xlabel('Distance (km)')
        ax.set_ylabel('Distance (km)')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label='Elevation (m)')
        
        # Overlay rivers
        if rivers is not None:
            river_y, river_x = np.where(rivers)
            if len(river_y) > 0:
                ax.scatter(
                    river_x * self.pixel_scale_m / 1000,
                    river_y * self.pixel_scale_m / 1000,
                    c='blue', s=1, alpha=0.6, label='Rivers'
                )
        
        # Overlay lakes
        if lakes is not None:
            lake_y, lake_x = np.where(lakes)
            if len(lake_y) > 0:
                ax.scatter(
                    lake_x * self.pixel_scale_m / 1000,
                    lake_y * self.pixel_scale_m / 1000,
                    c='cyan', s=2, alpha=0.7, label='Lakes', marker='s'
                )
        
        if rivers is not None or lakes is not None:
            ax.legend()
        
        return ax
    
    def plot_erosion_map(self, erosion: np.ndarray, title: str = "Erosion Map", ax: Optional[plt.Axes] = None):
        """Plot erosion map."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(
            erosion,
            cmap='Reds',
            origin='lower',
            extent=[0, self.grid_size * self.pixel_scale_m / 1000,
                   0, self.grid_size * self.pixel_scale_m / 1000]
        )
        ax.set_xlabel('Distance (km)')
        ax.set_ylabel('Distance (km)')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label='Erosion (m)')
        
        return ax
    
    def plot_time_series(self, results: Dict[str, Any], save_path: Optional[str] = None):
        """
        Create time series plot showing evolution of topography.
        
        Parameters:
        -----------
        results : dict
            Results dictionary from simulate()
        save_path : str, optional
            Path to save the figure
        """
        snapshots = results.get("snapshots", [])
        
        if len(snapshots) == 0:
            print("No snapshots to plot")
            return
        
        # Determine grid layout
        n_snapshots = len(snapshots)
        n_cols = min(4, n_snapshots)
        n_rows = (n_snapshots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        if n_snapshots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, snapshot in enumerate(snapshots):
            ax = axes[idx]
            self.plot_topography(
                elevation=snapshot["elevation"],
                rivers=snapshot["rivers"],
                lakes=snapshot["lakes"],
                title=f"t = {snapshot['time_years']:.1f} years",
                ax=ax
            )
        
        # Hide unused subplots
        for idx in range(n_snapshots, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig


def main():
    """Main function to run erosion simulation example."""
    print("=" * 60)
    print("Realistic Erosion Model Simulation")
    print("=" * 60)
    
    # Create erosion model
    model = ErosionModel(
        grid_size=256,
        pixel_scale_m=10.0,
        elev_range_m=700.0,
        random_seed=42
    )
    
    # Generate initial terrain
    model.generate_initial_terrain()
    
    # Run simulation
    results = model.simulate(
        num_years=10.0,
        annual_rainfall_mm=1200.0,
        time_step_years=0.1,
        save_snapshots=True,
        snapshot_interval_years=2.0
    )
    
    # Create visualization
    print("\nCreating visualizations...")
    
    # Initial vs final comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    model.plot_topography(
        elevation=results["initial_elevation"],
        title="Initial Topography",
        ax=axes[0]
    )
    
    model.plot_topography(
        elevation=results["final_elevation"],
        rivers=results["final_rivers"],
        lakes=results["final_lakes"],
        title="Final Topography (after erosion)",
        ax=axes[1]
    )
    
    plt.tight_layout()
    plt.savefig("erosion_comparison.png", dpi=150, bbox_inches='tight')
    print("Saved: erosion_comparison.png")
    
    # Erosion map
    fig, ax = plt.subplots(figsize=(12, 10))
    model.plot_erosion_map(results["total_erosion"], title="Total Erosion", ax=ax)
    plt.tight_layout()
    plt.savefig("erosion_map.png", dpi=150, bbox_inches='tight')
    print("Saved: erosion_map.png")
    
    # Time series
    model.plot_time_series(results, save_path="erosion_timeseries.png")
    print("Saved: erosion_timeseries.png")
    
    print("\nSimulation complete! Check the generated PNG files.")
    plt.show()


if __name__ == "__main__":
    main()
