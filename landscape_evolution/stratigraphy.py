"""
Stratigraphy Update Module

This module handles layer-aware erosion and deposition.

Key principles:
1. Erosion removes material from the topmost layer first
2. When a layer is exhausted, erosion continues into the layer below
3. Deposition adds material to the top layer (usually mobile sediment/alluvium)
4. Layer ordering is always maintained
5. No layer inversions allowed

This module bridges geomorphic processes with the subsurface stratigraphy.
"""

import numpy as np
from typing import Tuple, Optional
from .world_state import WorldState


class StratigraphyUpdater:
    """
    Handles layer-aware erosion and deposition.
    
    This class modifies the WorldState to account for erosion and
    deposition while maintaining geological consistency.
    """
    
    def __init__(self, min_layer_thickness: float = 0.01):
        """
        Initialize stratigraphy updater.
        
        Parameters
        ----------
        min_layer_thickness : float
            Minimum thickness (m) to consider a layer present.
            Layers thinner than this are effectively removed.
        """
        self.min_layer_thickness = min_layer_thickness
    
    def apply_erosion(
        self,
        world: WorldState,
        erosion_depth: np.ndarray,
        verbose: bool = False
    ) -> dict:
        """
        Apply erosion to the world state.
        
        Removes material from the topmost layers, working downward
        through the stratigraphy until the erosion depth is satisfied.
        
        Parameters
        ----------
        world : WorldState
            The world state to modify (modified in place)
        erosion_depth : np.ndarray
            Depth of material to erode at each (i,j) location (m).
            Positive values = erosion.
        verbose : bool
            If True, print diagnostic information
            
        Returns
        -------
        dict
            Statistics about erosion:
            - 'total_eroded': total volume eroded (m^3)
            - 'layers_exposed': dict of layer name -> number of cells where exposed
        """
        ny, nx = world.ny, world.nx
        
        # Track statistics
        total_eroded = 0.0
        layers_exposed = {name: 0 for name in world.layer_names}
        
        # Ensure erosion depth is positive
        erosion_depth = np.maximum(erosion_depth, 0.0)
        
        # Process each cell
        for i in range(ny):
            for j in range(nx):
                if erosion_depth[i, j] <= 0:
                    continue
                
                remaining_erosion = erosion_depth[i, j]
                
                # Erode through layers from top to bottom
                for layer_name in world.layer_names:
                    if remaining_erosion <= 0:
                        break
                    
                    # Current thickness of this layer
                    current_thickness = world.layer_thickness[layer_name][i, j]
                    
                    if current_thickness < self.min_layer_thickness:
                        # Layer already gone, move to next
                        continue
                    
                    # Erode from this layer
                    eroded_from_layer = min(remaining_erosion, current_thickness)
                    
                    world.layer_thickness[layer_name][i, j] -= eroded_from_layer
                    remaining_erosion -= eroded_from_layer
                    total_eroded += eroded_from_layer
                    
                    # If layer is now exposed at the surface
                    if world.layer_thickness[layer_name][i, j] < self.min_layer_thickness:
                        layers_exposed[layer_name] += 1
                
                # Update surface elevation
                world.surface_elev[i, j] -= erosion_depth[i, j]
        
        # Rebuild interfaces from thicknesses
        world.update_interfaces_from_thicknesses()
        
        # Enforce layer ordering
        world.enforce_layer_ordering()
        
        # Convert to volume (m^3)
        cell_area = world.pixel_scale_m ** 2
        total_volume = total_eroded * cell_area
        
        if verbose:
            print(f"Erosion applied: {total_volume:.2e} m³")
            print(f"Layers exposed: {layers_exposed}")
        
        return {
            'total_eroded': total_volume,
            'layers_exposed': layers_exposed
        }
    
    def apply_deposition(
        self,
        world: WorldState,
        deposition_depth: np.ndarray,
        target_layer: str = "Alluvium",
        verbose: bool = False
    ) -> dict:
        """
        Apply deposition to the world state.
        
        Adds material to the specified layer (typically mobile sediment or alluvium).
        
        Parameters
        ----------
        world : WorldState
            The world state to modify (modified in place)
        deposition_depth : np.ndarray
            Depth of material to deposit at each (i,j) location (m).
            Positive values = deposition.
        target_layer : str
            Name of the layer to add material to (must be in world.layer_names)
        verbose : bool
            If True, print diagnostic information
            
        Returns
        -------
        dict
            Statistics about deposition:
            - 'total_deposited': total volume deposited (m^3)
        """
        # Ensure deposition depth is positive
        deposition_depth = np.maximum(deposition_depth, 0.0)
        
        # Check that target layer exists
        if target_layer not in world.layer_names:
            raise ValueError(f"Target layer '{target_layer}' not in world.layer_names")
        
        # Add to layer thickness
        world.layer_thickness[target_layer] += deposition_depth
        
        # Update surface elevation
        world.surface_elev += deposition_depth
        
        # Rebuild interfaces from thicknesses
        world.update_interfaces_from_thicknesses()
        
        # Enforce layer ordering
        world.enforce_layer_ordering()
        
        # Statistics
        cell_area = world.pixel_scale_m ** 2
        total_volume = deposition_depth.sum() * cell_area
        
        if verbose:
            print(f"Deposition applied: {total_volume:.2e} m³ to {target_layer}")
        
        return {
            'total_deposited': total_volume
        }
    
    def apply_weathering(
        self,
        world: WorldState,
        weathering_depth: np.ndarray,
        source_layer: str,
        target_layer: str = "Saprolite",
        verbose: bool = False
    ) -> dict:
        """
        Apply weathering: convert bedrock to regolith.
        
        Removes material from source_layer and adds to target_layer.
        
        Parameters
        ----------
        world : WorldState
            The world state to modify (modified in place)
        weathering_depth : np.ndarray
            Depth of bedrock to weather (m)
        source_layer : str
            Layer to remove material from (e.g., "Sandstone")
        target_layer : str
            Layer to add weathered material to (e.g., "Saprolite")
        verbose : bool
            If True, print diagnostic information
            
        Returns
        -------
        dict
            Statistics about weathering
        """
        # Ensure weathering depth is positive
        weathering_depth = np.maximum(weathering_depth, 0.0)
        
        # Check that layers exist
        if source_layer not in world.layer_names:
            raise ValueError(f"Source layer '{source_layer}' not in world.layer_names")
        if target_layer not in world.layer_names:
            raise ValueError(f"Target layer '{target_layer}' not in world.layer_names")
        
        # Only weather where source layer exists
        source_thickness = world.layer_thickness[source_layer]
        actual_weathering = np.minimum(weathering_depth, source_thickness)
        
        # Remove from source
        world.layer_thickness[source_layer] -= actual_weathering
        
        # Add to target
        world.layer_thickness[target_layer] += actual_weathering
        
        # Rebuild interfaces
        world.update_interfaces_from_thicknesses()
        world.enforce_layer_ordering()
        
        # Statistics
        cell_area = world.pixel_scale_m ** 2
        total_volume = actual_weathering.sum() * cell_area
        
        if verbose:
            print(f"Weathering: {total_volume:.2e} m³ from {source_layer} to {target_layer}")
        
        return {
            'total_weathered': total_volume
        }
    
    def apply_combined_changes(
        self,
        world: WorldState,
        erosion_depth: np.ndarray,
        deposition_depth: np.ndarray,
        weathering_depth: Optional[np.ndarray] = None,
        deposition_layer: str = "Alluvium",
        verbose: bool = False
    ) -> dict:
        """
        Apply erosion, deposition, and optionally weathering in one step.
        
        This is the main interface for updating stratigraphy during a time step.
        
        Parameters
        ----------
        world : WorldState
            The world state to modify (modified in place)
        erosion_depth : np.ndarray
            Erosion depth (m, positive = erosion)
        deposition_depth : np.ndarray
            Deposition depth (m, positive = deposition)
        weathering_depth : np.ndarray, optional
            Weathering depth (m, positive = weathering).
            If None, no weathering applied.
        deposition_layer : str
            Layer to deposit into
        verbose : bool
            If True, print diagnostic information
            
        Returns
        -------
        dict
            Combined statistics
        """
        stats = {}
        
        # 1. Apply erosion
        if erosion_depth is not None and erosion_depth.max() > 0:
            erosion_stats = self.apply_erosion(world, erosion_depth, verbose=verbose)
            stats.update(erosion_stats)
        
        # 2. Apply deposition
        if deposition_depth is not None and deposition_depth.max() > 0:
            deposition_stats = self.apply_deposition(
                world, deposition_depth, target_layer=deposition_layer, verbose=verbose
            )
            stats.update(deposition_stats)
        
        # 3. Apply weathering (if provided)
        if weathering_depth is not None and weathering_depth.max() > 0:
            # For simplicity, weather from the top bedrock layer into saprolite
            # This is a simplified approach; more complex models could track
            # weathering from multiple source layers
            pass  # TODO: implement if needed
        
        return stats


def compute_layer_aware_erosion(
    world: WorldState,
    gross_erosion_depth: np.ndarray,
    updater: Optional[StratigraphyUpdater] = None
) -> dict:
    """
    Convenience function to apply layer-aware erosion.
    
    Parameters
    ----------
    world : WorldState
        World state (modified in place)
    gross_erosion_depth : np.ndarray
        Gross erosion depth (m)
    updater : StratigraphyUpdater, optional
        Stratigraphy updater. If None, creates default.
        
    Returns
    -------
    dict
        Erosion statistics
    """
    if updater is None:
        updater = StratigraphyUpdater()
    
    return updater.apply_erosion(world, gross_erosion_depth)


def compute_layer_aware_deposition(
    world: WorldState,
    deposition_depth: np.ndarray,
    layer_name: str = "Alluvium",
    updater: Optional[StratigraphyUpdater] = None
) -> dict:
    """
    Convenience function to apply layer-aware deposition.
    
    Parameters
    ----------
    world : WorldState
        World state (modified in place)
    deposition_depth : np.ndarray
        Deposition depth (m)
    layer_name : str
        Layer to deposit into
    updater : StratigraphyUpdater, optional
        Stratigraphy updater. If None, creates default.
        
    Returns
    -------
    dict
        Deposition statistics
    """
    if updater is None:
        updater = StratigraphyUpdater()
    
    return updater.apply_deposition(world, deposition_depth, target_layer=layer_name)


class StructuralGeometry:
    """
    Defines large-scale structural patterns (dip, folds, uplift zones).
    
    These can be applied to layer interfaces to create realistic
    geological structures.
    """
    
    def __init__(self, nx: int, ny: int, pixel_scale_m: float):
        """
        Initialize structural geometry.
        
        Parameters
        ----------
        nx, ny : int
            Grid dimensions
        pixel_scale_m : float
            Grid spacing (m)
        """
        self.nx = nx
        self.ny = ny
        self.pixel_scale_m = pixel_scale_m
    
    def apply_regional_dip(
        self,
        interface: np.ndarray,
        dip_direction_deg: float,
        dip_angle_deg: float
    ) -> np.ndarray:
        """
        Apply a regional dip to a layer interface.
        
        Parameters
        ----------
        interface : np.ndarray
            Layer interface elevation (m)
        dip_direction_deg : float
            Direction of dip (degrees, 0=east, 90=north)
        dip_angle_deg : float
            Angle of dip (degrees from horizontal, 0=flat, 90=vertical)
            
        Returns
        -------
        np.ndarray
            Modified interface with dip applied
        """
        # Convert angles to radians
        dip_dir_rad = np.deg2rad(dip_direction_deg)
        dip_angle_rad = np.deg2rad(dip_angle_deg)
        
        # Dip vector
        dx = np.cos(dip_dir_rad) * np.tan(dip_angle_rad)
        dy = np.sin(dip_dir_rad) * np.tan(dip_angle_rad)
        
        # Create coordinate grids
        y, x = np.ogrid[0:self.ny, 0:self.nx]
        x = x * self.pixel_scale_m
        y = y * self.pixel_scale_m
        
        # Apply dip
        dip_surface = x * dx + y * dy
        
        return interface + dip_surface
    
    def create_anticline(
        self,
        center_x: float,
        center_y: float,
        wavelength: float,
        amplitude: float
    ) -> np.ndarray:
        """
        Create an anticline (upward fold) pattern.
        
        Parameters
        ----------
        center_x, center_y : float
            Center of anticline (m)
        wavelength : float
            Wavelength of fold (m)
        amplitude : float
            Amplitude of fold (m)
            
        Returns
        -------
        np.ndarray
            Fold surface (elevation adjustment)
        """
        y, x = np.ogrid[0:self.ny, 0:self.nx]
        x = x * self.pixel_scale_m
        y = y * self.pixel_scale_m
        
        # Distance from center
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Anticline: positive in center, decays outward
        fold = amplitude * np.cos(2 * np.pi * dist / wavelength)
        fold = np.maximum(fold, 0)  # Only positive part (anticline)
        
        return fold.astype(np.float32)
    
    def create_syncline(
        self,
        center_x: float,
        center_y: float,
        wavelength: float,
        amplitude: float
    ) -> np.ndarray:
        """
        Create a syncline (downward fold) pattern.
        
        Parameters
        ----------
        center_x, center_y : float
            Center of syncline (m)
        wavelength : float
            Wavelength of fold (m)
        amplitude : float
            Amplitude of fold (m, negative)
            
        Returns
        -------
        np.ndarray
            Fold surface (elevation adjustment)
        """
        anticline = self.create_anticline(center_x, center_y, wavelength, amplitude)
        return -anticline  # Invert for syncline
    
    def __repr__(self):
        return f"StructuralGeometry(nx={self.nx}, ny={self.ny}, dx={self.pixel_scale_m}m)"
