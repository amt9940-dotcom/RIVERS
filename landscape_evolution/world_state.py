"""
World State Management Module

This module defines the core data structures that represent the "state of the world"
at any given time step in the landscape evolution simulation.

The world state includes:
- Surface elevation field
- Subsurface layer interfaces (tops of each geological layer)
- Material properties for each layer type
- Mobile sediment/regolith thickness
- Grid metadata (spacing, dimensions)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class MaterialProperties:
    """
    Material properties for a single layer type.
    
    These properties control how the layer behaves under erosion,
    weathering, and other geomorphic processes.
    """
    name: str
    erodibility: float  # Higher = easier to erode (m^2/kg or similar units)
    density: float  # kg/m^3
    permeability: float  # m/s or m^2 (affects groundwater flow)
    weathering_rate: float  # m/yr (conversion to regolith)
    color: str = "#808080"  # For visualization
    
    def __repr__(self):
        return (f"MaterialProperties({self.name}, "
                f"K={self.erodibility:.2e}, "
                f"ρ={self.density:.0f} kg/m³)")


# Default material property database
DEFAULT_MATERIAL_PROPERTIES = {
    # Unconsolidated surface materials (highly erodible)
    "Topsoil": MaterialProperties(
        name="Topsoil",
        erodibility=1e-3,
        density=1300,
        permeability=1e-5,
        weathering_rate=0.0,  # Already weathered
        color="#8B4513"
    ),
    "Colluvium": MaterialProperties(
        name="Colluvium",
        erodibility=8e-4,
        density=1600,
        permeability=1e-5,
        weathering_rate=0.0,
        color="#A0522D"
    ),
    "Alluvium": MaterialProperties(
        name="Alluvium",
        erodibility=7e-4,
        density=1700,
        permeability=1e-5,
        weathering_rate=0.0,
        color="#D2691E"
    ),
    
    # Weathered bedrock (moderately erodible)
    "Saprolite": MaterialProperties(
        name="Saprolite",
        erodibility=5e-4,
        density=1800,
        permeability=1e-6,
        weathering_rate=0.0,
        color="#CD853F"
    ),
    "WeatheredBR": MaterialProperties(
        name="WeatheredBR",
        erodibility=3e-4,
        density=2000,
        permeability=1e-7,
        weathering_rate=0.0,
        color="#DEB887"
    ),
    
    # Sedimentary rocks (variable erodibility)
    "Sandstone": MaterialProperties(
        name="Sandstone",
        erodibility=1e-4,
        density=2200,
        permeability=1e-7,
        weathering_rate=5e-5,  # 0.05 mm/yr
        color="#F4A460"
    ),
    "Shale": MaterialProperties(
        name="Shale",
        erodibility=2e-4,
        density=2400,
        permeability=1e-9,
        weathering_rate=8e-5,
        color="#696969"
    ),
    "Limestone": MaterialProperties(
        name="Limestone",
        erodibility=1.5e-4,
        density=2600,
        permeability=1e-8,
        weathering_rate=1e-4,  # Dissolves faster
        color="#D3D3D3"
    ),
    "Conglomerate": MaterialProperties(
        name="Conglomerate",
        erodibility=8e-5,
        density=2300,
        permeability=1e-6,
        weathering_rate=4e-5,
        color="#A0826D"
    ),
    
    # Crystalline basement (very resistant)
    "Basement": MaterialProperties(
        name="Basement",
        erodibility=2e-5,
        density=2700,
        permeability=1e-10,
        weathering_rate=1e-5,  # Very slow
        color="#4B0082"
    ),
    "Granite": MaterialProperties(
        name="Granite",
        erodibility=2.5e-5,
        density=2650,
        permeability=1e-10,
        weathering_rate=1.5e-5,
        color="#FFB6C1"
    ),
    "Gneiss": MaterialProperties(
        name="Gneiss",
        erodibility=1.8e-5,
        density=2750,
        permeability=1e-11,
        weathering_rate=8e-6,
        color="#8B7D7B"
    ),
}


class WorldState:
    """
    The complete state of the world at a given time step.
    
    This class maintains:
    - Current surface elevation
    - Layer interface elevations (tops of each layer)
    - Layer thicknesses
    - Mobile sediment thickness
    - Material properties
    - Grid metadata
    
    All spatial fields are 2D numpy arrays with shape (ny, nx).
    """
    
    def __init__(
        self,
        nx: int,
        ny: int,
        pixel_scale_m: float,
        layer_names: List[str],
        material_properties: Optional[Dict[str, MaterialProperties]] = None
    ):
        """
        Initialize world state.
        
        Parameters
        ----------
        nx, ny : int
            Grid dimensions
        pixel_scale_m : float
            Grid spacing in meters
        layer_names : List[str]
            Names of geological layers, from top (youngest) to bottom (oldest).
            Example: ["Topsoil", "Colluvium", "Saprolite", "WeatheredBR", 
                      "Sandstone", "Shale", "Basement"]
        material_properties : Dict[str, MaterialProperties], optional
            Material properties for each layer. If None, uses defaults.
        """
        self.nx = nx
        self.ny = ny
        self.pixel_scale_m = pixel_scale_m
        self.layer_names = layer_names
        
        # Use default properties if not provided
        if material_properties is None:
            self.material_properties = DEFAULT_MATERIAL_PROPERTIES
        else:
            self.material_properties = material_properties
        
        # Initialize spatial fields
        self.surface_elev = np.zeros((ny, nx), dtype=np.float32)
        
        # Layer interfaces: dict mapping layer name -> 2D array of top elevation
        self.layer_interfaces = {}
        for name in layer_names:
            self.layer_interfaces[name] = np.zeros((ny, nx), dtype=np.float32)
        
        # Layer thicknesses: dict mapping layer name -> 2D array of thickness
        self.layer_thickness = {}
        for name in layer_names:
            self.layer_thickness[name] = np.zeros((ny, nx), dtype=np.float32)
        
        # Mobile sediment (loose material that can move easily)
        self.mobile_sediment_thickness = np.zeros((ny, nx), dtype=np.float32)
        
        # Time tracking
        self.time = 0.0  # Current simulation time (years)
        
    def set_initial_topography(self, surface_elev: np.ndarray):
        """
        Set the initial surface elevation.
        
        Parameters
        ----------
        surface_elev : np.ndarray
            2D array of surface elevations (m)
        """
        assert surface_elev.shape == (self.ny, self.nx)
        self.surface_elev = surface_elev.copy()
        
    def set_layer_from_thickness(
        self,
        layer_name: str,
        thickness: np.ndarray,
        below_layer: Optional[str] = None
    ):
        """
        Set a layer's interface based on its thickness.
        
        Parameters
        ----------
        layer_name : str
            Name of the layer to set
        thickness : np.ndarray
            2D array of layer thickness (m)
        below_layer : str, optional
            Name of the layer immediately above this one.
            If None, places this layer below the surface.
        """
        assert thickness.shape == (self.ny, self.nx)
        assert layer_name in self.layer_names
        
        self.layer_thickness[layer_name] = thickness.copy()
        
        if below_layer is None:
            # Place directly below surface
            self.layer_interfaces[layer_name] = self.surface_elev - thickness
        else:
            # Place below the specified layer
            assert below_layer in self.layer_names
            self.layer_interfaces[layer_name] = (
                self.layer_interfaces[below_layer] - thickness
            )
    
    def set_layer_interface(self, layer_name: str, interface_elev: np.ndarray):
        """
        Directly set a layer's top interface elevation.
        
        Parameters
        ----------
        layer_name : str
            Name of the layer
        interface_elev : np.ndarray
            2D array of top interface elevation (m)
        """
        assert interface_elev.shape == (self.ny, self.nx)
        assert layer_name in self.layer_names
        self.layer_interfaces[layer_name] = interface_elev.copy()
        
    def update_thicknesses_from_interfaces(self):
        """
        Recompute layer thicknesses from interface elevations.
        
        Assumes layers are ordered from top (youngest) to bottom (oldest).
        """
        for i, name in enumerate(self.layer_names):
            if i == 0:
                # Topmost layer: thickness = surface - interface
                self.layer_thickness[name] = np.maximum(
                    0, self.surface_elev - self.layer_interfaces[name]
                )
            else:
                # Other layers: thickness = interface_above - interface_this
                name_above = self.layer_names[i - 1]
                self.layer_thickness[name] = np.maximum(
                    0, 
                    self.layer_interfaces[name_above] - self.layer_interfaces[name]
                )
    
    def update_interfaces_from_thicknesses(self):
        """
        Recompute layer interfaces from thicknesses.
        
        Assumes layers are ordered from top (youngest) to bottom (oldest).
        Builds interfaces downward from the surface.
        """
        cumulative_depth = np.zeros((self.ny, self.nx), dtype=np.float32)
        
        for name in self.layer_names:
            cumulative_depth += self.layer_thickness[name]
            self.layer_interfaces[name] = self.surface_elev - cumulative_depth
    
    def enforce_layer_ordering(self):
        """
        Ensure no layer inversions occur.
        
        Enforces that:
        - All layer tops are at or below the surface
        - Each layer top is at or below the layer above it
        - No negative thicknesses
        """
        for i, name in enumerate(self.layer_names):
            if i == 0:
                # First layer must be at or below surface
                self.layer_interfaces[name] = np.minimum(
                    self.layer_interfaces[name], self.surface_elev
                )
            else:
                # Each layer must be at or below the one above
                name_above = self.layer_names[i - 1]
                self.layer_interfaces[name] = np.minimum(
                    self.layer_interfaces[name],
                    self.layer_interfaces[name_above]
                )
        
        # Recompute thicknesses to ensure consistency
        self.update_thicknesses_from_interfaces()
        
        # Ensure no negative thicknesses
        for name in self.layer_names:
            self.layer_thickness[name] = np.maximum(0, self.layer_thickness[name])
    
    def get_top_layer_at(self, i: int, j: int, min_thickness: float = 0.01) -> str:
        """
        Get the name of the topmost layer at a given location.
        
        Parameters
        ----------
        i, j : int
            Grid indices
        min_thickness : float
            Minimum thickness to consider layer present (m)
            
        Returns
        -------
        str
            Name of the topmost layer, or "None" if no layers present
        """
        for name in self.layer_names:
            if self.layer_thickness[name][i, j] >= min_thickness:
                return name
        return "None"
    
    def get_top_layer_map(self, min_thickness: float = 0.01) -> np.ndarray:
        """
        Get a 2D map of the topmost layer at each location.
        
        Parameters
        ----------
        min_thickness : float
            Minimum thickness to consider layer present (m)
            
        Returns
        -------
        np.ndarray
            2D array of layer indices (0 = topmost layer in layer_names, etc.)
            Value is -1 where no layer is present.
        """
        top_layer = np.full((self.ny, self.nx), -1, dtype=np.int32)
        
        for idx, name in enumerate(self.layer_names):
            mask = self.layer_thickness[name] >= min_thickness
            top_layer[mask & (top_layer == -1)] = idx
            
        return top_layer
    
    def get_property_field(self, property_name: str) -> np.ndarray:
        """
        Get a 2D field of a material property based on the top layer.
        
        Parameters
        ----------
        property_name : str
            Name of the property (e.g., 'erodibility', 'density')
            
        Returns
        -------
        np.ndarray
            2D array of the property values
        """
        prop_field = np.zeros((self.ny, self.nx), dtype=np.float32)
        
        for name in self.layer_names:
            if name not in self.material_properties:
                continue
                
            # Where this layer is the topmost
            is_top = (self.layer_thickness[name] > 0.01)
            
            # Get the property value
            mat_prop = self.material_properties[name]
            value = getattr(mat_prop, property_name)
            
            # Only update where no other layer has been assigned
            mask = is_top & (prop_field == 0)
            prop_field[mask] = value
        
        return prop_field
    
    def summary(self) -> str:
        """
        Return a summary string of the world state.
        """
        lines = [
            f"WorldState Summary (t={self.time:.2f} years)",
            f"  Grid: {self.nx} × {self.ny}, spacing={self.pixel_scale_m:.1f} m",
            f"  Surface elevation: [{self.surface_elev.min():.1f}, {self.surface_elev.max():.1f}] m",
            f"  Layers ({len(self.layer_names)}):"
        ]
        
        for name in self.layer_names:
            thick = self.layer_thickness[name]
            lines.append(
                f"    {name:15s}: mean={thick.mean():.2f} m, "
                f"max={thick.max():.2f} m"
            )
        
        mobile_mean = self.mobile_sediment_thickness.mean()
        mobile_max = self.mobile_sediment_thickness.max()
        lines.append(
            f"  Mobile sediment: mean={mobile_mean:.2f} m, max={mobile_max:.2f} m"
        )
        
        return "\n".join(lines)
    
    def __repr__(self):
        return (f"WorldState(nx={self.nx}, ny={self.ny}, "
                f"nlayers={len(self.layer_names)}, t={self.time:.2f} yr)")
