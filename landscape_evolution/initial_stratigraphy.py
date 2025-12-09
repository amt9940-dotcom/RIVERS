"""
Initial Stratigraphy Generation Module

This module provides a bridge between your existing stratigraphy generation
code (in Project.ipynb) and the new landscape evolution framework.

Your original generate_stratigraphy() function is preserved in Project.ipynb
and contains sophisticated logic for:
- Regolith layers (Topsoil, Colluvium, Saprolite, WeatheredBR)
- Sedimentary rocks (Sandstone, Shale, Limestone, Conglomerate, etc.)
- Crystalline basement (Granite, Gneiss, Basalt)
- Energy-based facies rules
- Topography-dependent layer thicknesses

This module provides a simplified interface for initializing WorldState
from generated stratigraphy.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from .world_state import WorldState


def initialize_world_from_stratigraphy(
    world: WorldState,
    surface_elev: np.ndarray,
    thickness: Dict[str, np.ndarray],
    interfaces: Optional[Dict[str, np.ndarray]] = None
):
    """
    Initialize a WorldState from existing stratigraphy data.
    
    This function takes the output of your generate_stratigraphy()
    function and populates a WorldState object.
    
    Parameters
    ----------
    world : WorldState
        WorldState to initialize
    surface_elev : np.ndarray
        Surface elevation (m), shape (ny, nx)
    thickness : Dict[str, np.ndarray]
        Layer thicknesses (m) for each layer name
    interfaces : Dict[str, np.ndarray], optional
        Layer interfaces (m). If None, computed from thicknesses.
        
    Example
    -------
    >>> # After running your generate_stratigraphy():
    >>> # strata = generate_stratigraphy(z_norm, rng, ...)
    >>> world = WorldState(nx, ny, pixel_scale_m, layer_names)
    >>> initialize_world_from_stratigraphy(
    ...     world,
    ...     surface_elev=strata['surface_elev'],
    ...     thickness=strata['thickness']
    ... )
    """
    # Set surface elevation
    world.set_initial_topography(surface_elev)
    
    # Set layer thicknesses
    for layer_name in world.layer_names:
        if layer_name in thickness:
            world.layer_thickness[layer_name] = thickness[layer_name].copy()
        else:
            world.layer_thickness[layer_name] = np.zeros((world.ny, world.nx))
    
    # If interfaces provided, use them; otherwise compute from thicknesses
    if interfaces is not None:
        for layer_name in world.layer_names:
            if layer_name in interfaces:
                world.layer_interfaces[layer_name] = interfaces[layer_name].copy()
    else:
        # Compute interfaces from thicknesses
        world.update_interfaces_from_thicknesses()
    
    # Enforce consistency
    world.enforce_layer_ordering()


def create_simple_initial_stratigraphy(
    world: WorldState,
    surface_elev: np.ndarray,
    regolith_thickness_m: float = 2.0,
    saprolite_thickness_m: float = 5.0,
    bedrock_thickness_m: float = 100.0
):
    """
    Create a simple layered stratigraphy for testing/demo purposes.
    
    This creates a simple three-layer stratigraphy:
    1. Topsoil/regolith (thin)
    2. Saprolite (moderate)
    3. Bedrock (thick)
    
    Parameters
    ----------
    world : WorldState
        WorldState to initialize
    surface_elev : np.ndarray
        Surface elevation (m)
    regolith_thickness_m : float
        Thickness of topsoil layer (m)
    saprolite_thickness_m : float
        Thickness of saprolite layer (m)
    bedrock_thickness_m : float
        Thickness of bedrock layer (m)
    """
    world.set_initial_topography(surface_elev)
    
    # Create uniform thickness fields
    uniform_regolith = np.full((world.ny, world.nx), regolith_thickness_m, dtype=np.float32)
    uniform_saprolite = np.full((world.ny, world.nx), saprolite_thickness_m, dtype=np.float32)
    uniform_bedrock = np.full((world.ny, world.nx), bedrock_thickness_m, dtype=np.float32)
    
    # Assign to layers
    # Assume layer_names contains at least these layers
    if "Topsoil" in world.layer_names:
        world.layer_thickness["Topsoil"] = uniform_regolith.copy()
    
    if "Saprolite" in world.layer_names:
        world.layer_thickness["Saprolite"] = uniform_saprolite.copy()
    
    # Put remaining thickness in bedrock/basement layer
    bedrock_layer = None
    for name in ["Sandstone", "Basement", "Bedrock"]:
        if name in world.layer_names:
            bedrock_layer = name
            break
    
    if bedrock_layer:
        world.layer_thickness[bedrock_layer] = uniform_bedrock.copy()
    
    # Compute interfaces
    world.update_interfaces_from_thicknesses()
    world.enforce_layer_ordering()


def create_slope_dependent_stratigraphy(
    world: WorldState,
    surface_elev: np.ndarray,
    pixel_scale_m: float,
    base_regolith_m: float = 2.0,
    base_saprolite_m: float = 5.0,
    bedrock_thickness_m: float = 100.0
):
    """
    Create simple slope-dependent stratigraphy.
    
    Regolith and saprolite are thicker on gentle slopes, thinner on steep slopes.
    This is a simplified version of your sophisticated rules.
    
    Parameters
    ----------
    world : WorldState
        WorldState to initialize
    surface_elev : np.ndarray
        Surface elevation (m)
    pixel_scale_m : float
        Grid spacing (m)
    base_regolith_m : float
        Base regolith thickness (m)
    base_saprolite_m : float
        Base saprolite thickness (m)
    bedrock_thickness_m : float
        Bedrock thickness (m)
    """
    world.set_initial_topography(surface_elev)
    
    # Compute slope
    dy, dx = np.gradient(surface_elev, pixel_scale_m)
    slope = np.sqrt(dx**2 + dy**2)
    
    # Normalize slope (0 = flat, 1 = steep)
    slope_norm = np.clip(slope / (slope.max() + 1e-6), 0, 1)
    
    # Thickness decreases with slope
    slope_factor = np.exp(-3.0 * slope_norm)  # Exponential decay
    
    # Regolith thickness
    if "Topsoil" in world.layer_names:
        world.layer_thickness["Topsoil"] = base_regolith_m * slope_factor
    
    # Saprolite thickness
    if "Saprolite" in world.layer_names:
        world.layer_thickness["Saprolite"] = base_saprolite_m * slope_factor
    
    # Bedrock (uniform)
    bedrock_layer = None
    for name in ["Sandstone", "Basement", "Bedrock"]:
        if name in world.layer_names:
            bedrock_layer = name
            break
    
    if bedrock_layer:
        world.layer_thickness[bedrock_layer] = np.full(
            (world.ny, world.nx), bedrock_thickness_m, dtype=np.float32
        )
    
    # Compute interfaces
    world.update_interfaces_from_thicknesses()
    world.enforce_layer_ordering()


def note_on_original_code():
    """
    Note on original stratigraphy generation code.
    
    Your original generate_stratigraphy() function in Project.ipynb
    contains extensive logic for realistic stratigraphy including:
    
    1. Regolith Layers:
       - Topsoil with slope-dependent thickness
       - Colluvium with curvature and TWI controls
       - Saprolite with gentle-slope enhancement
       - Weathered bedrock rind
    
    2. Sedimentary Rocks:
       - Sandstone (fan toes and valley fills)
       - Conglomerate (alluvial fans)
       - Shale (valley fills)
       - Limestone (platforms)
       - Evaporite (closed basins)
    
    3. Crystalline Basement:
       - Granite, Gneiss, Basalt
       - Ancient crust
    
    4. Energy-Based Facies Rules:
       - High energy environments -> coarse sediments
       - Low energy environments -> fine sediments
       - Topographic control on facies distribution
    
    5. Interface Smoothing:
       - Different smoothing scales for different layers
       - Preserves geological realism
    
    To use your original code with the new framework:
    
    1. Run generate_stratigraphy() as before to get 'strata' dict
    2. Create a WorldState with the same layer names
    3. Use initialize_world_from_stratigraphy() to populate it
    4. Now you can run the landscape evolution simulator
    
    Example:
    --------
    # Generate terrain and stratigraphy with your original code
    z_norm, rng = quantum_seeded_topography(N=512)
    strata = generate_stratigraphy(z_norm, rng, elev_range_m=700.0, ...)
    
    # Create WorldState
    layer_names = list(strata['thickness'].keys())  # Or specify manually
    world = WorldState(512, 512, pixel_scale_m=10.0, layer_names=layer_names)
    
    # Initialize from generated stratigraphy
    initialize_world_from_stratigraphy(
        world,
        surface_elev=strata['surface_elev'],
        thickness=strata['thickness']
    )
    
    # Now you can evolve the landscape
    simulator = LandscapeEvolutionSimulator(world, tectonics, weather)
    history = simulator.run(total_time=10000.0, dt=10.0)
    """
    pass
