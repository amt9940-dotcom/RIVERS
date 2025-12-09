"""
Landscape Evolution Simulator

A modular framework for simulating landscape evolution with:
- Layer-aware stratigraphy
- Tectonic forcing
- Climate/weather patterns
- Water routing and hydrology
- Erosion and deposition processes
- Realistic structural geometry

Author: Refactored from existing terrain/stratigraphy code
"""

__version__ = "1.0.0"

# Core components
from .world_state import WorldState, MaterialProperties, DEFAULT_MATERIAL_PROPERTIES
from .forcing import TectonicUplift, WeatherGenerator, ClimateState, create_simple_forcing
from .hydrology import FlowRouter, compute_simple_drainage
from .geomorphic_processes import (
    ChannelErosion,
    HillslopeDiffusion,
    Weathering,
    SedimentTransport,
    GeomorphicEngine
)
from .stratigraphy import (
    StratigraphyUpdater,
    StructuralGeometry,
    compute_layer_aware_erosion,
    compute_layer_aware_deposition
)
from .simulator import (
    LandscapeEvolutionSimulator,
    SimulationHistory,
    create_simple_simulator
)
from .visualization import (
    plot_erosion_analysis,
    plot_erosion_rate_map
)

__all__ = [
    # World state
    'WorldState',
    'MaterialProperties',
    'DEFAULT_MATERIAL_PROPERTIES',
    
    # Forcing
    'TectonicUplift',
    'WeatherGenerator',
    'ClimateState',
    'create_simple_forcing',
    
    # Hydrology
    'FlowRouter',
    'compute_simple_drainage',
    
    # Geomorphic processes
    'ChannelErosion',
    'HillslopeDiffusion',
    'Weathering',
    'SedimentTransport',
    'GeomorphicEngine',
    
    # Stratigraphy
    'StratigraphyUpdater',
    'StructuralGeometry',
    'compute_layer_aware_erosion',
    'compute_layer_aware_deposition',
    
    # Simulator
    'LandscapeEvolutionSimulator',
    'SimulationHistory',
    'create_simple_simulator',
    
    # Visualization
    'plot_erosion_analysis',
    'plot_erosion_rate_map',
]
