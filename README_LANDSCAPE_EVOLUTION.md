# Landscape Evolution Simulator - Refactored Architecture

A modular, layer-aware landscape evolution simulator that integrates terrain generation, stratigraphy, and geomorphic processes.

## Overview

This project has been refactored from your original code to follow a clear conceptual architecture that separates concerns and makes the simulation more maintainable and extensible.

### Key Features

✅ **Clear World State** - Surface elevation, layer interfaces, material properties, mobile sediment  
✅ **External Forcing** - Tectonic uplift/subsidence, climate/weather patterns  
✅ **Water Routing** - D8 flow direction, flow accumulation, drainage networks  
✅ **Geomorphic Processes** - Channel erosion, hillslope diffusion, weathering, sediment transport  
✅ **Layer-Aware Stratigraphy** - Erosion removes from top layers, deposition adds material  
✅ **Time-Stepping Engine** - Integrates all processes over many time steps  
✅ **Preserved Original Code** - Your terrain and stratigraphy generation code is maintained  
✅ **Comprehensive Visualization** - Plot evolution, rivers, erosion/deposition, cross-sections  

## Architecture

### Module Structure

```
landscape_evolution/
├── __init__.py                    # Package initialization
├── world_state.py                 # Core data structures (WorldState, MaterialProperties)
├── forcing.py                     # External forcing (TectonicUplift, WeatherGenerator)
├── hydrology.py                   # Water routing (FlowRouter)
├── geomorphic_processes.py        # Erosion, deposition, weathering
├── stratigraphy.py                # Layer-aware updates, structural geometry
├── simulator.py                   # Time-stepping engine (LandscapeEvolutionSimulator)
├── terrain_generation.py          # Your existing terrain generation code
├── initial_stratigraphy.py        # Bridge to your stratigraphy generation
└── visualization.py               # Plotting functions
```

### Conceptual Flow

```
┌─────────────────────────────────────────────────────────────┐
│                      WORLD STATE                            │
│  • Surface elevation                                        │
│  • Layer interfaces (tops of each geological layer)        │
│  • Material properties (erodibility, density, etc.)        │
│  • Mobile sediment thickness                               │
└─────────────────────────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   EXTERNAL FORCING                          │
│  • Tectonic uplift/subsidence                              │
│  • Climate/weather (rainfall, wind, orographic effects)    │
└─────────────────────────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   WATER ROUTING                             │
│  • Flow directions (D8 steepest descent)                   │
│  • Flow accumulation (discharge proxy)                     │
│  • Channel network identification                          │
└─────────────────────────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                GEOMORPHIC PROCESSES                         │
│  • Channel erosion (stream power law)                      │
│  • Hillslope diffusion (soil creep)                        │
│  • Weathering (bedrock → regolith)                         │
│  • Sediment transport and deposition                       │
└─────────────────────────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              LAYER-AWARE STRATIGRAPHY UPDATE                │
│  • Erosion removes from topmost layers                     │
│  • Deposition adds to surface layers                       │
│  • Layer ordering enforced                                 │
│  • No layer inversions allowed                             │
└─────────────────────────────────────────────────────────────┘
                          ▼
                  TIME STEP COMPLETE
                  (repeat for next step)
```

## Quick Start

### Installation

```bash
# Navigate to project directory
cd /workspace

# The landscape_evolution package is ready to use
```

### Basic Usage

```python
import numpy as np
from landscape_evolution import (
    WorldState,
    LandscapeEvolutionSimulator,
    TectonicUplift,
    WeatherGenerator
)
from landscape_evolution.terrain_generation import quantum_seeded_topography, denormalize_elevation
from landscape_evolution.initial_stratigraphy import create_slope_dependent_stratigraphy
from landscape_evolution.visualization import plot_evolution_summary

# 1. Generate terrain
z_norm, rng = quantum_seeded_topography(N=256, random_seed=42)
surface_elev = denormalize_elevation(z_norm, (0, 1000))

# 2. Create world state
layer_names = ["Topsoil", "Saprolite", "Sandstone", "Basement"]
world = WorldState(256, 256, pixel_scale_m=100.0, layer_names=layer_names)
create_slope_dependent_stratigraphy(world, surface_elev, pixel_scale_m=100.0)

# 3. Set up forcing
tectonics = TectonicUplift(256, 256, 100.0)
tectonics.set_uniform_uplift(1e-3)  # 1 mm/yr

weather = WeatherGenerator(256, 256, 100.0, mean_annual_precip_m=1.0)

# 4. Create and run simulator
simulator = LandscapeEvolutionSimulator(world, tectonics, weather)
history = simulator.run(total_time=10000.0, dt=10.0)

# 5. Visualize
plot_evolution_summary(history, world)
```

See `Example_Landscape_Evolution.ipynb` for a complete walkthrough.

## Integration with Original Code

Your original terrain and stratigraphy generation code is preserved and can be used:

```python
# Generate terrain and stratigraphy with your original code
# (from Project.ipynb)
z_norm, rng = quantum_seeded_topography(N=512, random_seed=42)
strata = generate_stratigraphy(z_norm, rng, elev_range_m=700.0, ...)

# Initialize WorldState from it
from landscape_evolution.initial_stratigraphy import initialize_world_from_stratigraphy

layer_names = list(strata['thickness'].keys())
world = WorldState(512, 512, pixel_scale_m=10.0, layer_names=layer_names)
initialize_world_from_stratigraphy(
    world,
    surface_elev=strata['surface_elev'],
    thickness=strata['thickness']
)

# Now evolve the landscape!
simulator = LandscapeEvolutionSimulator(world, tectonics, weather)
history = simulator.run(total_time=10000.0, dt=10.0)
```

## Key Concepts

### 1. World State (State of the World at Each Time Step)

The `WorldState` class maintains the complete state:

- **Surface elevation**: 2D field of current ground surface (m)
- **Layer interfaces**: For each layer, the elevation of its top surface
- **Layer thicknesses**: Computed from interfaces
- **Material properties**: Erodibility, density, permeability, weathering rate for each layer type
- **Mobile sediment**: Loose material that can be easily transported

```python
world = WorldState(nx, ny, pixel_scale_m, layer_names)
world.set_initial_topography(surface_elev)
world.layer_thickness["Topsoil"] = thickness_field
world.enforce_layer_ordering()  # Ensure geological consistency
```

### 2. External Forcing (What Drives Change)

#### Tectonic Uplift/Subsidence

```python
tectonics = TectonicUplift(nx, ny, pixel_scale_m)
tectonics.set_uniform_uplift(1e-3)  # Uniform uplift
# or
tectonics.set_regional_pattern(center_rate=2e-3, edge_rate=0.0)
```

#### Climate/Weather

```python
weather = WeatherGenerator(
    nx, ny, pixel_scale_m,
    mean_annual_precip_m=1.0,
    wind_direction_deg=270.0,  # Wind from west
    orographic_factor=0.5       # Orographic enhancement strength
)

# Generates rainfall with orographic effects
climate_state = weather.generate_climate_state(surface_elev, time)
```

### 3. Water Routing

```python
router = FlowRouter(pixel_scale_m)
flow_dir, slope, flow_accum = router.compute_flow(surface_elev)

# Identify channels
channels = router.identify_channels(threshold_area_m2=1e5)

# Get stream power for erosion
discharge = router.get_discharge_proxy(rainfall, dt)
stream_power = router.get_stream_power(discharge, K=1e-4, m=0.5, n=1.0)
```

### 4. Geomorphic Processes

All processes are encapsulated in `GeomorphicEngine`:

```python
engine = GeomorphicEngine(pixel_scale_m)
results = engine.compute_all_processes(
    surface_elev=surface,
    drainage_area=area,
    slope=slope,
    erodibility=erodibility_field,
    weathering_rate=weathering_field,
    mobile_sediment=mobile_sed,
    dt=time_step
)
# Returns: channel_erosion, hillslope_change, weathering, deposition, etc.
```

Individual processes:
- **Channel Erosion**: E = K × A^m × S^n (stream power law)
- **Hillslope Diffusion**: ∂z/∂t = κ ∇²z (soil creep)
- **Weathering**: Converts bedrock to regolith (exponential decay with regolith thickness)
- **Sediment Transport**: Simple capacity-based model

### 5. Layer-Aware Stratigraphy Updates

The `StratigraphyUpdater` handles erosion and deposition while maintaining geological consistency:

```python
updater = StratigraphyUpdater()

# Erosion removes material from top layers first
updater.apply_erosion(world, erosion_depth)

# Deposition adds material to specified layer
updater.apply_deposition(world, deposition_depth, target_layer="Alluvium")

# Weathering converts bedrock to regolith
updater.apply_weathering(world, weathering_depth, source_layer="Sandstone", target_layer="Saprolite")
```

Key principles:
- Erosion works downward through layers
- Deposition adds to the top
- Layer ordering is always enforced
- No negative thicknesses or inversions

### 6. Time-Stepping Loop

The `LandscapeEvolutionSimulator` orchestrates everything:

```python
simulator = LandscapeEvolutionSimulator(
    world=world,
    tectonics=tectonics,
    weather=weather,
    snapshot_interval=50  # Save every 50 steps
)

history = simulator.run(total_time=10000.0, dt=10.0)
```

Each time step:
1. Apply tectonic uplift to surface and all layer interfaces
2. Generate rainfall for this time step (with orographic effects)
3. Route water over current surface
4. Get material properties at surface (from top layer)
5. Compute all geomorphic processes
6. Apply changes to stratigraphy in layer-aware manner
7. Enforce layer ordering and constraints
8. Advance time

## Visualization

The `visualization` module provides comprehensive plotting:

```python
from landscape_evolution.visualization import (
    plot_initial_vs_final,
    plot_erosion_deposition_maps,
    plot_river_network,
    plot_layer_exposure,
    plot_cross_section,
    plot_evolution_summary,
    plot_time_series
)

# Initial vs final
plot_initial_vs_final(history, pixel_scale_m)

# Erosion and deposition
plot_erosion_deposition_maps(history, pixel_scale_m)

# River network
plot_river_network(surface_elev, flow_accum, pixel_scale_m)

# Which layers are exposed
plot_layer_exposure(world)

# Cross-section
plot_cross_section(world, row=N//2)

# Comprehensive summary
plot_evolution_summary(history, world, flow_accum)

# Time series
plot_time_series(history)
```

## Customization

### Custom Material Properties

```python
from landscape_evolution import MaterialProperties

custom_props = {
    "MyRock": MaterialProperties(
        name="MyRock",
        erodibility=1e-4,
        density=2500,
        permeability=1e-8,
        weathering_rate=5e-5,
        color="#FF5733"
    )
}

world = WorldState(nx, ny, pixel_scale_m, layer_names, material_properties=custom_props)
```

### Custom Process Parameters

```python
from landscape_evolution import ChannelErosion, HillslopeDiffusion

channel_erosion = ChannelErosion(m=0.5, n=1.0, K_base=1e-5)
hillslope_diffusion = HillslopeDiffusion(kappa=0.01)

engine = GeomorphicEngine(
    pixel_scale_m,
    channel_erosion=channel_erosion,
    hillslope_diffusion=hillslope_diffusion
)
```

### Structural Geometry (Dip, Folds)

```python
from landscape_evolution import StructuralGeometry

structure = StructuralGeometry(nx, ny, pixel_scale_m)

# Apply regional dip to a layer
dipped_interface = structure.apply_regional_dip(
    interface, dip_direction_deg=90, dip_angle_deg=10
)

# Create anticline
anticline = structure.create_anticline(
    center_x=5000, center_y=5000, wavelength=2000, amplitude=100
)
```

## File Reference

### Core Modules

- **`world_state.py`**: WorldState, MaterialProperties, DEFAULT_MATERIAL_PROPERTIES
- **`forcing.py`**: TectonicUplift, WeatherGenerator, ClimateState
- **`hydrology.py`**: FlowRouter, compute_simple_drainage
- **`geomorphic_processes.py`**: ChannelErosion, HillslopeDiffusion, Weathering, SedimentTransport, GeomorphicEngine
- **`stratigraphy.py`**: StratigraphyUpdater, StructuralGeometry
- **`simulator.py`**: LandscapeEvolutionSimulator, SimulationHistory

### Integration Modules

- **`terrain_generation.py`**: Your existing terrain generation (quantum_seeded_topography, fractional_surface, domain_warp, etc.)
- **`initial_stratigraphy.py`**: Bridge to your stratigraphy generation (initialize_world_from_stratigraphy)
- **`visualization.py`**: All plotting functions

### Examples

- **`Example_Landscape_Evolution.ipynb`**: Complete walkthrough with multiple examples
- **`Project.ipynb`**: Your original notebook (preserved)

## Original Code Preservation

Your original code is fully preserved in:

1. **Terrain Generation**: Extracted to `terrain_generation.py`
   - `quantum_seeded_topography()`
   - `fractional_surface()`
   - `domain_warp()`
   - `ridged_mix()`
   - All helper functions

2. **Stratigraphy Generation**: Your full `generate_stratigraphy()` function remains in `Project.ipynb`
   - All your sophisticated layer generation rules
   - Energy-based facies assignment
   - Topography-dependent thicknesses
   - Interface smoothing

3. **Integration**: `initial_stratigraphy.py` provides bridge functions to use your original code with the new framework

## What Changed vs Original

### Added
- Clear separation of concerns (state, forcing, processes, stratigraphy)
- Time-stepping evolution engine
- Layer-aware erosion and deposition
- Comprehensive visualization suite
- Modular, extensible architecture

### Preserved
- Your terrain generation code
- Your stratigraphy generation rules
- All plotting functionality (enhanced)
- Geological realism and constraints

### Improved
- Code organization and readability
- Ability to evolve landscapes over time
- Integration of multiple processes
- Visualization of changes

## Next Steps

1. **Run the example notebook**: `Example_Landscape_Evolution.ipynb`
2. **Integrate your full stratigraphy**: Use `generate_stratigraphy()` from `Project.ipynb`
3. **Adjust parameters**: Tune erosion rates, uplift, climate for your application
4. **Add complexity**: Implement additional processes or forcing patterns
5. **Scale up**: Run longer simulations or larger domains

## Dependencies

- `numpy`: Array operations
- `scipy`: Filtering and convolution
- `matplotlib`: Visualization
- `numba`: Fast flow routing (optional but recommended)

## Questions?

The code is extensively commented. Key concepts are explained in docstrings. The example notebook provides step-by-step walkthroughs.

Your original `Project.ipynb` remains untouched and can be used alongside this new framework.
