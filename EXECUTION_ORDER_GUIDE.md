# Execution Order Guide for Erosion Simulator

## Option 1: Use the Complete Notebook (RECOMMENDED)

**File**: `Complete_Erosion_Simulator.ipynb`

**Just open and run cells in order from top to bottom!**

This single notebook contains everything you need. Each cell is numbered and explained.

---

## Option 2: Manual Assembly Order

If you want to build it yourself or understand the structure, here's the order:

### Step-by-Step Execution Order

#### 1. **Imports** (Must be first)
```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
```

#### 2. **Import Package** (After standard imports)
```python
from landscape_evolution import (
    WorldState,
    TectonicUplift,
    WeatherGenerator,
    FlowRouter,
    LandscapeEvolutionSimulator,
    plot_erosion_analysis,
    plot_erosion_rate_map
)

from landscape_evolution.terrain_generation import (
    quantum_seeded_topography,
    denormalize_elevation
)

from landscape_evolution.initial_stratigraphy import (
    create_slope_dependent_stratigraphy
)
```

#### 3. **Generate Terrain** (Before world state)
```python
N = 256
pixel_scale_m = 100.0
z_norm, rng = quantum_seeded_topography(N=N, random_seed=42)
surface_elev = denormalize_elevation(z_norm, (0, 1000))
```

#### 4. **Create World State** (After terrain)
```python
layer_names = ["Topsoil", "Saprolite", "Sandstone", "Basement"]
world = WorldState(N, N, pixel_scale_m, layer_names)
create_slope_dependent_stratigraphy(world, surface_elev, pixel_scale_m)
```

#### 5. **Set Up Forcing** (After world state)
```python
tectonics = TectonicUplift(N, N, pixel_scale_m)
tectonics.set_uniform_uplift(1e-3)

weather = WeatherGenerator(N, N, pixel_scale_m, mean_annual_precip_m=1.0)
```

#### 6. **Create Simulator** (After forcing)
```python
simulator = LandscapeEvolutionSimulator(
    world=world,
    tectonics=tectonics,
    weather=weather,
    snapshot_interval=50,
    verbose=True
)
```

#### 7. **Run Simulation** (After simulator created)
```python
history = simulator.run(total_time=5000.0, dt=10.0)
```

#### 8. **Compute Flow** (After simulation)
```python
flow_router = FlowRouter(pixel_scale_m)
flow_dir, slope, flow_accum = flow_router.compute_flow(world.surface_elev)
```

#### 9. **Plot Erosion Analysis** (After simulation)
```python
erosion = history.get_total_erosion()
plot_erosion_analysis(erosion, world.surface_elev, pixel_scale_m)
```

#### 10. **Plot Erosion Rate** (After flow computed)
```python
erosion_rate = erosion / 5000.0
plot_erosion_rate_map(erosion_rate, pixel_scale_m, flow_accum)
```

---

## Critical Dependencies

### What Must Come Before What?

```
Imports
  ↓
Terrain Generation
  ↓
World State Creation
  ↓
Forcing Setup (Tectonics + Weather)
  ↓
Simulator Creation
  ↓
Run Simulation
  ↓
Flow Computation (for rivers) ←─ Can be parallel with plots
  ↓
Visualizations (any order)
```

### Key Rules:

1. ✅ **Terrain before World State** - Need surface_elev to initialize
2. ✅ **World State before Forcing** - Need grid dimensions
3. ✅ **Forcing before Simulator** - Simulator needs both tectonics and weather
4. ✅ **Simulation before Plots** - Need history data to plot
5. ✅ **Flow before River Overlays** - Need flow_accum for river plots

### Optional Dependencies:

- **Flow computation** can happen anytime after simulation
- **Different plots** can be in any order after simulation
- **Save/export** can happen anytime after data is generated

---

## Common Errors and Order Issues

### ❌ ERROR: "NameError: name 'world' is not defined"
**Fix**: Run terrain generation and world state creation cells first

### ❌ ERROR: "NameError: name 'history' is not defined"
**Fix**: Run the simulation cell first

### ❌ ERROR: "NameError: name 'flow_accum' is not defined"
**Fix**: Run the flow routing cell before plotting rivers

### ❌ ERROR: "No module named 'landscape_evolution'"
**Fix**: Make sure you're in /workspace directory and package is installed
```bash
cd /workspace
pip install -r requirements.txt
```

### ❌ ERROR: "Need at least 2 snapshots"
**Fix**: Simulation didn't run or didn't save snapshots - check `simulator.run()` completed

---

## File Structure Reference

If you want to understand the package structure:

```
landscape_evolution/
├── __init__.py                 # Package exports
├── world_state.py             # WorldState class
├── forcing.py                 # TectonicUplift, WeatherGenerator
├── hydrology.py               # FlowRouter
├── geomorphic_processes.py    # Erosion/diffusion/weathering
├── stratigraphy.py            # Layer-aware updates
├── simulator.py               # LandscapeEvolutionSimulator
├── terrain_generation.py      # Terrain functions
├── initial_stratigraphy.py    # Initialization helpers
└── visualization.py           # All plotting functions
```

**You don't need to understand the internals - just import and use!**

---

## Quick Reference Card

### Minimal Working Example (Correct Order)

```python
# 1. Imports
import numpy as np
import matplotlib.pyplot as plt
from landscape_evolution import *
from landscape_evolution.terrain_generation import *
from landscape_evolution.initial_stratigraphy import *

# 2. Parameters
N = 256
pixel_scale_m = 100.0

# 3. Terrain
z_norm, rng = quantum_seeded_topography(N=N, random_seed=42)
surface = denormalize_elevation(z_norm, (0, 1000))

# 4. World
world = WorldState(N, N, pixel_scale_m, ["Topsoil", "Sandstone", "Basement"])
create_slope_dependent_stratigraphy(world, surface, pixel_scale_m)

# 5. Forcing
tectonics = TectonicUplift(N, N, pixel_scale_m)
tectonics.set_uniform_uplift(1e-3)
weather = WeatherGenerator(N, N, pixel_scale_m, mean_annual_precip_m=1.0)

# 6. Simulator
simulator = LandscapeEvolutionSimulator(world, tectonics, weather)

# 7. Run
history = simulator.run(total_time=5000.0, dt=10.0)

# 8. Flow
router = FlowRouter(pixel_scale_m)
_, _, flow_accum = router.compute_flow(world.surface_elev)

# 9. Plot
plot_erosion_analysis(history.get_total_erosion(), world.surface_elev, pixel_scale_m)
plot_erosion_rate_map(history.get_total_erosion()/5000, pixel_scale_m, flow_accum)
```

---

## Summary

**EASIEST**: Use `Complete_Erosion_Simulator.ipynb` - everything is in order!

**CUSTOM**: Follow the numbered steps above

**UNDERSTAND**: Read the package structure and dependencies

**TROUBLESHOOT**: Check the error fixes section
