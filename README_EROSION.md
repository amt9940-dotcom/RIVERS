# Erosion Simulation Model

A comprehensive, realistic erosion simulation system that models the evolution of landscapes over time through weathering, water erosion, and sediment transport.

## Overview

This erosion model simulates realistic landscape evolution with:

- **Multi-layer Geology**: Geological strata with different erodibility coefficients
- **Rainfall-driven Erosion**: Spatially-varying precipitation patterns with storms
- **Water Flow**: D8 flow routing and shallow water flow simulation
- **Sediment Transport**: Stream power law erosion with transport capacity
- **River Formation**: Automatic detection of drainage networks
- **Lake Formation**: Identification of closed depressions with standing water
- **Time-stepping**: Configurable simulation duration and time steps
- **Visualization**: Comprehensive plotting of topography, erosion, water features

## Features

### Geological Layers

The simulation supports multiple geological layers with realistic erodibility:

**Surface Materials** (most erodible):
- Topsoil (K = 0.0050)
- Clay, Silt, Sand
- Colluvium, Saprolite

**Sedimentary Rocks** (moderate):
- Sandstone, Shale, Limestone
- Conglomerate, Mudstone, Siltstone

**Crystalline Rocks** (resistant):
- Granite, Gneiss, Basalt
- Schist, Ancient Crust

**Basement** (very resistant):
- Basement rock (K = 0.0003)

### Erosion Processes

1. **Stream Power Erosion**: E = K × A^m × S^n
   - K: material erodibility
   - A: drainage area
   - S: slope

2. **Water Flow**: Manning's equation for overland flow

3. **Sediment Transport**: Capacity-limited transport with deposition

4. **Orographic Effects**: Elevation-dependent rainfall patterns

### Weather System

- Base rainfall patterns (orographic effects)
- Stochastic storm events
- Windward/leeward effects
- Spatial variability

## Files

- `erosion_simulation.py` - Core simulation engine
- `example_erosion_simulation.py` - Standalone examples with simple terrain
- `integrated_erosion_example.py` - Integration with "Rivers new" terrain generation
- `README_EROSION.md` - This file

## Installation

### Requirements

```bash
pip install numpy matplotlib scipy
```

### Optional (for quantum terrain generation):
```bash
pip install qiskit qiskit-aer
```

## Usage

### Quick Start

```python
from erosion_simulation import ErosionSimulation, run_erosion_simulation
import numpy as np

# Create simple terrain
N = 128
surface_elevation = np.random.randn(N, N).cumsum(axis=0).cumsum(axis=1) * 10

# Define layers
layer_interfaces = {
    "Topsoil": surface_elevation - 2,
    "Sandstone": surface_elevation - 20,
    "Granite": surface_elevation - 100,
    "Basement": surface_elevation - 500
}
layer_order = ["Topsoil", "Sandstone", "Granite", "Basement"]

# Run simulation
sim = run_erosion_simulation(
    surface_elevation=surface_elevation,
    layer_interfaces=layer_interfaces,
    layer_order=layer_order,
    n_years=1000.0,
    dt=5.0,
    pixel_scale_m=100.0
)
```

### Running Examples

**Example 1: Basic erosion simulation**
```bash
python example_erosion_simulation.py
```

**Example 2: Integrated simulation with Rivers new**
```bash
python integrated_erosion_example.py
```

### Step-by-Step Simulation

```python
from erosion_simulation import ErosionSimulation
import numpy as np

# Initialize
sim = ErosionSimulation(
    surface_elevation=my_terrain,
    layer_interfaces=my_layers,
    layer_order=layer_names,
    pixel_scale_m=100.0,
    uplift_rate=0.0001  # 0.1 mm/year
)

# Run time steps
for t in range(100):
    # Generate rainfall (mm)
    rainfall_map = 1000.0 * np.ones(sim.elevation.shape)
    
    # Perform one year of erosion
    sim.step(dt=1.0, rainfall_map=rainfall_map)
    
    # Check results
    print(f"Year {sim.current_time}: Erosion = {sim.get_total_erosion():.2e} m³")
```

### Visualization

```python
from erosion_simulation import plot_simulation_summary
import matplotlib.pyplot as plt

# Create comprehensive plot
fig = plot_simulation_summary(sim)
plt.show()

# Or individual plots
from erosion_simulation import plot_topography, plot_erosion_deposition, plot_water_features

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
plot_topography(sim, ax=axes[0])
plot_erosion_deposition(sim, ax=axes[1])
plot_water_features(sim, ax=axes[2])
plt.show()
```

## Configuration

### Simulation Parameters

- `n_years`: Total simulation duration (years)
- `dt`: Time step size (years)
- `pixel_scale_m`: Spatial resolution (meters/pixel)
- `uplift_rate`: Tectonic uplift rate (m/year)

### Erosion Parameters

- `transport_coefficient`: Sediment transport capacity (default: 0.01)
- `m`: Drainage area exponent in stream power law (default: 0.5)
- `n`: Slope exponent in stream power law (default: 1.0)

### Water Flow Parameters

- `n_manning`: Manning's roughness coefficient (default: 0.03)
- `infiltration_rate`: Water infiltration rate (m/hour, default: 0.01)
- `evaporation_rate`: Water evaporation rate (m/hour, default: 0.001)

### River/Lake Detection

- `threshold_accumulation`: Minimum flow accumulation for rivers (cells, default: 100)
- `min_depth`: Minimum water depth for lakes (m, default: 0.1)

## Output

### Simulation State

Access simulation results through the `ErosionSimulation` object:

```python
# Topography
current_elevation = sim.elevation
elevation_change = sim.get_elevation_change()

# Erosion statistics
total_erosion_m3 = sim.get_total_erosion()
total_deposition_m3 = sim.get_total_deposition()

# Water features
river_cells = sim.river_mask
lake_cells = sim.lake_mask

# Flow network
drainage_network = sim.flow_accumulation

# Surface materials
surface_materials = sim.get_surface_material()
erodibility_map = sim.get_erodibility_map()
```

### Saved Plots

The simulation automatically saves plots at specified intervals:
- `erosion_t{time}yr.png` - Summary plots at each interval
- `erosion_final.png` - Final state
- `surface_geology.png` - Surface material map

## Examples

### Example 1: Basic Erosion (500 years)

Creates a simple terrain and simulates 500 years of erosion with:
- 128×128 grid at 100m resolution
- Multiple geological layers
- Orographic rainfall pattern
- Stochastic storm events

**Output**: Shows drainage network development and river formation

### Example 2: Climate Change (2000 years)

Long-term simulation with varying climate:
- 500-year climate cycles
- Varying rainfall intensity
- Slow tectonic uplift

**Output**: Demonstrates landscape response to climate variations

### Example 3: Rock Type Comparison

Compares erosion in different rock types:
- Soft sedimentary terrain (shale, sandstone)
- Hard crystalline terrain (granite, gneiss)
- Same climate and duration

**Output**: Shows dramatic difference in erosion rates and patterns

## Integration with Rivers New

The `integrated_erosion_example.py` script integrates with the quantum-seeded terrain and weather generation from "Rivers new":

```python
python integrated_erosion_example.py
```

Features:
- Quantum-seeded realistic terrain (or classical fallback)
- Full geological stratigraphy
- Orographic weather patterns
- Storm simulation
- Multi-scale erosion processes

## Scientific Basis

### Stream Power Law

The model uses the stream power incision model:

```
E = K × A^m × S^n
```

Where:
- E: erosion rate (m/year)
- K: erodibility coefficient (material-dependent)
- A: upstream drainage area (m²)
- S: local slope (m/m)
- m, n: empirical exponents (typically m≈0.5, n≈1.0)

### Sediment Transport

Transport capacity is computed as:

```
Tc = Kt × A × S
```

Where Kt is the transport coefficient. When sediment load exceeds capacity, deposition occurs.

### Water Flow

Overland flow uses Manning's equation:

```
v = (1/n) × R^(2/3) × S^(1/2)
```

Where:
- v: flow velocity (m/s)
- n: Manning's roughness coefficient
- R: hydraulic radius ≈ water depth (m)
- S: slope (m/m)

### Material Properties

Erodibility coefficients are based on:
- Rock hardness
- Fracture density
- Chemical weathering susceptibility
- Cohesion and grain size (for sediments)

## Performance

### Typical Run Times (on modern CPU)

- 128×128 grid, 500 years, dt=2yr: ~2-5 minutes
- 256×256 grid, 1000 years, dt=5yr: ~10-20 minutes
- 512×512 grid, 2000 years, dt=10yr: ~1-2 hours

### Memory Usage

- 128×128: ~50 MB
- 256×256: ~200 MB
- 512×512: ~800 MB

### Optimization Tips

1. Use larger time steps (dt) for long simulations
2. Reduce grid resolution for initial exploration
3. Disable intermediate plotting (plot_interval=0)
4. Use fewer flow substeps if water depth is low

## Limitations

1. **2D Model**: Assumes uniform thickness in the vertical direction within each layer
2. **Simplified Physics**: Uses empirical erosion laws rather than full fluid dynamics
3. **No Bank Erosion**: Rivers erode vertically but not laterally
4. **No Glacial Processes**: Only water-driven erosion
5. **No Chemical Weathering**: Only mechanical erosion
6. **Static Vegetation**: No vegetation dynamics

## Future Enhancements

Potential additions:
- [ ] Lateral river migration
- [ ] Glacial erosion and deposition
- [ ] Chemical weathering
- [ ] Vegetation effects on erosion
- [ ] Landslides and mass wasting
- [ ] Coastal erosion
- [ ] 3D sediment tracking
- [ ] Multiple grain sizes

## References

### Stream Power Model
- Howard, A. D., & Kerby, G. (1983). Channel changes in badlands. Geological Society of America Bulletin, 94(6), 739-752.
- Whipple, K. X., & Tucker, G. E. (1999). Dynamics of the stream-power river incision model. Journal of Geophysical Research, 104(B8), 17661-17674.

### Sediment Transport
- Bagnold, R. A. (1966). An approach to the sediment transport problem from general physics. US Geological Survey Professional Paper 422-I.

### Landscape Evolution
- Tucker, G. E., & Hancock, G. R. (2010). Modelling landscape evolution. Earth Surface Processes and Landforms, 35(1), 28-50.

## License

This code is provided for research and educational purposes.

## Contact

For questions or issues, please refer to the documentation or create an issue.

---

**Last Updated**: December 2025
