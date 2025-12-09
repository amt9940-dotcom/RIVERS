# Erosion Model Summary

## Overview

I've created a comprehensive erosion model simulation system that integrates terrain generation, geological layer generation, and weather-driven erosion to simulate realistic landscape evolution over time.

## What Was Created

### 1. `erosion_model.py` - Main Erosion Model Class

This is the core simulation engine with the following capabilities:

**Key Features:**
- **Terrain Generation**: Uses functions from "Rivers new" to generate realistic random terrain
- **Layer System**: Manages multiple geological layers (Topsoil, Sandstone, Limestone, Basement, etc.) with different erodibility values
- **Water Flow Simulation**: Implements D8 flow direction algorithm for water routing
- **Erosion Physics**: Uses stream power law: `E = K × A^m × S^n × Erodibility`
- **River Detection**: Automatically identifies river networks from flow accumulation
- **Lake Detection**: Identifies lakes in depressions based on water depth
- **Time Evolution**: Tracks changes over time with configurable snapshots

**Main Methods:**
- `generate_initial_terrain()`: Creates initial terrain and stratigraphy
- `simulate()`: Runs erosion simulation for specified number of years
- `step()`: Performs one time step of erosion
- `plot_topography()`: Visualizes topography with rivers/lakes
- `plot_erosion_map()`: Visualizes erosion patterns
- `plot_time_series()`: Creates time series visualization

### 2. `run_erosion_demo.py` - Demo Script

A complete example showing how to:
- Initialize the model
- Run a 20-year simulation
- Generate multiple visualization outputs
- Display statistics about rivers, lakes, and erosion

### 3. `README_EROSION.md` - Documentation

Comprehensive documentation including:
- Quick start guide
- Configuration parameters
- Layer erodibility values
- Erosion physics explanation
- Customization examples
- Troubleshooting guide

## How It Works

### 1. Terrain Generation
- Uses quantum-seeded topography from "Rivers new" (or fallback if unavailable)
- Creates realistic elevation maps with proper scaling

### 2. Layer Generation
- Generates geological layers with realistic thicknesses
- Each layer has erodibility values (e.g., Topsoil=1.00, Basement=0.15)
- Layers are ordered from top to bottom

### 3. Weather/Rain Simulation
- Rainfall is distributed spatially (more on higher elevations - orographic effect)
- Temporal variation is added (seasonal-like patterns)
- Rainfall drives water flow and erosion

### 4. Water Flow
- D8 flow direction algorithm finds steepest descent
- Flow accumulation tracks total water through each cell
- Water depth is computed from rainfall and flow

### 5. Erosion
- Erosion rate depends on:
  - Flow accumulation (more water = more erosion)
  - Slope (steeper = more erosion)
  - Layer erodibility (softer materials erode faster)
- Erosion is applied to surface elevation
- Layer thicknesses are updated as material is eroded

### 6. River/Lake Formation
- **Rivers**: Detected from high flow accumulation (top 5% typically)
- **Lakes**: Detected from water depth in depressions (min 0.5m depth)

## Usage Example

```python
from erosion_model import ErosionModel

# Create and initialize
model = ErosionModel(grid_size=256, pixel_scale_m=10.0, elev_range_m=700.0)
model.generate_initial_terrain()

# Run simulation
results = model.simulate(
    num_years=20.0,
    annual_rainfall_mm=1200.0,
    time_step_years=0.1,
    save_snapshots=True,
    snapshot_interval_years=2.0
)

# Visualize
model.plot_topography(
    elevation=results["final_elevation"],
    rivers=results["final_rivers"],
    lakes=results["final_lakes"]
)
```

## Key Parameters You Can Adjust

### Simulation Time
- `num_years`: How long to simulate (e.g., 10, 20, 100 years)
- `time_step_years`: Time resolution (smaller = more accurate but slower)

### Rainfall
- `annual_rainfall_mm`: Annual rainfall amount (e.g., 800-3000 mm/year)
- Affects erosion rate and river/lake formation

### Grid Resolution
- `grid_size`: Grid dimensions (128, 256, 512, etc.)
- `pixel_scale_m`: Physical size per pixel (5-20 meters typical)

### Erodibility
- Modify `model.layer_properties[layer_name]["erodibility"]` to change how fast different materials erode

## Output Files Generated

When you run `run_erosion_demo.py`, it creates:

1. **erosion_comparison.png**: Side-by-side initial vs final topography
2. **erosion_map.png**: Total erosion over simulation period
3. **erosion_timeseries.png**: Evolution over time (multiple snapshots)
4. **erosion_analysis.png**: Comprehensive 6-panel analysis

## Integration with "Rivers new"

The model attempts to import functions from "Rivers new":
- `quantum_seeded_topography()`: Terrain generation
- `generate_stratigraphy()`: Layer generation
- `compute_top_material_map()`: Material identification

If these aren't available, the model uses simplified fallback functions, so it works independently.

## Realistic Features

1. **Layer Erodibility**: Different geological materials erode at realistic rates
2. **Flow-Dependent Erosion**: More water = more erosion (realistic)
3. **Slope-Dependent Erosion**: Steeper slopes erode faster
4. **River Formation**: Rivers form naturally in high-flow areas
5. **Lake Formation**: Lakes form in depressions where water accumulates
6. **Time Evolution**: You can see how topography changes over time

## Next Steps

To use the model:

1. **Run the demo**: `python3 run_erosion_demo.py`
2. **Modify parameters**: Adjust years, rainfall, grid size in the demo script
3. **Experiment**: Try different erodibility values, time steps, etc.
4. **Customize**: Create your own simulation scripts using the `ErosionModel` class

## Technical Details

### Erosion Algorithm
- Stream power law with m=0.5, n=1.0
- Maximum erosion rate capped at 0.1 m/year
- Erodibility values range from 0.02 (very resistant) to 1.05 (highly erodible)

### Flow Routing
- D8 (8-direction) flow direction
- Flow accumulation computed from high to low elevation
- Water depth includes both direct rainfall and accumulated flow

### River/Lake Detection
- Rivers: Flow accumulation > 95th percentile (configurable)
- Lakes: Water depth > 0.5m in depressions (configurable)
- Both use morphological operations to clean up noise

## Limitations & Future Improvements

Current limitations:
- Simplified flow routing (could use multiple flow direction)
- No sediment deposition (only erosion)
- No chemical weathering (especially for limestone)
- No tectonic uplift or base level changes

Potential improvements:
- Add sediment transport and deposition
- Implement chemical weathering
- Add tectonic processes
- More sophisticated lake dynamics
- Better integration with weather generation from "Rivers new"

## Summary

You now have a working erosion model that:
✅ Generates realistic terrain
✅ Creates geological layers with different erodibility
✅ Simulates weather-driven erosion
✅ Forms rivers and lakes naturally
✅ Shows topography evolution over time
✅ Provides comprehensive visualizations

The model is ready to use and can be customized for different scenarios, time periods, and geological settings!
