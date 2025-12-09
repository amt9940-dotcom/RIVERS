# Realistic Erosion Model Simulation

This erosion model simulates realistic erosion of Earth's layers over time, creating rivers and lakes from weather-driven rain storms. It integrates terrain generation, geological layer generation, and weather/rain generation to create a comprehensive erosion simulation.

## Features

- **Random Terrain Generation**: Uses quantum-seeded topography generation for realistic terrain
- **Geological Layers**: Multiple layers with different erodibility values (Topsoil, Sandstone, Limestone, Basement, etc.)
- **Weather-Driven Erosion**: Rain storms drive water flow and erosion
- **River Formation**: Automatic detection and visualization of river networks
- **Lake Formation**: Detection of lakes in depressions
- **Time Evolution**: Track topography changes over time with snapshots
- **Realistic Erodibility**: Different materials erode at different rates based on geological properties

## Files

- `erosion_model.py`: Main erosion model class with all simulation logic
- `run_erosion_demo.py`: Demo script showing how to use the model
- `Rivers new`: Source file containing terrain, layer, and weather generation functions

## Quick Start

### Basic Usage

```python
from erosion_model import ErosionModel

# Create model
model = ErosionModel(
    grid_size=256,          # 256x256 grid
    pixel_scale_m=10.0,      # Each pixel = 10 meters
    elev_range_m=700.0,      # Elevation range 0-700m
    random_seed=42          # For reproducibility
)

# Generate initial terrain
model.generate_initial_terrain()

# Run simulation
results = model.simulate(
    num_years=20.0,              # Simulate 20 years
    annual_rainfall_mm=1200.0,   # 1200 mm/year rainfall
    time_step_years=0.1,         # 0.1 year time steps
    save_snapshots=True,         # Save elevation snapshots
    snapshot_interval_years=2.0  # Every 2 years
)

# Visualize results
model.plot_topography(
    elevation=results["final_elevation"],
    rivers=results["final_rivers"],
    lakes=results["final_lakes"],
    title="Final Topography"
)
```

### Run Demo

```bash
python run_erosion_demo.py
```

This will:
1. Generate initial terrain and layers
2. Run a 20-year erosion simulation
3. Create visualization files:
   - `erosion_comparison.png`: Initial vs Final topography
   - `erosion_map.png`: Total erosion map
   - `erosion_timeseries.png`: Evolution over time
   - `erosion_analysis.png`: Detailed analysis

## Configuration Parameters

### Model Initialization

- `grid_size` (int): Grid resolution (e.g., 256 for 256x256)
- `pixel_scale_m` (float): Physical size of one pixel in meters
- `elev_range_m` (float): Elevation range in meters
- `random_seed` (int, optional): Random seed for reproducibility

### Simulation Parameters

- `num_years` (float): Number of years to simulate
- `annual_rainfall_mm` (float): Annual rainfall in millimeters
- `time_step_years` (float): Time step for simulation (smaller = more accurate but slower)
- `save_snapshots` (bool): Whether to save elevation snapshots
- `snapshot_interval_years` (float): Interval between snapshots

## Layer Erodibility

The model includes realistic erodibility values for different geological layers:

| Layer | Erodibility | Description |
|-------|-------------|-------------|
| Topsoil | 1.00 | Highly erodible, organic-rich |
| Subsoil | 0.85 | Less erodible than topsoil |
| Colluvium | 0.90 | Very erodible slope wash |
| Sandstone | 0.30 | Moderately resistant |
| Shale | 0.45 | Weak, easily eroded |
| Limestone | 0.28 | Strong but chemically erodible |
| Basement | 0.15 | Very resistant crystalline rock |
| Basalt | 0.12 | Extremely resistant |

Erodibility values range from 0.0 (completely resistant) to 1.0+ (highly erodible).

## Erosion Physics

The model uses a simplified stream power law:

```
Erosion Rate = K × (Flow Accumulation)^m × (Slope)^n × Erodibility
```

Where:
- `K`: Base erosion coefficient
- `m`: Flow exponent (typically 0.5)
- `n`: Slope exponent (typically 1.0)
- Flow Accumulation: Total water flowing through each cell
- Slope: Local topographic slope
- Erodibility: Material-dependent erodibility factor

## River and Lake Detection

### Rivers
Rivers are detected from flow accumulation maps. Cells with flow accumulation above a threshold (default: 95th percentile) are identified as river channels.

### Lakes
Lakes are detected from water depth maps. Depressions with water depth above a minimum threshold (default: 0.5 m) are identified as lakes.

## Visualization

The model provides several visualization functions:

- `plot_topography()`: Plot elevation map with rivers and lakes overlaid
- `plot_erosion_map()`: Plot erosion map
- `plot_time_series()`: Create time series showing evolution

## Customization

### Adjusting Erodibility

You can modify layer erodibility values:

```python
model.layer_properties["Sandstone"]["erodibility"] = 0.5  # Make sandstone more erodible
```

### Custom Rainfall Patterns

You can provide custom rainfall maps:

```python
# Create custom rainfall pattern
custom_rainfall = np.ones((256, 256)) * 0.001  # 1 mm uniform
custom_rainfall[100:150, 100:150] = 0.005  # More rain in center

# Use in simulation
model.step(custom_rainfall, dt_years=0.1)
```

### Different Time Steps

For faster simulation with less detail:
```python
results = model.simulate(
    num_years=100.0,
    time_step_years=1.0,  # Larger time step
    ...
)
```

For more detailed simulation:
```python
results = model.simulate(
    num_years=10.0,
    time_step_years=0.01,  # Smaller time step
    ...
)
```

## Output Structure

The `simulate()` method returns a dictionary with:

- `initial_elevation`: Initial topography
- `final_elevation`: Final topography after erosion
- `total_erosion`: Cumulative erosion map
- `final_rivers`: Binary map of river network
- `final_lakes`: Binary map of lakes
- `snapshots`: List of elevation snapshots over time
- `snapshot_times`: List of times for each snapshot

Each snapshot contains:
- `time_years`: Time in years
- `elevation`: Elevation map at this time
- `rivers`: River network at this time
- `lakes`: Lakes at this time
- `erosion_rate`: Erosion rate map
- `flow_accumulation`: Flow accumulation map

## Examples

### Example 1: Short-term Erosion (5 years)

```python
model = ErosionModel(grid_size=128, pixel_scale_m=20.0)
model.generate_initial_terrain()
results = model.simulate(num_years=5.0, annual_rainfall_mm=800.0)
```

### Example 2: Long-term Erosion (100 years)

```python
model = ErosionModel(grid_size=512, pixel_scale_m=5.0)
model.generate_initial_terrain()
results = model.simulate(
    num_years=100.0,
    annual_rainfall_mm=1500.0,
    time_step_years=0.5,
    snapshot_interval_years=10.0
)
```

### Example 3: High Rainfall Scenario

```python
model = ErosionModel(grid_size=256)
model.generate_initial_terrain()
results = model.simulate(
    num_years=10.0,
    annual_rainfall_mm=3000.0  # Very high rainfall
)
```

## Dependencies

- numpy
- matplotlib
- scipy

## Notes

- The model integrates with functions from "Rivers new" if available
- Falls back to simplified versions if "Rivers new" functions are not available
- Simulation time increases with grid size and smaller time steps
- For large grids (>512x512), consider using larger time steps

## Troubleshooting

**Import Error from "Rivers new":**
- The model will use fallback functions if "Rivers new" cannot be imported
- This is fine for basic functionality

**Slow Simulation:**
- Reduce grid size
- Increase time step
- Reduce number of snapshots

**No Rivers/Lakes Appearing:**
- Increase simulation time
- Increase rainfall amount
- Check that flow accumulation is working (plot flow_accumulation)

## Future Enhancements

Potential improvements:
- Sediment transport and deposition
- More sophisticated flow routing (multiple flow direction)
- Chemical weathering (especially for limestone)
- Tectonic uplift
- Base level changes
- More detailed lake dynamics
