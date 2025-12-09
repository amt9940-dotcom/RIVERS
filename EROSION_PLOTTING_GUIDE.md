# Erosion Plotting Guide

This guide explains how to visualize erosion output from the landscape evolution simulator.

## Available Functions

The `landscape_evolution.visualization` module provides two dedicated erosion plotting functions:

### 1. `plot_erosion_analysis()` - Comprehensive Erosion Analysis

Creates a multi-panel figure showing:
- **Erosion map**: Spatial distribution of cumulative erosion
- **Histogram**: Distribution of erosion values
- **Cross-section profile**: Erosion along a transect
- **Erosion vs elevation scatter**: Relationship between erosion and current elevation

**Usage:**

```python
from landscape_evolution.visualization import plot_erosion_analysis

plot_erosion_analysis(
    erosion=cumulative_erosion,          # 2D array of erosion depth (m)
    surface_elev=final_surface,          # 2D array of current elevation (m)
    pixel_scale_m=100.0,                 # Grid spacing
    row_for_profile=128,                 # Optional: row for cross-section
    save_path='erosion_analysis.png'     # Optional: save to file
)
```

**Output:**
- Large map showing where erosion occurred
- Statistics box with total, mean, max, std
- Histogram of erosion values with mean/median lines
- Profile showing erosion intensity along a transect
- Scatter plot showing erosion vs elevation relationship

### 2. `plot_erosion_rate_map()` - Erosion Rate with River Overlay

Creates a focused plot of erosion rate, optionally overlaying the river network.

**Usage:**

```python
from landscape_evolution.visualization import plot_erosion_rate_map

plot_erosion_rate_map(
    erosion_rate=erosion_rate,           # 2D array of erosion rate (m/yr)
    pixel_scale_m=100.0,                 # Grid spacing
    flow_accum=flow_accumulation,        # Optional: for river overlay
    save_path='erosion_rate.png'         # Optional: save to file
)
```

**Output:**
- Left panel: Erosion rate map
- Right panel (if flow_accum provided): Erosion rate with rivers in blue

## Complete Example

Here's a complete workflow from simulation to erosion visualization:

```python
import numpy as np
from landscape_evolution import (
    WorldState,
    LandscapeEvolutionSimulator,
    TectonicUplift,
    WeatherGenerator,
    FlowRouter,
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

# 1. Generate terrain
N = 256
pixel_scale_m = 100.0
z_norm, rng = quantum_seeded_topography(N=N, random_seed=42)
surface_elev = denormalize_elevation(z_norm, (0, 1000))

# 2. Create world state
layer_names = ["Topsoil", "Saprolite", "Sandstone", "Basement"]
world = WorldState(N, N, pixel_scale_m, layer_names)
create_slope_dependent_stratigraphy(world, surface_elev, pixel_scale_m)

# 3. Set up forcing
tectonics = TectonicUplift(N, N, pixel_scale_m)
tectonics.set_uniform_uplift(1e-3)  # 1 mm/yr

weather = WeatherGenerator(N, N, pixel_scale_m, mean_annual_precip_m=1.0)

# 4. Run simulation
simulator = LandscapeEvolutionSimulator(world, tectonics, weather)
history = simulator.run(total_time=10000.0, dt=10.0)

# 5. Get erosion data
cumulative_erosion = history.get_total_erosion()

# 6. Compute flow for river overlay
router = FlowRouter(pixel_scale_m)
flow_dir, slope, flow_accum = router.compute_flow(world.surface_elev)

# 7. Plot erosion analysis
plot_erosion_analysis(
    erosion=cumulative_erosion,
    surface_elev=world.surface_elev,
    pixel_scale_m=pixel_scale_m,
    save_path='erosion_analysis.png'
)

# 8. Plot erosion rate with rivers
erosion_rate = cumulative_erosion / 10000.0  # Average rate over simulation
plot_erosion_rate_map(
    erosion_rate=erosion_rate,
    pixel_scale_m=pixel_scale_m,
    flow_accum=flow_accum,
    save_path='erosion_rate_rivers.png'
)
```

## Quick Example Script

Run the provided example script:

```bash
cd /workspace
python3 example_erosion_plots.py
```

This will:
1. Generate terrain
2. Run a 5,000-year simulation
3. Create two erosion plots
4. Save them to your workspace

Output files:
- `/workspace/erosion_analysis.png`
- `/workspace/erosion_rate_map.png`

## Understanding the Outputs

### Erosion Map
- **Red colors** = High erosion (more material removed)
- **White/light** = Low or no erosion
- Look for patterns:
  - Erosion concentrated in valleys/channels
  - Ridges show less erosion
  - Rivers create linear erosion patterns

### Histogram
- Shows distribution of erosion values
- **Blue dashed line** = Mean erosion
- **Green dashed line** = Median erosion
- Typically right-skewed (most cells have low erosion, few have high)

### Cross-Section Profile
- Shows erosion intensity along a transect
- **Peaks** = Locations of high erosion (often channels)
- **Valleys** = Less erosion (ridges, interfluves)

### Erosion vs Elevation Scatter
- Shows relationship between current elevation and erosion
- Often shows:
  - **Negative correlation**: Higher elevations → Less erosion (ridges)
  - **Positive correlation**: Lower elevations → More erosion (valleys)
- Trend line and R² value show strength of relationship

### River Overlay
- **Blue colors** = High flow accumulation (rivers)
- Rivers often align with high erosion zones
- Shows causal relationship: water flow → erosion

## Customization

### Change Colormap

```python
plot_erosion_analysis(
    erosion=erosion,
    surface_elev=surface,
    pixel_scale_m=100,
    cmap='hot'  # Try: 'hot', 'YlOrRd', 'Reds', 'RdYlBu_r'
)
```

### Different Profile Location

```python
plot_erosion_analysis(
    erosion=erosion,
    surface_elev=surface,
    pixel_scale_m=100,
    row_for_profile=200  # Specify row index
)
```

### Adjust Figure Size

```python
plot_erosion_analysis(
    erosion=erosion,
    surface_elev=surface,
    pixel_scale_m=100,
    figsize=(20, 12)  # Width, height in inches
)
```

## Advanced: Tracking Erosion Rate Over Time

If you want instantaneous erosion rates (not just cumulative), modify the simulator to track them:

```python
# In your simulation loop
erosion_rates = []
for i in range(n_steps):
    # ... simulation step ...
    
    # Track erosion rate this step
    step_erosion = processes['channel_erosion']  # From geomorphic engine
    erosion_rates.append(step_erosion.copy())

# Later, analyze rates
mean_erosion_rate = np.mean(erosion_rates, axis=0)
max_erosion_rate = np.max(erosion_rates, axis=0)

plot_erosion_rate_map(mean_erosion_rate, pixel_scale_m)
```

## Integration with Other Visualizations

Combine erosion plots with other views:

```python
from landscape_evolution.visualization import (
    plot_initial_vs_final,
    plot_erosion_deposition_maps,
    plot_evolution_summary
)

# 1. Initial vs final topography
plot_initial_vs_final(history, pixel_scale_m)

# 2. Erosion and deposition together
plot_erosion_deposition_maps(history, pixel_scale_m)

# 3. Detailed erosion analysis
plot_erosion_analysis(history.get_total_erosion(), world.surface_elev, pixel_scale_m)

# 4. Comprehensive summary (includes erosion)
plot_evolution_summary(history, world, flow_accum)
```

## Exporting Data for External Analysis

To export erosion data for use in other tools:

```python
import numpy as np

# Get erosion data
erosion = history.get_total_erosion()

# Save as text file
np.savetxt('erosion.txt', erosion, fmt='%.3f')

# Save as binary (faster, smaller)
np.save('erosion.npy', erosion)

# Save as GeoTIFF (requires gdal/rasterio)
# from osgeo import gdal
# driver = gdal.GetDriverByName('GTiff')
# dataset = driver.Create('erosion.tif', N, N, 1, gdal.GDT_Float32)
# dataset.GetRasterBand(1).WriteArray(erosion)
# dataset = None
```

## Tips for Interpretation

1. **High erosion in channels**: Expected - water concentrates in valleys
2. **Low erosion on ridges**: Expected - ridges are topographic highs with less water
3. **Spatial patterns**: Look for drainage basin boundaries, watershed divides
4. **Magnitude**: Compare to uplift rate - if erosion >> uplift, landscape is lowering
5. **Rate variability**: High variability suggests complex interactions (e.g., layer exposure)

## Troubleshooting

**Q: Erosion map is all white/no data**
- Check that simulation actually ran (`history.times` should have multiple entries)
- Verify erosion is occurring (print `erosion.max()`)
- May need longer simulation time or higher erosion rates

**Q: Profile shows no variation**
- Try a different row (`row_for_profile`)
- Check if erosion is uniform (unlikely but possible)

**Q: Scatter plot is empty**
- Erosion may be zero everywhere (simulation too short)
- Check `erosion[erosion > 0].size` to count eroded cells

**Q: Rivers don't align with erosion**
- Flow accumulation computed on *final* surface
- Rivers may have migrated during simulation
- Recompute flow on *initial* surface to see original drainage

## See Also

- `README_LANDSCAPE_EVOLUTION.md` - Main documentation
- `Example_Landscape_Evolution.ipynb` - Full tutorial
- `example_erosion_plots.py` - Runnable example script
- `visualization.py` - Source code with all plotting functions
