# New Erosion Plotting Features

## Summary

I've added two comprehensive erosion visualization functions to the `landscape_evolution` package.

## What's New

### 1. `plot_erosion_analysis()` - Multi-Panel Erosion Analysis

**Location**: `landscape_evolution/visualization.py`

**What it shows**:
- ✅ **Large erosion map** with statistics (total, mean, max, std)
- ✅ **Histogram** of erosion values with mean/median lines
- ✅ **Cross-section profile** showing erosion along a transect (with cyan line marking location on map)
- ✅ **Erosion vs elevation scatter** with trend line and R² value

**Usage**:
```python
from landscape_evolution import plot_erosion_analysis

plot_erosion_analysis(
    erosion=cumulative_erosion,       # 2D array (m)
    surface_elev=final_surface,       # 2D array (m)
    pixel_scale_m=100.0,              # Grid spacing
    row_for_profile=128,              # Optional: which row to profile
    save_path='erosion_analysis.png'  # Optional: save location
)
```

### 2. `plot_erosion_rate_map()` - Erosion Rate with Optional River Overlay

**Location**: `landscape_evolution/visualization.py`

**What it shows**:
- ✅ **Erosion rate map** (m/yr)
- ✅ **Optional river network overlay** in blue (if flow_accum provided)
- ✅ **Side-by-side comparison** (with/without rivers)

**Usage**:
```python
from landscape_evolution import plot_erosion_rate_map

plot_erosion_rate_map(
    erosion_rate=erosion_rate,          # 2D array (m/yr)
    pixel_scale_m=100.0,                # Grid spacing
    flow_accum=flow_accumulation,       # Optional: for rivers
    save_path='erosion_rate.png'        # Optional: save location
)
```

## Files Added/Modified

### Modified Files:
1. **`landscape_evolution/visualization.py`**
   - Added `plot_erosion_analysis()` function
   - Added `plot_erosion_rate_map()` function

2. **`landscape_evolution/__init__.py`**
   - Exported new functions for easy import

### New Files:
3. **`example_erosion_plots.py`**
   - Runnable example showing how to use both functions
   - Complete workflow from terrain generation to erosion plotting
   - Saves output to workspace

4. **`EROSION_PLOTTING_GUIDE.md`**
   - Complete documentation
   - Multiple examples
   - Customization tips
   - Troubleshooting guide

5. **`NEW_EROSION_PLOTTING_FEATURES.md`** (this file)
   - Quick reference summary

## Quick Start

### Option 1: Run the Example Script

```bash
cd /workspace
python3 example_erosion_plots.py
```

This will:
- Generate terrain (256×256 grid)
- Run 5,000 year simulation
- Create and save both erosion plots
- Output: `erosion_analysis.png` and `erosion_rate_map.png`

### Option 2: Use in Your Code

```python
# After running a simulation...
from landscape_evolution import plot_erosion_analysis, plot_erosion_rate_map, FlowRouter

# Get erosion from history
erosion = history.get_total_erosion()

# Compute flow for river overlay
router = FlowRouter(pixel_scale_m)
_, _, flow_accum = router.compute_flow(world.surface_elev)

# Plot comprehensive analysis
plot_erosion_analysis(erosion, world.surface_elev, pixel_scale_m)

# Plot erosion rate with rivers
rate = erosion / total_time
plot_erosion_rate_map(rate, pixel_scale_m, flow_accum)
```

### Option 3: In Jupyter Notebook

```python
%matplotlib inline
import matplotlib.pyplot as plt
from landscape_evolution import plot_erosion_analysis

# ... run simulation ...

# Plot in notebook
plot_erosion_analysis(
    erosion=history.get_total_erosion(),
    surface_elev=world.surface_elev,
    pixel_scale_m=pixel_scale_m
)
plt.show()
```

## Key Features

### `plot_erosion_analysis()`

**Statistics Box** on map shows:
- Total eroded material (m)
- Mean erosion (m)
- Maximum erosion (m)
- Standard deviation (m)

**Histogram** shows:
- Distribution of erosion values
- Blue line: mean erosion
- Green line: median erosion

**Profile** shows:
- Erosion depth along cross-section
- Filled area for visual impact
- Corresponds to cyan line on map

**Scatter Plot** shows:
- Relationship between elevation and erosion
- Colored by erosion intensity
- Trend line with R² correlation
- Helps understand erosion patterns

### `plot_erosion_rate_map()`

**Single View Mode** (no flow_accum):
- Just the erosion rate map
- Clean, focused visualization

**Dual View Mode** (with flow_accum):
- Left: Erosion rate alone
- Right: Erosion rate + rivers in blue
- Shows spatial correlation between water and erosion

## Example Output

Running `example_erosion_plots.py` produces:

**erosion_analysis.png** contains:
```
┌─────────────────────┬────────────────┐
│  Erosion Map        │  Histogram     │
│  (with stats box    │  (with mean/   │
│   and profile line) │   median)      │
│                     ├────────────────┤
│                     │  Profile       │
│                     │  (erosion vs   │
│                     │   distance)    │
├─────────────────────┴────────────────┤
│  Erosion vs Elevation Scatter        │
│  (with trend line and R²)            │
└──────────────────────────────────────┘
```

**erosion_rate_map.png** contains:
```
┌──────────────────┬──────────────────┐
│  Erosion Rate    │  Erosion Rate    │
│  (plain)         │  + Rivers        │
│                  │  (blue overlay)  │
└──────────────────┴──────────────────┘
```

## Integration with Existing Code

These functions integrate seamlessly with your existing workflow:

```python
# Your existing simulation code
from landscape_evolution import LandscapeEvolutionSimulator
simulator = LandscapeEvolutionSimulator(world, tectonics, weather)
history = simulator.run(total_time=10000, dt=10)

# NEW: Add erosion analysis
from landscape_evolution import plot_erosion_analysis
plot_erosion_analysis(
    history.get_total_erosion(),
    world.surface_elev,
    pixel_scale_m
)
```

## Customization Examples

### Different Color Schemes

```python
# Hot colors (yellow → red → black)
plot_erosion_analysis(erosion, surface, pixel_scale_m, cmap='hot')

# Yellow-Orange-Red
plot_erosion_analysis(erosion, surface, pixel_scale_m, cmap='YlOrRd')

# Diverging (for positive/negative)
plot_erosion_analysis(erosion, surface, pixel_scale_m, cmap='RdBu_r')
```

### Custom Profile Location

```python
# Profile through specific feature
plot_erosion_analysis(
    erosion, surface, pixel_scale_m,
    row_for_profile=200  # Northern part of domain
)
```

### Larger/Smaller Figures

```python
# Large figure for presentations
plot_erosion_analysis(
    erosion, surface, pixel_scale_m,
    figsize=(20, 14)
)

# Small figure for papers
plot_erosion_analysis(
    erosion, surface, pixel_scale_m,
    figsize=(10, 7)
)
```

## Documentation

- **Detailed Guide**: See `EROSION_PLOTTING_GUIDE.md`
- **Function Docstrings**: In `landscape_evolution/visualization.py`
- **Example Code**: `example_erosion_plots.py`
- **Tutorial Notebook**: `Example_Landscape_Evolution.ipynb` (can add erosion examples)

## Dependencies

These functions use:
- ✅ numpy (already required)
- ✅ matplotlib (already required)
- ✅ scipy (for trend line in scatter plot)

All dependencies are in `requirements.txt`.

## Next Steps

1. **Try the example**: `python3 example_erosion_plots.py`
2. **Read the guide**: `EROSION_PLOTTING_GUIDE.md`
3. **Integrate into your workflow**: Add to your existing simulation code
4. **Customize**: Adjust colors, sizes, profiles to your needs

## Questions?

- Check docstrings: `help(plot_erosion_analysis)`
- Read the guide: `EROSION_PLOTTING_GUIDE.md`
- Look at examples: `example_erosion_plots.py`
