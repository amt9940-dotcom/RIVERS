# Erosion Simulation - Quick Start Guide

## How to Run

### Step 1: Run Your Existing Terrain Code
Execute cells 0-9 in `Project.ipynb` (your quantum-seeded terrain and storm simulation)

### Step 2: Load Erosion Modules
Execute cells 10-15 (these just define functions, very fast)

### Step 3: Run the Simulation
Execute cell 16 - this will:
- Take 5-10 minutes for typical grid sizes
- Print progress updates every 50 years
- Automatically generate all three required plots

## Expected Outputs

### 1. Erosion Results (6-panel figure)
- Initial terrain
- Final terrain after 500 years
- Elevation change map (red = lowered, blue = raised)
- Total erosion (red = more erosion)
- Total deposition (blue = more deposition)
- Average rainfall (if storm simulation was run)

### 2. Rainfall Distribution (2-panel figure)
*Only if you ran the storm simulation (cells 2-9)*
- Left: Spatial rainfall map
- Right: Rainfall vs elevation scatter plot

### 3. Final Terrain with Water Features (2-panel figure)
- Left: Hillshaded final terrain
- Right: Same terrain with:
  - Rivers in light blue
  - Lakes in darker blue

## Key Variables After Running

```python
# Final elevation map
final_surface = erosion_simulation_results["final_surface"]

# How much was eroded/deposited
erosion_total = erosion_simulation_results["erosion_total"]
deposition_total = erosion_simulation_results["deposition_total"]

# Water features
river_mask = erosion_simulation_results["river_mask"]
lake_mask = erosion_simulation_results["lake_mask"]
```

## Quick Tweaks

### Make erosion faster:
Change in cell 16:
```python
"K_base": 2e-5,  # → try 5e-5 or 1e-4
"kappa": 0.015,  # → try 0.03
```

### Make erosion slower:
```python
"K_base": 2e-5,  # → try 1e-5 or 5e-6
"kappa": 0.015,  # → try 0.01
```

### Show more/fewer rivers:
Change in cell 16:
```python
detect_rivers(
    final_flow_accum,
    threshold_percentile=95.0,  # Lower = more rivers (try 90)
    min_accumulation=100.0       # Lower = more rivers (try 50)
)
```

### Detect more/fewer lakes:
```python
detect_lakes(
    final_surface,
    min_depth=0.5,      # Lower = more lakes (try 0.3)
    min_area_cells=4    # Lower = more small lakes (try 2)
)
```

## Troubleshooting

**Problem**: "NameError: name 'surface_elev' is not defined"
**Solution**: Run cells 0-1 first (terrain generation)

**Problem**: "NameError: name 'interfaces' is not defined"  
**Solution**: Run cell 0 completely (generates stratigraphy)

**Problem**: No rainfall map shown
**Solution**: Normal if you didn't run cells 2-9 (storm simulation). The erosion will still work with uniform rainfall.

**Problem**: Simulation too slow
**Solution**: 
- Reduce grid size in terrain generation
- Or reduce `n_years` to 100 or 200 for testing

**Problem**: Results look unrealistic
**Solution**: Check that `pixel_scale_m` matches your terrain scale (default 100m per cell)

## What Each Module Does

| Cell | Module | What It Does |
|------|--------|--------------|
| 10 | Flow Routing | Computes where water flows |
| 11 | Stream Power | Calculates erosion in channels |
| 12 | Layer Updates | Removes material from layers |
| 13 | Time Loop | Runs simulation for 500 years |
| 14 | Water Detection | Finds rivers and lakes |
| 15 | Visualization | Creates all plots |
| 16 | Main Execution | Runs everything together |

## Typical Workflow

1. **Generate terrain** (cells 0-1): 30 seconds
2. **Run storm simulation** (cells 2-9): 2-5 minutes  
3. **Run erosion simulation** (cells 10-16): 5-10 minutes

**Total time**: ~10-15 minutes for a complete run

## FAQ

**Q: Does this modify my original terrain?**
A: No, it makes a copy. Your original `surface_elev` is unchanged.

**Q: Can I run erosion without the storm simulation?**
A: Yes, it will use uniform rainfall automatically.

**Q: Can I change the number of years?**
A: Yes, edit `"n_years": 500` in cell 16 (try 100 for faster testing).

**Q: Why are some areas not eroding?**
A: They may be at local highs (ridges) or have resistant rock (low erodibility).

**Q: Can I export the results?**
A: Yes:
```python
import numpy as np
np.save('final_terrain.npy', erosion_simulation_results['final_surface'])
```

**Q: How do I re-run with different parameters?**
A: Just edit the parameters in cell 16 and re-run that cell. No need to re-run cells 10-15.

## Performance Tips

- **256×256 grid**: ~1-2 minutes (good for testing)
- **512×512 grid**: ~5-10 minutes (standard)
- **1024×1024 grid**: ~30-60 minutes (high resolution)

Time scales approximately with (grid_size)²

## Integration Notes

✅ Works with your quantum-seeded terrain
✅ Works with your wind-driven storms
✅ Works with your layer-aware stratigraphy
✅ Does not modify any existing code
✅ All new code is in cells 10-16

You can continue using all your existing terrain generation features while adding erosion simulation on top.
