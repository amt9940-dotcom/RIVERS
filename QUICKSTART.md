# Erosion Model Quickstart Guide

## What's New

You now have a complete **landscape evolution / erosion model** integrated into `Project.ipynb`!

## How to Use

### 1. Open the Notebook

Open `Project.ipynb` in Jupyter and look for the new cells (10-13):

- **Cell 10**: Erosion model code (auto-runs when you execute it)
- **Cell 11**: Complete demo simulation (run this to see it in action!)
- **Cell 12**: Advanced integration examples
- **Cell 13**: Quick reference documentation

### 2. Run the Demo

Simply execute **Cell 11** in the notebook. It will:

1. Generate quantum-seeded terrain (256×256 grid)
2. Create layered stratigraphy
3. Run 50 epochs of erosion (50,000 years)
4. Show before/after visualizations
5. Display statistics and cross-sections

**Expected runtime**: ~5 minutes on a typical machine

### 3. Experiment

Try modifying parameters in Cell 11:

```python
# Make erosion faster
K_channel = 5e-6      # More channel incision

# Make terrain smoother
D_hillslope = 0.01    # More hillslope smoothing

# Add more uplift
uplift_rate = 0.0002  # Stronger tectonic uplift

# Run longer
num_epochs = 100      # More time steps
```

## Basic Usage Pattern

```python
import numpy as np
import copy

# 1. Generate terrain
z_norm, rng = quantum_seeded_topography(N=256, beta=3.0, random_seed=42)
strata = generate_stratigraphy(z_norm, elev_range_m=2000, pixel_scale_m=100, rng=rng)

# 2. Save initial state
strata_initial = copy.deepcopy(strata)

# 3. Run erosion
history = run_erosion_simulation(
    strata,
    pixel_scale_m=100,
    num_epochs=50,
    dt=1000,
    uplift_rate=0.0001,
    K_channel=1e-6,
    D_hillslope=0.005
)

# 4. Visualize
fig = plot_erosion_evolution(strata_initial, strata, history[-1], 100)
plt.show()
```

## What the Model Does

1. **Routes water** over your terrain (flow direction + accumulation)
2. **Erodes channels** where water flows (stream-power model)
3. **Smooths hillslopes** (diffusion)
4. **Transports sediment** and deposits it in valleys
5. **Updates stratigraphy** (removes from top layers, maintains ordering)
6. **Applies tectonic uplift** (raises terrain)

## Key Features

- **Layer-aware**: Automatically uses different erosion rates for different rock types
- **Physically-based**: Stream power + diffusion equations from geomorphology
- **Weather-integrated**: Can use your existing storm/wind generators
- **Visualizations**: Before/after maps, cross-sections, diagnostics

## Parameters

Start with these safe values:

```python
K_channel = 1e-6        # Channel erosion (lower = more stable)
D_hillslope = 0.005     # Hillslope diffusion
uplift_rate = 0.0001    # Tectonic uplift (0.1 mm/year)
dt = 1000               # Time step (years)
num_epochs = 50         # Number of steps
```

## Troubleshooting

**Problem**: Erosion too fast (terrain collapses)
- **Solution**: Reduce `K_channel` to 1e-7 or lower

**Problem**: Nothing happens
- **Solution**: Increase `K_channel` to 1e-5 or add more uplift

**Problem**: Simulation too slow
- **Solution**: Use smaller grid (N=128) or fewer epochs

## Documentation

- **EROSION_MODEL_README.md**: Complete API reference and examples
- **EROSION_IMPLEMENTATION_NOTES.md**: Technical details and design decisions
- **Cell 13 in notebook**: Quick reference guide

## Next Steps

1. Run Cell 11 to see the demo
2. Experiment with different parameters
3. Try weather-driven rainfall (Cell 12)
4. Run longer simulations to see mature landscapes
5. Add your own forcing functions

Enjoy evolving your quantum-seeded terrain! 🏔️ → 🌊
