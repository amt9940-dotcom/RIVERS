# How to Use the Erosion Model in VS Code

Since your `Project.ipynb` is too large to open, I've extracted the code into **standalone Python files** that you can use directly in VS Code.

## 📁 Files Created

```
workspace/
├── terrain_generator.py          ← Your existing terrain generation code
├── erosion_model.py               ← New erosion model engine
├── erosion_demo.py                ← Ready-to-run example
├── QUICKSTART.md                  ← Quick reference guide
├── EROSION_MODEL_README.md        ← Full documentation
└── HOW_TO_USE.md                  ← This file
```

## 🚀 Quick Start (3 Steps)

### Step 1: Open in VS Code

Open your workspace folder in VS Code:
```bash
cd /workspace
code .
```

### Step 2: Install Dependencies

Make sure you have the required packages:
```bash
pip install numpy scipy matplotlib
```

### Step 3: Run the Demo

```bash
python erosion_demo.py
```

That's it! The demo will:
1. Generate a 256×256 terrain
2. Run 50 epochs of erosion (50,000 years)
3. Create before/after visualizations
4. Save images: `erosion_results.png` and `erosion_cross_section.png`

**Expected runtime:** ~5-10 minutes on a typical computer

---

## 📝 File Descriptions

### `terrain_generator.py` (53 KB)
Your **existing** terrain generation code from `Project.ipynb` Cell 0:
- `quantum_seeded_topography()` - Generates quantum-seeded terrain
- `generate_stratigraphy()` - Creates layered stratigraphy
- All your weather, wind, and structural functions

**You can use this file as-is** - it's your original code extracted for easier use.

### `erosion_model.py` (26 KB)
The **new** erosion model engine with 15 functions:

**Water Routing:**
- `compute_flow_direction_d8()` - D8 steepest descent
- `compute_flow_accumulation()` - Flow accumulation with topological sorting
- `route_flow_simple()` - Combined routing

**Erosion:**
- `get_top_layer_at_surface()` - Determines exposed layer
- `get_effective_erodibility()` - Gets layer-specific K value
- `channel_incision_stream_power()` - Stream power erosion
- `hillslope_diffusion()` - Diffusive smoothing

**Sediment Transport:**
- `compute_sediment_transport()` - Routes sediment downslope

**Stratigraphy Updates:**
- `update_stratigraphy_with_erosion()` - Removes from top layers
- `update_stratigraphy_with_deposition()` - Adds to alluvium

**Forcing:**
- `apply_uplift()` - Tectonic uplift

**Time-Stepping:**
- `run_erosion_epoch()` - Single time step
- `run_erosion_simulation()` - Multiple epochs

**Visualization:**
- `plot_erosion_evolution()` - Before/after comparison
- `plot_cross_section_evolution()` - Stratigraphic sections

### `erosion_demo.py`
A complete, ready-to-run example showing how to use everything together.

---

## 💻 Using in Your Own Code

### Basic Usage

```python
# Import what you need
from terrain_generator import quantum_seeded_topography, generate_stratigraphy
from erosion_model import run_erosion_simulation, plot_erosion_evolution
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
import matplotlib.pyplot as plt
fig = plot_erosion_evolution(strata_initial, strata, history[-1], 100)
plt.show()
```

### Custom Rainfall

```python
# Define a custom rainfall function
def my_rainfall(epoch):
    """Generate spatially variable rainfall for each epoch."""
    ny, nx = strata["surface_elev"].shape
    
    # Example: More rain in the north
    rainfall = np.ones((ny, nx))
    for i in range(ny):
        rainfall[i, :] = 0.5 + (i / ny)  # 0.5 to 1.5 m/year
    
    return rainfall

# Use it in simulation
history = run_erosion_simulation(
    strata,
    pixel_scale_m=100,
    num_epochs=50,
    dt=1000,
    rainfall_func=my_rainfall,  # Custom rainfall
    uplift_rate=0.0001,
    K_channel=1e-6,
    D_hillslope=0.005
)
```

### Weather-Driven Rainfall

```python
# Use your existing weather generators
from terrain_generator import build_wind_structures, compute_orographic_low_pressure

def weather_driven_rainfall(epoch):
    """Use existing weather system to generate rainfall."""
    surface_elev = strata["surface_elev"]
    
    # Build wind structures
    wind_structs = build_wind_structures(
        surface_elev, 
        pixel_scale_m=100, 
        base_wind_dir_deg=270
    )
    
    # Compute orographic low pressure
    low_pressure = compute_orographic_low_pressure(
        surface_elev,
        rng,
        pixel_scale_m=100,
        base_wind_dir_deg=270,
        wind_structs=wind_structs
    )
    
    # Convert to rainfall
    base_rainfall = 0.5  # m/year
    rainfall = base_rainfall * (1.0 + 2.0 * low_pressure["low_pressure_likelihood"])
    
    return rainfall

# Use in simulation
history = run_erosion_simulation(
    strata,
    pixel_scale_m=100,
    num_epochs=100,
    dt=500,
    rainfall_func=weather_driven_rainfall,
    uplift_rate=0.0001,
    K_channel=1e-6,
    D_hillslope=0.005
)
```

---

## ⚙️ Key Parameters

### Recommended Starting Values

```python
K_channel = 1e-6        # Channel erosion (1e-7 to 1e-5)
D_hillslope = 0.005     # Hillslope diffusion (0.001 to 0.01 m²/year)
uplift_rate = 0.0001    # Tectonic uplift (1e-5 to 1e-3 m/year)
dt = 1000               # Time step (500 to 5000 years)
num_epochs = 50         # Number of steps
```

### Adjusting Parameters

**More erosion:**
- Increase `K_channel` to 5e-6
- Decrease `uplift_rate` to 1e-5

**Less erosion / more stable:**
- Decrease `K_channel` to 1e-7
- Increase `uplift_rate` to 0.0002

**Faster runs:**
- Use smaller grid: `N=128`
- Fewer epochs: `num_epochs=25`

**Higher quality:**
- Use larger grid: `N=512`
- More epochs: `num_epochs=100`

---

## 🔧 Troubleshooting

### Problem: Import errors

```
ModuleNotFoundError: No module named 'numpy'
```

**Solution:** Install dependencies
```bash
pip install numpy scipy matplotlib
```

### Problem: Erosion too fast (terrain collapses)

**Solution:** Reduce erosion coefficient
```python
K_channel = 1e-7  # Lower value
```

### Problem: Nothing happens / no erosion visible

**Solution:** Increase erosion or reduce uplift
```python
K_channel = 1e-5    # Higher erosion
uplift_rate = 1e-5  # Lower uplift
```

### Problem: Script too slow

**Solution:** Use smaller grid or fewer epochs
```python
N = 128           # Smaller grid
num_epochs = 25   # Fewer steps
```

### Problem: "Can't open Project.ipynb"

**Solution:** You don't need to! Use the extracted Python files instead:
- `terrain_generator.py` - All terrain functions
- `erosion_model.py` - All erosion functions
- `erosion_demo.py` - Ready-to-run example

---

## 📊 What the Code Does

Each erosion epoch (time step):

1. **Uplift**: Raises surface and all rock layers
2. **Water routing**: Computes where water flows (discharge Q, slope S)
3. **Channel erosion**: Carves valleys where Q is high (stream power model)
4. **Hillslope smoothing**: Rounds hilltops (diffusion)
5. **Sediment transport**: Moves eroded material downslope
6. **Deposition**: Deposits sediment in valleys
7. **Stratigraphy update**: Removes from top layers, adds to alluvium

**Layer-aware:** Automatically uses different erosion rates for different rocks:
- Topsoil (erodibility = 1.0) erodes fast
- Sandstone (erodibility = 0.3) erodes slowly
- Basement (erodibility = 0.15) erodes very slowly

---

## 📚 Documentation

- **QUICKSTART.md** - Quick reference
- **EROSION_MODEL_README.md** - Complete API reference
- **EROSION_IMPLEMENTATION_NOTES.md** - Technical details

---

## 🎯 Next Steps

1. **Run the demo:**
   ```bash
   python erosion_demo.py
   ```

2. **Experiment with parameters** in `erosion_demo.py`

3. **Create your own script** using the examples above

4. **Integrate with your existing code** by importing from `terrain_generator.py` and `erosion_model.py`

---

## 💡 Tips

- Start with **small grids (N=128)** to test quickly
- Use **lower K_channel (1e-7)** for first tests to ensure stability
- **Save intermediate results** if running long simulations
- **Visualize often** to see how terrain evolves

---

## ✅ Summary

You now have three Python files that work together:

1. **terrain_generator.py** - Your existing terrain code (ready to use)
2. **erosion_model.py** - New erosion engine (ready to use)
3. **erosion_demo.py** - Complete working example (run this!)

**No need to open the large notebook** - everything is extracted and ready to use in VS Code!

Happy eroding! 🏔️ → 🌊
