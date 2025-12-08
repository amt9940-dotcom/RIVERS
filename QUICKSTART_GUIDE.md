# Erosion Simulation - Quick Start Guide

## What You Have

A complete, realistic erosion simulation system that models landscape evolution over time with:

✓ **Multi-layer geology** with different rock types and erodibility  
✓ **Realistic erosion physics** using stream power law  
✓ **Water flow simulation** with rainfall and storms  
✓ **River and lake formation** from realistic drainage patterns  
✓ **Time-stepped simulation** - run for hundreds or thousands of years  
✓ **Beautiful visualizations** of topography, erosion, and water features  

## Files Created

```
/workspace/
├── erosion_simulation.py           # Core simulation engine
├── example_erosion_simulation.py   # Standalone examples
├── integrated_erosion_example.py   # Integration with "Rivers new"
├── test_erosion.py                 # Test suite
├── README_EROSION.md               # Full documentation
└── QUICKSTART_GUIDE.md            # This file
```

## Quick Start (3 Steps)

### 1. Run the Test

```bash
cd /workspace
python3 test_erosion.py
```

This verifies everything works correctly.

### 2. Run Basic Example

```bash
python3 example_erosion_simulation.py
```

This runs a 500-year erosion simulation with:
- Realistic terrain generation
- Multiple geological layers
- Orographic rainfall patterns
- River and lake formation

**Output files:**
- `erosion_t*.png` - Progress snapshots
- `erosion_final.png` - Final results
- `surface_geology.png` - Surface material map

### 3. Run Integrated Example

```bash
python3 integrated_erosion_example.py
```

This integrates with your "Rivers new" code for quantum-seeded terrain.

## Understanding the Output

### Topography Maps
- **Elevation colors**: Brown/green = terrain elevation
- **Blue overlays**: Rivers (light blue) and lakes (cyan)
- Shows how landscape changes over time

### Erosion/Deposition Maps
- **Red areas**: Net deposition (sediment buildup)
- **Blue areas**: Net erosion (material removed)
- **Intensity**: Darker = more change

### Flow Accumulation
- **Bright areas**: Main drainage channels (rivers)
- **Dark areas**: Ridge tops and divides
- Shows the natural drainage network

### Statistics
- **Total Erosion**: Volume of material removed (km³)
- **Total Deposition**: Volume of material deposited (km³)
- **River/Lake Cells**: Number of cells with flowing/standing water

## Customizing Your Simulation

### Basic Parameters

```python
from erosion_simulation import run_erosion_simulation

sim = run_erosion_simulation(
    surface_elevation=my_terrain,      # Your terrain array
    layer_interfaces=my_layers,        # Layer elevations
    layer_order=layer_names,           # Layer names (top to bottom)
    
    # Simulation parameters
    n_years=1000.0,                    # Total time to simulate
    dt=5.0,                            # Time step (years)
    pixel_scale_m=100.0,               # Resolution (meters/pixel)
    
    # Physical parameters
    rainfall_mm_per_year=1200.0,       # Average rainfall
    uplift_rate=0.0001,                # Tectonic uplift (m/year)
    
    # Output control
    plot_interval=20,                  # Plot every N steps (0=no plots)
)
```

### Changing Time Scale

**Fast simulation** (hundreds of years):
```python
n_years=500.0
dt=5.0        # 100 time steps
```

**Long simulation** (thousands of years):
```python
n_years=5000.0
dt=25.0       # 200 time steps
```

**Very detailed** (slow but accurate):
```python
n_years=1000.0
dt=1.0        # 1000 time steps
```

### Changing Climate

**Wet climate** (more erosion):
```python
rainfall_mm_per_year=2000.0
storm_frequency=0.3
```

**Dry climate** (less erosion):
```python
rainfall_mm_per_year=500.0
storm_frequency=0.05
```

**Variable climate** (climate cycles):
```python
def variable_rainfall(time_years):
    base = 1000.0
    cycle = 0.5 * np.sin(2 * np.pi * time_years / 500.0)
    return base * (1.0 + cycle)
```

### Changing Terrain Size

**Small (fast, testing)**:
```python
N = 64          # 64x64 grid
pixel_scale_m = 200.0    # 200m resolution
# Domain: 12.8 km x 12.8 km
# Runtime: ~30 seconds
```

**Medium (balanced)**:
```python
N = 128         # 128x128 grid
pixel_scale_m = 100.0    # 100m resolution
# Domain: 12.8 km x 12.8 km
# Runtime: ~2-5 minutes
```

**Large (detailed, slow)**:
```python
N = 256         # 256x256 grid
pixel_scale_m = 50.0     # 50m resolution
# Domain: 12.8 km x 12.8 km
# Runtime: ~15-30 minutes
```

**Very Large (production)**:
```python
N = 512         # 512x512 grid
pixel_scale_m = 25.0     # 25m resolution
# Domain: 12.8 km x 12.8 km
# Runtime: ~1-2 hours
```

## Creating Your Own Layers

### Simple 3-Layer Example

```python
import numpy as np

# Create terrain
N = 128
terrain = np.random.randn(N, N).cumsum(axis=0).cumsum(axis=1) * 20
terrain = terrain - terrain.min() + 100  # Shift to positive elevation

# Define layers (from surface down)
layer_order = ["Topsoil", "Sandstone", "Basement"]
layer_interfaces = {
    "Topsoil": terrain - 2,              # 2m topsoil
    "Sandstone": terrain - 50,           # 48m sandstone
    "Basement": terrain - 500,           # 450m to basement
}
```

### Realistic Multi-Layer Stack

```python
layer_order = [
    "Topsoil",      # Soil cover (0-2m)
    "Subsoil",      # Weathered soil (2-6m)
    "Saprolite",    # Weathered rock (6-26m)
    "Sandstone",    # Sedimentary layer (26-106m)
    "Shale",        # Fine sediments (106-206m)
    "Limestone",    # Carbonate layer (206-326m)
    "Granite",      # Crystalline rock (326-726m)
    "Basement",     # Deep basement (726m+)
]

# Each layer interface = elevation of its top surface
```

### Layer Properties

The simulation automatically uses these erodibility values:

| Material | K (erodibility) | Typical Erosion Rate |
|----------|-----------------|---------------------|
| Sand | 0.0070 | Very Fast |
| Topsoil | 0.0050 | Fast |
| Shale | 0.0025 | Moderate |
| Sandstone | 0.0020 | Moderate |
| Limestone | 0.0018 | Slow |
| Granite | 0.0008 | Very Slow |
| Basement | 0.0003 | Extremely Slow |

## Advanced Usage

### Step-by-Step Control

```python
from erosion_simulation import ErosionSimulation
import numpy as np

# Initialize
sim = ErosionSimulation(
    surface_elevation=terrain,
    layer_interfaces=layers,
    layer_order=layer_names,
    pixel_scale_m=100.0
)

# Run your own time loop
for year in range(1000):
    # Custom rainfall for this year
    rainfall = generate_my_rainfall(year)
    
    # One time step
    sim.step(dt=1.0, rainfall_map=rainfall)
    
    # Access state
    if year % 100 == 0:
        print(f"Year {year}:")
        print(f"  Elevation range: {sim.elevation.min():.1f} - {sim.elevation.max():.1f} m")
        print(f"  Rivers: {np.sum(sim.river_mask)} cells")
        print(f"  Total erosion: {sim.get_total_erosion()/1e9:.4f} km³")
```

### Custom Rainfall Patterns

```python
def my_rainfall_generator(time_years):
    """Generate custom rainfall map."""
    
    # Base pattern (orographic - more rain on mountains)
    base_rain = 1000.0 * (0.5 + 0.5 * elevation_normalized)
    
    # Add seasonal variation
    season = np.sin(2 * np.pi * time_years)
    seasonal_factor = 0.7 + 0.6 * season
    
    # Add storm events
    if random.random() < 0.1:  # 10% chance of storm
        storm_center = (random.randint(0, N), random.randint(0, N))
        storm_intensity = 5.0
        # ... add storm pattern
    
    return base_rain * seasonal_factor
```

### Accessing Results

```python
# Current state
current_elevation = sim.elevation
rivers = sim.river_mask
lakes = sim.lake_mask

# Changes
elevation_change = sim.get_elevation_change()
total_erosion_m3 = sim.get_total_erosion()
total_deposition_m3 = sim.get_total_deposition()

# Flow network
drainage_area = sim.flow_accumulation * sim.cell_area_m2

# Surface properties
surface_materials = sim.get_surface_material()  # Array of material names
erodibility = sim.get_erodibility_map()        # Array of K values

# Time
current_time = sim.current_time  # in years
```

## Example Scenarios

### 1. Mountain Range Erosion

```python
# High relief terrain with hard rock
terrain = create_mountain_range(height=2000, width=50000)
layers = {
    "Topsoil": terrain - 1,
    "Granite": terrain - 100,
    "Basement": terrain - 1000,
}

sim = run_erosion_simulation(
    surface_elevation=terrain,
    layer_interfaces=layers,
    layer_order=["Topsoil", "Granite", "Basement"],
    n_years=10000,           # 10,000 years
    dt=50,                   # 50 year steps
    rainfall_mm_per_year=2000,  # Heavy rainfall
    uplift_rate=0.001,       # Active uplift (1 mm/year)
)
```

### 2. Valley Incision

```python
# Gentle terrain with soft sediments
terrain = create_gentle_slope(gradient=0.01)
layers = {
    "Sand": terrain - 2,
    "Clay": terrain - 10,
    "Silt": terrain - 25,
    "Sandstone": terrain - 100,
    "Basement": terrain - 500,
}

sim = run_erosion_simulation(
    surface_elevation=terrain,
    layer_interfaces=layers,
    layer_order=["Sand", "Clay", "Silt", "Sandstone", "Basement"],
    n_years=5000,
    dt=25,
    rainfall_mm_per_year=1200,
    uplift_rate=0.0,  # No uplift
)
```

### 3. Badlands Formation

```python
# Highly erodible terrain (like American Southwest)
terrain = create_plateau_with_canyons(height=1500)
layers = {
    "Topsoil": terrain - 0.5,    # Very thin soil
    "Mudstone": terrain - 20,     # Soft mudstone
    "Shale": terrain - 100,       # Easily eroded
    "Sandstone": terrain - 200,   # Resistant cap
    "Basement": terrain - 1000,
}

sim = run_erosion_simulation(
    surface_elevation=terrain,
    layer_interfaces=layers,
    layer_order=["Topsoil", "Mudstone", "Shale", "Sandstone", "Basement"],
    n_years=2000,
    dt=10,
    rainfall_mm_per_year=400,   # Arid climate
    storm_frequency=0.3,         # Infrequent intense storms
)
```

## Interpreting Results

### What to Look For

1. **Rivers form in valleys**
   - Check: Do rivers follow topographic lows?
   - Expected: Tree-like branching patterns

2. **Erosion on steep slopes**
   - Check: Is erosion highest on steep slopes?
   - Expected: Blue (erosion) on hillsides

3. **Deposition in valleys**
   - Check: Is sediment accumulating in valleys?
   - Expected: Red (deposition) in lowlands

4. **Material exposure**
   - Check: Are deeper layers exposed on hillslopes?
   - Expected: Different materials at different elevations

5. **Conservation**
   - Check: Does erosion ≈ deposition?
   - Expected: Small difference (some sediment lost to boundaries)

### Troubleshooting

**No rivers forming:**
- Increase simulation time
- Increase rainfall
- Lower river detection threshold

**Too much/too little erosion:**
- Adjust erodibility coefficients
- Change rainfall amounts
- Modify time step size

**Unrealistic patterns:**
- Check layer ordering (top to bottom)
- Verify elevation units (meters)
- Ensure positive elevations

## Tips for Best Results

1. **Start small**: Test with N=64 before scaling up
2. **Plot often**: Use plot_interval to see evolution
3. **Check units**: Elevation in meters, rainfall in mm
4. **Be patient**: Large grids take time
5. **Experiment**: Try different parameters to understand behavior

## Getting Help

If something isn't working:

1. Run `python3 test_erosion.py` to verify installation
2. Check that your terrain has reasonable elevations (0-3000m typical)
3. Verify layer interfaces decrease with depth
4. Make sure all arrays are the same shape
5. Check the full documentation in `README_EROSION.md`

## What's Next?

- Try the comparison examples (different rock types)
- Experiment with climate variations
- Create your own terrain and layer combinations
- Integrate with the quantum terrain generator
- Export results for further analysis

---

**Have fun exploring landscape evolution!**

For detailed documentation, see `README_EROSION.md`
