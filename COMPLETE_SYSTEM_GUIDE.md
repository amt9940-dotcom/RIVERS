# COMPLETE EROSION SYSTEM WITH REALISTIC LAYERS & EPOCH VISUALIZATION

## 🎯 What's New

### ✅ Fixed Issues
1. **Non-uniform erosion**: Different materials (Topsoil, Subsoil, Colluvium, Saprolite, Weathered Bedrock, Basement) have different erodibility
2. **Realistic layer generation**: Uses sophisticated hillslope & weathering rules from original Project.ipynb
3. **Epoch-by-epoch visualization**: Shows erosion progress after each time period
4. **Non-uniform rain**: Wind physics creates wet windward slopes and dry leeward sides

## 📋 Files Overview

### **cells_00_to_09_WITH_LAYERS.py** (NEW)
- **Purpose**: Generate terrain with 6 realistic layers
- **Features**:
  - Topsoil: Thicker on gentle slopes, thin on steep slopes
  - Subsoil: Thicker in mid-elevation areas
  - Colluvium: Gravity deposits in valleys
  - Saprolite: Weathered bedrock on stable ridges
  - Weathered Bedrock: Patchy, partially weathered
  - Basement: Unweathered crystalline rock
- **Output**: `GLOBAL_STRATA`, `GLOBAL_WEATHER_DATA`, `GLOBAL_RAIN_TIMESERIES`

### **cell_10_constants.py** (Updated)
- **Purpose**: Define erosion parameters
- **Erodibility values**:
  ```
  Topsoil:      2.0× (most erodible)
  Subsoil:      1.5×
  Colluvium:    1.8×
  Saprolite:    1.2×
  WeatheredBR:  0.8× (resistant)
  Basement:     0.3× (very resistant)
  ```

### **cell_11_flow_direction.py** → **cell_18_visualization.py**
- **Purpose**: Erosion simulation components (unchanged)
- Same advanced erosion physics as before

### **cell_19_demonstration_EPOCHS.py** (NEW)
- **Purpose**: Run erosion simulation with epoch-by-epoch visualization
- **Features**:
  - Shows terrain evolution over multiple epochs
  - Tracks surface material exposure over time
  - Analyzes erosion rates and patterns
  - Generates comprehensive plots

## 🚀 Quick Start

### Step 1: Copy Cells 0-9 (Terrain + Layers + Weather)
```python
# In Jupyter, create a new code cell and paste:
# [Copy entire contents of cells_00_to_09_WITH_LAYERS.py]
```

**What it does**:
- Generates 256×256 terrain with quantum-seeded randomness
- Creates 6 different surface layers with realistic distribution
- Simulates 100 years of weather with EAST wind (90°)
- Wind creates orographic precipitation (wet windward, dry leeward)
- Stores data in global variables for erosion simulator

**Output**:
- Terrain elevation map
- Surface material map (shows which layer is exposed)
- Total rain map (non-uniform due to wind)
- Layer thickness maps (Topsoil, Colluvium, Saprolite)

### Step 2: Copy Erosion Constants (Cell 10)
```python
# Create a new code cell and paste:
# [Copy entire contents of cell_10_constants.py]
```

**What it does**:
- Defines erodibility for all 6 layers
- Sets time acceleration (10×), rain boost (100×)
- Configures erosion, transport, and diffusion parameters

### Step 3: Copy Erosion Components (Cells 11-18)
```python
# Create separate cells for each file:
# Cell 11: [cell_11_flow_direction.py]
# Cell 12: [cell_12_discharge.py]
# Cell 13: [cell_13_erosion_pass_a.py]
# Cell 14: [cell_14_sediment_transport.py]
# Cell 15: [cell_15_hillslope_diffusion.py]
# Cell 16: [cell_16_river_lake_detection.py]
# Cell 17: [cell_17_main_simulation.py]
# Cell 18: [cell_18_visualization.py]
```

**What they do**:
- Cell 11: D8 flow direction algorithm
- Cell 12: Runoff and discharge calculation
- Cell 13: Erosion (with half-loss rule)
- Cell 14: Sediment transport and deposition
- Cell 15: Hillslope diffusion (soil creep)
- Cell 16: River and lake detection
- Cell 17: Main simulation orchestration
- Cell 18: Visualization functions

### Step 4: Run Epoch Demonstration (Cell 19)
```python
# Create a new code cell and paste:
# [Copy entire contents of cell_19_demonstration_EPOCHS.py]
```

**What it does**:
- Runs 5 epochs × 20 years = 100 sim years (1000 real years with 10× acceleration)
- Shows snapshots after each epoch
- Generates comprehensive visualizations

## 📊 Expected Output

### Plot 1: Epoch-by-Epoch Evolution (3 rows × 6 columns)

**Row 1: Elevation**
- Shows terrain height at Year 0, 20, 40, 60, 80, 100
- Notice valleys deepening, ridges lowering

**Row 2: Surface Material**
- Color-coded: Topsoil (brown) → Subsoil (orange) → Colluvium (green) → Saprolite (purple) → Weathered BR (pink) → Basement (red)
- Watch as Topsoil erodes away, exposing deeper layers
- Valleys expose hard Basement, ridges keep Saprolite

**Row 3: Erosion Depth**
- Hot colors = more erosion
- Shows cumulative erosion from initial state
- Notice non-uniform pattern (valleys erode more)

### Plot 2: Erosion Rate Analysis (3 panels)

**Panel 1: Average Erosion Over Time**
- Line plot showing mean erosion increasing
- Should see consistent erosion rate

**Panel 2: Maximum Erosion Over Time**
- Shows deepest point eroded
- Valleys deepen dramatically

**Panel 3: Erosion Distribution**
- Histogram of erosion depths
- Some areas erode little, valleys erode a lot

### Plot 3: Surface Material Evolution (2 panels)

**Panel 1: Stacked Area Chart**
- Shows how much of each material is exposed over time
- Topsoil decreases, deeper layers increase

**Panel 2: Percentage Coverage**
- Line plot for each layer
- Watch Topsoil decline, Basement increase

## 🔬 Key Physics

### Why Erosion is Non-Uniform

1. **Different Materials**:
   - Topsoil: Erodes quickly (2.0× erodibility)
   - Basement: Resists erosion (0.3× erodibility)
   - As Topsoil erodes away → exposes harder layers → erosion slows

2. **Non-Uniform Rain**:
   - EAST wind (90°) hits barriers (ridges)
   - Windward slopes: More rain → more erosion
   - Leeward slopes: Rain shadow → less erosion
   - Channels: Rain funnels → concentrated erosion

3. **Topographic Feedback**:
   - Valleys get more water (high discharge Q) → erode faster
   - Ridges get less water → erode slower
   - Creates self-reinforcing drainage network

### Layer Exposure Pattern

**Initial State**:
- Surface is mostly Topsoil and Subsoil
- Topsoil on gentle slopes
- Colluvium in valleys

**After 50 Years**:
- Topsoil eroding rapidly
- Saprolite exposed on ridges
- Colluvium accumulating in valley bottoms

**After 100 Years**:
- Topsoil mostly gone
- Weathered Bedrock exposed on ridges
- Basement exposed in deeply eroded valleys
- Colluvium fills low areas

## 🎨 Interpreting the Plots

### Elevation Maps
- **Dark green/blue**: Low elevation (valleys, basins)
- **Yellow/brown**: Mid elevation
- **White/tan**: High elevation (ridges)
- **Watch**: Valleys deepen, overall elevation decreases

### Surface Material Maps
- **Brown**: Topsoil (erodible, disappears quickly)
- **Orange**: Subsoil (moderately erodible)
- **Green**: Colluvium (deposited in valleys)
- **Purple**: Saprolite (exposed on ridges)
- **Pink**: Weathered Bedrock (resistant)
- **Red**: Basement (very resistant, exposed in deep valleys)

### Erosion Depth Maps
- **White/yellow**: Little erosion (<1 m)
- **Orange/red**: Moderate erosion (1-5 m)
- **Dark red/black**: Heavy erosion (>5 m)
- **Pattern**: Valleys show heavy erosion, ridges show light erosion

## ⚙️ Customization

### Adjust Erosion Speed
In `cell_10_constants.py`:
```python
RAIN_BOOST = 200.0  # Double the erosion power (was 100.0)
TIME_ACCELERATION = 20.0  # Double the time acceleration (was 10.0)
```

### Adjust Number of Epochs
In `cell_19_demonstration_EPOCHS.py`:
```python
num_epochs = 10  # More snapshots (was 5)
years_per_epoch = 10  # Smaller time steps (was 20)
```

### Adjust Layer Erodibility
In `cell_10_constants.py`:
```python
ERODIBILITY_MAP = {
    "Topsoil": 3.0,  # Even more erodible (was 2.0)
    "Basement": 0.1,  # Even more resistant (was 0.3)
    # ... adjust others as needed
}
```

### Change Wind Direction
In `cells_00_to_09_WITH_LAYERS.py`:
```python
wind_dir_deg = 180.0  # South wind (was 90.0 for East)
```

## 🔍 Validation Checklist

After running the full system, verify:

- [ ] **Initial terrain shows 6 different layers**
  - Check surface material map has multiple colors
  - Topsoil should be widespread
  - Colluvium should appear in valleys
  - Saprolite should appear on ridges

- [ ] **Rain is non-uniform**
  - Check total rain map
  - Should see wet windward (west) sides of ridges
  - Should see dry leeward (east) sides (rain shadow)
  - Should see rain streaks along valleys

- [ ] **Erosion is non-uniform**
  - Check erosion depth map
  - Valleys should erode more (darker colors)
  - Ridges should erode less (lighter colors)
  - Pattern should follow rain and material distribution

- [ ] **Layers evolve over time**
  - Check material exposure plots
  - Topsoil percentage should decrease
  - Deeper layer percentages should increase
  - By Year 100, should see significant Basement exposure

- [ ] **Epoch plots show progression**
  - Each epoch should show visible change
  - Valleys should progressively deepen
  - Surface materials should progressively change
  - Erosion depth should progressively increase

## 🐛 Troubleshooting

### "GLOBAL_STRATA not found"
- **Cause**: Cells 0-9 not run
- **Fix**: Run `cells_00_to_09_WITH_LAYERS.py` first

### "Uniform erosion everywhere"
- **Cause**: All materials have same erodibility, or rain is uniform
- **Fix**: Check `ERODIBILITY_MAP` in cell 10, check rain map from cells 0-9

### "No visible erosion"
- **Cause**: Erosion parameters too weak
- **Fix**: Increase `RAIN_BOOST` or `BASE_K` in cell 10

### "Erosion too fast / unstable"
- **Cause**: Erosion parameters too strong
- **Fix**: Decrease `RAIN_BOOST` or increase `MAX_ERODE_PER_STEP`

### "All layers look the same in plots"
- **Cause**: Colormap issue
- **Fix**: Ensure using `cmap='tab10'` for material plots

### "Plots don't show after each epoch"
- **Cause**: Missing `plt.show()` calls
- **Fix**: Check `cell_19_demonstration_EPOCHS.py` has `plt.show()` after each figure

## 📈 Expected Performance

- **Terrain generation**: ~5-10 seconds
- **Weather simulation (100 years)**: ~10-20 seconds
- **Erosion per epoch (20 years)**: ~30-60 seconds
- **Total runtime**: ~5-10 minutes for full demonstration

## 🎓 Scientific Basis

### Layer Distribution Rules

**Topsoil**:
- Thickest on gentle slopes (soil accumulates)
- Thinnest on steep slopes (soil slides off)
- Moderate on ridges (stable but thin)

**Subsoil**:
- Thickest in mid-elevation areas
- Thinner at high elevations (less weathering)
- Thinner in valleys (erosion)

**Colluvium**:
- Only in valleys and at slope bases
- Gravity-driven deposits
- Thickest at valley junctions

**Saprolite**:
- Thickest on stable ridges and interfluves
- Represents deep weathering
- Stripped from valleys by erosion

**Weathered Bedrock**:
- Patchy distribution
- Partially weathered but still resistant
- More common at high elevations

**Basement**:
- Infinite thickness (unbreakable foundation)
- Only exposed after deep erosion
- Appears in deeply incised valleys

### Erosion Physics

**Stream Power Law**:
```
Erosion Rate = K × Q^m × S^n × Erodibility
```
Where:
- K = erosion coefficient
- Q = discharge (water flux)
- S = slope
- m, n = exponents (~0.5, 1.0)

**Half-Loss Rule**:
- 50% of eroded material is transported downstream
- 50% is "lost" from the system
- Allows valleys to deepen permanently

**Transport Capacity**:
- Steep + high Q → high capacity → no deposition
- Flat + low Q → low capacity → deposition
- Creates alluvial fans, floodplains

## 📚 References

- **Stream Power Law**: Howard & Kerby (1983), Whipple & Tucker (1999)
- **Hillslope Diffusion**: Culling (1960), Roering et al. (1999)
- **Orographic Precipitation**: Roe (2005), Smith & Barstad (2004)
- **Landscape Evolution**: Tucker & Hancock (2010), Whipple (2004)

## ✅ Summary

You now have:
1. ✅ Realistic layer generation (6 materials with different erodibility)
2. ✅ Non-uniform rain (wind creates wet/dry patterns)
3. ✅ Non-uniform erosion (different materials + different rain)
4. ✅ Epoch-by-epoch visualization (see erosion progress)
5. ✅ Material exposure tracking (watch layers appear/disappear)
6. ✅ Comprehensive analysis (rates, distributions, patterns)

**Result**: A scientifically realistic erosion simulation that shows how terrain evolves over time, with different materials eroding at different rates, and wind-driven rain creating non-uniform erosion patterns.
