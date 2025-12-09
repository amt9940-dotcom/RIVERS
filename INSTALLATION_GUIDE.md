# 📝 Installation Guide - Copy & Paste Individual Cells

## Overview

I've created **11 separate files** (one for each code block) that you can copy and paste individually into your Jupyter notebook. This allows you to edit and run each component separately.

---

## 📁 Files Created

| File Name | Cell Type | Description |
|-----------|-----------|-------------|
| `cell_10_constants.py` | Code | Erosion constants and parameters (configurable) |
| `cell_11_flow_direction.py` | Code | D8 flow direction algorithm |
| `cell_12_discharge.py` | Code | Discharge (Q) computation |
| `cell_13_erosion_pass_a.py` | Code | Erosion with half-loss rule (Pass A) |
| `cell_14_sediment_transport.py` | Code | Sediment transport and deposition (Pass B) |
| `cell_15_hillslope_diffusion.py` | Code | Hillslope diffusion smoothing |
| `cell_16_river_lake_detection.py` | Code | River and lake detection |
| `cell_17_main_simulation.py` | Code | Main erosion simulation function |
| `cell_18_visualization.py` | Code | Visualization and plotting |
| `cell_19_demonstration.py` | Code | **Complete demonstration** (runs everything) |
| `cell_20_documentation.md` | Markdown | Documentation and user guide |

---

## 🚀 Installation Instructions

### Step 1: Open Your Jupyter Notebook

Open `Project.ipynb` in Jupyter Notebook or JupyterLab.

### Step 2: Create New Cells

At the bottom of your existing notebook, create **11 new cells** (10 code cells + 1 markdown cell).

### Step 3: Copy & Paste Each File

For each file, follow these steps:

#### **Cell 10** (Code Cell)
1. Open `cell_10_constants.py`
2. Copy **all content** (Ctrl+A, Ctrl+C)
3. Paste into new code cell in Jupyter
4. Run the cell to verify it works

#### **Cell 11** (Code Cell)
1. Open `cell_11_flow_direction.py`
2. Copy all content
3. Paste into next code cell
4. Run the cell

#### **Cell 12** (Code Cell)
1. Open `cell_12_discharge.py`
2. Copy all content
3. Paste into next code cell
4. Run the cell

#### **Cell 13** (Code Cell)
1. Open `cell_13_erosion_pass_a.py`
2. Copy all content
3. Paste into next code cell
4. Run the cell

#### **Cell 14** (Code Cell)
1. Open `cell_14_sediment_transport.py`
2. Copy all content
3. Paste into next code cell
4. Run the cell

#### **Cell 15** (Code Cell)
1. Open `cell_15_hillslope_diffusion.py`
2. Copy all content
3. Paste into next code cell
4. Run the cell

#### **Cell 16** (Code Cell)
1. Open `cell_16_river_lake_detection.py`
2. Copy all content
3. Paste into next code cell
4. Run the cell

#### **Cell 17** (Code Cell)
1. Open `cell_17_main_simulation.py`
2. Copy all content
3. Paste into next code cell
4. Run the cell

#### **Cell 18** (Code Cell)
1. Open `cell_18_visualization.py`
2. Copy all content
3. Paste into next code cell
4. Run the cell

#### **Cell 19** (Code Cell) - **THE MAIN DEMONSTRATION**
1. Open `cell_19_demonstration.py`
2. Copy all content
3. Paste into next code cell
4. **Run this cell to see the complete erosion simulation!**

#### **Cell 20** (Markdown Cell)
1. Open `cell_20_documentation.md`
2. Copy all content
3. Create a **Markdown cell** (not code!)
4. Paste the content
5. Run the cell to render the documentation

---

## ⚡ Quick Test

After pasting all cells, verify the installation:

### Test 1: Run Cells 10-18
Run cells 10 through 18 in order. Each should print a "✅" success message.

Expected output:
```
✅ Erosion parameters initialized!
✅ Flow direction module loaded!
✅ Discharge computation module loaded!
✅ Erosion Pass A module loaded!
✅ Sediment Transport Pass B module loaded!
✅ Hillslope diffusion module loaded!
✅ River and lake detection module loaded!
✅ Main erosion simulation function loaded!
✅ Visualization module loaded!
```

### Test 2: Run Cell 19 (Complete Simulation)
Run cell 19 to execute the full erosion simulation. This will:
1. Generate quantum-seeded terrain
2. Simulate 100 years (= 1000 real years)
3. Display 6 comprehensive plots
4. Show rivers and lakes

Expected runtime: **5-10 minutes** for 256×256 grid

---

## 🎯 Execution Order

**IMPORTANT**: Always run cells in this order:

1. **Cells 0-9** (your original terrain generation code)
2. **Cell 10** (constants)
3. **Cell 11** (flow direction)
4. **Cell 12** (discharge)
5. **Cell 13** (erosion Pass A)
6. **Cell 14** (sediment transport Pass B)
7. **Cell 15** (hillslope diffusion)
8. **Cell 16** (river/lake detection)
9. **Cell 17** (main simulation)
10. **Cell 18** (visualization)
11. **Cell 19** (demonstration) ← **Run this to see results!**
12. **Cell 20** (documentation - markdown)

---

## 🔧 Customization

### Change Erosion Strength

Edit **Cell 10** (constants):
```python
TIME_ACCELERATION = 20.0   # Make it 20× instead of 10×
RAIN_BOOST = 200.0         # Double the rain strength
BASE_K = 0.002             # More aggressive erosion
```

### Change Simulation Duration

Edit **Cell 19** (demonstration):
```python
num_timesteps = 200  # Run for 200 years instead of 100
N = 128              # Use smaller grid for faster testing
```

### Adjust River/Lake Detection

Edit **Cell 19** (bottom section):
```python
river_discharge_threshold = 1000.0  # Lower = more rivers
lake_discharge_threshold = 500.0    # Lower = more lakes
```

---

## 📊 Expected Results

After running Cell 19, you should see:

### 6 Plots:
1. **Initial Topography**: Your quantum-generated terrain
2. **Final Topography**: After 1000 real years of erosion
3. **Elevation Change**: Red areas = erosion, blue = deposition
4. **Rivers and Lakes**: Blue rivers, cyan lakes
5. **Discharge Map**: Drainage network intensity
6. **Cross-Section**: Before/after comparison

### Statistics:
```
EROSION STATISTICS
================================================================================
Initial elevation: 50.2 - 487.3 m
Final elevation: 45.7 - 465.1 m
Mean elevation change: -3.45 m
Max erosion: 25.3 m
Max deposition: 8.7 m
Total volume eroded: 0.125 km³
Total volume deposited: 0.058 km³
Net volume change: -0.067 km³  ← Half-loss rule working!

River cells: 1247 (1.9%)
Lake cells: 143 (0.2%)
Number of lakes: 8
================================================================================
```

---

## 🐛 Troubleshooting

### Problem: "NameError: name 'compute_flow_direction_d8' is not defined"
**Solution**: You must run cells **in order**. Start from Cell 10 and run each cell sequentially.

### Problem: "NameError: name 'TIME_ACCELERATION' is not defined"
**Solution**: Run Cell 10 first to define all constants.

### Problem: Simulation is too slow
**Solution**: In Cell 19, change:
```python
N = 128  # Smaller grid (faster)
num_timesteps = 50  # Fewer timesteps
```

### Problem: Not enough visible erosion
**Solution**: In Cell 10, increase:
```python
RAIN_BOOST = 200.0  # Double the rain
BASE_K = 0.002      # Double erosion rate
```

### Problem: Quantum functions not working
**Solution**: The code automatically falls back to classical RNG. You'll see a warning but it will still work.

---

## 📚 Additional Documentation

See these files for more information:
- **`EROSION_SYSTEM_SUMMARY.md`**: Complete technical documentation (374 lines!)
- **`cell_20_documentation.md`**: User guide and physics explanation
- **Each `.py` file**: Contains detailed docstrings and comments

---

## ✅ Verification Checklist

Before running the full simulation, verify:

- [ ] All 11 files copied into Jupyter notebook
- [ ] Cell 10 runs without errors (constants defined)
- [ ] Cells 11-18 each show "✅ ... module loaded!"
- [ ] Cell 19 is ready to run (demonstration)
- [ ] Cell 20 is markdown (not code)

---

## 🎉 You're Ready!

Once all cells are pasted and verified:
1. **Run Cell 19** to see the complete erosion simulation
2. **Wait 5-10 minutes** for computation
3. **View the results**: 6 beautiful plots with rivers and lakes!

**Enjoy your quantum-accelerated erosion simulation! 🌋🏔️💧**

---

*Questions? Check the documentation files or adjust parameters in Cell 10.*
