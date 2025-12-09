# How to Use the Erosion Model - Step by Step

## Quick Answer

**Just run this ONE file:**
```bash
python3 erosion_with_rivers_weather.py
```

This file automatically:
1. Imports from your "Rivers new" code
2. Uses YOUR weather system (not simplified)
3. Runs the erosion simulation
4. Shows results

---

## Files You Need (Already in /workspace)

### ✅ Files You Must Have:

1. **`Rivers new`** - Your existing code (already there)
   - Contains terrain, weather, stratigraphy

2. **`erosion_simulation.py`** - Core erosion engine (I created)
   - The main physics and algorithms

3. **`erosion_with_rivers_weather.py`** - Integration script (I just created)
   - **THIS IS THE ONE TO RUN** ← START HERE

### 📚 Optional Files (for reference/examples):

4. `example_erosion_simulation.py` - Standalone examples (doesn't need Rivers new)
5. `integrated_erosion_example.py` - Old integration (uses simplified weather)
6. `test_erosion.py` - Tests

---

## Usage Options

### Option 1: Use Your Full Weather System (RECOMMENDED)

```bash
cd /workspace
python3 erosion_with_rivers_weather.py
```

**What it does:**
- ✅ Uses quantum-seeded terrain from "Rivers new"
- ✅ Uses full stratigraphy from "Rivers new"  
- ✅ Uses YOUR sophisticated weather system from "Rivers new"
- ✅ Applies erosion physics
- ✅ Creates rivers and lakes
- ✅ Shows beautiful plots

**Runtime:** ~5-15 minutes for 50 years, 128×128 grid

---

### Option 2: Quick Test (Standalone)

```bash
python3 example_erosion_simulation.py
```

**What it does:**
- Uses simple terrain generator (no quantum)
- Uses simplified rainfall (not your weather)
- Faster, but less realistic

**Use this for:** Testing, learning how erosion works

---

### Option 3: Write Your Own Integration

```python
#!/usr/bin/env python3
"""
my_erosion_script.py - Custom integration
"""

# Import the erosion engine
from erosion_simulation import ErosionSimulation

# Import from your Rivers new code
# (Load it however you normally do)
from your_imports import (
    quantum_seeded_topography,
    generate_stratigraphy,
    generate_storm_weather_fields,
    # ... etc
)

# 1. Generate terrain (your code)
surface_elevation, rng = quantum_seeded_topography(N=256)

# 2. Generate layers (your code)
strata = generate_stratigraphy(surface_elevation, rng=rng)

# 3. Initialize erosion
sim = ErosionSimulation(
    surface_elevation=surface_elevation,
    layer_interfaces=strata['interfaces'],
    layer_order=list(strata['interfaces'].keys()),
    pixel_scale_m=100.0
)

# 4. Run simulation with YOUR weather
for year in range(100):
    # Generate weather for this year (your code)
    storms = generate_storm_schedule_for_year(year, ...)
    
    # Convert storms to rainfall map
    year_rainfall = accumulate_rainfall_from_storms(storms, ...)
    
    # Apply erosion
    sim.step(dt=1.0, rainfall_map=year_rainfall)
    
    print(f"Year {year}: {np.sum(sim.river_mask)} river cells")

# 5. Visualize
from erosion_simulation import plot_simulation_summary
plot_simulation_summary(sim)
```

---

## What Each File Does

### Files You RUN:

| File | Purpose | Weather System | Runtime |
|------|---------|----------------|---------|
| **`erosion_with_rivers_weather.py`** | **Full integration** | **YOUR weather** ✓ | 5-15 min |
| `example_erosion_simulation.py` | Standalone examples | Simple/fake | 2-5 min |
| `test_erosion.py` | Verify installation | None | 30 sec |

### Files You IMPORT:

| File | Contains | Used By |
|------|----------|---------|
| **`erosion_simulation.py`** | Core engine | ALL other files |
| **`Rivers new`** | Your existing code | `erosion_with_rivers_weather.py` |

---

## Step-by-Step Instructions

### Step 1: Test Installation

```bash
cd /workspace
python3 test_erosion.py
```

**Expected output:**
```
============================================================
TESTING EROSION SIMULATION
============================================================
...
ALL TESTS PASSED ✓
```

If tests fail, you need to install packages:
```bash
pip3 install --user numpy matplotlib scipy
```

---

### Step 2: Run Full Integration

```bash
python3 erosion_with_rivers_weather.py
```

**Expected output:**
```
Loading Rivers new components...
✓ Successfully loaded all Rivers new components
  ✓ Terrain generation
  ✓ Stratigraphy
  ✓ Weather generation

================================================================================
EROSION SIMULATION WITH RIVERS NEW WEATHER SYSTEM
================================================================================

--- STEP 1: Generate Quantum-Seeded Terrain ---
✓ Terrain generated: 0.0 to 1500.0 m

--- STEP 2: Generate Stratigraphy ---
✓ Stratigraphy generated: 16 layers

--- STEP 3: Analyze Wind Structures ---
✓ Wind structures identified

--- STEP 4: Compute Orographic Low Pressure ---
✓ Low-pressure likelihood map computed

--- STEP 5: Initialize Erosion Simulation ---
✓ Erosion simulation initialized

--- STEP 6: Run Weather + Erosion Simulation ---
Simulating 50 years...

--- Year 1/50 ---
  Generated 6 storms for year 1
    Storm 1: 45.3 mm average rainfall
    Storm 2: 67.8 mm average rainfall
    ...
  Total year rainfall: 1243.5 mm

[... progress continues ...]

✓ Weather + erosion simulation complete!

--- STEP 7: Results and Visualization ---
Final Statistics:
  Duration: 50.0 years
  Total erosion: 0.0234 km³
  River cells: 1523 (11.84%)
  Lake cells: 87 (0.68%)
  
✓ Saved: erosion_with_rivers_weather.png

✓ This simulation used:
  • Your quantum-seeded terrain generation
  • Your stratigraphy system
  • Your sophisticated weather/storm generation ← KEY!
  • Erosion physics from erosion_simulation.py
```

**Output files:**
- `erosion_with_rivers_weather.png` - Complete visualization

---

### Step 3: Customize Parameters

Edit `erosion_with_rivers_weather.py` at the bottom:

```python
sim = run_erosion_with_rivers_weather(
    N=128,                      # Change to 256 for more detail
    pixel_scale_m=100.0,        # Resolution
    n_years=50,                 # Change to 100, 500, etc.
    base_wind_dir_deg=225.0,    # Wind direction (225 = SW)
    mean_annual_rain_mm=1200.0, # Average rainfall
    random_seed=42              # Change for different terrain
)
```

---

## Understanding the Integration

### What "Rivers new" Provides:

```
Rivers new
│
├─ quantum_seeded_topography()
│  └─ Generates realistic terrain
│
├─ generate_stratigraphy()  
│  └─ Creates geological layers
│
├─ build_wind_structures()
│  └─ Identifies mountains, valleys, wind patterns
│
├─ compute_orographic_low_pressure()
│  └─ Calculates where storms form
│
├─ generate_storm_schedule_for_year()
│  └─ Creates storm events for each year
│
└─ generate_storm_weather_fields()
   └─ Generates detailed rainfall from each storm
      ├─ Orographic effects
      ├─ Wind barriers
      ├─ Storm tracks
      └─ Spatial rainfall patterns
```

### What Erosion Simulation Adds:

```
erosion_simulation.py
│
├─ ErosionSimulation class
│  │
│  ├─ Takes rainfall from "Rivers new" ← INTEGRATION POINT
│  │
│  ├─ Computes water flow
│  ├─ Calculates erosion (stream power law)
│  ├─ Transports sediment
│  ├─ Updates elevation
│  ├─ Detects rivers
│  └─ Detects lakes
│
└─ Visualization functions
```

### How They Connect:

```
Flow Diagram:
=============

1. "Rivers new" generates terrain
              ↓
2. "Rivers new" creates layers
              ↓
3. "Rivers new" analyzes wind structures
              ↓
4. FOR EACH YEAR:
   ├─ "Rivers new" generates storms
   ├─ "Rivers new" calculates rainfall map
   │              ↓
   ├─ erosion_simulation receives rainfall ← CONNECTION
   ├─ erosion_simulation erodes terrain
   ├─ erosion_simulation updates elevation
   └─ erosion_simulation finds rivers/lakes
              ↓
5. Visualization shows results
```

---

## Import Structure

### If you want to use in your own code:

```python
# Your script.py

# Import erosion engine
from erosion_simulation import (
    ErosionSimulation,           # Main class
    plot_simulation_summary,     # Visualization
    plot_topography,             # Plot terrain
    ERODIBILITY                  # Material properties
)

# Import from Rivers new (however you normally do it)
# ... your imports ...

# Then use them together:
terrain = quantum_seeded_topography(...)
sim = ErosionSimulation(terrain, ...)
sim.step(rainfall_map=your_rainfall)
```

---

## File Dependencies

```
Dependency Tree:
================

erosion_with_rivers_weather.py (MAIN FILE - RUN THIS)
├── imports: erosion_simulation.py
│   └── imports: numpy, matplotlib, scipy
│
└── imports: Rivers new
    └── imports: numpy, matplotlib, qiskit (optional)
```

**Order doesn't matter** - just run `erosion_with_rivers_weather.py`

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'numpy'"

```bash
pip3 install --user numpy matplotlib scipy
```

### "Rivers new components not available"

The script will tell you what's missing:
```
⚠ Some Rivers new components not available
  Terrain: True
  Strata: False  ← This one failed
  Weather: True
```

Check that "Rivers new" file exists and has those functions.

### "Simulation too slow"

Reduce parameters in the script:
```python
N=64,          # Smaller grid (was 128)
n_years=25,    # Fewer years (was 50)
```

### "Weather generation error"

The script has fallback behavior - it will use simplified rainfall if your weather system fails for any storm.

---

## Summary

### To use YOUR existing weather system:

```bash
python3 erosion_with_rivers_weather.py
```

**That's it!** This one file does everything.

### File Priority:

1. **`erosion_with_rivers_weather.py`** ← USE THIS (your full weather)
2. `example_erosion_simulation.py` ← Use only for quick tests (fake weather)
3. `erosion_simulation.py` ← Don't run directly (it's a library)

### What you DON'T need to modify:

- ✓ Rivers new (leave as-is)
- ✓ erosion_simulation.py (leave as-is)

### What you CAN modify:

- ✓ `erosion_with_rivers_weather.py` - Change parameters at bottom
- ✓ Or write your own script importing from both

---

## Need Help?

**Quick test:** `python3 test_erosion.py`  
**Full run:** `python3 erosion_with_rivers_weather.py`  
**Documentation:** See `README_EROSION.md` and `QUICKSTART_GUIDE.md`

---

**TL;DR: Just run `erosion_with_rivers_weather.py` - it uses YOUR weather system!**
