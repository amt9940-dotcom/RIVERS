# 🚀 QUICK START - Complete Erosion System

## What You Get

✅ **Non-uniform erosion** - Different materials (6 layers) erode at different rates  
✅ **Non-uniform rain** - Wind creates wet windward slopes, dry leeward slopes  
✅ **Epoch-by-epoch plots** - See erosion progress over time  
✅ **Realistic physics** - Stream power law, sediment transport, hillslope diffusion  

## Copy-Paste Order

### 1️⃣ Terrain + Layers + Weather (NEW!)
**File**: `cells_00_to_09_WITH_LAYERS.py`

Creates:
- 256×256 terrain with quantum randomness
- 6 realistic layers: Topsoil, Subsoil, Colluvium, Saprolite, Weathered Bedrock, Basement
- 100 years of weather with EAST wind
- Non-uniform rain (windward wet, leeward dry)

**Output**: Terrain map, Surface material map, Rain map, Layer thickness maps

---

### 2️⃣ Erosion Constants
**File**: `cell_10_constants.py`

Sets:
- Erodibility for each layer (Topsoil: 2.0×, Basement: 0.3×)
- Time acceleration: 10× (100 sim years = 1000 real years)
- Rain boost: 100× (extreme erosion)

---

### 3️⃣ Erosion Components (Cells 11-18)
Copy each file into a separate cell:

| Cell | File | Purpose |
|------|------|---------|
| 11 | `cell_11_flow_direction.py` | D8 flow direction |
| 12 | `cell_12_discharge.py` | Water flux calculation |
| 13 | `cell_13_erosion_pass_a.py` | Material removal (half-loss rule) |
| 14 | `cell_14_sediment_transport.py` | Sediment transport & deposition |
| 15 | `cell_15_hillslope_diffusion.py` | Soil creep |
| 16 | `cell_16_river_lake_detection.py` | River/lake identification |
| 17 | `cell_17_main_simulation.py` | Main simulation loop |
| 18 | `cell_18_visualization.py` | Plotting functions |

---

### 4️⃣ Epoch Demonstration (NEW!)
**File**: `cell_19_demonstration_EPOCHS.py`

Runs:
- 5 epochs × 20 years = 100 sim years (1000 real years)
- Shows snapshots after each epoch

**Output**:
- **Row 1**: Elevation at Years 0, 20, 40, 60, 80, 100
- **Row 2**: Surface material (which layer exposed)
- **Row 3**: Erosion depth (cumulative)
- **Plot 2**: Erosion rate analysis
- **Plot 3**: Material exposure tracking

---

## Expected Results

### Initial State (Year 0)
- Surface: Mostly Topsoil (brown) and Subsoil (orange)
- Colluvium (green) in valleys
- Saprolite (purple) on ridges

### Mid-Simulation (Year 50)
- Topsoil eroding rapidly
- Deeper layers exposed on slopes
- Valleys deepening

### Final State (Year 100)
- Topsoil mostly gone
- Weathered Bedrock (pink) on ridges
- Basement (red) exposed in deep valleys
- Colluvium accumulated in low areas

---

## Key Features

### 1. Non-Uniform Rain
```
Rain = base_storm × barrier_factor × channel_factor

Windward slopes (west-facing): 1.5-2.5× more rain
Leeward slopes (east-facing): 0.5-0.8× less rain (rain shadow)
Channels (valleys): 1.2-1.5× more rain (funneling)
```

### 2. Layer-Dependent Erosion
```
Erosion Rate = BASE_K × Q^0.5 × S^1.0 × Erodibility

Topsoil:      Erodibility = 2.0 → erodes quickly
Subsoil:      Erodibility = 1.5 → moderate
Colluvium:    Erodibility = 1.8 → moderate-high
Saprolite:    Erodibility = 1.2 → moderate-low
WeatheredBR:  Erodibility = 0.8 → resistant
Basement:     Erodibility = 0.3 → very resistant
```

### 3. Epoch Visualization
Shows:
- Terrain evolution over time
- Which layers are exposed
- How much erosion occurred
- Erosion rate changes
- Material exposure percentages

---

## Validation

✅ **Non-uniform rain**: Rain map shows wet windward, dry leeward  
✅ **Non-uniform erosion**: Erosion depth map shows valleys erode more  
✅ **Layer exposure**: Material map changes from Topsoil → deeper layers  
✅ **Epoch progression**: Each epoch shows visible change  

---

## Customization

### More/Fewer Epochs
In `cell_19_demonstration_EPOCHS.py`:
```python
num_epochs = 10  # (was 5)
years_per_epoch = 10  # (was 20)
```

### Faster/Slower Erosion
In `cell_10_constants.py`:
```python
RAIN_BOOST = 200.0  # (was 100.0) - double erosion power
TIME_ACCELERATION = 20.0  # (was 10.0) - double time scale
```

### Change Wind Direction
In `cells_00_to_09_WITH_LAYERS.py`:
```python
wind_dir_deg = 180.0  # South wind (was 90.0 for East)
```

---

## Troubleshooting

**"GLOBAL_STRATA not found"**  
→ Run cells 0-9 first (`cells_00_to_09_WITH_LAYERS.py`)

**"Uniform erosion everywhere"**  
→ Check erodibility map has different values, check rain map is non-uniform

**"No visible erosion"**  
→ Increase `RAIN_BOOST` or `BASE_K` in cell 10

**"Erosion too fast"**  
→ Decrease `RAIN_BOOST` or increase `MAX_ERODE_PER_STEP` in cell 10

---

## Runtime

- Terrain generation: ~5-10 s
- Weather simulation: ~10-20 s
- Erosion (per epoch): ~30-60 s
- **Total**: ~5-10 minutes

---

## Why It Works

### Non-Uniform Erosion Sources

1. **Different Materials**  
   Topsoil erodes fast → exposes Basement → erosion slows

2. **Wind-Driven Rain**  
   Windward slopes get more rain → erode faster  
   Leeward slopes get less rain → erode slower

3. **Topographic Feedback**  
   Valleys have high Q → erode fast → deepen  
   Ridges have low Q → erode slow → remain high

### Result
Realistic, non-uniform erosion that:
- Deepens valleys where rain concentrates
- Preserves ridges where hard rock exposed
- Creates drainage networks
- Evolves over time as layers are exposed

---

## 📁 All Files

| # | File | Purpose |
|---|------|---------|
| 0-9 | `cells_00_to_09_WITH_LAYERS.py` | Terrain + Layers + Weather ⭐ |
| 10 | `cell_10_constants.py` | Erosion parameters |
| 11 | `cell_11_flow_direction.py` | Flow direction (D8) |
| 12 | `cell_12_discharge.py` | Discharge (Q) |
| 13 | `cell_13_erosion_pass_a.py` | Erosion (half-loss) |
| 14 | `cell_14_sediment_transport.py` | Transport & deposition |
| 15 | `cell_15_hillslope_diffusion.py` | Hillslope diffusion |
| 16 | `cell_16_river_lake_detection.py` | River/lake detection |
| 17 | `cell_17_main_simulation.py` | Main simulation |
| 18 | `cell_18_visualization.py` | Visualization |
| 19 | `cell_19_demonstration_EPOCHS.py` | Epoch demonstration ⭐ |

⭐ = NEW files with realistic layers and epoch visualization

---

## 🎓 Summary

**Before**: Uniform erosion, single terrain generation, no material variation  
**After**: Non-uniform erosion, realistic layers, epoch-by-epoch visualization

**Result**: Scientifically accurate erosion simulation showing:
- Different materials erode at different rates
- Wind creates rain patterns that drive erosion
- Terrain evolves over time as layers are exposed
- Valleys deepen, ridges resist, drainage networks form

---

## 📖 Full Documentation

See `COMPLETE_SYSTEM_GUIDE.md` for:
- Detailed physics explanations
- Layer generation rules
- Customization options
- Validation checklist
- Scientific references
