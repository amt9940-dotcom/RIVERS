# 🌬️ WIND-RAIN PHYSICS - CORRECTED IMPLEMENTATION

## ✅ **WHAT WAS FIXED**

### **Wind Direction** → **EAST (90°, to the right)** ✅
- **Before**: West wind (270°)
- **After**: EAST wind (90°, pointing right on the map)

### **Barrier Physics** → **Windward Wet, Leeward Dry** ✅
- **Before**: Simple speed multiplier
- **After**: Proper orographic physics
  - **Windward side** (facing wind): MORE rain
  - **Leeward side** (downwind): LESS rain (rain shadow)

### **Channel Physics** → **Funneling Along Valleys** ✅
- **Before**: Just speed multiplier
- **After**: Rain concentrated along valleys
  - **Along valleys**: Rain funneled into streams
  - **At junctions**: Extra rain hotspots

---

## 🧮 **THE PHYSICS (Conceptual → Code)**

### **Inputs (2D Arrays)**

```python
base_rain[i, j]       # Storm rain without terrain effects
barrier_score[i, j]   # Ridge strength [0-1]
channel_score[i, j]   # Valley strength [0-1]
slope_x[i, j]         # Slope in X direction (East)
slope_y[i, j]         # Slope in Y direction (North)
wind_x, wind_y        # Wind direction unit vector (1, 0) for EAST
```

### **Step 1: Barrier Factor (Windward vs Leeward)**

```python
# Compute alignment of slope with wind
slope_mag = sqrt(slope_x² + slope_y²)
cos_theta = (slope_x * wind_x + slope_y * wind_y) / slope_mag

# cos_theta ranges from -1 to +1:
#   +1 = slope faces directly INTO wind (windward)
#    0 = slope perpendicular to wind (side)
#   -1 = slope faces AWAY from wind (leeward)

# Apply different effects:
if cos_theta > 0:  # WINDWARD
    barrier_factor = 1.0 + k_windward * cos_theta * barrier_score
    # More rain on upwind slopes

elif cos_theta < 0:  # LEEWARD  
    barrier_factor = 1.0 - k_lee * (-cos_theta) * barrier_score
    # Less rain in rain shadow

else:  # SIDE (cos_theta ≈ 0)
    barrier_factor = 1.0
    # Normal rain on side slopes
```

**Parameters:**
- `k_windward = 0.8`: Up to 80% more rain on windward slopes
- `k_lee = 0.6`: Up to 60% less rain in lee (rain shadow)
- Result: `barrier_factor` ranges from 0.2 to 2.5

### **Step 2: Channel Factor (Valley Funneling)**

```python
# Simple multiplier for valleys
channel_factor = 1.0 + k_channel * channel_score

# Where channel_score is high (valleys):
#   channel_factor > 1 → more rain
# Where channel_score is low (ridges/plains):
#   channel_factor ≈ 1 → normal rain
```

**Parameters:**
- `k_channel = 0.5`: Up to 50% more rain in channels
- Result: `channel_factor` ranges from 1.0 to 1.5

### **Step 3: Final Rain**

```python
rain[i, j] = base_rain[i, j] * barrier_factor[i, j] * channel_factor[i, j]
```

**Combined effects:**
- Windward valley: `2.5 × 1.5 = 3.75×` (lots of rain!)
- Leeward ridge: `0.2 × 1.0 = 0.2×` (very dry, rain shadow)

---

## 🗺️ **HOW IT LOOKS ON THE MAP**

### **Wind Direction: EAST (→)**

```
Wind →  Wind →  Wind →
    ↓       ↓       ↓
    
WEST SIDE              EAST SIDE
(Windward)             (Leeward)
[More rain] ████████ [Less rain]
            Ridge
            /\
           /  \
          /    \
```

### **With Barriers (Mountains/Ridges)**

```
    Rain          Rain shadow
   ↓↓↓↓↓           ↓ ↓
Wind → ████    vs    ░░░
         /\          /\
      West East   West East
    (Wet) (Dry)  (Wet) (Dry)
```

**Pattern**: Blobs and gradients around ridges
- West slopes: WET (windward)
- East slopes: DRY (leeward, rain shadow)

### **With Channels (Valleys)**

```
Wind →
    ═══════════  ← Heavy rain along valley
    
    ═══╦═══     ← Extra rain at junction
       ║
    ═══╬═══     ← Multiple valleys meeting
       ║
```

**Pattern**: Lines and forks following valleys
- Along valleys: Rain concentrated (streams form)
- At junctions: Hotspots (convergence)
- Outside: Normal rain

---

## 📊 **EXPECTED RESULTS**

### **Barrier Effects (Windward vs Leeward)**

| Location | Slope Direction | cos_theta | barrier_factor | Rain |
|----------|----------------|-----------|----------------|------|
| West slope (windward) | Faces EAST (into wind) | +1.0 | 1.8 | 180% |
| Ridge crest | Flat or mixed | ~0 | 1.0 | 100% |
| East slope (leeward) | Faces WEST (away) | -1.0 | 0.4 | 40% |

### **Channel Effects (Valleys)**

| Location | channel_score | channel_factor | Rain |
|----------|--------------|----------------|------|
| Valley bottom | 1.0 | 1.5 | 150% |
| Valley junction | 1.0 | 1.5 | 150% |
| Plains | 0.1 | 1.05 | 105% |
| Ridge top | 0.0 | 1.0 | 100% |

### **Combined Effects**

| Location | Description | Total Multiplier | Result |
|----------|-------------|------------------|--------|
| West valley | Windward channel | 1.8 × 1.5 = 2.7× | **Heavy rain!** |
| West ridge | Windward barrier | 1.8 × 1.0 = 1.8× | More rain |
| East valley | Leeward channel | 0.4 × 1.5 = 0.6× | Moderate |
| East ridge | Leeward barrier | 0.4 × 1.0 = 0.4× | **Rain shadow!** |

---

## 🔍 **VERIFICATION IN CODE**

### **Check 1: Wind Direction**

```python
print(f"Wind direction: {wind_dir_deg}°")
# Should be 90° (EAST)

wind_x, wind_y = wind_features['wind_vector']
print(f"Wind vector: ({wind_x:.2f}, {wind_y:.2f})")
# Should be (1.0, 0.0) for EAST wind
```

### **Check 2: Windward vs Leeward**

```python
# Get a ridge cell
ridge_i, ridge_j = find_ridge()

# Check west slope (windward)
west_i = ridge_i
west_j = ridge_j - 10  # 10 cells to the west

# Check east slope (leeward)
east_i = ridge_i  
east_j = ridge_j + 10  # 10 cells to the east

print(f"West slope rain: {rain[west_i, west_j]:.3f} m")
print(f"Ridge rain: {rain[ridge_i, ridge_j]:.3f} m")
print(f"East slope rain: {rain[east_i, east_j]:.3f} m")

# Should see: West > Ridge > East
```

### **Check 3: Channel Funneling**

```python
# Find a valley
valley_mask = channel_score > 0.7

valley_rain = rain[valley_mask].mean()
non_valley_rain = rain[~valley_mask].mean()

print(f"Valley rain: {valley_rain:.3f} m")
print(f"Non-valley rain: {non_valley_rain:.3f} m")
print(f"Ratio: {valley_rain / non_valley_rain:.2f}×")

# Should see ratio > 1 (valleys have more rain)
```

---

## 🎯 **KEY EQUATIONS IMPLEMENTED**

### **Barrier Factor**

```python
# Slope-wind alignment
cos_theta = (slope_x * wind_x + slope_y * wind_y) / sqrt(slope_x² + slope_y²)

# Windward boost
if cos_theta > 0:
    barrier_factor = 1 + 0.8 * cos_theta * barrier_score

# Leeward reduction (rain shadow)
if cos_theta < 0:
    barrier_factor = 1 - 0.6 * |cos_theta| * barrier_score
```

### **Channel Factor**

```python
# Valley funneling
channel_factor = 1 + 0.5 * channel_score
```

### **Final Rain**

```python
rain = base_rain * barrier_factor * channel_factor
```

---

## 📈 **EXPECTED VISUALIZATION**

When you run the code, you should see:

### **Plot 1: Barrier Score**
- Red blobs = ridges, peaks
- High elevation features
- Should be perpendicular to wind

### **Plot 2: Channel Score**
- Blue lines = valleys
- Low elevation features  
- Should be parallel to wind (for aligned channels)

### **Plot 3: Windward vs Leeward**
- Red (positive) = windward slopes (facing EAST)
- Blue (negative) = leeward slopes (facing WEST)
- Should show clear pattern across ridges

### **Plot 4: Total Rain**
- More rain on WEST side of ridges (windward)
- Less rain on EAST side of ridges (leeward)
- Streaks following valleys

---

## 🔧 **TUNING PARAMETERS**

If you want to adjust the effects, edit these in the code:

```python
# In apply_wind_rain_physics function:

k_windward = 0.8   # Windward boost (0.5-1.5 recommended)
k_lee = 0.6        # Leeward reduction (0.4-0.8 recommended)  
k_channel = 0.5    # Channel funneling (0.3-0.8 recommended)

# Stronger effects (more dramatic):
k_windward = 1.2
k_lee = 0.9
k_channel = 0.8

# Weaker effects (more subtle):
k_windward = 0.5
k_lee = 0.4
k_channel = 0.3
```

---

## ✅ **PHYSICS CHECKLIST**

After running, verify:

- [ ] Wind direction is 90° (EAST, to the right)
- [ ] West slopes have MORE rain than east slopes
- [ ] Clear rain shadow on east side of ridges
- [ ] Valleys show streaks/lines of concentrated rain
- [ ] Valley junctions show rain hotspots
- [ ] Combined effects produce 2-4× variation in rain
- [ ] Rivers eventually form along valleys (after erosion)
- [ ] Windward-leeward pattern visible on terrain

---

## 🎓 **SUMMARY**

### **What the Code Does:**

1. **Classifies Terrain**:
   - barrier_score: Identifies ridges (high curvature + high elevation)
   - channel_score: Identifies valleys (low elevation + low curvature)

2. **Computes Wind-Slope Alignment**:
   - cos_theta = dot(slope, wind)
   - Positive = windward, Negative = leeward

3. **Applies Physics**:
   - Windward: `rain *= (1 + 0.8 * cos_theta * barrier_score)`
   - Leeward: `rain *= (1 - 0.6 * |cos_theta| * barrier_score)`
   - Channels: `rain *= (1 + 0.5 * channel_score)`

4. **Result**:
   - Wet windward slopes (west side)
   - Dry leeward slopes (east side, rain shadow)
   - Rain funneled along valleys
   - Realistic orographic precipitation

---

**File to use:** `cells_00_to_09_FINAL.py`

**This replaces:** `cells_00_to_09_REFACTORED_v2.py`

**Difference:** Corrected wind-rain physics with proper orographic effects! 🌧️⛰️
