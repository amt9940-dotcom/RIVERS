# ⚡ QUICK START GUIDE

## ✅ **ALL ISSUES FIXED - FINAL VERSION**

### **What's Corrected:**
1. ✅ **One terrain map** (not two)
2. ✅ **Only 4 layers** (no unused code)
3. ✅ **Project33-style terrain**
4. ✅ **Wind goes EAST** (to the right)
5. ✅ **Barriers**: Wet windward, dry leeward (rain shadow)
6. ✅ **Channels**: Rain funneled in valleys
7. ✅ **Quantum random rain** within each storm
8. ✅ **Weather drives erosion** (not artificial)

---

## 📦 **FILES TO USE (12 Total)**

### **CELL 0** (Terrain + Weather):
```
cells_00_to_09_FINAL.py (21 KB)
```
- Generates terrain ONCE
- Generates 100 years of weather
- Wind: EAST (90°)
- Physics: Windward wet, leeward dry
- Creates: GLOBAL_STRATA, GLOBAL_RAIN_TIMESERIES

### **CELLS 1-9** (Erosion Modules):
```
cell_10_constants.py
cell_11_flow_direction.py
cell_12_discharge.py
cell_13_erosion_pass_a.py
cell_14_sediment_transport.py
cell_15_hillslope_diffusion.py
cell_16_river_lake_detection.py
cell_17_main_simulation.py
cell_18_visualization.py
```

### **CELL 10** (Demonstration):
```
cell_19_demonstration_FIXED.py (13 KB)
```
- Uses terrain from GLOBAL_STRATA
- Uses weather from GLOBAL_RAIN_TIMESERIES
- Validates integration
- Shows results

### **CELL 11** (Documentation):
```
cell_20_documentation.md
```
- User guide (markdown)

---

## 🚀 **SUPER QUICK SETUP (3 Steps)**

### **Step 1**: Create 12 Cells in Jupyter
- 11 CODE cells
- 1 MARKDOWN cell (last one)

### **Step 2**: Copy & Paste Files
1. `cells_00_to_09_FINAL.py` → Cell 0
2. `cell_10_constants.py` → Cell 1
3. `cell_11_flow_direction.py` → Cell 2
4. `cell_12_discharge.py` → Cell 3
5. `cell_13_erosion_pass_a.py` → Cell 4
6. `cell_14_sediment_transport.py` → Cell 5
7. `cell_15_hillslope_diffusion.py` → Cell 6
8. `cell_16_river_lake_detection.py` → Cell 7
9. `cell_17_main_simulation.py` → Cell 8
10. `cell_18_visualization.py` → Cell 9
11. `cell_19_demonstration_FIXED.py` → Cell 10
12. `cell_20_documentation.md` → Cell 11 (MARKDOWN!)

### **Step 3**: Run in Order
```
Cell 0  → Run (30s)   - Generates terrain & weather
Cells 1-9  → Run each (instant) - Loads functions
Cell 10 → Run (5-10 min) - Erosion simulation
Cell 11 → Run (instant) - Documentation
```

---

## 📊 **WHAT YOU'LL SEE**

### **Cell 0 Output:**
```
✓ Terrain generated: 256×256
✓ Weather generated: 100 years
  Wind direction: 90° (EAST → to the right)
  Wind barriers: 8234 cells
  Wind channels: 12456 cells
✓ GLOBAL_STRATA created
✓ GLOBAL_RAIN_TIMESERIES created

[Shows 6 plots]:
1. Terrain elevation
2. Barrier score (ridges in red)
3. Channel score (valleys in blue)
4. Total rain (100 years)
5. Windward vs Leeward (red=wet, blue=dry)
6. Year 1 rain
```

### **Cell 10 Output:**
```
✓ Found GLOBAL_STRATA (terrain from Cell 0)
✓ Found GLOBAL_RAIN_TIMESERIES (weather from Cell 0)
✓ Terrain matches: True ✅
✓ Weather matches: True ✅

[Runs erosion simulation...]

VALIDATION:
  ✓ Same terrain used
  ✓ Same weather used
  ✓ Windward slopes: MORE rain
  ✓ Leeward slopes: LESS rain (rain shadow)
  ✓ Channels: Rain funneled
  ✓ Rivers form in valleys

[Shows 10+ plots]
```

---

## 🌬️ **WIND PHYSICS (Key Points)**

### **Wind Direction:**
```
        North ↑
             |
West ← - - - + - - - → EAST (Wind direction: 90°)
             |
        South ↓
```

### **Barrier Effect (Mountains/Ridges):**
```
Rain ↓↓↓     Rain ↓
  ████        ░░
Wind→ /\    Wind→ /\
   West East  West East
   (Wet)(Dry) (Wet)(Dry)
```
- **West slopes** (windward): **MORE rain**
- **East slopes** (leeward): **LESS rain** (rain shadow)

### **Channel Effect (Valleys):**
```
     Rain ↓↓↓↓↓
Wind → ═════════ (Valley aligned with wind)
       ↓↓↓↓↓
     Heavy rain along valley
```
- **Along valleys**: Rain **concentrated**
- **At junctions**: Rain **hotspots**

---

## 🔍 **VERIFICATION CHECKLIST**

After running Cell 10, check:

- [ ] Console shows "Terrain matches: True"
- [ ] Console shows "Weather matches: True"
- [ ] Plots show wind arrow pointing EAST →
- [ ] West slopes have more rain than east slopes
- [ ] Clear rain shadow visible on east side of ridges
- [ ] Valleys show streaks of concentrated rain
- [ ] Rivers eventually align with valleys
- [ ] Combined effects show 2-4× rain variation

---

## 🔧 **CUSTOMIZATION**

### **Change Grid Size** (Cell 0 line ~490):
```python
N = 128   # Faster (1-2 min erosion)
N = 256   # Default (5-10 min erosion)
N = 512   # Detailed (30-60 min erosion)
```

### **Change Wind Direction** (Cell 0 line ~495):
```python
wind_dir_deg = 0.0    # North
wind_dir_deg = 90.0   # EAST (default)
wind_dir_deg = 180.0  # South
wind_dir_deg = 270.0  # West
```

### **Change Erosion Strength** (Cell 1):
```python
TIME_ACCELERATION = 20.0  # 2× faster
RAIN_BOOST = 200.0        # 2× stronger
```

### **Change Wind Effects** (Cell 0 around line 300):
```python
k_windward = 1.2  # Stronger windward boost
k_lee = 0.9       # Stronger rain shadow
k_channel = 0.8   # Stronger valley funneling
```

---

## 📚 **DOCUMENTATION FILES**

- **`QUICK_START.md`** ← You are here!
- **`WIND_PHYSICS_EXPLAINED.md`** ← Physics details
- **`README_FINAL.md`** ← Complete overview
- **`UPDATED_INSTALLATION_GUIDE.md`** ← Detailed setup

---

## ⚠️ **COMMON MISTAKES**

### **Mistake 1: Running Cell 0 Twice**
❌ **Problem**: Generates NEW terrain (different map)
✅ **Solution**: Restart kernel if you ran Cell 0 twice

### **Mistake 2: Wrong File**
❌ **Problem**: Using old `cells_00_to_09_REFACTORED_v2.py`
✅ **Solution**: Use `cells_00_to_09_FINAL.py` (has corrected wind physics)

### **Mistake 3: Cell 11 as Code**
❌ **Problem**: Markdown file in CODE cell
✅ **Solution**: Change Cell 11 to MARKDOWN type

### **Mistake 4: Skipping Cells 1-9**
❌ **Problem**: Erosion functions not loaded
✅ **Solution**: Run ALL cells in order (0→1→2→...→10)

---

## 🎯 **SUCCESS CRITERIA**

You know it's working when:

1. ✅ Cell 0 shows "GLOBAL VARIABLES CREATED"
2. ✅ Cell 0 plots show wind arrow pointing EAST →
3. ✅ Cell 0 plots show red (wet) west slopes, blue (dry) east slopes
4. ✅ Cells 1-9 all show "✅ module loaded!"
5. ✅ Cell 10 shows "Terrain matches: True"
6. ✅ Cell 10 shows "Weather matches: True"
7. ✅ Cell 10 shows windward>leeward rain difference
8. ✅ Cell 10 shows rivers in valleys
9. ✅ All plots render correctly
10. ✅ No errors in any cell

---

## 🎉 **YOU'RE DONE!**

If all checks pass, you have:
- ✅ ONE terrain map (used everywhere)
- ✅ Correct wind physics (EAST wind, wet windward, dry leeward)
- ✅ Rain funneled in valleys
- ✅ Quantum random rain
- ✅ Complete erosion simulation
- ✅ Rivers and lakes
- ✅ Full validation

**Start copying files into Jupyter now!** 🚀

---

## 📞 **NEED HELP?**

1. Check console for error messages
2. Verify you're using `cells_00_to_09_FINAL.py` (not old version)
3. Make sure Cell 0 ran successfully
4. Read `WIND_PHYSICS_EXPLAINED.md` for physics details
5. Check `README_FINAL.md` for complete overview

---

**Time to completion: ~15 minutes** (3 min setup + 10 min computation + 2 min review)

**Good luck!** 🌟
