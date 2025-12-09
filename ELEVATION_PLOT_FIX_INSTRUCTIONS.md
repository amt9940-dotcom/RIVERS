# How to Fix the "AFTER: Elevation Shows Dots" Issue

## What I've Done

I've added **extensive diagnostic output** to track down the exact bug causing the AFTER elevation plot to show dots instead of a proper terrain map.

### Files Updated:
1. ✅ **NOTEBOOK_CELL_2_erosion_model.py** - Added debug tracking for surface changes
2. ✅ **NOTEBOOK_CELL_3_weather_driven.py** - Added comprehensive diagnostics at every step

### Files Created:
1. 📄 **DEBUGGING_ELEVATION_PLOT.md** - Complete guide to interpreting diagnostic output
2. 📄 **MINIMAL_TEST_CELL.py** - Simple test to verify basic erosion works
3. 📄 **TEST_EROSION.py** - Basic array operation test
4. 📄 **TEST_FULL_EROSION.py** - Test with deep copy

---

## Step-by-Step Instructions

### Step 1: Run the Minimal Test First

This will verify the basic erosion logic works:

1. Create a **new notebook cell**
2. Copy the contents of **`MINIMAL_TEST_CELL.py`** into it
3. Run the cell
4. **Expected result:** You should see 3 plots:
   - BEFORE: A peaked terrain (1000-1200m)
   - AFTER: Same terrain, slightly lower (990-1190m)
   - DIFFERENCE: Red at peak (erosion)

**If this test FAILS:** The problem is fundamental (basic Python/NumPy issue)

**If this test PASSES:** The erosion logic works! The problem is specific to Cell 3.

---

### Step 2: Update Cell 2 and Cell 3

1. **Replace Cell 2** with the new **`NOTEBOOK_CELL_2_erosion_model.py`**
2. **Replace Cell 3** with the new **`NOTEBOOK_CELL_3_weather_driven.py`**
3. **Run Cell 1** (terrain generator)
4. **Run Cell 2** (erosion model)
5. **Run Cell 3** (demo)

---

### Step 3: Read the Diagnostic Output

When Cell 3 runs, you'll see lots of debug output. Look for these key sections:

#### ✓ Section: "Initial state captured"
```
✓ Initial state captured:
  Surface range: 800.23 - 1187.45 m    ← Should be in km range
  Surface mean: 1003.21 m
```

#### ✓ Section: "First epoch diagnostics"
```
DEBUG: First epoch diagnostics:
  Erosion (channel): 0.023456 m avg, 2.345678 m max
  Total erosion: 0.024690 m avg, 2.345678 m max
  Deposition: 0.012345 m avg, 1.234567 m max
  Surface change: -2.345678 to +1.234567 m
```

#### ✓ Section: "Final state"
```
✓ Final state:
  Surface range: 795.12 - 1190.23 m    ← Should STILL be in km range!
  Surface mean: 1001.06 m
  Has NaN: False
  Has Inf: False
```

#### ✓ Section: "Change statistics"
```
✓ Change statistics:
  Min change: -50.23 m
  Max change: +30.45 m
  Mean change: -2.15 m
  Cells changed: 2467 / 2500
```

#### ✓ Section: "Verifying data integrity"
```
DEBUG: Verifying data integrity before plotting:
  Are they the same object? False    ← MUST be False!
  Initial mean: 1003.21 m
  Final mean: 1001.06 m
  Difference: -2.15 m                ← Should be negative (net erosion)
```

---

## What Different Outputs Mean

### 🟢 GOOD OUTPUT (Everything Working):
```
Initial surface: 800-1200 m
Final surface: 795-1205 m
Mean change: -5.23 m
Are they the same object? False
```
✅ **Result:** AFTER plot should show a proper terrain map, slightly modified

---

### 🔴 BAD OUTPUT #1 (Plotting BasementFloor):
```
Final surface: -50000.0 - -48000.0 m    ← HUGE NEGATIVE!
```
❌ **Problem:** Plotting the wrong array (BasementFloor interface instead of surface)

**Fix:** Check line ~417 in Cell 3:
```python
# Should be:
ax.imshow(strata["surface_elev"], ...)
# NOT:
ax.imshow(strata["interfaces"]["BasementFloor"], ...)
```

---

### 🔴 BAD OUTPUT #2 (Plotting Erosion Depth):
```
Final surface: -5.0 - +3.0 m    ← Small positive/negative values
Mean: -2.15 m
```
❌ **Problem:** Plotting erosion depth, not elevation

**Fix:** Look for a line like:
```python
after_elev = np.zeros_like(...) - erosion    # WRONG!
# Should be:
after_elev = surface_elev.copy() - erosion   # RIGHT!
```

---

### 🔴 BAD OUTPUT #3 (Shallow Copy):
```
Are they the same object? True    ← Both point to same array!
Difference: 0.00 m
```
❌ **Problem:** Shallow copy instead of deep copy

**Fix:** Line ~163 in Cell 3:
```python
# Should be:
strata_initial = copy.deepcopy(strata)
# NOT:
strata_initial = strata.copy()       # Too shallow!
strata_initial = strata              # Just a reference!
```

---

### 🔴 BAD OUTPUT #4 (Erosion Too Small):
```
Total erosion: 0.000023 m avg    ← Tiny!
Mean change: -0.000023 m
```
❌ **Problem:** Erosion coefficients too small, or dt too small

**Fix:** Line ~289-296 in Cell 3, increase parameters:
```python
K_channel = 1e-5        # Try 1e-4 or 1e-3 (bigger = more erosion)
D_hillslope = 0.01      # Try 0.1 (bigger = more smoothing)
dt = 1000.0             # Try 10000.0 (longer timesteps)
num_epochs = 25         # Try 50 or 100 (more iterations)
```

---

### 🔴 BAD OUTPUT #5 (Balanced by Uplift):
```
Total erosion: 5.00 m avg
Mean change: 0.00 m    ← Net change is zero!
```
❌ **Problem:** Uplift cancels out erosion

**Fix:** Line ~291 in Cell 3, reduce uplift:
```python
uplift_rate = 0.0001    # Try 0.0 to see pure erosion first
```

---

## Common Bug Signatures (Quick Reference)

| Symptom | Likely Cause |
|---------|-------------|
| Final surface: **-50000 to -48000 m** | Plotting BasementFloor instead of surface |
| Final surface: **-5 to +5 m** | Plotting erosion depth instead of elevation |
| Same object? **True** | Shallow copy (not deep copy) |
| Mean change: **~0.00001 m** | Erosion coefficients too small |
| Mean change: **0.00 m exactly** | Uplift balances erosion |
| Has NaN: **True** | Numerical instability (check discharge) |
| Cells changed: **0** | No erosion happened at all |

---

## What to Send Me

If it still doesn't work, paste these specific lines from the output:

```
1. Initial surface range: ??? - ??? m
2. Final surface range: ??? - ??? m
3. Mean change: ??? m
4. Are they the same object? ???
5. First epoch erosion: ??? m
```

With just those 5 lines, I can tell you EXACTLY what's wrong!

---

## Expected Behavior (When Working)

**BEFORE Plot:**
- Shows full terrain with mountains and valleys
- Range: ~800-1200 m
- Color: Brown mountains, green valleys (terrain colormap)

**AFTER Plot:**
- Shows similar terrain but subtly modified
- Range: ~795-1205 m (close to BEFORE, but not identical)
- Color: Same terrain colormap
- **NOT dots!** Should be a continuous elevation map

**DIFFERENCE Plot:**
- Shows red (erosion) in channels and valleys
- Shows blue (deposition) in basins
- Shows white (no change) in flat areas
- Range: -50 to +30 m (much smaller than elevation range)

---

## Next Steps

1. ✅ Run **MINIMAL_TEST_CELL.py** first to verify basics work
2. ✅ Update Cell 2 and Cell 3 with the new diagnostic versions
3. ✅ Run the updated cells and read the output
4. ✅ Read **DEBUGGING_ELEVATION_PLOT.md** for detailed explanations
5. ✅ Send me the 5 key diagnostic lines if it still doesn't work

Good luck! The diagnostics will reveal exactly what's wrong.
