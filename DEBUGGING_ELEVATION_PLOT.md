# Debugging the "AFTER: Elevation" Plot Issue

## What I've Added

I've added extensive diagnostic output to **NOTEBOOK_CELL_2_erosion_model.py** and **NOTEBOOK_CELL_3_weather_driven.py** to track down why the AFTER elevation plot might show dots instead of a proper terrain map.

## Run the Updated Cells

1. **Update Cell 2** with the new `NOTEBOOK_CELL_2_erosion_model.py`
2. **Update Cell 3** with the new `NOTEBOOK_CELL_3_weather_driven.py`
3. **Run both cells**

## What to Look For in the Output

### Section 1: Initial State Check
```
✓ Initial state captured:
  Surface range: 800.00 - 1200.00 m
  Surface mean: 1000.00 m
```
This verifies the initial terrain is correct (hundreds to thousands of meters).

---

### Section 2: First Epoch Diagnostics
```
DEBUG: First epoch diagnostics:
  Erosion (channel): 0.023456 m avg, 2.345678 m max
  Erosion (hillslope): 0.001234 m avg, 0.056789 m max
  Total erosion: 0.024690 m avg, 2.345678 m max
  Deposition: 0.012345 m avg, 1.234567 m max
  Surface change: -2.345678 to +1.234567 m
  Mean change: -0.012345 m
```

**What this tells you:**
- **If erosion/deposition values are ~0.0**: The simulation isn't actually eroding/depositing much
- **If surface change is ~0.0**: The terrain isn't changing (erosion too small, or balanced by deposition/uplift)

---

### Section 3: Final State Check
```
✓ Final state:
  Surface range: 795.00 - 1205.00 m    ← Should still be in km range!
  Surface mean: 1000.50 m
  Has NaN: False
  Has Inf: False
  BasementFloor range: -50000.0 - -48000.0 m (for comparison)
```

**RED FLAGS:**
- ❌ If surface range is **-100000 to -99000**: You're plotting the **BasementFloor** by mistake!
- ❌ If surface range is **-5.0 to +5.0**: You're plotting **erosion depth**, not elevation!
- ❌ If "Has NaN" is **True**: The simulation produced invalid values
- ✓ If surface range is **800-1200 m**: This is CORRECT (similar to initial)

---

### Section 4: Change Statistics
```
✓ Change statistics:
  Min change: -50.23 m
  Max change: +30.45 m
  Mean change: -2.15 m
  Cells changed: 2467 / 2500
```

**What this tells you:**
- **Min/Max change**: Shows the range of erosion and deposition
- **Mean change**: Net elevation change (should be small but negative if erosion > uplift)
- **Cells changed**: How many cells were modified (should be most of them)

**RED FLAGS:**
- ❌ If mean change is **-10000 m**: Erosion was applied with wrong units!
- ❌ If cells changed is **0 / 2500**: No erosion happened at all!

---

### Section 5: Array Identity Check
```
DEBUG: Verifying data integrity before plotting:
  strata_initial['surface_elev'] id: 140234567890
  strata['surface_elev'] id: 140234567999
  Are they the same object? False
  Initial mean: 1000.00 m
  Final mean: 997.85 m
  Difference: -2.15 m
```

**RED FLAGS:**
- ❌ If "Are they the same object?" is **True**: The deep copy failed! Both arrays point to the same data!
- ❌ If "Difference" is **0.00 m**: The terrain didn't change at all!

---

## Common Bugs and Their Signatures

### Bug 1: Plotting BasementFloor Instead of Surface
**Signature:**
```
Surface range: -50000.0 - -48000.0 m
```
**What to check:** Make sure the plot uses `strata["surface_elev"]`, not `strata["interfaces"]["BasementFloor"]`

---

### Bug 2: Plotting Erosion Depth Instead of Elevation
**Signature:**
```
Surface range: -50.0 - +30.0 m
Mean: -2.15 m
```
**What to check:** Look for a line like `after_elev = np.zeros_like(...) - erosion`

---

### Bug 3: Shallow Copy (Not Deep Copy)
**Signature:**
```
Are they the same object? True
Difference: 0.00 m
```
**What to check:** Make sure you use `copy.deepcopy(strata)`, not `strata.copy()` or `strata_initial = strata`

---

### Bug 4: Erosion Too Small to See
**Signature:**
```
Total erosion: 0.000023 m avg
Mean change: -0.000023 m
```
**What to check:** Increase `K_channel` or `D_hillslope`, or run more epochs, or increase `dt`

---

### Bug 5: Erosion Balanced by Uplift
**Signature:**
```
Total erosion: 5.00 m avg
Deposition: 2.00 m avg
Mean change: 0.00 m
```
**What to check:** Uplift is too strong, set `uplift_rate = 0` to see pure erosion

---

## What I Expect

If everything is working correctly, you should see:
- ✓ Initial surface range: ~800-1200 m
- ✓ Final surface range: ~795-1205 m (slightly different)
- ✓ Mean change: ~-2 to -10 m (small but measurable)
- ✓ Arrays are separate objects
- ✓ AFTER plot shows a terrain map similar to BEFORE but with subtle changes

If the AFTER plot still shows "dots," **paste the diagnostic output** and I'll tell you exactly what's wrong!

---

## Quick Fix Checklist

Before running, verify in Cell 3:

```python
# ✓ Deep copy (not shallow)
strata_initial = copy.deepcopy(strata)

# ✓ Simulation modifies strata in place
history = run_erosion_simulation(strata=strata, ...)

# ✓ Plot the correct arrays
ax.imshow(strata_initial["surface_elev"], ...)  # BEFORE
ax.imshow(strata["surface_elev"], ...)          # AFTER (not strata_initial!)
```

---

## Next Steps

1. Run the updated cells
2. Look at the diagnostic output
3. Tell me what the ranges are for:
   - Initial surface range
   - Final surface range
   - Mean change
   - First epoch erosion/deposition

I'll pinpoint the exact bug based on those numbers!
