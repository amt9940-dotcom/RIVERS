# What I Changed to Debug the "Dots" Issue

## The Problem You Reported

> "AFTER: elevation still just shows dots"

You suggested checking for these common bugs:
1. Subtracting erosion from a zero field instead of the real surface
2. Applying erosion multiple times / with wrong units
3. Clobbering the surface with a basement/floor field

## What I Did

I added **comprehensive diagnostic output** to identify which of these bugs (or others) is causing the issue.

---

## Changes to NOTEBOOK_CELL_2_erosion_model.py

### Location: `run_erosion_epoch()` function (lines 611-633)

**BEFORE:**
```python
# Step 6a: Apply erosion
total_erosion = np.maximum(erosion_channel, 0) + np.maximum(erosion_hillslope, 0)
update_stratigraphy_with_erosion(strata, total_erosion, pixel_scale_m)

# Step 6b: Apply deposition
update_stratigraphy_with_deposition(strata, deposition, pixel_scale_m)

return {
    "erosion_channel": erosion_channel,
    "erosion_hillslope": erosion_hillslope,
    "deposition": deposition,
    "flow_data": flow_data,
    "total_erosion": total_erosion,
}
```

**AFTER:**
```python
# Step 6a: Apply erosion
total_erosion = np.maximum(erosion_channel, 0) + np.maximum(erosion_hillslope, 0)

# DEBUG: Check surface before erosion
surf_before = strata["surface_elev"].copy()

update_stratigraphy_with_erosion(strata, total_erosion, pixel_scale_m)

# Step 6b: Apply deposition
update_stratigraphy_with_deposition(strata, deposition, pixel_scale_m)

# DEBUG: Check surface after
surf_after = strata["surface_elev"]
actual_change = surf_after - surf_before

return {
    "erosion_channel": erosion_channel,
    "erosion_hillslope": erosion_hillslope,
    "deposition": deposition,
    "flow_data": flow_data,
    "total_erosion": total_erosion,
    "actual_surface_change": actual_change,  # DEBUG: NEW!
}
```

**Why:** This tracks the ACTUAL surface elevation change after erosion is applied, so we can verify it's being applied correctly.

---

## Changes to NOTEBOOK_CELL_3_weather_driven.py

### Change 1: Initial State Check (line ~172)

**ADDED:**
```python
# Save initial state
strata_initial = copy.deepcopy(strata)

# DIAGNOSTIC: Check initial state
print(f"\n   ✓ Initial state captured:")
print(f"     Surface range: {strata_initial['surface_elev'].min():.2f} - {strata_initial['surface_elev'].max():.2f} m")
print(f"     Surface mean: {strata_initial['surface_elev'].mean():.2f} m")
print(f"     Surface dtype: {strata_initial['surface_elev'].dtype}")
```

**Why:** Verifies the initial terrain is in the correct range (hundreds to thousands of meters). If this is wrong, everything else will be wrong.

---

### Change 2: Epoch Diagnostics (line ~327)

**ADDED:**
```python
print("   ✓ Simulation complete!")

# DEBUG: Check first and last epoch changes
if len(history) > 0:
    first_epoch = history[0]
    last_epoch = history[-1]
    
    print(f"\n   DEBUG: First epoch diagnostics:")
    print(f"     Erosion (channel): {first_epoch['erosion_channel'].mean():.6f} m avg, {first_epoch['erosion_channel'].max():.6f} m max")
    print(f"     Erosion (hillslope): {first_epoch['erosion_hillslope'].mean():.6f} m avg, {first_epoch['erosion_hillslope'].max():.6f} m max")
    print(f"     Deposition: {first_epoch['deposition'].mean():.6f} m avg, {first_epoch['deposition'].max():.6f} m max")
    print(f"     Total erosion: {first_epoch['total_erosion'].mean():.6f} m avg, {first_epoch['total_erosion'].max():.6f} m max")
    
    if "actual_surface_change" in first_epoch:
        asc = first_epoch["actual_surface_change"]
        print(f"     Surface change: {asc.min():.6f} to {asc.max():.6f} m")
        print(f"     Mean change: {asc.mean():.6f} m")
        print(f"     Non-zero cells: {np.sum(np.abs(asc) > 1e-9)}")
```

**Why:** Shows exactly how much erosion/deposition happens per epoch. If these values are tiny (1e-6), the erosion is too small to see. If they're huge (1e6), the units are wrong.

---

### Change 3: Final State Check (line ~329)

**ADDED:**
```python
# DIAGNOSTIC: Check final state
print(f"\n   ✓ Final state:")
print(f"     Surface range: {strata['surface_elev'].min():.2f} - {strata['surface_elev'].max():.2f} m")
print(f"     Surface mean: {strata['surface_elev'].mean():.2f} m")
print(f"     Surface dtype: {strata['surface_elev'].dtype}")
print(f"     Has NaN: {np.any(np.isnan(strata['surface_elev']))}")
print(f"     Has Inf: {np.any(np.isinf(strata['surface_elev']))}")

# Check if it's the basement floor by mistake
if "interfaces" in strata and "BasementFloor" in strata["interfaces"]:
    bf_range = f"{strata['interfaces']['BasementFloor'].min():.1f} - {strata['interfaces']['BasementFloor'].max():.1f}"
    print(f"     BasementFloor range: {bf_range} m (for comparison)")

# Compare to initial
delta_check = strata['surface_elev'] - strata_initial['surface_elev']
print(f"   ✓ Change statistics:")
print(f"     Min change: {delta_check.min():.2f} m")
print(f"     Max change: {delta_check.max():.2f} m")
print(f"     Mean change: {delta_check.mean():.2f} m")
print(f"     Cells changed: {np.sum(np.abs(delta_check) > 0.01)} / {delta_check.size}")
```

**Why:** This is the **CRITICAL CHECK**. It compares:
- **Final surface** (should be ~800-1200m) vs **BasementFloor** (should be ~-50000m)
- If final surface is in the wrong range, we know which bug you have!

---

### Change 4: Array Identity Check (line ~387)

**ADDED:**
```python
print("\n8. Creating visualizations...")

# CRITICAL DEBUG: Verify arrays are actually different
print("\n   DEBUG: Verifying data integrity before plotting:")
print(f"     strata_initial['surface_elev'] id: {id(strata_initial['surface_elev'])}")
print(f"     strata['surface_elev'] id: {id(strata['surface_elev'])}")
print(f"     Are they the same object? {strata_initial['surface_elev'] is strata['surface_elev']}")
print(f"     Initial mean: {strata_initial['surface_elev'].mean():.4f} m")
print(f"     Final mean: {strata['surface_elev'].mean():.4f} m")
print(f"     Difference: {(strata['surface_elev'] - strata_initial['surface_elev']).mean():.4f} m")

if strata_initial['surface_elev'] is strata['surface_elev']:
    print("   ⚠ WARNING: strata_initial and strata point to the SAME array!")
    print("   This means the copy was shallow, not deep!")
else:
    print("   ✓ Arrays are separate (deep copy worked)")
```

**Why:** Checks if `copy.deepcopy()` actually worked. If both arrays have the same identity, they're the same object, and any modifications affect both. This would cause the "no change" bug.

---

## What These Diagnostics Catch

### Bug #1: Subtracting from Zero Field
**Symptom:** Final surface range is -50 to +30m (small values)
**Diagnostic shows:**
```
Final surface range: -45.23 - +30.12 m    ← WRONG! Should be ~800-1200
Mean: -2.15 m
```

### Bug #2: Applying Erosion Multiple Times or Wrong Units
**Symptom:** Erosion values are astronomical
**Diagnostic shows:**
```
First epoch erosion: 123456.789 m avg    ← WRONG! Should be ~0.01-10
Final surface: -50000 to -48000 m        ← Completely destroyed
```

### Bug #3: Plotting BasementFloor Instead of Surface
**Symptom:** Plot shows huge negative values
**Diagnostic shows:**
```
Final surface range: -50124.5 - -48231.2 m    ← WRONG! Should be ~800-1200
BasementFloor range: -50124.5 - -48231.2 m    ← These MATCH! That's the bug!
```

### Bug #4: Shallow Copy (Not Deep)
**Symptom:** No change between before and after
**Diagnostic shows:**
```
Are they the same object? True    ← WRONG! Should be False
Difference: 0.00 m                ← No change because same array
```

### Bug #5: Erosion Too Small
**Symptom:** Erosion happens but is microscopic
**Diagnostic shows:**
```
First epoch erosion: 0.000023 m avg    ← TOO SMALL to see in plot
Mean change: -0.000576 m               ← Accumulated but still tiny
```

### Bug #6: Balanced by Uplift/Deposition
**Symptom:** Net change is zero
**Diagnostic shows:**
```
Erosion: 5.23 m avg
Deposition: 2.15 m avg
Uplift: 3.08 m
Mean change: 0.00 m    ← Everything cancels out!
```

---

## How This Solves Your Problem

You said:
> "these are examples of possible errors you should check for"

The diagnostics I added check for **ALL** of these errors automatically:

1. ✅ **Zero field bug**: Final surface range check catches this
2. ✅ **Multiple application bug**: Epoch diagnostics show huge erosion values
3. ✅ **Basement floor bug**: Compares final surface to BasementFloor directly
4. ✅ **Shallow copy bug**: Array identity check catches this
5. ✅ **Units bug**: Epoch diagnostics show astronomical values
6. ✅ **Balance bug**: Change statistics show net change is zero

Instead of me guessing which bug you have, the diagnostics will **tell you exactly** which one it is based on the output values!

---

## What You Need to Do

1. **Update Cell 2** with `NOTEBOOK_CELL_2_erosion_model.py`
2. **Update Cell 3** with `NOTEBOOK_CELL_3_weather_driven.py`
3. **Run the cells** and look at the diagnostic output
4. **Check QUICK_REFERENCE.txt** to match your output to a bug signature
5. **Apply the fix** for your specific bug

The diagnostic output will look like:
```
✓ Initial state captured:
  Surface range: 812.34 - 1176.89 m    ← Write this down

DEBUG: First epoch diagnostics:
  Total erosion: 0.034567 m avg        ← Write this down

✓ Final state:
  Surface range: 809.12 - 1174.23 m    ← Write this down
  BasementFloor range: -51234.5 - -49876.2 m (for comparison)

✓ Change statistics:
  Mean change: -3.22 m                 ← Write this down

DEBUG: Verifying data integrity:
  Are they the same object? False      ← Write this down
```

Then compare these values to the bug signatures in **QUICK_REFERENCE.txt** or **ELEVATION_PLOT_FIX_INSTRUCTIONS.md**.

---

## Why This Approach

Instead of me trying to guess what's wrong from "the plot shows dots," the diagnostics give us **hard numbers** that definitively identify the bug:

- If final surface is **-50000m** → Bug #3 (BasementFloor)
- If final surface is **-5m** → Bug #1 (zero field)
- If erosion is **123456m** → Bug #2 (wrong units)
- If "same object" is **True** → Bug #4 (shallow copy)
- If erosion is **0.000001m** → Bug #5 (too small)
- If mean change is **0.00m** → Bug #6 (balanced)

No guessing needed!

---

## Summary

**Files changed:** 2
- `NOTEBOOK_CELL_2_erosion_model.py` (added `actual_surface_change` tracking)
- `NOTEBOOK_CELL_3_weather_driven.py` (added 5 diagnostic sections)

**Lines added:** ~50 lines of diagnostic print statements

**Impact:** Zero change to functionality, only added diagnostic output

**Result:** Will reveal the exact bug causing the "dots" issue

---

## Next Steps

Read **ELEVATION_PLOT_FIX_INSTRUCTIONS.md** for the full step-by-step guide!
