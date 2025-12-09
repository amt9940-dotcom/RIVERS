# 📋 How to Paste Code into Your Notebook

Since your `Project.ipynb` is too large to open, here's how to use the code in a **NEW, EMPTY notebook**:

---

## ✅ Step-by-Step Instructions

### Step 1: Create a New Notebook

In VS Code or Jupyter:
1. Create a **new notebook** (not the old Project.ipynb)
2. Name it something like `erosion_test.ipynb`

### Step 2: Paste Cell 1 - Terrain Generator

1. Create a new code cell
2. Open the file **`NOTEBOOK_CELL_1_terrain_generator.py`**
3. **Copy ALL the contents** (Ctrl+A, Ctrl+C)
4. **Paste into the cell** (Ctrl+V)
5. **Run the cell** (Shift+Enter)

You should see:
```
✓ Terrain generation functions loaded!
  Main functions:
    - quantum_seeded_topography()
    - generate_stratigraphy()
```

### Step 3: Paste Cell 2 - Erosion Model

1. Create another new code cell
2. Open the file **`NOTEBOOK_CELL_2_erosion_model.py`**
3. **Copy ALL the contents**
4. **Paste into the cell**
5. **Run the cell**

You should see:
```
✓ Erosion model engine loaded successfully!
  Main functions:
    - run_erosion_epoch(): run one time step
    - run_erosion_simulation(): run multiple time steps
    - plot_erosion_evolution(): visualize results
    - plot_cross_section_evolution(): show layer changes
✓ Erosion model loaded!
```

### Step 4: Paste Cell 3 - Demo

1. Create another new code cell
2. Open the file **`NOTEBOOK_CELL_3_demo.py`**
3. **Copy ALL the contents**
4. **Paste into the cell**
5. **Run the cell**

This will:
- Generate a 128×128 terrain
- Run 25 epochs of erosion
- Show visualizations
- Takes ~2-5 minutes

---

## 🎯 What You'll Have

After running all 3 cells, your new notebook will have:

```
Cell 1: [TERRAIN GENERATOR] ✓ Run this first
Cell 2: [EROSION MODEL]     ✓ Run this second
Cell 3: [DEMO]              ✓ Run this third
```

---

## 📦 Required Packages

Make sure these are installed:
```bash
pip install numpy scipy matplotlib
```

If you don't have scipy, install it:
```bash
pip install scipy
```

---

## 🚨 Common Errors & Fixes

### Error: "No module named 'scipy'"

**Fix:** Install scipy
```bash
pip install scipy
```

### Error: "name 'quantum_seeded_topography' is not defined"

**Fix:** You forgot to run Cell 1 first. Run cells in order: 1, then 2, then 3.

### Error: "name 'run_erosion_simulation' is not defined"

**Fix:** You forgot to run Cell 2. Run it before running Cell 3.

### Error: Notebook is slow or crashes

**Fix:** In Cell 3, change:
```python
N = 64  # Make grid smaller
num_epochs = 10  # Fewer epochs
```

---

## 🎨 After It Works

Once you successfully run all 3 cells and see the visualizations, you can:

1. **Experiment with parameters** in Cell 3:
   ```python
   N = 256           # Bigger grid (slower but prettier)
   num_epochs = 50   # More time steps
   K_channel = 5e-6  # More erosion
   uplift_rate = 0.0002  # More uplift
   ```

2. **Create new cells** to try different scenarios

3. **Save your new notebook** - it will be much smaller than the old one!

---

## 💡 Pro Tip

If you want to keep your old Project.ipynb data but can't open it:

1. Use this new notebook for erosion experiments
2. The old notebook still contains all your original data
3. You can extract specific parts later if needed

---

## 📁 File Reference

Files to paste (in order):
1. **`NOTEBOOK_CELL_1_terrain_generator.py`** → Paste into Cell 1
2. **`NOTEBOOK_CELL_2_erosion_model.py`** → Paste into Cell 2
3. **`NOTEBOOK_CELL_3_demo.py`** → Paste into Cell 3

---

## ✅ Checklist

- [ ] Created new notebook
- [ ] Installed scipy (`pip install scipy`)
- [ ] Pasted Cell 1 and ran it
- [ ] Pasted Cell 2 and ran it  
- [ ] Pasted Cell 3 and ran it
- [ ] Saw visualizations appear
- [ ] Experimented with parameters

---

## 🆘 Still Having Issues?

If you still get errors, share:
1. Which cell is failing (1, 2, or 3)
2. The exact error message
3. Your Python version (`python --version`)

I'll help you fix it!
