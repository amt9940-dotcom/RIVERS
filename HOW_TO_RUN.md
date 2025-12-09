# How to Run the Erosion Simulator

You have **TWO options** to run the complete erosion simulator:

---

## ⚡ Option 1: Python Script (EASIEST - Just Run It!)

**File**: `run_erosion_simulator.py`

**How to run**:
```bash
cd /workspace
python3 run_erosion_simulator.py
```

**What happens**:
- Runs automatically from start to finish
- Prints progress to terminal
- Saves 7 PNG images with all the plots
- Takes ~5 minutes to complete

**Output files**:
- `01_initial_terrain.png`
- `02_erosion_analysis.png` ← **Main erosion plot**
- `03_erosion_rate_rivers.png` ← **Erosion + rivers**
- `04_initial_vs_final.png`
- `05_erosion_deposition.png`
- `06_river_network.png`
- `07_cross_section.png`

**Perfect if you just want to run it and get results!**

---

## 📓 Option 2: Jupyter Notebook (Interactive)

**File**: `Complete_Erosion_Simulator.ipynb`

**How to run**:
```bash
cd /workspace
jupyter notebook Complete_Erosion_Simulator.ipynb
```

Then in the browser:
- Click each cell and press `Shift+Enter` to run
- OR: Cell menu → Run All

**What happens**:
- You can run each step individually
- See plots inline in the notebook
- Modify parameters between steps
- More interactive and educational

**Perfect if you want to explore and experiment!**

---

## 🚫 Common Error

**If you see**: `NameError: name 'null' is not defined`

**Reason**: You tried to run the `.ipynb` file with `python3` directly.

**Fix**: Use one of the options above!
- `.ipynb` files must be opened in Jupyter
- `.py` files can be run with python3

---

## 🎯 Quick Comparison

| Feature | Python Script | Jupyter Notebook |
|---------|--------------|------------------|
| **File** | `run_erosion_simulator.py` | `Complete_Erosion_Simulator.ipynb` |
| **Command** | `python3 run_erosion_simulator.py` | `jupyter notebook ...` |
| **Interaction** | Automatic, no interaction | Interactive, run cell-by-cell |
| **Output** | Saves PNG files | Shows plots inline |
| **Best for** | Quick results | Learning & experimenting |
| **Modification** | Edit script, re-run all | Edit cells, re-run changed parts |

---

## 📋 Prerequisites

Both options need:

1. **Python packages installed**:
   ```bash
   cd /workspace
   pip install -r requirements.txt
   ```

2. **In the /workspace directory**:
   ```bash
   cd /workspace
   ```

---

## 🔍 Execution Order (Both Options)

Both follow the same order internally:

1. Import packages
2. Set parameters (N=256, pixel_scale=100m, etc.)
3. Generate terrain
4. Initialize world state with layers
5. Set up forcing (tectonics + weather)
6. Create simulator
7. **Run simulation** (this takes ~5 minutes)
8. Compute water routing
9. Create erosion plots
10. Create additional visualizations

---

## 💡 Tips

### For Python Script:
- **Run it once** to see if everything works
- **Edit parameters** at the top of the script:
  ```python
  N = 256  # Change to 128 for faster, 512 for more detail
  total_time = 5000.0  # Change to 1000 or 10000
  ```
- **Re-run** to see the effect

### For Jupyter Notebook:
- **Run all cells** once to see the full workflow
- **Go back** and change parameters in earlier cells
- **Re-run** from that cell forward
- **Save notebook** to keep your changes

---

## 🐛 Troubleshooting

### Error: "No module named 'landscape_evolution'"
```bash
cd /workspace
pip install -r requirements.txt
```

### Error: "No module named 'numpy'"
```bash
pip install numpy scipy matplotlib numba
```

### Error: "jupyter: command not found"
```bash
pip install jupyter
```

### Script runs but no plots appear
- Plots are saved as PNG files in `/workspace`
- Look for `01_initial_terrain.png`, etc.

### Simulation is too slow
- Edit the script/notebook
- Change `N = 256` to `N = 128`
- Or change `total_time = 5000.0` to `total_time = 1000.0`

---

## ✅ Expected Runtime

With default parameters (N=256, 5000 years):
- **Terrain generation**: ~10 seconds
- **Simulation**: ~3-5 minutes
- **Plotting**: ~30 seconds
- **Total**: ~5-6 minutes

---

## 📊 What You'll Get

Both options produce the same results:

### Main Erosion Analysis Plot (4-panel)
- Top-left: Erosion map with stats
- Top-right: Histogram of erosion values
- Bottom-right: Cross-section profile
- Bottom: Erosion vs elevation scatter

### Erosion Rate + Rivers (2-panel)
- Left: Erosion rate map
- Right: Erosion rate with blue rivers overlaid

### Plus: Initial/final comparison, deposition, cross-sections, etc.

---

## 🎓 Next Steps

After running once:

1. **Change parameters** and run again
2. **Compare results** from different runs
3. **Save output images** for your reports
4. **Read the guides**:
   - `EROSION_PLOTTING_GUIDE.md`
   - `README_LANDSCAPE_EVOLUTION.md`

---

## Summary

**Quick results?** → Use `python3 run_erosion_simulator.py`

**Want to explore?** → Use Jupyter notebook

**Got the error?** → Don't run .ipynb files with python3!

**Both work!** Choose what fits your workflow.
