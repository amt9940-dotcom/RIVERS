# START HERE - Quick Guide to Running the Erosion Simulator

## 🚀 Fastest Way to Run

### Option 1: Complete Notebook (RECOMMENDED)

**Open this file**: `Complete_Erosion_Simulator.ipynb`

**Then**: Run cells from top to bottom (Shift+Enter)

**That's it!** Everything is in the correct order.

---

## 📋 What You Get

After running the notebook, you'll see:

1. ✅ **Initial terrain** generated
2. ✅ **World state** with geological layers initialized
3. ✅ **Simulation progress** printed to screen
4. ✅ **Erosion analysis plot** (4 panels):
   - Erosion map with statistics
   - Histogram of erosion values
   - Cross-section profile
   - Erosion vs elevation scatter
5. ✅ **Erosion rate with rivers** (2 panels):
   - Erosion rate map
   - Erosion rate + river network overlay
6. ✅ **Additional plots**: Initial vs final, deposition, cross-sections

---

## 📁 File Guide

### Main Files:

| File | Purpose |
|------|---------|
| **`Complete_Erosion_Simulator.ipynb`** | ⭐ **RUN THIS!** All-in-one notebook |
| `EXECUTION_ORDER_GUIDE.md` | Order to run things if building manually |
| `requirements.txt` | Python dependencies |

### Documentation:

| File | What's Inside |
|------|---------------|
| `README_LANDSCAPE_EVOLUTION.md` | Complete architecture documentation |
| `EROSION_PLOTTING_GUIDE.md` | Erosion visualization guide |
| `NEW_EROSION_PLOTTING_FEATURES.md` | New erosion plot features |
| `REFACTORING_SUMMARY.md` | What changed from original code |

### Examples:

| File | Purpose |
|------|---------|
| `Example_Landscape_Evolution.ipynb` | General tutorial |
| `example_erosion_plots.py` | Python script version |

### Package Files:

| Directory | Contents |
|-----------|----------|
| `landscape_evolution/` | The main Python package (9 modules) |
| `Project.ipynb` | Your original notebook (preserved) |

---

## 🔧 Setup (One Time Only)

If you haven't installed dependencies yet:

```bash
cd /workspace
pip install -r requirements.txt
```

This installs:
- numpy
- scipy
- matplotlib
- numba (optional but recommended)

---

## ⚡ Quick Start Commands

### Using Jupyter Notebook:
```bash
cd /workspace
jupyter notebook Complete_Erosion_Simulator.ipynb
```

Then run cells top-to-bottom.

### Using Python Script:
```bash
cd /workspace
python3 example_erosion_plots.py
```

This will generate `erosion_analysis.png` and `erosion_rate_map.png`.

---

## 📊 What the Plots Show

### Erosion Analysis (4-panel):
- **Top-left**: Erosion map with stats box
  - Red = high erosion
  - Cyan line = cross-section location
- **Top-right**: Histogram with mean/median
- **Bottom-right**: Cross-section profile
- **Bottom**: Erosion vs elevation scatter with trend

### Erosion Rate + Rivers (2-panel):
- **Left**: Erosion rate only
- **Right**: Erosion rate + rivers in blue
- Shows correlation: rivers = erosion

---

## 🎯 Parameters You Can Change

In the notebook, look for these cells:

### Grid Size:
```python
N = 256  # Try: 128 (faster), 512 (slower, more detail)
```

### Simulation Time:
```python
total_time = 5000.0  # Try: 1000, 10000, 50000
dt = 10.0            # Time step size
```

### Uplift Rate:
```python
tectonics.set_uniform_uplift(1e-3)  # Try: 5e-4, 2e-3, 5e-3
```

### Rainfall:
```python
mean_annual_precip_m=1.0  # Try: 0.5, 2.0, 3.0
```

### Random Seed:
```python
random_seed=42  # Change for different terrain
```

---

## 🐛 Troubleshooting

### Problem: "No module named 'landscape_evolution'"
**Solution**: Install requirements
```bash
cd /workspace
pip install -r requirements.txt
```

### Problem: "No module named 'numpy'"
**Solution**: Install dependencies
```bash
pip install numpy scipy matplotlib numba
```

### Problem: Cells run out of order
**Solution**: Restart kernel and run from top
- In Jupyter: Kernel → Restart & Run All

### Problem: Plots don't show
**Solution**: Add this at top of notebook
```python
%matplotlib inline
```

### Problem: Simulation is slow
**Solution**: Reduce grid size or time
```python
N = 128           # Instead of 256
total_time = 1000 # Instead of 5000
```

---

## 📖 Learning Path

### Beginner:
1. Run `Complete_Erosion_Simulator.ipynb` as-is
2. Look at the plots, understand what they show
3. Read cell explanations

### Intermediate:
1. Change parameters (N, total_time, uplift_rate)
2. Run again, compare results
3. Try different random seeds

### Advanced:
1. Read `EXECUTION_ORDER_GUIDE.md`
2. Build custom workflow
3. Integrate your own stratigraphy from `Project.ipynb`

---

## 🎓 Key Concepts

The simulator works like this:

```
Initial Terrain
    ↓
Add Geological Layers (stratigraphy)
    ↓
Set External Forcing (uplift + rain)
    ↓
Run Time-Stepping Loop:
  - Water flows downhill
  - Erosion removes material (channels, hillslopes)
  - Deposition adds material (valleys, basins)
  - Uplift raises everything
  - Repeat for thousands of years
    ↓
Analyze Results (erosion maps, plots)
```

---

## 💡 Tips

1. **Start small**: Use N=128 for quick tests
2. **Save often**: Notebook auto-saves, but be careful
3. **Check progress**: `verbose=True` shows progress
4. **Export data**: Uncomment save lines in last cell
5. **Compare runs**: Change one parameter at a time

---

## 📞 Need Help?

1. **Read the docs**:
   - `EROSION_PLOTTING_GUIDE.md` - Detailed plotting guide
   - `EXECUTION_ORDER_GUIDE.md` - Step-by-step order
   - `README_LANDSCAPE_EVOLUTION.md` - Full documentation

2. **Check examples**:
   - Look at working code in `Complete_Erosion_Simulator.ipynb`
   - Run `example_erosion_plots.py`

3. **Look at docstrings**:
   ```python
   help(plot_erosion_analysis)
   help(LandscapeEvolutionSimulator)
   ```

---

## ✅ Checklist

Before you start:
- [ ] In `/workspace` directory
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Opened `Complete_Erosion_Simulator.ipynb`

To run:
- [ ] Run cells from top to bottom
- [ ] Wait for simulation to complete (~5 minutes)
- [ ] View plots

To customize:
- [ ] Change parameters in cells
- [ ] Rerun modified cells
- [ ] Compare results

---

## 🎉 You're Ready!

Open `Complete_Erosion_Simulator.ipynb` and start running cells!

The first few cells will import modules and generate terrain.
The middle cells will run the simulation (this takes a few minutes).
The final cells will show beautiful erosion analysis plots.

**Enjoy exploring landscape evolution!** 🏔️🌊
