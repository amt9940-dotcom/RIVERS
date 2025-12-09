# VS Code Guide - Quantum Erosion

## 🎯 Problem Solved!

The Jupyter notebook (`.ipynb` file) shows as JSON strings in VS Code's text editor. 

**Solution:** Use the Python script versions instead!

---

## 📁 Files for VS Code

### ✅ Use These (Regular Python Scripts)

1. **`quantum_erosion_3blocks.py`** ⭐ MAIN FILE
   - Complete 3-block system
   - Run directly in VS Code
   - All code is executable

2. **`quick_start.py`**
   - Simple demo
   - Easy to customize
   - Great starting point

3. **`test_quantum_erosion.py`**
   - Tests all components
   - Run first to verify

4. **`run_quantum_erosion_demo.py`**
   - Standalone demo
   - No imports needed

### ❌ Don't Edit in Text Editor

- **`quantum_erosion_enhanced.ipynb`** - This is for Jupyter only!
  - In VS Code, use Jupyter extension to open it properly
  - Or use the `.py` files instead

---

## 🚀 How to Use in VS Code

### Method 1: Run the Main Script

```bash
# Open VS Code terminal (Ctrl+` or View → Terminal)
python3 quantum_erosion_3blocks.py
```

This runs the complete demo automatically:
- ✅ Generates terrain (Block 1)
- ✅ Runs quantum erosion (Block 2)
- ✅ Creates visualizations (Block 3)
- ✅ Saves PNG images

### Method 2: Quick Start

```bash
python3 quick_start.py
```

Simplified version, easy to customize.

### Method 3: Interactive (VS Code Python Interactive)

1. Open `quantum_erosion_3blocks.py` in VS Code
2. Click "Run Cell" buttons (appear above code blocks)
3. Or press `Shift+Enter` on each section

### Method 4: Import as Module

Create a new file (e.g., `my_simulation.py`):

```python
from quantum_erosion_3blocks import (
    quantum_seeded_topography,
    QuantumErosionSimulator,
    plot_terrain_comparison,
)

# Your custom code here
z, rng = quantum_seeded_topography(N=128, random_seed=42)
elevation = z * 500.0

sim = QuantumErosionSimulator(elevation, pixel_scale_m=10.0)
sim.run(n_steps=5, quantum_mode='amplitude')

plot_terrain_comparison(elevation, sim.elevation, 10.0)
```

---

## 🎨 VS Code Settings (Optional)

### Install Recommended Extensions

1. **Python** (by Microsoft) - Essential
2. **Jupyter** (by Microsoft) - For `.ipynb` files
3. **Pylance** - Better IntelliSense

### Enable Interactive Mode

Add to `settings.json`:
```json
{
    "jupyter.interactiveWindow.textEditor.executeSelection": true,
    "python.terminal.executeInFileDir": true
}
```

---

## 📊 File Structure Overview

```
workspace/
├── quantum_erosion_3blocks.py    ← Main script (use this!)
├── quick_start.py                ← Simple demo
├── test_quantum_erosion.py       ← Test suite
├── run_quantum_erosion_demo.py   ← Standalone demo
│
├── quantum_erosion_enhanced.ipynb  ← For Jupyter (not text editor)
│
└── Documentation/
    ├── START_HERE.md
    ├── VSCODE_GUIDE.md (this file)
    ├── QUANTUM_EROSION_README.md
    └── ...
```

---

## 🔧 Editing the Code

### Open the Right File

❌ **Don't:** Open `.ipynb` in text editor (shows JSON)
✅ **Do:** Open `.py` files in text editor

### Structure of quantum_erosion_3blocks.py

```python
# ==============================================================================
# BLOCK 1: QUANTUM RNG + TERRAIN GENERATION
# ==============================================================================

# All Block 1 functions here...
def quantum_seeded_topography(...):
    ...

print("✓ Block 1 loaded")

# ==============================================================================
# BLOCK 2: QUANTUM EROSION PHYSICS
# ==============================================================================

# All Block 2 functions here...
class QuantumErosionSimulator:
    ...

print("✓ Block 2 loaded")

# ==============================================================================
# BLOCK 3: DEMO + VISUALIZATION
# ==============================================================================

# All Block 3 functions here...
if __name__ == '__main__':
    # Demo code runs here
    ...
```

### Customize Parameters

Find this section at the bottom:

```python
if __name__ == '__main__':
    # EDIT THESE:
    N = 128                    # Grid size
    pixel_scale_m = 10.0       # Cell size
    elev_range_m = 500.0       # Elevation range
    
    # Create simulator
    sim = QuantumErosionSimulator(
        elevation=initial_elevation,
        pixel_scale_m=pixel_scale_m,
        K_base=5e-4,    # ← Change erosion strength
        kappa=0.01      # ← Change diffusion
    )
    
    sim.run(
        n_steps=5,                  # ← More steps = more erosion
        quantum_mode='amplitude',   # ← Try 'simple' or 'entangled'
        verbose=True
    )
```

---

## 🐛 Common Issues

### Issue 1: "ModuleNotFoundError"

**Problem:** Missing packages

**Solution:**
```bash
pip install numpy scipy matplotlib qiskit qiskit-aer
```

### Issue 2: Code doesn't run when I click

**Problem:** VS Code not configured for Python

**Solution:**
1. Install Python extension
2. Select Python interpreter: `Ctrl+Shift+P` → "Python: Select Interpreter"
3. Choose your Python installation

### Issue 3: Plots don't show

**Problem:** Backend issue

**Solution:** Add this at the top:
```python
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
import matplotlib.pyplot as plt
```

### Issue 4: Slow quantum operations

**Problem:** Large grids with quantum circuits

**Solution:** Reduce grid size:
```python
N = 64  # Instead of 128 or 256
quantum_mode = 'simple'  # Instead of 'entangled'
```

---

## 💡 Tips for VS Code

### Run Specific Sections

1. Select code you want to run
2. Right-click → "Run Selection/Line in Python Terminal"
3. Or press `Shift+Enter`

### Debug Mode

1. Set breakpoint: Click left of line number
2. Press `F5` or Debug → Start Debugging
3. Step through code to understand flow

### Multi-File Projects

Create separate files for different experiments:

```
my_project/
├── experiment1.py
├── experiment2.py
├── experiment3.py
└── quantum_erosion_3blocks.py  (import this)
```

Each experiment imports and uses the main module.

---

## 📈 Workflow Example

### 1. First Time Setup

```bash
# In VS Code terminal:
cd /workspace
pip install numpy scipy matplotlib qiskit qiskit-aer
python3 test_quantum_erosion.py  # Verify installation
```

### 2. Run Default Demo

```bash
python3 quantum_erosion_3blocks.py
```

Check output PNG files in workspace folder.

### 3. Customize Parameters

Edit `quantum_erosion_3blocks.py` at the bottom:
- Change `N`, `n_steps`, `quantum_mode`
- Save file (`Ctrl+S`)
- Run again

### 4. Create Your Own Script

Copy `quick_start.py` to `my_experiment.py`:
- Modify parameters
- Add your analysis code
- Run and iterate

---

## 🎓 Learning the Code

### Start with Block 3 (Demo)

Read from bottom up:
1. Look at the `if __name__ == '__main__':` section
2. See how functions are called
3. Understand the workflow

### Then Block 1 (Terrain)

Understand terrain generation:
1. `quantum_seeded_topography()` - Main function
2. `fractional_surface()` - Fractal generation
3. `domain_warp()` - Texture
4. `ridged_mix()` - Sharpening

### Finally Block 2 (Erosion)

Most complex:
1. `create_quantum_erosion_mask_*()` - Quantum decisions
2. `route_flow()` - Water routing
3. `compute_stream_power_erosion()` - Erosion physics
4. `route_sediment()` - Sediment transport
5. `QuantumErosionSimulator` - Complete system

---

## 📊 Output Files

After running, you'll get:

```
workspace/
├── initial_terrain.png           ← Starting terrain
├── terrain_comparison.png        ← Before/after/change
├── flow_and_erosion.png          ← Discharge, slope, erosion
└── quantum_mask_effect.png       ← Rain, quantum mask, actual
```

Open these in VS Code or any image viewer.

---

## 🔄 Jupyter vs Python Script

### Want to use Jupyter in VS Code?

1. Install Jupyter extension
2. Open `quantum_erosion_enhanced.ipynb`
3. VS Code automatically detects it
4. Click "Run Cell" buttons

**But if you prefer code editing:** Use the `.py` files!

---

## ✅ Quick Reference Card

| Task | Command |
|------|---------|
| **Run main script** | `python3 quantum_erosion_3blocks.py` |
| **Quick demo** | `python3 quick_start.py` |
| **Test** | `python3 test_quantum_erosion.py` |
| **Open in Jupyter** | Right-click `.ipynb` → Open With → Jupyter |
| **Run selection** | Select code → `Shift+Enter` |
| **Debug** | `F5` |
| **Terminal** | `Ctrl+`` (backtick) |

---

## 🎉 You're Ready!

The `.py` files are regular Python scripts - no JSON, no strings, just clean code you can edit and run in VS Code!

**Start here:**
```bash
python3 quick_start.py
```

Then explore and customize `quantum_erosion_3blocks.py`!
