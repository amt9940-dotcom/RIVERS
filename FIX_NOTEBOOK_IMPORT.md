# Fix: ModuleNotFoundError in Jupyter Notebook

## The Problem

```python
ModuleNotFoundError: No module named 'erosion_simulation'
```

This happens because Jupyter doesn't know where to find `erosion_simulation.py`.

---

## Solution 1: Add to Python Path (Quick Fix)

Add this cell **at the very beginning** of your notebook:

```python
# FIRST CELL - Add workspace to path
import sys
from pathlib import Path

workspace = Path("/workspace")
if str(workspace) not in sys.path:
    sys.path.insert(0, str(workspace))
    print(f"✓ Added {workspace} to Python path")

# Now imports will work
from erosion_simulation import ErosionSimulation
```

---

## Solution 2: Use the Notebook I Created

I created a complete working notebook for you:

```bash
# Open this file in Jupyter:
/workspace/erosion_notebook.ipynb
```

This notebook has:
- ✓ Path setup included
- ✓ All code cells ready
- ✓ Step-by-step erosion simulation
- ✓ Visualizations included

---

## Solution 3: Run from Command Line Instead

Instead of a notebook, run the Python script:

```bash
cd /workspace
python3 erosion_simple_working.py
```

This works immediately without import issues.

---

## Solution 4: Change Working Directory

If you're in a Jupyter notebook, add this at the top:

```python
import os
os.chdir('/workspace')

# Now import works
from erosion_simulation import ErosionSimulation
```

---

## Complete Working Example for Notebook

Here's a complete cell you can copy-paste:

```python
# === CELL 1: Setup ===
import sys
from pathlib import Path

# Add workspace to path
workspace = Path("/workspace")
sys.path.insert(0, str(workspace))

# Imports
import numpy as np
import matplotlib.pyplot as plt
from erosion_simulation import ErosionSimulation, plot_simulation_summary

print("✓ Ready!")

# === CELL 2: Generate Terrain ===
def generate_terrain(N=128):
    kx = np.fft.fftfreq(N)
    ky = np.fft.rfftfreq(N)
    K = np.sqrt(kx[:, None]**2 + ky[None, :]**2)
    K[0, 0] = np.inf
    amp = 1.0 / (K ** 1.5)
    phase = np.random.uniform(0, 2*np.pi, size=(N, ky.size))
    spec = amp * (np.cos(phase) + 1j*np.sin(phase))
    spec[0, 0] = 0.0
    z = np.fft.irfftn(spec, s=(N, N))
    return (z - z.min()) / (z.max() - z.min()) * 1000.0

terrain = generate_terrain(N=64)
plt.imshow(terrain, cmap='terrain')
plt.colorbar()
plt.title('Terrain')
plt.show()

# === CELL 3: Setup Layers ===
layers = {
    "Topsoil": terrain - 2,
    "Sandstone": terrain - 50,
    "Granite": terrain - 200,
    "Basement": terrain - 1000,
}
layer_order = ["Topsoil", "Sandstone", "Granite", "Basement"]

# === CELL 4: Run Simulation ===
sim = ErosionSimulation(
    surface_elevation=terrain,
    layer_interfaces=layers,
    layer_order=layer_order,
    pixel_scale_m=100.0
)

# Run 20 years
for year in range(20):
    rainfall = np.ones_like(terrain) * 1000.0
    sim.step(dt=1.0, rainfall_map=rainfall)
    if (year + 1) % 5 == 0:
        print(f"Year {year+1}: {np.sum(sim.river_mask)} rivers")

# === CELL 5: Visualize ===
plot_simulation_summary(sim)
plt.show()
```

---

## Why This Happens

Python's `import` statement looks for modules in:
1. Current directory
2. Directories in `sys.path`
3. Standard library locations

When you run code in Jupyter, the current directory might not be `/workspace`, so Python can't find `erosion_simulation.py`.

Adding `/workspace` to `sys.path` fixes this.

---

## Quick Reference

| Method | Code | When to Use |
|--------|------|-------------|
| Add to path | `sys.path.insert(0, "/workspace")` | In notebook |
| Change dir | `os.chdir("/workspace")` | In notebook |
| Use full notebook | Open `erosion_notebook.ipynb` | Easiest |
| Run script | `python3 erosion_simple_working.py` | Command line |

---

## Files You Can Use

| File | Type | Status |
|------|------|--------|
| `erosion_notebook.ipynb` | Jupyter Notebook | ✅ Ready to use |
| `erosion_simple_working.py` | Python script | ✅ Ready to run |
| `erosion_simulation.py` | Module | ✅ Import this |

---

**Recommendation:** Use `erosion_notebook.ipynb` - it's already set up correctly!
