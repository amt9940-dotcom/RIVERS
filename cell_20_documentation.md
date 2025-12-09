# 🌍 ADVANCED EROSION SIMULATION SYSTEM

## Overview

This notebook now includes a **complete physics-based erosion simulation** with the following features:

### ✨ Key Features

1. **Time Acceleration**: 10× speed-up (100 sim years = 1000 real years)
2. **Extreme Rain Physics**: 100× rain boost for visible erosion
3. **Quantum Optimization**: Uses quantum RNG where efficient (terrain seed, rainfall variability)
4. **Realistic Erosion Physics**:
   - D8 flow direction algorithm
   - Discharge accumulation (stream power)
   - Erosion with **half-loss rule** (50% material deleted → valleys deepen)
   - Capacity-based sediment transport
   - Hillslope diffusion
5. **River and Lake Detection**: Automatic identification of hydrological features
6. **Layer-Aware Erosion**: Different rock types erode at different rates
7. **Comprehensive Visualization**: Before/after maps, rivers, lakes, cross-sections

---

## 📋 How to Use

### Quick Start (Run All Cells)

Simply execute all cells in order (Cell 0 → Cell 19). The final demonstration cell will:
- Generate quantum-seeded terrain
- Apply 100 years of erosion (= 1000 real years)
- Show rivers and lakes
- Display comprehensive plots

### Advanced Usage

#### Change Erosion Parameters

Edit **Cell 10** to adjust:
```python
TIME_ACCELERATION = 10.0  # 1 sim year = 10 real years
RAIN_BOOST = 100.0        # Rain strength multiplier
BASE_K = 0.001            # Erosion coefficient
MAX_ERODE_PER_STEP = 0.5  # Max erosion per year [m]
```

#### Change Simulation Duration

Edit **Cell 19**:
```python
num_timesteps = 100  # Number of years to simulate
```

#### Adjust Grid Resolution

For faster testing or higher detail:
```python
N = 128   # Small/fast (128×128)
N = 256   # Medium (256×256, default)
N = 512   # Large/slow (512×512, high detail)
```

---

## 🧪 Physics Implementation

### Erosion Pipeline (Each Timestep)

1. **Rain Boost**: `rain_effective = rain × 100`
2. **Runoff**: `runoff = rain × (1 - infiltration_fraction)`
3. **Flow Direction**: Steepest descent (D8)
4. **Discharge**: Accumulate water from upstream: `Q = local_runoff + Σ(upstream_Q)`
5. **Erosion (Pass A)**:
   - Erosion power: `E = K × Q^m × S^n × erodibility`
   - **Half-loss rule**: Only 50% of eroded material moves downstream
   - Other 50% deleted from system → enables valley formation
6. **Transport (Pass B)**:
   - Capacity: `C = K × Q^m × S^n`
   - If sediment > capacity: deposit excess
   - Else: carry all sediment downstream
7. **Hillslope Diffusion**: Smooth sharp features

### Key Innovation: Half-Loss Rule

```python
eroded_material = erosion_depth  # Total material removed
sediment_to_move = 0.5 × eroded_material  # Only half moves
sediment_lost = 0.5 × eroded_material     # Half deleted forever
```

This is **critical** for realistic landscape evolution:
- Without it: erosion and deposition cancel out
- With it: Net volume loss → valleys deepen, lakes form

---

## 📊 Output Plots

The simulation produces 6 plots:

1. **Initial Topography**: Starting terrain
2. **Final Topography**: After erosion
3. **Elevation Change**: Red = erosion, Blue = deposition
4. **Rivers and Lakes**: Hydrological features overlay
5. **Discharge Map**: Shows drainage network intensity
6. **Cross-Section**: Direct comparison of before/after

---

## 🔬 Quantum Computing Integration

Quantum RNG is used **efficiently** for:
- **Terrain generation seed** (Cell 0): One-time quantum seed
- **Rainfall spatial variability** (Cell 19): Quantum-generated spatial patterns

Classical computation is used for:
- **Flow routing**: Deterministic algorithm (D8)
- **Erosion physics**: Deterministic equations
- **Sediment transport**: Deterministic capacity calculation

This is the **optimal balance**: Use quantum where randomness matters, classical for deterministic physics.

---

## 🎯 Validation

To verify the 10× time acceleration:
1. Run 100 years simulation
2. Check total erosion depth
3. Compare to expected ~1000 years of natural erosion
4. Typical natural erosion rates: 0.01-0.1 mm/yr → 10-100 mm over 1000 years

The simulation should show visible valley formation, channel networks, and lake basins.

---

## 🐛 Troubleshooting

**Issue**: Simulation is too slow
- **Solution**: Reduce `N` (grid size) in Cell 19

**Issue**: Not enough visible erosion
- **Solution**: Increase `RAIN_BOOST` or `BASE_K` in Cell 10

**Issue**: Too much erosion (terrain collapses)
- **Solution**: Decrease `MAX_ERODE_PER_STEP` in Cell 10

**Issue**: Quantum RNG not working
- **Solution**: Code automatically falls back to classical RNG

---

## 📚 References

Physics based on:
- Stream power erosion: Howard (1994)
- Sediment transport: Willgoose et al. (1991)
- Half-loss rule: Custom implementation for valley formation
- D8 flow routing: O'Callaghan & Mark (1984)

---

## ✅ Cell Summary

| Cell | Description |
|------|-------------|
| 0-9  | Original terrain and weather system |
| 10   | Erosion constants and parameters |
| 11   | Flow direction (D8 algorithm) |
| 12   | Discharge computation |
| 13   | Erosion with half-loss rule (Pass A) |
| 14   | Sediment transport (Pass B) |
| 15   | Hillslope diffusion |
| 16   | River and lake detection |
| 17   | Main simulation function |
| 18   | Visualization and analysis |
| 19   | **DEMONSTRATION** (run this!) |
| 20   | This documentation |

---

**Ready to run!** Execute Cell 19 to see the complete erosion simulation in action. 🚀
