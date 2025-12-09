#!/usr/bin/env python3
"""
QUICK START - Quantum Erosion Simulation
Simple script you can run directly in VS Code
"""

# Import the 3-block system
# (All blocks are loaded automatically)
from quantum_erosion_3blocks import (
    # Block 1: Terrain
    quantum_seeded_topography,
    
    # Block 2: Erosion
    QuantumErosionSimulator,
    
    # Block 3: Visualization
    plot_terrain_comparison,
)

import numpy as np

print("="*80)
print("QUICK START DEMO")
print("="*80)

# Generate terrain (Block 1)
print("\n1. Generating terrain...")
z_norm, rng = quantum_seeded_topography(N=64, random_seed=42)
elevation = z_norm * 500.0  # Scale to 500m
print(f"   Terrain: 64×64, range {elevation.min():.1f}-{elevation.max():.1f}m")

# Run quantum erosion (Block 2)
print("\n2. Running quantum erosion...")
sim = QuantumErosionSimulator(
    elevation=elevation,
    pixel_scale_m=10.0,
    K_base=5e-4,
    kappa=0.01
)

sim.run(
    n_steps=3,
    quantum_mode='amplitude',  # Try: 'simple', 'entangled', or 'amplitude'
    verbose=True
)

# Visualize results (Block 3)
print("\n3. Creating visualizations...")
plot_terrain_comparison(elevation, sim.elevation, 10.0)

print("\n" + "="*80)
print("✓ DONE!")
print("="*80)
print("\nTry changing:")
print("  - N=64 to N=128 (bigger terrain)")
print("  - n_steps=3 to n_steps=10 (more erosion)")
print("  - quantum_mode='amplitude' to 'simple' or 'entangled'")
print("  - K_base=5e-4 to 1e-3 (more erosion) or 1e-4 (less)")
