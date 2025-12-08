#!/usr/bin/env python3
"""
Erosion Simulation - Notebook Compatible Version

This file can be run in a Jupyter notebook by adding the workspace to the path.
"""

# Add workspace to Python path (for notebooks)
import sys
from pathlib import Path

# Get the workspace directory
workspace = Path("/workspace")
if str(workspace) not in sys.path:
    sys.path.insert(0, str(workspace))
    print(f"✓ Added {workspace} to Python path")

# Now import should work
try:
    from erosion_simulation import (
        ErosionSimulation,
        plot_simulation_summary,
        plot_topography,
        ERODIBILITY
    )
    print("✓ Successfully imported erosion_simulation")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    print(f"\nMake sure erosion_simulation.py is in: {workspace}")
    sys.exit(1)

import numpy as np
import matplotlib.pyplot as plt

# Rest of your code here...
print("\n" + "=" * 80)
print("READY TO RUN EROSION SIMULATION IN NOTEBOOK")
print("=" * 80)
