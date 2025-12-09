"""
EROSION SYSTEM - CONSTANTS AND PARAMETERS

This cell defines all constants for the advanced erosion simulation:
- Time acceleration (1 sim year = 10 real years)
- Rain boost factor (100×)
- Erosion coefficients
- Transport capacity parameters
- Hillslope diffusion rate
"""

import numpy as np

# ============================================================================
# TIME ACCELERATION
# ============================================================================
# 1 simulated year = 10 real years of erosion
TIME_ACCELERATION = 10.0
print(f"⚡ TIME ACCELERATION: {TIME_ACCELERATION}×")
print(f"   Simulating 100 years will equal {100 * TIME_ACCELERATION:.0f} real years")

# ============================================================================
# RAIN BOOST (extreme erosion power)
# ============================================================================
# Each unit of rain behaves like 100× the erosive power
RAIN_BOOST = 100.0
print(f"🌧️  RAIN BOOST: {RAIN_BOOST}×")

# ============================================================================
# EROSION PARAMETERS
# ============================================================================
# Base erosion coefficient (K in stream power law)
BASE_K = 0.001  # m^(1-2m) / yr, adjusted for boosted rain

# Maximum erosion per timestep (for stability)
MAX_ERODE_PER_STEP = 0.5  # meters per year

# Flat cell erosion coefficient (for ponding/lake scouring)
FLAT_K = 0.0005  # Half of BASE_K for flat areas

# Slope threshold to distinguish flat vs downslope cells
SLOPE_THRESHOLD = 0.001  # ~0.1% grade

# Stream power exponents (standard values)
M_DISCHARGE = 0.5  # Discharge exponent
N_SLOPE = 1.0      # Slope exponent

print(f"\n🏔️  EROSION PARAMETERS:")
print(f"   Base K: {BASE_K}")
print(f"   Max erode/step: {MAX_ERODE_PER_STEP} m/yr")
print(f"   Flat K: {FLAT_K}")
print(f"   Slope threshold: {SLOPE_THRESHOLD}")

# ============================================================================
# SEDIMENT TRANSPORT
# ============================================================================
# Half-loss rule: 50% of eroded material is removed from system
HALF_LOSS_FRACTION = 0.5

# Transport capacity coefficient
CAPACITY_K = 0.01  # kg/m^3 or dimensionless, tunable

# Capacity exponents (often similar to erosion exponents)
CAPACITY_M = 0.5
CAPACITY_N = 1.0

print(f"\n🪨  SEDIMENT TRANSPORT:")
print(f"   Half-loss fraction: {HALF_LOSS_FRACTION} (50% deleted)")
print(f"   Capacity K: {CAPACITY_K}")

# ============================================================================
# INFILTRATION
# ============================================================================
# Simple constant infiltration rate (mm/day or fraction of rain)
# For simplicity: fraction of rain that infiltrates
INFILTRATION_FRACTION = 0.3  # 30% infiltrates, 70% becomes runoff

print(f"\n💧  INFILTRATION:")
print(f"   Infiltration fraction: {INFILTRATION_FRACTION} (30% absorbed)")

# ============================================================================
# HILLSLOPE DIFFUSION
# ============================================================================
# Diffusion coefficient for gentle slope creep (m^2/yr)
DIFFUSION_K = 0.01  # Small value for gentle smoothing

print(f"\n⛰️   HILLSLOPE DIFFUSION:")
print(f"   Diffusion K: {DIFFUSION_K} m²/yr")

# ============================================================================
# LAYER-AWARE ERODIBILITY
# ============================================================================
# Different rock types erode at different rates
# Relative erodibility multipliers (1.0 = baseline)
ERODIBILITY_MAP = {
    "Topsoil": 2.0,        # Very erodible
    "Subsoil": 1.5,        # Moderately erodible
    "Colluvium": 1.8,      # Loose material, erodible
    "Alluvium": 2.0,       # River deposits, erodible
    "Saprolite": 1.2,      # Weathered but still firm
    "WeatheredBR": 0.8,    # Weathered bedrock, resistant
    "Sandstone": 0.6,      # Moderately resistant
    "Shale": 1.0,          # Baseline erodibility
    "Limestone": 0.7,      # Somewhat resistant
    "Basement": 0.3,       # Very resistant
    "BasementFloor": 0.1,  # Nearly unbreakable
}

print(f"\n🪨  LAYER ERODIBILITY:")
for layer, erod in ERODIBILITY_MAP.items():
    print(f"   {layer:15s}: {erod:.1f}×")

print("\n✅ Erosion parameters initialized!")
