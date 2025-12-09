#!/usr/bin/env python3
"""
Quick demo script showing the 3-block structure
Demonstrates quantum erosion simulation in action
"""

print("=" * 80)
print("QUANTUM EROSION SIMULATION - 3-BLOCK DEMO")
print("=" * 80)
print()
print("This demo mirrors the 3-block structure of the Jupyter notebook:")
print("  BLOCK 1: Quantum RNG + Terrain Generation")
print("  BLOCK 2: Quantum Erosion Physics")
print("  BLOCK 3: Visualization + Demo")
print()

# ==============================================================================
# BLOCK 1: QUANTUM RNG + TERRAIN GENERATION
# ==============================================================================

print("BLOCK 1: Loading Quantum RNG + Terrain Generation...")

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy import ndimage
import os
import time
import hashlib

# Qiskit
try:
    from qiskit import QuantumCircuit
    try:
        from qiskit_aer import Aer
    except ImportError:
        from qiskit import Aer
    HAVE_QISKIT = True
    print("  ✓ Qiskit available")
except ImportError:
    HAVE_QISKIT = False
    print("  ⚠ Qiskit not available, using classical fallback")

# Quantum RNG
def qrng_uint32(n, nbits=32):
    if not HAVE_QISKIT:
        return np.random.default_rng().integers(0, 2**32, size=n, dtype=np.uint32)
    from qiskit import QuantumCircuit
    try:
        from qiskit_aer import Aer
    except ImportError:
        from qiskit import Aer
    qc = QuantumCircuit(nbits, nbits)
    qc.h(range(nbits))
    qc.measure(range(nbits), range(nbits))
    backend = Aer.get_backend('qasm_simulator')
    seed_sim = int.from_bytes(os.urandom(4), 'little')
    job = backend.run(qc, shots=n, memory=True, seed_simulator=seed_sim)
    result = job.result()
    memory = result.get_memory(qc)
    return np.array([np.uint32(int(bits[::-1], 2)) for bits in memory], dtype=np.uint32)

def rng_from_qrng(n_seeds=4, random_seed=None):
    if random_seed is not None:
        return np.random.default_rng(int(random_seed))
    seeds = qrng_uint32(n_seeds).tobytes()
    mix = seeds + os.urandom(16) + int(time.time_ns()).to_bytes(8, 'little')
    h = hashlib.blake2b(mix, digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(h, 'little'))

# Terrain generation
def fractional_surface(N, beta=3.1, rng=None):
    rng = rng or np.random.default_rng()
    kx = np.fft.fftfreq(N)
    ky = np.fft.rfftfreq(N)
    K = np.sqrt(kx[:, None]**2 + ky[None, :]**2)
    K[0, 0] = np.inf
    amp = 1.0 / (K ** (beta/2))
    phase = rng.uniform(0, 2*np.pi, size=(N, ky.size))
    spec = amp * (np.cos(phase) + 1j*np.sin(phase))
    spec[0, 0] = 0.0
    z = np.fft.irfftn(spec, s=(N, N))
    lo, hi = np.percentile(z, [2, 98])
    return np.clip((z - lo)/(hi - lo + 1e-12), 0, 1)

def bilinear_sample(img, X, Y):
    N = img.shape[0]
    x0 = np.floor(X).astype(int) % N
    y0 = np.floor(Y).astype(int) % N
    x1 = (x0+1) % N
    y1 = (y0+1) % N
    dx = X - np.floor(X)
    dy = Y - np.floor(Y)
    return ((1-dx)*(1-dy)*img[x0,y0] + dx*(1-dy)*img[x1,y0] +
            (1-dx)*dy*img[x0,y1] + dx*dy*img[x1,y1])

def domain_warp(z, rng, amp=0.12, beta=3.0):
    N = z.shape[0]
    u = fractional_surface(N, beta=beta, rng=rng)*2 - 1
    v = fractional_surface(N, beta=beta, rng=rng)*2 - 1
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    Xw = (ii + amp*N*u) % N
    Yw = (jj + amp*N*v) % N
    return bilinear_sample(z, Xw, Yw)

def ridged_mix(z, alpha=0.18):
    ridged = 1.0 - np.abs(2.0*z - 1.0)
    out = (1-alpha)*z + alpha*ridged
    lo, hi = np.percentile(out, [2, 98])
    return np.clip((out - lo)/(hi - lo + 1e-12), 0, 1)

def quantum_seeded_topography(N=256, beta=3.1, warp_amp=0.12, 
                              ridged_alpha=0.18, random_seed=None):
    rng = rng_from_qrng(n_seeds=4, random_seed=random_seed)
    base_low = fractional_surface(N, beta=beta, rng=rng)
    base_high = fractional_surface(N, beta=beta-0.4, rng=rng)
    z = 0.65*base_low + 0.35*base_high
    z = domain_warp(z, rng=rng, amp=warp_amp, beta=beta)
    z = ridged_mix(z, alpha=ridged_alpha)
    return z, rng

print("  ✓ BLOCK 1 loaded")

# ==============================================================================
# BLOCK 2: QUANTUM EROSION PHYSICS
# ==============================================================================

print("\nBLOCK 2: Loading Quantum Erosion Physics...")

# Quantum erosion mask (amplitude mode)
def create_quantum_erosion_mask_amplitude(rain_field, threshold=0.1):
    ny, nx = rain_field.shape
    erosion_mask = np.zeros((ny, nx), dtype=bool)
    active_cells = rain_field > threshold
    
    if not HAVE_QISKIT or np.sum(active_cells) == 0:
        n_active = np.sum(active_cells)
        erosion_mask[active_cells] = np.random.rand(n_active) > 0.5
        return erosion_mask
    
    from qiskit import QuantumCircuit
    try:
        from qiskit_aer import Aer
    except ImportError:
        from qiskit import Aer
    
    backend = Aer.get_backend('qasm_simulator')
    active_indices = np.argwhere(active_cells)
    rain_max = rain_field.max()
    
    for idx, (i, j) in enumerate(active_indices):
        rain_val = rain_field[i, j]
        rain_norm = min(rain_val / rain_max, 1.0)
        qc = QuantumCircuit(1, 1)
        angle = np.pi * rain_norm
        qc.ry(angle, 0)
        qc.measure(0, 0)
        job = backend.run(qc, shots=1, memory=True)
        result = job.result()
        measurement = int(result.get_memory(qc)[0])
        erosion_mask[i, j] = (measurement == 1)
    
    return erosion_mask

# Flow routing
def compute_flow_direction_d8(elevation, pixel_scale_m):
    ny, nx = elevation.shape
    flow_dir = np.full((ny, nx), -1, dtype=np.int8)
    receivers = np.full((ny, nx, 2), -1, dtype=np.int32)
    di = np.array([-1, -1, 0, 1, 1, 1, 0, -1])
    dj = np.array([0, 1, 1, 1, 0, -1, -1, -1])
    distances = np.array([1, np.sqrt(2), 1, np.sqrt(2), 
                         1, np.sqrt(2), 1, np.sqrt(2)]) * pixel_scale_m
    
    for i in range(ny):
        for j in range(nx):
            z_center = elevation[i, j]
            steepest_slope = 0.0
            steepest_dir = -1
            for k in range(8):
                ni = i + di[k]
                nj = j + dj[k]
                if ni < 0 or ni >= ny or nj < 0 or nj >= nx:
                    continue
                dz = z_center - elevation[ni, nj]
                slope = dz / distances[k]
                if slope > steepest_slope:
                    steepest_slope = slope
                    steepest_dir = k
            if steepest_dir >= 0:
                flow_dir[i, j] = steepest_dir
                receivers[i, j, 0] = i + di[steepest_dir]
                receivers[i, j, 1] = j + dj[steepest_dir]
    return flow_dir, receivers

def compute_flow_accumulation(elevation, flow_dir, receivers, pixel_scale_m, rainfall=None):
    ny, nx = elevation.shape
    cell_area = pixel_scale_m ** 2
    if rainfall is not None:
        runoff = rainfall * 0.5
        water = runoff * cell_area
    else:
        water = np.ones((ny, nx)) * cell_area
    discharge = water.copy()
    indices = [(i, j) for i in range(ny) for j in range(nx)]
    indices_sorted = sorted(indices, key=lambda idx: elevation[idx], reverse=True)
    for (i, j) in indices_sorted:
        if flow_dir[i, j] >= 0:
            ni, nj = receivers[i, j]
            if 0 <= ni < ny and 0 <= nj < nx:
                discharge[ni, nj] += discharge[i, j]
    return discharge

def route_flow(elevation, pixel_scale_m, rainfall=None):
    flow_dir, receivers = compute_flow_direction_d8(elevation, pixel_scale_m)
    discharge = compute_flow_accumulation(elevation, flow_dir, receivers, pixel_scale_m, rainfall)
    dy, dx = np.gradient(elevation, pixel_scale_m)
    slope = np.sqrt(dx**2 + dy**2)
    slope = np.maximum(slope, 1e-6)
    return {'flow_dir': flow_dir, 'receivers': receivers, 'discharge': discharge, 'slope': slope}

# Erosion
def compute_stream_power_erosion(discharge, slope, K_base, m=0.5, n=1.0):
    Q_norm = discharge / (discharge.max() + 1e-12)
    erosion = K_base * (Q_norm ** m) * (slope ** n)
    return erosion

def apply_hillslope_diffusion(elevation, pixel_scale_m, kappa, dt):
    laplacian = (
        np.roll(elevation, -1, axis=0) + np.roll(elevation, 1, axis=0) +
        np.roll(elevation, -1, axis=1) + np.roll(elevation, 1, axis=1) -
        4 * elevation
    ) / (pixel_scale_m ** 2)
    delta_h = kappa * laplacian * dt
    return elevation + delta_h

def route_sediment(elevation, flow_dir, receivers, erosion_potential, 
                   erosion_mask, pixel_scale_m, transport_capacity_factor=1.2):
    ny, nx = elevation.shape
    erosion_actual = np.zeros((ny, nx))
    sediment_supply = np.zeros((ny, nx))
    indices = [(i, j) for i in range(ny) for j in range(nx)]
    indices_sorted = sorted(indices, key=lambda idx: elevation[idx], reverse=True)
    
    for (i, j) in indices_sorted:
        capacity = erosion_potential[i, j] * transport_capacity_factor
        supply = sediment_supply[i, j]
        can_erode = erosion_mask[i, j]
        
        if supply > capacity:
            deposit = supply - capacity
            erosion_actual[i, j] = -deposit
            sediment_out = capacity
        elif can_erode:
            erode_amount = min(erosion_potential[i, j], capacity - supply)
            erosion_actual[i, j] = erode_amount
            sediment_out = supply + erode_amount
        else:
            erosion_actual[i, j] = 0.0
            sediment_out = supply
        
        if flow_dir[i, j] >= 0:
            ni, nj = receivers[i, j]
            if 0 <= ni < ny and 0 <= nj < nx:
                sediment_supply[ni, nj] += sediment_out
    
    return erosion_actual

print("  ✓ BLOCK 2 loaded")

# ==============================================================================
# BLOCK 3: DEMO
# ==============================================================================

print("\nBLOCK 3: Running Demo...\n")

# Parameters
N = 64  # Small size for quick demo
pixel_scale_m = 10.0
elev_range_m = 500.0

print(f"1. Generating {N}×{N} terrain...")
z_norm, rng = quantum_seeded_topography(N=N, beta=3.2, random_seed=42)
initial_elevation = z_norm * elev_range_m
print(f"   ✓ Elevation range: {initial_elevation.min():.1f} - {initial_elevation.max():.1f} m")

print(f"\n2. Running quantum erosion simulation...")
elevation = initial_elevation.copy()

# Simple 3-step simulation
n_steps = 3
K_base = 5e-4
kappa = 0.01

for step in range(n_steps):
    print(f"   Step {step+1}/{n_steps}:")
    
    # Generate rainfall
    rainfall = np.ones((N, N)) * 1.0 + np.random.randn(N, N) * 0.2
    rainfall = np.maximum(rainfall, 0)
    rainfall = ndimage.gaussian_filter(rainfall, sigma=2)
    
    # Flow routing
    flow = route_flow(elevation, pixel_scale_m, rainfall)
    
    # Erosion potential
    erosion_potential = compute_stream_power_erosion(
        flow['discharge'], flow['slope'], K_base, m=0.5, n=1.0
    )
    
    # Quantum decision (amplitude mode)
    erosion_mask = create_quantum_erosion_mask_amplitude(rainfall, threshold=0.5)
    
    # Route sediment
    erosion_actual = route_sediment(
        elevation, flow['flow_dir'], flow['receivers'],
        erosion_potential, erosion_mask, pixel_scale_m
    )
    
    # Apply erosion
    elevation -= erosion_actual * 1.0
    
    # Diffusion
    elevation = apply_hillslope_diffusion(elevation, pixel_scale_m, kappa, 1.0)
    
    # Stats
    n_eroded = np.sum(erosion_actual > 0)
    n_deposited = np.sum(erosion_actual < 0)
    total_erosion = np.sum(erosion_actual[erosion_actual > 0])
    total_deposition = -np.sum(erosion_actual[erosion_actual < 0])
    quantum_fraction = np.sum(erosion_mask) / erosion_mask.size
    
    print(f"     Erosion: {total_erosion:.3f} m ({n_eroded} cells)")
    print(f"     Deposition: {total_deposition:.3f} m ({n_deposited} cells)")
    print(f"     Quantum mask: {100*quantum_fraction:.1f}% active")

# Final statistics
erosion_map = initial_elevation - elevation
total_change = np.sum(np.abs(erosion_map))

print(f"\n3. Results:")
print(f"   Initial mean elevation: {initial_elevation.mean():.2f} m")
print(f"   Final mean elevation: {elevation.mean():.2f} m")
print(f"   Total landscape change: {total_change:.2f} m")
print(f"   Net erosion: {np.sum(erosion_map):.3f} m")
print(f"   Relief change: {(initial_elevation.max()-initial_elevation.min()):.1f} → {(elevation.max()-elevation.min()):.1f} m")

print("\n" + "=" * 80)
print("✓ DEMO COMPLETE!")
print("=" * 80)
print()
print("The 3-block structure allows modular, organized code:")
print("  • BLOCK 1: Reusable terrain generation")
print("  • BLOCK 2: Flexible erosion physics")
print("  • BLOCK 3: Easy visualization and experimentation")
print()
print("For full visualization, run: jupyter notebook quantum_erosion_enhanced.ipynb")
