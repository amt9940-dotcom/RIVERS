#!/usr/bin/env python3
"""
QUANTUM EROSION SIMULATION - 3 BLOCKS
Python script version (runs directly in VS Code)

Structure:
  Block 1: Quantum RNG + Terrain Generation
  Block 2: Quantum Erosion Physics
  Block 3: Demo + Visualization
"""

# ==============================================================================
# BLOCK 1: QUANTUM RNG + TERRAIN GENERATION
# ==============================================================================

print("="*80)
print("BLOCK 1: QUANTUM RNG + TERRAIN GENERATION")
print("="*80)

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import os
import time
import hashlib

# Qiskit imports
try:
    from qiskit import QuantumCircuit
    try:
        from qiskit_aer import Aer
    except ImportError:
        from qiskit import Aer
    HAVE_QISKIT = True
    print("✓ Qiskit available")
except ImportError:
    HAVE_QISKIT = False
    print("⚠ Qiskit not available, using classical fallback")

# ------------------------------------------------------------------------------
# Quantum Random Number Generation
# ------------------------------------------------------------------------------

def qrng_uint32(n, nbits=32):
    """
    Generate n random uint32 values using Qiskit Hadamard gates.
    
    Creates quantum circuit with Hadamard gates on all qubits,
    measures them, and converts bitstrings to integers.
    """
    if not HAVE_QISKIT:
        return np.random.default_rng().integers(0, 2**32, size=n, dtype=np.uint32)
    
    from qiskit import QuantumCircuit
    try:
        from qiskit_aer import Aer
    except ImportError:
        from qiskit import Aer
    
    qc = QuantumCircuit(nbits, nbits)
    qc.h(range(nbits))  # Hadamard on all qubits
    qc.measure(range(nbits), range(nbits))
    
    backend = Aer.get_backend('qasm_simulator')
    seed_sim = int.from_bytes(os.urandom(4), 'little')
    job = backend.run(qc, shots=n, memory=True, seed_simulator=seed_sim)
    result = job.result()
    memory = result.get_memory(qc)
    
    return np.array([np.uint32(int(bits[::-1], 2)) for bits in memory], dtype=np.uint32)


def rng_from_qrng(n_seeds=4, random_seed=None):
    """Create NumPy RNG seeded with quantum randomness."""
    if random_seed is not None:
        return np.random.default_rng(int(random_seed))
    
    seeds = qrng_uint32(n_seeds).tobytes()
    mix = seeds + os.urandom(16) + int(time.time_ns()).to_bytes(8, 'little')
    h = hashlib.blake2b(mix, digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(h, 'little'))

# ------------------------------------------------------------------------------
# Fractal Terrain Generation
# ------------------------------------------------------------------------------

def fractional_surface(N, beta=3.1, rng=None):
    """Generate fractal surface with power-law spectrum."""
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
    """Bilinear interpolation for domain warping."""
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
    """Apply domain warping for micro-relief texture."""
    N = z.shape[0]
    u = fractional_surface(N, beta=beta, rng=rng)*2 - 1
    v = fractional_surface(N, beta=beta, rng=rng)*2 - 1
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    Xw = (ii + amp*N*u) % N
    Yw = (jj + amp*N*v) % N
    return bilinear_sample(z, Xw, Yw)


def ridged_mix(z, alpha=0.18):
    """Apply ridged fractal mixing for sharp features."""
    ridged = 1.0 - np.abs(2.0*z - 1.0)
    out = (1-alpha)*z + alpha*ridged
    lo, hi = np.percentile(out, [2, 98])
    return np.clip((out - lo)/(hi - lo + 1e-12), 0, 1)


def quantum_seeded_topography(N=256, beta=3.1, warp_amp=0.12, 
                              ridged_alpha=0.18, random_seed=None):
    """
    Generate terrain using quantum RNG.
    
    Args:
        N: grid size
        beta: power-law exponent (3.0-3.5)
        warp_amp: domain warp strength (0.10-0.15)
        ridged_alpha: ridge sharpening (0.15-0.20)
        random_seed: for reproducibility
    
    Returns:
        z_norm: normalized elevation (0-1)
        rng: random generator used
    """
    rng = rng_from_qrng(n_seeds=4, random_seed=random_seed)
    base_low = fractional_surface(N, beta=beta, rng=rng)
    base_high = fractional_surface(N, beta=beta-0.4, rng=rng)
    z = 0.65*base_low + 0.35*base_high
    z = domain_warp(z, rng=rng, amp=warp_amp, beta=beta)
    z = ridged_mix(z, alpha=ridged_alpha)
    return z, rng


print("✓ Block 1 loaded: Terrain generation functions ready")
print()

# ==============================================================================
# BLOCK 2: QUANTUM EROSION PHYSICS
# ==============================================================================

print("="*80)
print("BLOCK 2: QUANTUM EROSION PHYSICS")
print("="*80)

# ------------------------------------------------------------------------------
# Quantum Erosion Decision System (3 modes)
# ------------------------------------------------------------------------------

def create_quantum_erosion_mask(rain_field, threshold=0.1, batch_size=1000):
    """
    MODE 1: Simple Hadamard (independent decisions)
    
    For each cell with rain > threshold:
      |0⟩ → H → (|0⟩+|1⟩)/√2 → Measure → 0 or 1
    """
    ny, nx = rain_field.shape
    erosion_mask = np.zeros((ny, nx), dtype=bool)
    active_cells = rain_field > threshold
    n_active = np.sum(active_cells)
    
    if n_active == 0:
        return erosion_mask
    
    if not HAVE_QISKIT:
        erosion_mask[active_cells] = np.random.rand(n_active) > 0.5
        return erosion_mask
    
    from qiskit import QuantumCircuit
    try:
        from qiskit_aer import Aer
    except ImportError:
        from qiskit import Aer
    
    active_indices = np.argwhere(active_cells)
    backend = Aer.get_backend('qasm_simulator')
    
    for start_idx in range(0, n_active, batch_size):
        end_idx = min(start_idx + batch_size, n_active)
        batch_n = end_idx - start_idx
        
        qc = QuantumCircuit(batch_n, batch_n)
        qc.h(range(batch_n))
        qc.measure(range(batch_n), range(batch_n))
        
        job = backend.run(qc, shots=1, memory=True)
        result = job.result()
        bitstring = result.get_memory(qc)[0][::-1]
        decisions = np.array([int(b) for b in bitstring], dtype=bool)
        
        batch_indices = active_indices[start_idx:end_idx]
        for k, (i, j) in enumerate(batch_indices):
            erosion_mask[i, j] = decisions[k]
    
    return erosion_mask


def create_quantum_erosion_mask_entangled(rain_field, threshold=0.1, 
                                          entanglement_radius=2):
    """
    MODE 2: Entangled (correlated decisions via CNOT)
    
    Neighboring cells entangled → spatial correlation
    """
    ny, nx = rain_field.shape
    erosion_mask = np.zeros((ny, nx), dtype=bool)
    active_cells = rain_field > threshold
    
    if not HAVE_QISKIT or np.sum(active_cells) == 0:
        return create_quantum_erosion_mask(rain_field, threshold)
    
    from qiskit import QuantumCircuit
    try:
        from qiskit_aer import Aer
    except ImportError:
        from qiskit import Aer
    
    backend = Aer.get_backend('qasm_simulator')
    processed = np.zeros((ny, nx), dtype=bool)
    
    for i in range(0, ny, entanglement_radius):
        for j in range(0, nx, entanglement_radius):
            i_end = min(i + entanglement_radius, ny)
            j_end = min(j + entanglement_radius, nx)
            
            local_active = active_cells[i:i_end, j:j_end]
            n_local = np.sum(local_active)
            
            if n_local == 0 or n_local > 10:
                continue
            
            qc = QuantumCircuit(n_local, n_local)
            qc.h(range(n_local))
            for q in range(n_local - 1):
                qc.cx(q, q+1)  # Entangle neighbors
            qc.measure(range(n_local), range(n_local))
            
            job = backend.run(qc, shots=1, memory=True)
            result = job.result()
            bitstring = result.get_memory(qc)[0][::-1]
            decisions = np.array([int(b) for b in bitstring], dtype=bool)
            
            local_indices = np.argwhere(local_active)
            for k, (li, lj) in enumerate(local_indices):
                gi, gj = i + li, j + lj
                if not processed[gi, gj]:
                    erosion_mask[gi, gj] = decisions[k]
                    processed[gi, gj] = True
    
    remaining = active_cells & ~processed
    if np.any(remaining):
        remaining_mask = create_quantum_erosion_mask(
            np.where(remaining, rain_field, 0), threshold
        )
        erosion_mask[remaining] = remaining_mask[remaining]
    
    return erosion_mask


def create_quantum_erosion_mask_amplitude(rain_field, threshold=0.1):
    """
    MODE 3: Amplitude encoding (rain-intensity modulated) ⭐ BEST
    
    Ry(π × rain_normalized) → higher rain = higher erosion probability
    """
    ny, nx = rain_field.shape
    erosion_mask = np.zeros((ny, nx), dtype=bool)
    active_cells = rain_field > threshold
    
    if not HAVE_QISKIT or np.sum(active_cells) == 0:
        return create_quantum_erosion_mask(rain_field, threshold)
    
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

# ------------------------------------------------------------------------------
# Flow Routing
# ------------------------------------------------------------------------------

def compute_flow_direction_d8(elevation, pixel_scale_m):
    """Compute D8 flow direction (steepest descent)."""
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


def compute_flow_accumulation(elevation, flow_dir, receivers, 
                              pixel_scale_m, rainfall=None):
    """Compute discharge (upslope contributing area × runoff)."""
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
    """Complete flow routing pipeline."""
    flow_dir, receivers = compute_flow_direction_d8(elevation, pixel_scale_m)
    discharge = compute_flow_accumulation(elevation, flow_dir, receivers, 
                                         pixel_scale_m, rainfall)
    dy, dx = np.gradient(elevation, pixel_scale_m)
    slope = np.sqrt(dx**2 + dy**2)
    slope = np.maximum(slope, 1e-6)
    
    return {
        'flow_dir': flow_dir,
        'receivers': receivers,
        'discharge': discharge,
        'slope': slope,
    }

# ------------------------------------------------------------------------------
# Erosion Physics
# ------------------------------------------------------------------------------

def compute_stream_power_erosion(discharge, slope, K_base, m=0.5, n=1.0):
    """Stream power erosion: E = K * Q^m * S^n"""
    Q_norm = discharge / (discharge.max() + 1e-12)
    erosion = K_base * (Q_norm ** m) * (slope ** n)
    return erosion


def apply_hillslope_diffusion(elevation, pixel_scale_m, kappa, dt):
    """Hillslope diffusion: ∂h/∂t = κ ∇²h"""
    laplacian = (
        np.roll(elevation, -1, axis=0) +
        np.roll(elevation, 1, axis=0) +
        np.roll(elevation, -1, axis=1) +
        np.roll(elevation, 1, axis=1) -
        4 * elevation
    ) / (pixel_scale_m ** 2)
    delta_h = kappa * laplacian * dt
    return elevation + delta_h


def route_sediment(elevation, flow_dir, receivers, erosion_potential, 
                   erosion_mask, pixel_scale_m, transport_capacity_factor=1.2):
    """Route sediment downstream with capacity constraints."""
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

# ------------------------------------------------------------------------------
# Quantum Erosion Simulator Class
# ------------------------------------------------------------------------------

class QuantumErosionSimulator:
    """Complete quantum erosion simulation system."""
    
    def __init__(self, elevation, pixel_scale_m, 
                 K_base=5e-4, m=0.5, n=1.0, kappa=0.01):
        self.elevation = elevation.copy()
        self.initial_elevation = elevation.copy()
        self.pixel_scale_m = pixel_scale_m
        self.K_base = K_base
        self.m = m
        self.n = n
        self.kappa = kappa
        self.history = []
        
    def generate_rainfall(self, mean=1.0, std=0.3, seed=None):
        """Generate spatially variable rainfall field."""
        ny, nx = self.elevation.shape
        rng = np.random.default_rng(seed)
        rain = rng.normal(mean, std, size=(ny, nx))
        rain = ndimage.gaussian_filter(rain, sigma=5)
        rain = np.maximum(rain, 0.0)
        return rain
    
    def step(self, rainfall, dt=1.0, quantum_mode='amplitude', 
             rain_threshold=0.1, verbose=False):
        """Single erosion timestep."""
        if verbose:
            print(f"  Routing flow...")
        
        flow = route_flow(self.elevation, self.pixel_scale_m, rainfall)
        
        if verbose:
            print(f"  Computing erosion potential...")
        erosion_potential = compute_stream_power_erosion(
            flow['discharge'], flow['slope'], self.K_base, self.m, self.n
        )
        
        if verbose:
            print(f"  Creating quantum erosion mask ({quantum_mode})...")
        
        if quantum_mode == 'entangled':
            erosion_mask = create_quantum_erosion_mask_entangled(
                rainfall, rain_threshold
            )
        elif quantum_mode == 'amplitude':
            erosion_mask = create_quantum_erosion_mask_amplitude(
                rainfall, rain_threshold
            )
        else:
            erosion_mask = create_quantum_erosion_mask(
                rainfall, rain_threshold
            )
        
        if verbose:
            print(f"  Routing sediment...")
        erosion_actual = route_sediment(
            self.elevation, flow['flow_dir'], flow['receivers'],
            erosion_potential, erosion_mask, self.pixel_scale_m
        )
        
        self.elevation -= erosion_actual * dt
        
        if verbose:
            print(f"  Applying hillslope diffusion...")
        self.elevation = apply_hillslope_diffusion(
            self.elevation, self.pixel_scale_m, self.kappa, dt
        )
        
        eroded = erosion_actual > 0
        deposited = erosion_actual < 0
        
        stats = {
            'erosion_actual': erosion_actual,
            'erosion_mask': erosion_mask,
            'discharge': flow['discharge'],
            'slope': flow['slope'],
            'total_erosion_m': np.sum(erosion_actual[eroded]),
            'total_deposition_m': -np.sum(erosion_actual[deposited]),
            'mean_erosion_m': np.mean(erosion_actual[eroded]) if np.any(eroded) else 0,
            'n_eroded_cells': np.sum(eroded),
            'n_deposited_cells': np.sum(deposited),
            'quantum_erosion_fraction': np.sum(erosion_mask) / erosion_mask.size,
        }
        
        self.history.append(stats)
        
        if verbose:
            print(f"  Erosion: {stats['total_erosion_m']:.3f} m total, "
                  f"{stats['n_eroded_cells']} cells")
            print(f"  Deposition: {stats['total_deposition_m']:.3f} m total, "
                  f"{stats['n_deposited_cells']} cells")
            print(f"  Quantum mask: {100*stats['quantum_erosion_fraction']:.1f}% active")
        
        return stats
    
    def run(self, n_steps=10, mean_rainfall=1.0, dt=1.0, 
            quantum_mode='amplitude', verbose=True):
        """Run multiple erosion steps."""
        if verbose:
            print(f"\nRunning quantum erosion simulation...")
            print(f"  Steps: {n_steps}")
            print(f"  Quantum mode: {quantum_mode}")
            print(f"  Mean rainfall: {mean_rainfall} m/year")
        
        for step_i in range(n_steps):
            if verbose:
                print(f"\nStep {step_i+1}/{n_steps}:")
            
            rainfall = self.generate_rainfall(mean=mean_rainfall, seed=step_i)
            self.step(rainfall, dt=dt, quantum_mode=quantum_mode, verbose=verbose)
        
        if verbose:
            total_change = np.sum(np.abs(self.elevation - self.initial_elevation))
            print(f"\n✓ Simulation complete!")
            print(f"  Total landscape change: {total_change:.2f} m")
    
    def get_erosion_map(self):
        """Get cumulative erosion map (positive = eroded, negative = deposited)."""
        return self.initial_elevation - self.elevation


print("✓ Block 2 loaded: Quantum erosion physics ready")
print()

# ==============================================================================
# BLOCK 3: DEMO + VISUALIZATION
# ==============================================================================

print("="*80)
print("BLOCK 3: DEMO + VISUALIZATION")
print("="*80)

# ------------------------------------------------------------------------------
# Visualization Functions
# ------------------------------------------------------------------------------

def plot_terrain_comparison(initial_elev, final_elev, pixel_scale_m, figsize=(18, 6)):
    """Plot before/after terrain comparison."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    im1 = axes[0].imshow(initial_elev, cmap='terrain', origin='lower')
    axes[0].set_title('Initial Terrain', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('X (cells)')
    axes[0].set_ylabel('Y (cells)')
    plt.colorbar(im1, ax=axes[0], label='Elevation (m)')
    
    im2 = axes[1].imshow(final_elev, cmap='terrain', origin='lower')
    axes[1].set_title('Final Terrain (After Erosion)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('X (cells)')
    axes[1].set_ylabel('Y (cells)')
    plt.colorbar(im2, ax=axes[1], label='Elevation (m)')
    
    erosion_map = initial_elev - final_elev
    vmax = np.percentile(np.abs(erosion_map), 98)
    im3 = axes[2].imshow(erosion_map, cmap='RdBu_r', origin='lower',
                        vmin=-vmax, vmax=vmax)
    axes[2].set_title('Cumulative Erosion/Deposition', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('X (cells)')
    axes[2].set_ylabel('Y (cells)')
    cbar = plt.colorbar(im3, ax=axes[2], label='Change (m)')
    
    plt.tight_layout()
    plt.savefig('terrain_comparison.png', dpi=150, bbox_inches='tight')
    print("  Saved: terrain_comparison.png")
    plt.show()
    
    print(f"\nTerrain Statistics:")
    print(f"  Initial elevation: {initial_elev.min():.1f} - {initial_elev.max():.1f} m")
    print(f"  Final elevation: {final_elev.min():.1f} - {final_elev.max():.1f} m")
    print(f"  Total erosion: {np.sum(erosion_map[erosion_map > 0]):.2f} m")
    print(f"  Total deposition: {-np.sum(erosion_map[erosion_map < 0]):.2f} m")


def plot_flow_and_erosion(discharge, slope, erosion_map, figsize=(18, 6)):
    """Plot flow patterns and erosion."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    discharge_log = np.log10(discharge + 1)
    im1 = axes[0].imshow(discharge_log, cmap='Blues', origin='lower')
    axes[0].set_title('Water Discharge (log scale)', fontsize=14, fontweight='bold')
    plt.colorbar(im1, ax=axes[0], label='log₁₀(Q + 1)')
    
    im2 = axes[1].imshow(slope, cmap='hot', origin='lower')
    axes[1].set_title('Topographic Slope', fontsize=14, fontweight='bold')
    plt.colorbar(im2, ax=axes[1], label='Slope (m/m)')
    
    vmax = np.percentile(np.abs(erosion_map), 98)
    im3 = axes[2].imshow(erosion_map, cmap='RdBu_r', origin='lower',
                        vmin=-vmax, vmax=vmax)
    axes[2].set_title('Erosion Pattern', fontsize=14, fontweight='bold')
    plt.colorbar(im3, ax=axes[2], label='Erosion (m)')
    
    plt.tight_layout()
    plt.savefig('flow_and_erosion.png', dpi=150, bbox_inches='tight')
    print("  Saved: flow_and_erosion.png")
    plt.show()


def plot_quantum_mask_effect(rainfall, erosion_mask, erosion_actual, figsize=(18, 6)):
    """Visualize quantum mask effect."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    im1 = axes[0].imshow(rainfall, cmap='Blues', origin='lower')
    axes[0].set_title('Rainfall Field', fontsize=14, fontweight='bold')
    plt.colorbar(im1, ax=axes[0], label='Rain (m/year)')
    
    im2 = axes[1].imshow(erosion_mask, cmap='RdYlGn_r', origin='lower')
    axes[1].set_title('Quantum Erosion Mask\n(Hadamard Decision)', 
                     fontsize=14, fontweight='bold')
    cbar = plt.colorbar(im2, ax=axes[1], ticks=[0, 1])
    cbar.set_ticklabels(['No Erosion', 'Erosion'])
    
    vmax = np.percentile(np.abs(erosion_actual), 98)
    im3 = axes[2].imshow(erosion_actual, cmap='RdBu_r', origin='lower',
                        vmin=-vmax, vmax=vmax)
    axes[2].set_title('Actual Erosion/Deposition', fontsize=14, fontweight='bold')
    plt.colorbar(im3, ax=axes[2], label='Change (m)')
    
    plt.tight_layout()
    plt.savefig('quantum_mask_effect.png', dpi=150, bbox_inches='tight')
    print("  Saved: quantum_mask_effect.png")
    plt.show()

# ------------------------------------------------------------------------------
# Run Demo
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    print("\nRunning demo...")
    
    # Parameters
    N = 128
    pixel_scale_m = 10.0
    elev_range_m = 500.0
    seed = 42
    
    print(f"\n1. Generating {N}×{N} terrain...")
    z_norm, rng = quantum_seeded_topography(N=N, beta=3.2, random_seed=seed)
    initial_elevation = z_norm * elev_range_m
    print(f"   ✓ Elevation range: {initial_elevation.min():.1f} - {initial_elevation.max():.1f} m")
    
    # Visualize initial terrain
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(initial_elevation, cmap='terrain', origin='lower')
    ax.set_title('Initial Quantum-Seeded Terrain', fontsize=16, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Elevation (m)')
    plt.tight_layout()
    plt.savefig('initial_terrain.png', dpi=150, bbox_inches='tight')
    print("   Saved: initial_terrain.png")
    plt.show()
    
    print(f"\n2. Running quantum erosion simulation...")
    sim = QuantumErosionSimulator(
        elevation=initial_elevation,
        pixel_scale_m=pixel_scale_m,
        K_base=5e-4,
        m=0.5,
        n=1.0,
        kappa=0.01
    )
    
    sim.run(
        n_steps=5,
        mean_rainfall=1.0,
        dt=1.0,
        quantum_mode='amplitude',
        verbose=True
    )
    
    print(f"\n3. Visualizing results...")
    final_elevation = sim.elevation
    erosion_map = sim.get_erosion_map()
    
    plot_terrain_comparison(initial_elevation, final_elevation, pixel_scale_m)
    
    if len(sim.history) > 0:
        last_step = sim.history[-1]
        plot_flow_and_erosion(
            last_step['discharge'],
            last_step['slope'],
            last_step['erosion_actual']
        )
        
        rainfall = sim.generate_rainfall(mean=1.0, seed=len(sim.history)-1)
        plot_quantum_mask_effect(
            rainfall,
            last_step['erosion_mask'],
            last_step['erosion_actual']
        )
    
    print("\n" + "="*80)
    print("✓ DEMO COMPLETE!")
    print("="*80)
    print("\nCheck the generated PNG files:")
    print("  - initial_terrain.png")
    print("  - terrain_comparison.png")
    print("  - flow_and_erosion.png")
    print("  - quantum_mask_effect.png")
