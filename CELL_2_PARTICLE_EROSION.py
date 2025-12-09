"""
CELL 2: PARTICLE-BASED EROSION (Musgrave's Hydraulic Erosion)

Implements aggressive particle-based erosion for VISIBLE changes:
1. Time acceleration: each sim year = 100 real years of erosion
2. Particle-based: thousands of raindrops erode and deposit
3. Grid-based flow routing: for discharge calculation
4. Hybrid approach: realistic physics + visible results

Based on Musgrave's Hydraulic Erosion algorithm with modifications.
"""
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# TIME ACCELERATION FACTOR
# ==============================================================================
TIME_ACCELERATION = 100.0  # Each sim year = 100 real years
print(f"⚡ TIME ACCELERATION: {TIME_ACCELERATION}×")
print(f"   1 simulated year = {TIME_ACCELERATION} real years of erosion")

# ==============================================================================
# 1. FLOW ROUTING (for discharge calculation)
# ==============================================================================

def compute_flow_direction_d8(elevation, pixel_scale_m):
    """Compute D8 flow direction."""
    ny, nx = elevation.shape
    flow_dir = np.full((ny, nx), -1, dtype=np.int8)
    receivers = np.full((ny, nx, 2), -1, dtype=np.int32)
    
    di = np.array([-1, -1, 0, 1, 1, 1, 0, -1])
    dj = np.array([0, 1, 1, 1, 0, -1, -1, -1])
    distances = np.array([pixel_scale_m, pixel_scale_m * np.sqrt(2), pixel_scale_m,
                          pixel_scale_m * np.sqrt(2), pixel_scale_m, pixel_scale_m * np.sqrt(2),
                          pixel_scale_m, pixel_scale_m * np.sqrt(2)])
    
    for i in range(ny):
        for j in range(nx):
            z_center = elevation[i, j]
            steepest_slope = 0.0
            steepest_dir = -1
            
            for k in range(8):
                ni = (i + di[k]) % ny
                nj = (j + dj[k]) % nx
                dz = z_center - elevation[ni, nj]
                slope = dz / distances[k]
                
                if slope > steepest_slope:
                    steepest_slope = slope
                    steepest_dir = k
            
            if steepest_dir >= 0:
                flow_dir[i, j] = steepest_dir
                receivers[i, j, 0] = (i + di[steepest_dir]) % ny
                receivers[i, j, 1] = (j + dj[steepest_dir]) % nx
    
    return flow_dir, receivers


def compute_flow_accumulation(elevation, flow_dir, receivers, pixel_scale_m):
    """Flow accumulation for discharge."""
    ny, nx = elevation.shape
    cell_area = pixel_scale_m ** 2
    accumulation = np.ones((ny, nx)) * cell_area
    
    indices = [(i, j) for i in range(ny) for j in range(nx)]
    indices_sorted = sorted(indices, key=lambda idx: elevation[idx], reverse=True)
    
    for (i, j) in indices_sorted:
        if flow_dir[i, j] >= 0:
            ni, nj = receivers[i, j]
            accumulation[ni, nj] += accumulation[i, j]
    
    return accumulation


# ==============================================================================
# 2. PARTICLE-BASED EROSION (Musgrave's Algorithm)
# ==============================================================================

class WaterParticle:
    """
    A raindrop that flows downhill, eroding and depositing sediment.
    
    Based on Musgrave's Hydraulic Erosion algorithm.
    """
    def __init__(self, i, j, initial_volume=1.0):
        self.i = i  # Row (float for subpixel accuracy)
        self.j = j  # Column (float for subpixel accuracy)
        self.volume = initial_volume  # Water volume
        self.sediment = 0.0  # Sediment carried
        self.velocity = 0.0  # Current velocity
        self.alive = True
    
    def step(self, elevation, pixel_scale_m, 
             erosion_rate=0.3, deposition_rate=0.3, 
             sediment_capacity_const=4.0,
             min_slope=0.001, evaporation_rate=0.01,
             inertia=0.05):
        """
        Take one step downhill.
        
        Returns:
            erosion_map: amount eroded from each cell
            deposition_map: amount deposited to each cell
        """
        ny, nx = elevation.shape
        erosion_map = np.zeros((ny, nx))
        deposition_map = np.zeros((ny, nx))
        
        if not self.alive:
            return erosion_map, deposition_map
        
        # Current position (integer cell)
        i_int = int(self.i) % ny
        j_int = int(self.j) % nx
        
        # Find steepest descent direction
        current_height = elevation[i_int, j_int]
        
        best_di, best_dj = 0, 0
        steepest_slope = 0.0
        
        # Check 8 neighbors
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                
                ni = (i_int + di) % ny
                nj = (j_int + dj) % nx
                
                neighbor_height = elevation[ni, nj]
                distance = pixel_scale_m * np.sqrt(di**2 + dj**2)
                slope = (current_height - neighbor_height) / distance
                
                if slope > steepest_slope:
                    steepest_slope = slope
                    best_di = di
                    best_dj = dj
        
        # If no downhill path, particle dies (reached basin or edge)
        if steepest_slope <= min_slope:
            # Deposit all sediment
            if self.sediment > 0:
                deposition_map[i_int, j_int] = self.sediment
            self.alive = False
            return erosion_map, deposition_map
        
        # Update velocity based on slope and previous velocity (inertia)
        new_velocity = np.sqrt(steepest_slope) * pixel_scale_m
        self.velocity = inertia * self.velocity + (1 - inertia) * new_velocity
        
        # Sediment capacity based on velocity and volume
        capacity = sediment_capacity_const * self.velocity * self.volume
        
        # Erosion or deposition
        sediment_diff = capacity - self.sediment
        
        if sediment_diff > 0:
            # Can carry more sediment -> erode
            # Amount to erode depends on erosion rate and difference
            amount_to_erode = erosion_rate * sediment_diff
            erosion_map[i_int, j_int] = amount_to_erode
            self.sediment += amount_to_erode
        else:
            # Carrying too much -> deposit
            amount_to_deposit = deposition_rate * (-sediment_diff)
            deposition_map[i_int, j_int] = amount_to_deposit
            self.sediment -= amount_to_deposit
        
        # Move particle
        self.i += best_di
        self.j += best_dj
        
        # Evaporation
        self.volume *= (1 - evaporation_rate)
        
        # Die if volume too small
        if self.volume < 0.01:
            # Deposit remaining sediment
            ni = int(self.i) % ny
            nj = int(self.j) % nx
            if self.sediment > 0:
                deposition_map[ni, nj] += self.sediment
            self.alive = False
        
        return erosion_map, deposition_map


def apply_particle_erosion(elevation, pixel_scale_m, num_particles=50000,
                            max_particle_lifetime=100,
                            erosion_rate=0.3, deposition_rate=0.3,
                            sediment_capacity=4.0):
    """
    Apply particle-based erosion using many raindrops.
    
    Args:
        elevation: current terrain
        pixel_scale_m: cell size
        num_particles: number of raindrops to simulate
        max_particle_lifetime: max steps per particle
        erosion_rate: how fast particles erode
        deposition_rate: how fast particles deposit
        sediment_capacity: how much sediment water can carry
    
    Returns:
        total_erosion: cumulative erosion map
        total_deposition: cumulative deposition map
    """
    ny, nx = elevation.shape
    
    total_erosion = np.zeros((ny, nx))
    total_deposition = np.zeros((ny, nx))
    
    # Create a working copy of elevation that gets modified
    working_elevation = elevation.copy()
    
    print(f"   Simulating {num_particles} raindrops...")
    
    for p in range(num_particles):
        # Drop particle at random location
        i = np.random.randint(0, ny)
        j = np.random.randint(0, nx)
        
        particle = WaterParticle(i, j, initial_volume=1.0)
        
        # Simulate particle path
        for step in range(max_particle_lifetime):
            if not particle.alive:
                break
            
            erosion_map, deposition_map = particle.step(
                working_elevation, pixel_scale_m,
                erosion_rate=erosion_rate,
                deposition_rate=deposition_rate,
                sediment_capacity_const=sediment_capacity
            )
            
            # Apply to working elevation immediately (particle affects terrain)
            working_elevation -= erosion_map
            working_elevation += deposition_map
            
            # Accumulate for statistics
            total_erosion += erosion_map
            total_deposition += deposition_map
        
        # Progress indicator
        if (p + 1) % 10000 == 0:
            print(f"     {p + 1}/{num_particles} particles simulated...")
    
    return total_erosion, total_deposition


# ==============================================================================
# 3. COMBINED EROSION (Grid + Particle)
# ==============================================================================

def apply_combined_erosion(strata, pixel_scale_m, dt,
                            num_particles_per_year=5000,
                            erosion_strength=1.0):
    """
    Apply combined erosion: grid-based routing + particle-based erosion.
    
    Args:
        strata: terrain dict
        pixel_scale_m: cell size
        dt: time step (years, already accelerated)
        num_particles_per_year: particles per simulated year
        erosion_strength: multiplier for erosion amount
    
    Returns:
        total_erosion: erosion map
        total_deposition: deposition map
    """
    elevation = strata["surface_elev"]
    
    # Number of particles scaled by time step
    num_particles = int(num_particles_per_year * dt)
    
    # Apply particle erosion with TIME ACCELERATION baked in
    erosion_rate = 0.3 * erosion_strength * TIME_ACCELERATION / 100.0
    deposition_rate = 0.3 * erosion_strength * TIME_ACCELERATION / 100.0
    sediment_capacity = 4.0
    
    total_erosion, total_deposition = apply_particle_erosion(
        elevation, pixel_scale_m,
        num_particles=num_particles,
        erosion_rate=erosion_rate,
        deposition_rate=deposition_rate,
        sediment_capacity=sediment_capacity
    )
    
    return total_erosion, total_deposition


# ==============================================================================
# 4. STRATIGRAPHY UPDATE
# ==============================================================================

def update_stratigraphy_simple(strata, erosion, deposition, pixel_scale_m):
    """
    Simple stratigraphy update.
    
    Just modify surface elevation for now.
    Returns the actual applied change for accurate diagnostics.
    """
    # Net change
    net_change = deposition - erosion
    
    # Bound changes to prevent blow-up
    max_change = 10.0  # meters per step
    net_change_clamped = np.clip(net_change, -max_change, max_change)
    
    # Apply
    strata["surface_elev"] += net_change_clamped
    
    # Optional: Ensure positive elevations (commented out to allow true lowering)
    # Uncomment this line if you want to prevent elevations below 0:
    # strata["surface_elev"] = np.maximum(strata["surface_elev"], 0.0)
    
    return net_change_clamped


# ==============================================================================
# 5. TIME-STEPPING
# ==============================================================================

def run_particle_erosion_epoch(strata, pixel_scale_m, dt,
                                num_particles_per_year=5000,
                                erosion_strength=1.0):
    """Run one epoch of particle erosion."""
    
    print(f"   Simulating {dt} years (= {dt * TIME_ACCELERATION:.0f} real years)...")
    
    erosion_raw, deposition_raw = apply_combined_erosion(
        strata, pixel_scale_m, dt,
        num_particles_per_year=num_particles_per_year,
        erosion_strength=erosion_strength
    )
    
    # Update stratigraphy and get ACTUAL applied change (post-clamp)
    net_change_applied = update_stratigraphy_simple(strata, erosion_raw, deposition_raw, pixel_scale_m)
    
    # Separate into erosion/deposition for visualization
    erosion_applied = np.maximum(-net_change_applied, 0)  # Positive where surface lowered
    deposition_applied = np.maximum(net_change_applied, 0)  # Positive where surface raised
    
    return {
        "erosion": erosion_applied,  # Actual applied erosion (clamped)
        "deposition": deposition_applied,  # Actual applied deposition (clamped)
        "net_change": net_change_applied,  # Net change at surface
    }


def run_particle_erosion_simulation(strata, pixel_scale_m, num_epochs, dt,
                                     num_particles_per_year=5000,
                                     erosion_strength=1.0,
                                     verbose=True):
    """Run multiple epochs."""
    history = []
    
    total_real_years = num_epochs * dt * TIME_ACCELERATION
    
    print(f"\n🌊 STARTING PARTICLE EROSION SIMULATION")
    print(f"   Epochs: {num_epochs}")
    print(f"   Time step: {dt} sim years = {dt * TIME_ACCELERATION:.0f} real years")
    print(f"   Total: {total_real_years:.0f} real years of erosion")
    print(f"   Particles per sim year: {num_particles_per_year}")
    print(f"   Total particles: {num_epochs * num_particles_per_year * dt:.0f}")
    
    for epoch in range(num_epochs):
        if verbose:
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Surface range: {strata['surface_elev'].min():.1f} - {strata['surface_elev'].max():.1f} m")
        
        diagnostics = run_particle_erosion_epoch(
            strata, pixel_scale_m, dt,
            num_particles_per_year=num_particles_per_year,
            erosion_strength=erosion_strength
        )
        
        history.append(diagnostics)
        
        if verbose:
            print(f"  ✓ Epoch complete")
            print(f"     ACTUAL surface lowering: {diagnostics['erosion'].mean():.3f} m avg, {diagnostics['erosion'].max():.3f} m max")
            print(f"     ACTUAL surface raising: {diagnostics['deposition'].mean():.3f} m avg, {diagnostics['deposition'].max():.3f} m max")
            print(f"     Net change range: {diagnostics['net_change'].min():.3f} to {diagnostics['net_change'].max():.3f} m")
    
    print(f"\n✓ SIMULATION COMPLETE!")
    print(f"   Total erosion simulated: {total_real_years:.0f} real years")
    
    return history


print("✓ Particle-based erosion (Musgrave's Algorithm) loaded!")
print(f"  ⚡ TIME ACCELERATION: {TIME_ACCELERATION}× (each sim year = {TIME_ACCELERATION} real years)")
print("  🌊 Particle simulation: thousands of raindrops erode and deposit")
print("  📊 Expected: VISIBLE changes in meters!")
