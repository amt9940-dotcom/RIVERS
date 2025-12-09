"""
Landscape Evolution Simulator - Time-Stepping Engine

This module integrates all components into a coherent time-stepping framework
that evolves the landscape over time.

The simulator:
1. Manages the world state
2. Applies external forcing (tectonics, climate)
3. Routes water over the surface
4. Computes erosion and deposition
5. Updates stratigraphy in a layer-aware manner
6. Enforces geological constraints
7. Tracks history for analysis and visualization
"""

import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
import time as pytime

from .world_state import WorldState
from .forcing import TectonicUplift, WeatherGenerator
from .hydrology import FlowRouter
from .geomorphic_processes import GeomorphicEngine
from .stratigraphy import StratigraphyUpdater


@dataclass
class SimulationHistory:
    """
    Tracks the history of a simulation for analysis and plotting.
    """
    times: List[float]  # Simulation times (years)
    surface_snapshots: List[np.ndarray]  # Surface elevation at each snapshot
    erosion_maps: List[np.ndarray]  # Cumulative erosion
    deposition_maps: List[np.ndarray]  # Cumulative deposition
    top_layer_maps: List[np.ndarray]  # Top layer at each snapshot
    
    def __init__(self):
        self.times = []
        self.surface_snapshots = []
        self.erosion_maps = []
        self.deposition_maps = []
        self.top_layer_maps = []
    
    def add_snapshot(
        self,
        time: float,
        surface: np.ndarray,
        erosion: np.ndarray,
        deposition: np.ndarray,
        top_layer: np.ndarray
    ):
        """Add a snapshot to history."""
        self.times.append(time)
        self.surface_snapshots.append(surface.copy())
        self.erosion_maps.append(erosion.copy())
        self.deposition_maps.append(deposition.copy())
        self.top_layer_maps.append(top_layer.copy())
    
    def get_total_erosion(self) -> np.ndarray:
        """Get total cumulative erosion."""
        if len(self.erosion_maps) > 0:
            return self.erosion_maps[-1]
        return None
    
    def get_total_deposition(self) -> np.ndarray:
        """Get total cumulative deposition."""
        if len(self.deposition_maps) > 0:
            return self.deposition_maps[-1]
        return None
    
    def get_net_change(self) -> np.ndarray:
        """Get net elevation change (deposition - erosion)."""
        if len(self.surface_snapshots) >= 2:
            return self.surface_snapshots[-1] - self.surface_snapshots[0]
        return None


class LandscapeEvolutionSimulator:
    """
    Main simulator class that orchestrates landscape evolution.
    
    This integrates:
    - World state (surface, layers, properties)
    - External forcing (tectonics, climate)
    - Hydrology (water routing)
    - Geomorphic processes (erosion, deposition, weathering)
    - Stratigraphy updates (layer-aware changes)
    """
    
    def __init__(
        self,
        world: WorldState,
        tectonics: TectonicUplift,
        weather: WeatherGenerator,
        geomorphic_engine: Optional[GeomorphicEngine] = None,
        flow_router: Optional[FlowRouter] = None,
        strat_updater: Optional[StratigraphyUpdater] = None,
        snapshot_interval: int = 10,
        verbose: bool = True
    ):
        """
        Initialize the simulator.
        
        Parameters
        ----------
        world : WorldState
            Initial world state
        tectonics : TectonicUplift
            Tectonic forcing
        weather : WeatherGenerator
            Weather/climate generator
        geomorphic_engine : GeomorphicEngine, optional
            Geomorphic processes. If None, creates default.
        flow_router : FlowRouter, optional
            Flow routing. If None, creates default.
        strat_updater : StratigraphyUpdater, optional
            Stratigraphy updater. If None, creates default.
        snapshot_interval : int
            Number of time steps between snapshots
        verbose : bool
            Print progress messages
        """
        self.world = world
        self.tectonics = tectonics
        self.weather = weather
        self.verbose = verbose
        self.snapshot_interval = snapshot_interval
        
        # Initialize sub-components
        if geomorphic_engine is None:
            self.geomorphic_engine = GeomorphicEngine(world.pixel_scale_m)
        else:
            self.geomorphic_engine = geomorphic_engine
        
        if flow_router is None:
            self.flow_router = FlowRouter(world.pixel_scale_m)
        else:
            self.flow_router = flow_router
        
        if strat_updater is None:
            self.strat_updater = StratigraphyUpdater()
        else:
            self.strat_updater = strat_updater
        
        # History tracking
        self.history = SimulationHistory()
        
        # Cumulative erosion and deposition
        self.cumulative_erosion = np.zeros((world.ny, world.nx), dtype=np.float32)
        self.cumulative_deposition = np.zeros((world.ny, world.nx), dtype=np.float32)
        
        # Random number generator for weather
        self.rng = np.random.default_rng()
        
        # Performance tracking
        self.step_count = 0
        self.wall_time_start = pytime.time()
    
    def step(self, dt: float) -> dict:
        """
        Advance the simulation by one time step.
        
        This is the core of the time-stepping loop:
        1. Apply tectonic uplift
        2. Generate weather/rainfall
        3. Route water over surface
        4. Compute erosion and deposition
        5. Update stratigraphy
        6. Enforce constraints
        
        Parameters
        ----------
        dt : float
            Time step size (years)
            
        Returns
        -------
        dict
            Statistics for this time step
        """
        stats = {}
        
        # 1. Apply tectonic uplift to surface and layers
        if self.verbose and self.step_count % 10 == 0:
            print(f"Step {self.step_count}: t={self.world.time:.2f} yr")
        
        # Uplift surface
        self.world.surface_elev = self.tectonics.apply_uplift(
            self.world.surface_elev, dt
        )
        
        # Uplift all layer interfaces (everything moves up together)
        for layer_name in self.world.layer_names:
            self.world.layer_interfaces[layer_name] = self.tectonics.apply_uplift(
                self.world.layer_interfaces[layer_name], dt
            )
        
        stats['uplift_applied'] = True
        
        # 2. Generate rainfall for this time step
        climate = self.weather.generate_climate_state(
            self.world.surface_elev,
            self.world.time,
            rng=self.rng
        )
        rainfall = climate.rainfall
        
        # 3. Route water over the surface
        flow_dir, slope, flow_accum = self.flow_router.compute_flow(
            self.world.surface_elev,
            fill_depressions=False
        )
        
        drainage_area = self.flow_router.drainage_area
        stats['max_flow_accum'] = flow_accum.max()
        stats['mean_slope'] = slope.mean()
        
        # 4. Get material properties at the surface
        erodibility = self.world.get_property_field('erodibility')
        weathering_rate = self.world.get_property_field('weathering_rate')
        
        # 5. Compute all geomorphic processes
        processes = self.geomorphic_engine.compute_all_processes(
            surface_elev=self.world.surface_elev,
            drainage_area=drainage_area,
            slope=slope,
            erodibility=erodibility,
            weathering_rate=weathering_rate,
            mobile_sediment=self.world.mobile_sediment_thickness,
            dt=dt
        )
        
        # Extract process results
        channel_erosion_m = processes['channel_erosion']
        hillslope_change_m = processes['hillslope_change']
        weathering_m = processes['weathering']
        deposition_m = processes['deposition']
        
        stats['mean_channel_erosion'] = channel_erosion_m.mean()
        stats['max_channel_erosion'] = channel_erosion_m.max()
        stats['mean_deposition'] = deposition_m.mean()
        
        # 6. Apply changes to stratigraphy in a layer-aware manner
        
        # Erosion removes material from top layers
        net_erosion = np.maximum(channel_erosion_m, 0)
        erosion_stats = self.strat_updater.apply_erosion(
            self.world,
            net_erosion,
            verbose=False
        )
        
        # Deposition adds material (usually to topsoil or alluvium)
        net_deposition = np.maximum(deposition_m, 0)
        deposition_stats = self.strat_updater.apply_deposition(
            self.world,
            net_deposition,
            target_layer=self.world.layer_names[0],  # Top layer
            verbose=False
        )
        
        # Also apply hillslope diffusion (modify surface elevation directly)
        self.world.surface_elev += hillslope_change_m
        
        # Update mobile sediment (simplified)
        self.world.mobile_sediment_thickness += processes['regolith_change']
        self.world.mobile_sediment_thickness = np.maximum(
            self.world.mobile_sediment_thickness, 0
        )
        
        # 7. Enforce layer ordering and constraints
        self.world.enforce_layer_ordering()
        
        # 8. Track cumulative changes
        self.cumulative_erosion += net_erosion
        self.cumulative_deposition += net_deposition
        
        # 9. Update time
        self.world.time += dt
        self.step_count += 1
        
        # 10. Optionally save snapshot
        if self.step_count % self.snapshot_interval == 0:
            self._save_snapshot()
        
        return stats
    
    def run(
        self,
        total_time: float,
        dt: float,
        save_final: bool = True
    ) -> SimulationHistory:
        """
        Run the simulation for a specified duration.
        
        Parameters
        ----------
        total_time : float
            Total simulation time (years)
        dt : float
            Time step size (years)
        save_final : bool
            Save final snapshot even if not at snapshot interval
            
        Returns
        -------
        SimulationHistory
            History of the simulation
        """
        # Save initial state
        self._save_snapshot()
        
        # Compute number of steps
        n_steps = int(np.ceil(total_time / dt))
        
        if self.verbose:
            print(f"\nStarting simulation:")
            print(f"  Total time: {total_time} years")
            print(f"  Time step: {dt} years")
            print(f"  Number of steps: {n_steps}")
            print(f"  Snapshot interval: {self.snapshot_interval} steps")
            print()
        
        # Time-stepping loop
        for i in range(n_steps):
            self.step(dt)
            
            # Progress report
            if self.verbose and (i + 1) % 100 == 0:
                elapsed = pytime.time() - self.wall_time_start
                steps_per_sec = self.step_count / elapsed
                print(f"  Step {i+1}/{n_steps} "
                      f"(t={self.world.time:.1f} yr, "
                      f"{steps_per_sec:.1f} steps/sec)")
        
        # Save final state if requested
        if save_final and (self.step_count % self.snapshot_interval != 0):
            self._save_snapshot()
        
        if self.verbose:
            elapsed = pytime.time() - self.wall_time_start
            print(f"\nSimulation complete!")
            print(f"  Final time: {self.world.time:.2f} years")
            print(f"  Total steps: {self.step_count}")
            print(f"  Wall time: {elapsed:.1f} seconds")
            print(f"  Performance: {self.step_count / elapsed:.1f} steps/sec")
        
        return self.history
    
    def _save_snapshot(self):
        """Save a snapshot of the current state to history."""
        top_layer = self.world.get_top_layer_map()
        
        self.history.add_snapshot(
            time=self.world.time,
            surface=self.world.surface_elev,
            erosion=self.cumulative_erosion,
            deposition=self.cumulative_deposition,
            top_layer=top_layer
        )
        
        if self.verbose:
            print(f"  Snapshot saved at t={self.world.time:.2f} yr")
    
    def get_current_state(self) -> dict:
        """
        Get current state as a dictionary (for inspection/debugging).
        
        Returns
        -------
        dict
            Current state information
        """
        return {
            'time': self.world.time,
            'surface_elev': self.world.surface_elev.copy(),
            'cumulative_erosion': self.cumulative_erosion.copy(),
            'cumulative_deposition': self.cumulative_deposition.copy(),
            'top_layer': self.world.get_top_layer_map(),
            'mobile_sediment': self.world.mobile_sediment_thickness.copy()
        }
    
    def reset(self):
        """Reset the simulator (clear history, reset counters)."""
        self.history = SimulationHistory()
        self.cumulative_erosion[:] = 0
        self.cumulative_deposition[:] = 0
        self.step_count = 0
        self.wall_time_start = pytime.time()
        
        if self.verbose:
            print("Simulator reset")
    
    def __repr__(self):
        return (f"LandscapeEvolutionSimulator("
                f"t={self.world.time:.2f} yr, "
                f"steps={self.step_count}, "
                f"grid={self.world.nx}×{self.world.ny})")


def create_simple_simulator(
    nx: int = 256,
    ny: int = 256,
    pixel_scale_m: float = 100.0,
    layer_names: Optional[List[str]] = None,
    uplift_rate: float = 1e-3,
    mean_precip: float = 1.0,
    verbose: bool = True
) -> LandscapeEvolutionSimulator:
    """
    Create a simple simulator with default settings.
    
    Parameters
    ----------
    nx, ny : int
        Grid dimensions
    pixel_scale_m : float
        Grid spacing (m)
    layer_names : List[str], optional
        Layer names. If None, uses default layers.
    uplift_rate : float
        Uniform uplift rate (m/yr)
    mean_precip : float
        Mean annual precipitation (m/yr)
    verbose : bool
        Verbosity
        
    Returns
    -------
    LandscapeEvolutionSimulator
        Configured simulator
    """
    # Default layers if not provided
    if layer_names is None:
        layer_names = [
            "Topsoil",
            "Colluvium",
            "Saprolite",
            "WeatheredBR",
            "Sandstone",
            "Shale",
            "Basement"
        ]
    
    # Create world state
    world = WorldState(nx, ny, pixel_scale_m, layer_names)
    
    # Create forcing
    tectonics = TectonicUplift(nx, ny, pixel_scale_m)
    tectonics.set_uniform_uplift(uplift_rate)
    
    weather = WeatherGenerator(
        nx, ny, pixel_scale_m,
        mean_annual_precip_m=mean_precip,
        wind_direction_deg=270.0,
        orographic_factor=0.5
    )
    
    # Create simulator
    simulator = LandscapeEvolutionSimulator(
        world=world,
        tectonics=tectonics,
        weather=weather,
        verbose=verbose
    )
    
    return simulator
