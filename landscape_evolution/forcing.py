"""
External Forcing Module

This module defines external drivers that push the landscape evolution system:
- Tectonic uplift/subsidence
- Climate and weather patterns

These forcings provide the energy and boundary conditions for geomorphic processes.
"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class TectonicUplift:
    """
    Defines tectonic uplift/subsidence patterns.
    
    Positive values = uplift (land rises)
    Negative values = subsidence (land sinks)
    """
    
    def __init__(self, nx: int, ny: int, pixel_scale_m: float):
        """
        Initialize uplift field.
        
        Parameters
        ----------
        nx, ny : int
            Grid dimensions
        pixel_scale_m : float
            Grid spacing (m)
        """
        self.nx = nx
        self.ny = ny
        self.pixel_scale_m = pixel_scale_m
        
        # Uplift rate field (m/yr)
        self.uplift_rate = np.zeros((ny, nx), dtype=np.float32)
    
    def set_uniform_uplift(self, rate_m_per_year: float):
        """
        Set uniform uplift across the entire domain.
        
        Parameters
        ----------
        rate_m_per_year : float
            Uplift rate in m/yr (positive = uplift, negative = subsidence)
        """
        self.uplift_rate[:, :] = rate_m_per_year
    
    def set_regional_pattern(
        self, 
        center_rate: float = 1e-3,
        edge_rate: float = 0.0,
        center_frac: float = 0.5
    ):
        """
        Set a simple regional uplift pattern (high in center, low at edges).
        
        Parameters
        ----------
        center_rate : float
            Uplift rate at the center (m/yr)
        edge_rate : float
            Uplift rate at the edges (m/yr)
        center_frac : float
            Fraction of domain that experiences center_rate (0-1)
        """
        # Create radial distance from center
        y, x = np.ogrid[0:self.ny, 0:self.nx]
        cy, cx = self.ny / 2, self.nx / 2
        
        # Normalized distance from center (0 at center, 1 at corners)
        dist = np.sqrt(((y - cy) / self.ny)**2 + ((x - cx) / self.nx)**2)
        dist = dist / dist.max()
        
        # Smooth transition from center to edge
        weight = np.clip((1 - dist / center_frac), 0, 1)
        self.uplift_rate = edge_rate + (center_rate - edge_rate) * weight
    
    def set_block_uplift(
        self,
        x_min: float, x_max: float,
        y_min: float, y_max: float,
        rate: float
    ):
        """
        Set uplift in a rectangular block.
        
        Parameters
        ----------
        x_min, x_max, y_min, y_max : float
            Block boundaries (in meters from origin)
        rate : float
            Uplift rate inside the block (m/yr)
        """
        # Convert to indices
        i_min = int(y_min / self.pixel_scale_m)
        i_max = int(y_max / self.pixel_scale_m)
        j_min = int(x_min / self.pixel_scale_m)
        j_max = int(x_max / self.pixel_scale_m)
        
        # Clip to domain
        i_min = max(0, min(i_min, self.ny - 1))
        i_max = max(0, min(i_max, self.ny))
        j_min = max(0, min(j_min, self.nx - 1))
        j_max = max(0, min(j_max, self.nx))
        
        self.uplift_rate[i_min:i_max, j_min:j_max] = rate
    
    def apply_uplift(self, surface_elev: np.ndarray, dt: float) -> np.ndarray:
        """
        Apply uplift to surface elevation for a time step.
        
        Parameters
        ----------
        surface_elev : np.ndarray
            Current surface elevation (m)
        dt : float
            Time step (years)
            
        Returns
        -------
        np.ndarray
            Updated surface elevation
        """
        return surface_elev + self.uplift_rate * dt
    
    def __repr__(self):
        rate_min = self.uplift_rate.min()
        rate_max = self.uplift_rate.max()
        rate_mean = self.uplift_rate.mean()
        return (f"TectonicUplift(rate: [{rate_min:.2e}, {rate_max:.2e}] m/yr, "
                f"mean={rate_mean:.2e} m/yr)")


@dataclass
class ClimateState:
    """
    Represents climate/weather state at a given time.
    
    This includes:
    - Rainfall patterns
    - Wind direction and speed
    - Temperature (optional, for future weathering models)
    """
    
    # Rainfall intensity (m/yr or mm/hr depending on time scale)
    rainfall: np.ndarray
    
    # Wind direction (degrees, 0 = east, 90 = north)
    wind_direction: float = 90.0
    
    # Wind speed (m/s)
    wind_speed: float = 5.0
    
    # Mean annual temperature (°C, optional)
    temperature: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Validate shapes."""
        if self.rainfall is not None:
            self.ny, self.nx = self.rainfall.shape
            
            if self.temperature is not None:
                assert self.temperature.shape == (self.ny, self.nx)


class WeatherGenerator:
    """
    Generates time-varying rainfall and weather patterns.
    
    This is a simplified weather generator that incorporates:
    - Base climate (mean annual precipitation)
    - Orographic enhancement (rain on windward slopes, shadow on leeward)
    - Storm variability
    """
    
    def __init__(
        self,
        nx: int,
        ny: int,
        pixel_scale_m: float,
        mean_annual_precip_m: float = 1.0,
        wind_direction_deg: float = 270.0,  # West wind (from west)
        orographic_factor: float = 0.5
    ):
        """
        Initialize weather generator.
        
        Parameters
        ----------
        nx, ny : int
            Grid dimensions
        pixel_scale_m : float
            Grid spacing (m)
        mean_annual_precip_m : float
            Mean annual precipitation (m/yr)
        wind_direction_deg : float
            Prevailing wind direction (degrees, 0=east, 90=north, 270=west)
        orographic_factor : float
            Strength of orographic enhancement (0-1)
        """
        self.nx = nx
        self.ny = ny
        self.pixel_scale_m = pixel_scale_m
        self.mean_annual_precip_m = mean_annual_precip_m
        self.wind_direction_deg = wind_direction_deg
        self.orographic_factor = orographic_factor
        
        # Base rainfall (uniform)
        self.base_rainfall = np.full(
            (ny, nx), mean_annual_precip_m, dtype=np.float32
        )
    
    def compute_orographic_effect(
        self,
        surface_elev: np.ndarray,
        wind_dir_deg: Optional[float] = None
    ) -> np.ndarray:
        """
        Compute orographic rainfall enhancement/suppression.
        
        Windward slopes get enhanced rainfall, leeward slopes get rain shadow.
        
        Parameters
        ----------
        surface_elev : np.ndarray
            Surface elevation (m)
        wind_dir_deg : float, optional
            Wind direction (degrees). If None, uses default.
            
        Returns
        -------
        np.ndarray
            Orographic factor (1.0 = no change, >1 = enhancement, <1 = shadow)
        """
        if wind_dir_deg is None:
            wind_dir_deg = self.wind_direction_deg
        
        # Convert wind direction to radians
        wind_rad = np.deg2rad(wind_dir_deg)
        
        # Wind unit vector (direction wind is blowing TO)
        wx = np.cos(wind_rad)
        wy = np.sin(wind_rad)
        
        # Compute terrain gradient
        dy, dx = np.gradient(surface_elev, self.pixel_scale_m)
        
        # Dot product of wind direction with upslope direction
        # Positive = windward (upslope), negative = leeward (downslope)
        upslope_component = dx * wx + dy * wy
        
        # Normalize by typical slope scale
        slope_scale = np.std(np.sqrt(dx**2 + dy**2)) + 1e-6
        upslope_norm = upslope_component / (slope_scale + 1e-6)
        
        # Orographic factor: enhance on windward, suppress on leeward
        oro_factor = 1.0 + self.orographic_factor * np.tanh(upslope_norm * 2.0)
        
        # Ensure positive
        oro_factor = np.clip(oro_factor, 0.2, 2.5)
        
        return oro_factor.astype(np.float32)
    
    def generate_rainfall_field(
        self,
        surface_elev: np.ndarray,
        time_years: float,
        storm_intensity_factor: float = 1.0,
        rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        """
        Generate a rainfall field for a given time step.
        
        Parameters
        ----------
        surface_elev : np.ndarray
            Current surface elevation (m)
        time_years : float
            Current simulation time (years)
        storm_intensity_factor : float
            Multiplier for storm intensity (1.0 = normal, >1 = stronger storms)
        rng : np.random.Generator, optional
            Random number generator for variability
            
        Returns
        -------
        np.ndarray
            Rainfall rate (m/yr)
        """
        if rng is None:
            rng = np.random.default_rng()
        
        # Base rainfall
        rainfall = self.base_rainfall.copy()
        
        # Apply orographic effect
        oro_factor = self.compute_orographic_effect(surface_elev)
        rainfall *= oro_factor
        
        # Add storm variability (spatial)
        # Simple approach: add smooth random field scaled by intensity
        storm_noise = self._generate_smooth_noise(rng, wavelength_frac=0.3)
        rainfall *= (1.0 + 0.3 * storm_intensity_factor * storm_noise)
        
        # Ensure positive
        rainfall = np.maximum(rainfall, 0.0)
        
        return rainfall
    
    def _generate_smooth_noise(
        self,
        rng: np.random.Generator,
        wavelength_frac: float = 0.2
    ) -> np.ndarray:
        """
        Generate smooth random noise field using simple filtering.
        
        Parameters
        ----------
        rng : np.random.Generator
            Random number generator
        wavelength_frac : float
            Characteristic wavelength as fraction of domain size
            
        Returns
        -------
        np.ndarray
            Smooth noise field (mean ≈ 0, std ≈ 1)
        """
        from scipy.ndimage import gaussian_filter
        
        # Generate white noise
        noise = rng.standard_normal((self.ny, self.nx))
        
        # Smooth it
        sigma = wavelength_frac * min(self.nx, self.ny) / 2.35
        smooth_noise = gaussian_filter(noise, sigma=sigma, mode='wrap')
        
        # Normalize
        smooth_noise = (smooth_noise - smooth_noise.mean()) / (smooth_noise.std() + 1e-6)
        
        return smooth_noise.astype(np.float32)
    
    def generate_climate_state(
        self,
        surface_elev: np.ndarray,
        time_years: float,
        rng: Optional[np.random.Generator] = None
    ) -> ClimateState:
        """
        Generate complete climate state for a time step.
        
        Parameters
        ----------
        surface_elev : np.ndarray
            Current surface elevation
        time_years : float
            Current simulation time
        rng : np.random.Generator, optional
            Random number generator
            
        Returns
        -------
        ClimateState
            Complete climate state
        """
        rainfall = self.generate_rainfall_field(surface_elev, time_years, rng=rng)
        
        return ClimateState(
            rainfall=rainfall,
            wind_direction=self.wind_direction_deg,
            wind_speed=5.0,
            temperature=None
        )
    
    def __repr__(self):
        return (f"WeatherGenerator(precip={self.mean_annual_precip_m:.2f} m/yr, "
                f"wind_dir={self.wind_direction_deg:.0f}°, "
                f"oro_factor={self.orographic_factor:.2f})")


def create_simple_forcing(
    nx: int,
    ny: int,
    pixel_scale_m: float,
    uplift_rate: float = 1e-3,
    mean_precip: float = 1.0,
    wind_direction: float = 270.0
) -> Tuple[TectonicUplift, WeatherGenerator]:
    """
    Create simple uniform forcing conditions.
    
    Parameters
    ----------
    nx, ny : int
        Grid dimensions
    pixel_scale_m : float
        Grid spacing (m)
    uplift_rate : float
        Uniform uplift rate (m/yr)
    mean_precip : float
        Mean annual precipitation (m/yr)
    wind_direction : float
        Wind direction (degrees)
        
    Returns
    -------
    TectonicUplift, WeatherGenerator
        Tectonic and weather forcing objects
    """
    # Tectonic uplift
    uplift = TectonicUplift(nx, ny, pixel_scale_m)
    uplift.set_uniform_uplift(uplift_rate)
    
    # Weather
    weather = WeatherGenerator(
        nx, ny, pixel_scale_m,
        mean_annual_precip_m=mean_precip,
        wind_direction_deg=wind_direction,
        orographic_factor=0.5
    )
    
    return uplift, weather
