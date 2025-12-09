#!/usr/bin/env python3
"""
Fast and Realistic Weather Generation System

This module provides optimized weather generation with realistic physics:
- Wind-driven storm motion
- Orographic effects (mountains force uplift)
- Rain shadows (lee side gets less rain)
- Topographic steering of storms
- Wind barrier and channel effects
- Valley funneling
- Ridge blocking

Significantly faster than complex simulations while maintaining physical realism.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Dict, Tuple, Optional


class FastWeatherSystem:
    """
    Fast and realistic weather generation optimized for erosion simulations.
    
    Key features:
    - Wind-driven storm tracks
    - Orographic precipitation
    - Topographic steering
    - Vectorized operations for speed
    """
    
    def __init__(
        self,
        terrain_elevation: np.ndarray,
        pixel_scale_m: float = 100.0,
        prevailing_wind_dir_deg: float = 270.0,  # From west
        wind_speed_ms: float = 15.0,
        mean_annual_rainfall_mm: float = 1000.0,
        storm_frequency_per_year: float = 12.0,
    ):
        """
        Initialize weather system.
        
        Parameters:
        -----------
        terrain_elevation : ndarray
            Surface elevation (m)
        pixel_scale_m : float
            Spatial resolution (m/pixel)
        prevailing_wind_dir_deg : float
            Direction FROM which wind blows (0=E, 90=N, 180=W, 270=S)
        wind_speed_ms : float
            Typical wind speed (m/s)
        mean_annual_rainfall_mm : float
            Average annual rainfall
        storm_frequency_per_year : float
            Average number of storms per year
        """
        self.terrain = np.array(terrain_elevation, dtype=np.float64)
        self.ny, self.nx = self.terrain.shape
        self.pixel_scale_m = pixel_scale_m
        self.prevailing_wind_dir = prevailing_wind_dir_deg
        self.wind_speed = wind_speed_ms
        self.mean_annual_rain = mean_annual_rainfall_mm
        self.storm_frequency = storm_frequency_per_year
        
        # Precompute terrain properties for speed
        self._precompute_terrain_properties()
        
        # Precompute wind interactions
        self._precompute_wind_effects()
    
    def _precompute_terrain_properties(self):
        """Precompute terrain gradients and curvature for speed."""
        # Elevation normalized [0, 1]
        self.elev_norm = (self.terrain - self.terrain.min()) / \
                        (self.terrain.max() - self.terrain.min() + 1e-9)
        
        # Gradients (dy, dx)
        self.dy, self.dx = np.gradient(self.terrain, self.pixel_scale_m, self.pixel_scale_m)
        
        # Slope magnitude
        self.slope = np.sqrt(self.dx**2 + self.dy**2)
        
        # Aspect (direction of steepest descent)
        self.aspect = np.arctan2(self.dy, self.dx)
        
        # Curvature (convex/concave)
        up = np.roll(self.terrain, -1, axis=0)
        down = np.roll(self.terrain, 1, axis=0)
        left = np.roll(self.terrain, 1, axis=1)
        right = np.roll(self.terrain, -1, axis=1)
        self.curvature = (up + down + left + right - 4.0 * self.terrain) / \
                        (self.pixel_scale_m ** 2)
        
        # Identify valleys (concave, low elevation)
        self.valley_mask = (self.curvature > 0.001) & (self.elev_norm < 0.6)
        
        # Identify ridges (convex, high elevation)
        self.ridge_mask = (self.curvature < -0.001) & (self.elev_norm > 0.4)
    
    def _precompute_wind_effects(self):
        """Precompute how wind interacts with topography."""
        # Convert wind direction to radians (FROM direction)
        wind_rad = np.deg2rad(self.prevailing_wind_dir)
        
        # Wind vector components (direction wind is coming FROM)
        self.wind_x = np.cos(wind_rad)
        self.wind_y = np.sin(wind_rad)
        
        # Storm motion (opposite of wind FROM direction)
        self.storm_dir_x = -self.wind_x
        self.storm_dir_y = -self.wind_y
        
        # Compute windward vs leeward for each cell
        # Positive = windward (facing into wind), Negative = leeward (sheltered)
        dot_product = self.dx * self.wind_x + self.dy * self.wind_y
        self.windward_factor = dot_product / (self.slope + 1e-9)
        
        # Normalize to [0, 1] where 1 = fully windward, 0 = fully leeward
        self.windward_01 = np.clip((self.windward_factor + 1.0) / 2.0, 0, 1)
        
        # Base orographic multiplier (more rain on mountains and windward slopes)
        # This combines elevation and windward effects
        self.orographic_base = (
            0.3 +                          # Minimum (valleys get some rain)
            0.4 * self.elev_norm +        # Elevation effect (mountains get more)
            0.3 * self.windward_01        # Windward effect (facing wind gets more)
        )
        
        # Rain shadow factor (lee sides get much less)
        self.rain_shadow = np.where(
            self.windward_factor < -0.3,  # Strong leeward
            0.4,                           # Only 40% of normal rain
            1.0                            # Normal rain
        )
        
        # Topographic steering: how much terrain deflects wind
        # High slopes perpendicular to wind = strong deflection
        perp_slope = np.abs(self.dx * self.wind_y - self.dy * self.wind_x)
        self.steering_strength = np.clip(perp_slope / 0.1, 0, 1)
        
        # Valley channeling effect
        # Wind funnels through valleys aligned with wind direction
        valley_alignment = np.abs(np.cos(self.aspect - wind_rad))
        self.valley_funnel = self.valley_mask * valley_alignment * 1.5
        
        # Ridge blocking effect  
        # Ridges perpendicular to wind block flow
        ridge_perp = 1.0 - np.abs(np.cos(self.aspect - wind_rad))
        self.ridge_block = self.ridge_mask * ridge_perp * 0.5
    
    def generate_base_rainfall_pattern(self) -> np.ndarray:
        """
        Generate base rainfall pattern considering all topographic effects.
        
        Returns:
        --------
        rainfall : ndarray
            Base rainfall distribution (normalized, multiply by amount)
        """
        # Start with orographic base
        rainfall = self.orographic_base.copy()
        
        # Apply rain shadow
        rainfall *= self.rain_shadow
        
        # Enhance rainfall in funneling valleys
        rainfall += self.valley_funnel
        
        # Reduce rainfall behind blocking ridges
        rainfall *= (1.0 - self.ridge_block)
        
        # Smooth slightly for realism (air masses are continuous)
        rainfall = gaussian_filter(rainfall, sigma=1.5)
        
        # Normalize to mean of 1.0
        rainfall = rainfall / (rainfall.mean() + 1e-9)
        
        return rainfall
    
    def generate_storm(
        self,
        storm_intensity: float = 1.0,
        storm_duration_hours: float = 24.0,
        deviation_deg: float = 0.0,
        rng: Optional[np.random.Generator] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate a single storm event with realistic wind-driven motion.
        
        Parameters:
        -----------
        storm_intensity : float
            Storm strength multiplier (1.0 = average)
        storm_duration_hours : float
            How long storm lasts
        deviation_deg : float
            Deviation from prevailing wind direction (-180 to 180)
        rng : np.random.Generator
            Random number generator for reproducibility
        
        Returns:
        --------
        storm : dict
            'total_rainfall_mm' : total rainfall from this storm
            'max_intensity' : peak rainfall rate
            'track' : storm center positions over time
        """
        if rng is None:
            rng = np.random.default_rng()
        
        # Storm direction (prevailing + deviation)
        storm_dir = self.prevailing_wind_dir + deviation_deg
        storm_rad = np.deg2rad(storm_dir)
        
        # Storm motion vector (storms move WITH wind)
        motion_x = -np.cos(storm_rad) * self.wind_speed  # Note: opposite of FROM dir
        motion_y = -np.sin(storm_rad) * self.wind_speed
        
        # Storm initial position (upwind edge of domain)
        # Start storm coming FROM the wind direction
        if np.abs(motion_x) > np.abs(motion_y):
            # Horizontal motion dominant
            if motion_x > 0:  # Moving right (wind from left)
                start_col = 0
                start_row = rng.integers(self.ny // 4, 3 * self.ny // 4)
            else:  # Moving left (wind from right)
                start_col = self.nx - 1
                start_row = rng.integers(self.ny // 4, 3 * self.ny // 4)
        else:
            # Vertical motion dominant
            if motion_y > 0:  # Moving up (wind from below)
                start_row = 0
                start_col = rng.integers(self.nx // 4, 3 * self.nx // 4)
            else:  # Moving down (wind from above)
                start_row = self.ny - 1
                start_col = rng.integers(self.nx // 4, 3 * self.nx // 4)
        
        # Storm characteristics
        storm_radius_pixels = rng.uniform(15, 40)  # Storm size
        
        # Time steps for storm evolution
        dt_hours = 1.0  # 1 hour resolution
        n_steps = int(storm_duration_hours / dt_hours)
        
        # Initialize total rainfall
        total_rainfall = np.zeros_like(self.terrain)
        
        # Track storm center
        track = []
        
        # Simulate storm motion and rainfall
        current_row = float(start_row)
        current_col = float(start_col)
        
        for step in range(n_steps):
            # Current storm center (in pixels)
            center_row = int(current_row)
            center_col = int(current_col)
            
            # Check if still in domain
            if not (0 <= center_row < self.ny and 0 <= center_col < self.nx):
                break
            
            track.append((center_row, center_col))
            
            # Storm intensity varies over lifetime (starts weak, peaks, weakens)
            time_frac = step / float(n_steps)
            lifecycle_factor = np.sin(np.pi * time_frac)  # 0 -> 1 -> 0
            current_intensity = storm_intensity * lifecycle_factor
            
            # Create storm rainfall pattern
            rows = np.arange(self.ny)
            cols = np.arange(self.nx)
            R, C = np.meshgrid(rows, cols, indexing='ij')
            
            # Distance from storm center
            dist_sq = (R - current_row)**2 + (C - current_col)**2
            
            # Gaussian storm pattern
            storm_pattern = np.exp(-dist_sq / (2 * storm_radius_pixels**2))
            
            # Apply topographic effects
            # Windward slopes get MORE rain (orographic lift)
            # Leeward slopes get LESS rain (rain shadow)
            topo_modified = storm_pattern * self.orographic_base * self.rain_shadow
            
            # Enhanced rainfall on steep windward slopes
            steep_windward = (self.slope > 0.05) & (self.windward_factor > 0.3)
            topo_modified[steep_windward] *= 1.5
            
            # Reduced rainfall in rain shadow
            strong_leeward = self.windward_factor < -0.4
            topo_modified[strong_leeward] *= 0.3
            
            # Rainfall rate for this timestep (mm/hour)
            rainfall_rate = topo_modified * current_intensity * 10.0  # mm/hour
            
            # Accumulate (rainfall in mm for this hour)
            total_rainfall += rainfall_rate * dt_hours
            
            # Update storm position
            # Topographic steering: mountains deflect storm path
            if 0 <= center_row < self.ny and 0 <= center_col < self.nx:
                # Get local steering
                local_steering = self.steering_strength[center_row, center_col]
                
                # Deflection perpendicular to main motion
                deflect_x = -motion_y * local_steering * 0.3
                deflect_y = motion_x * local_steering * 0.3
                
                # Update position with motion + deflection
                pixels_per_hour = self.wind_speed * 3600 / self.pixel_scale_m
                current_col += (motion_x / np.abs(motion_x + 1e-9)) * pixels_per_hour * dt_hours
                current_col += deflect_x * pixels_per_hour * dt_hours
                current_row += (motion_y / np.abs(motion_y + 1e-9)) * pixels_per_hour * dt_hours
                current_row += deflect_y * pixels_per_hour * dt_hours
            else:
                break
        
        return {
            'total_rainfall_mm': total_rainfall,
            'max_intensity': float(total_rainfall.max()),
            'track': track,
            'duration_hours': storm_duration_hours
        }
    
    def generate_annual_rainfall(
        self,
        year: int = 0,
        rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        """
        Generate total rainfall for one year including multiple storms.
        
        Parameters:
        -----------
        year : int
            Year number (for seeding)
        rng : np.random.Generator
            Random number generator
        
        Returns:
        --------
        annual_rainfall : ndarray
            Total rainfall in mm for the year
        """
        if rng is None:
            rng = np.random.default_rng(year * 1000)
        
        # Base pattern (background rainfall)
        base_pattern = self.generate_base_rainfall_pattern()
        
        # Background rainfall (light continuous rain)
        background_amount = 0.3 * self.mean_annual_rain  # 30% from background
        annual_rainfall = base_pattern * background_amount
        
        # Generate storms
        n_storms = rng.poisson(self.storm_frequency)
        n_storms = max(1, min(n_storms, 30))  # Limit to 1-30 storms
        
        storm_total = 0.7 * self.mean_annual_rain  # 70% from storms
        
        for i in range(n_storms):
            # Storm properties
            intensity = rng.uniform(0.5, 2.0)  # Variable intensity
            duration = rng.uniform(12, 48)     # 12-48 hours
            deviation = rng.uniform(-45, 45)   # Deviation from prevailing wind
            
            # Generate storm
            storm = self.generate_storm(
                storm_intensity=intensity,
                storm_duration_hours=duration,
                deviation_deg=deviation,
                rng=rng
            )
            
            # Add storm rainfall
            # Scale so total storm rainfall matches target
            storm_rain = storm['total_rainfall_mm']
            if storm_rain.sum() > 0:
                scale = (storm_total / n_storms) / storm_rain.mean()
                annual_rainfall += storm_rain * scale * (self.terrain.size / 1e6)  # Normalize
        
        # Add small-scale variability
        noise = 0.9 + 0.2 * rng.random(size=self.terrain.shape)
        annual_rainfall *= noise
        
        return annual_rainfall
    
    def update_terrain(self, new_elevation: np.ndarray):
        """
        Update terrain elevation and recompute wind effects.
        
        Call this after erosion modifies the terrain.
        
        Parameters:
        -----------
        new_elevation : ndarray
            Updated surface elevation
        """
        self.terrain = np.array(new_elevation, dtype=np.float64)
        self._precompute_terrain_properties()
        self._precompute_wind_effects()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_weather_system(
    terrain: np.ndarray,
    pixel_scale_m: float = 100.0,
    climate: str = "temperate",
    wind_from: str = "west"
) -> FastWeatherSystem:
    """
    Create weather system with preset climates and wind directions.
    
    Parameters:
    -----------
    terrain : ndarray
        Surface elevation
    pixel_scale_m : float
        Spatial resolution
    climate : str
        "arid", "temperate", "wet", "tropical"
    wind_from : str
        "north", "south", "east", "west", "northwest", "southwest", etc.
    
    Returns:
    --------
    weather : FastWeatherSystem
        Configured weather system
    """
    # Climate presets
    climate_params = {
        "arid": {
            "mean_annual_rainfall_mm": 300.0,
            "storm_frequency_per_year": 5.0,
            "wind_speed_ms": 12.0
        },
        "semi-arid": {
            "mean_annual_rainfall_mm": 600.0,
            "storm_frequency_per_year": 8.0,
            "wind_speed_ms": 13.0
        },
        "temperate": {
            "mean_annual_rainfall_mm": 1000.0,
            "storm_frequency_per_year": 12.0,
            "wind_speed_ms": 15.0
        },
        "wet": {
            "mean_annual_rainfall_mm": 1800.0,
            "storm_frequency_per_year": 18.0,
            "wind_speed_ms": 16.0
        },
        "tropical": {
            "mean_annual_rainfall_mm": 2500.0,
            "storm_frequency_per_year": 25.0,
            "wind_speed_ms": 10.0
        }
    }
    
    # Wind direction mapping
    wind_directions = {
        "north": 180,     # Wind FROM north
        "northeast": 225,
        "east": 270,
        "southeast": 315,
        "south": 0,
        "southwest": 45,
        "west": 90,
        "northwest": 135
    }
    
    params = climate_params.get(climate.lower(), climate_params["temperate"])
    wind_dir = wind_directions.get(wind_from.lower(), 270)  # Default west
    
    return FastWeatherSystem(
        terrain_elevation=terrain,
        pixel_scale_m=pixel_scale_m,
        prevailing_wind_dir_deg=wind_dir,
        **params
    )


if __name__ == "__main__":
    print("Fast Realistic Weather Generation System")
    print("=" * 60)
    print("\nFeatures:")
    print("  • Wind-driven storm motion")
    print("  • Orographic precipitation")
    print("  • Rain shadows")
    print("  • Topographic steering")
    print("  • Valley funneling")
    print("  • Ridge blocking")
    print("\nOptimized for speed with vectorized operations.")
