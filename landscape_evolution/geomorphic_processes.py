"""
Geomorphic Processes Module

This module implements the physical processes that modify the landscape:
- Channel erosion (stream power law)
- Hillslope diffusion (soil creep, mass wasting)
- Weathering (bedrock → regolith conversion)
- Sediment transport and deposition

These processes operate on the surface and must be aware of material
properties (erodibility, etc.) and stratigraphy.
"""

import numpy as np
from typing import Tuple, Optional
from scipy.ndimage import convolve


class ChannelErosion:
    """
    Channel erosion using stream power law.
    
    Erosion rate: E = K × A^m × S^n
    where:
    - K = erodibility
    - A = drainage area (or discharge)
    - S = slope
    - m, n = empirical exponents
    """
    
    def __init__(
        self,
        m: float = 0.5,
        n: float = 1.0,
        K_base: float = 1e-5
    ):
        """
        Initialize channel erosion parameters.
        
        Parameters
        ----------
        m : float
            Drainage area exponent (typically 0.4-0.6)
        n : float
            Slope exponent (typically 0.8-1.2)
        K_base : float
            Base erodibility coefficient (will be modified by material properties)
        """
        self.m = m
        self.n = n
        self.K_base = K_base
    
    def compute_erosion_rate(
        self,
        drainage_area: np.ndarray,
        slope: np.ndarray,
        erodibility: np.ndarray,
        threshold_area: float = 1e4
    ) -> np.ndarray:
        """
        Compute channel erosion rate.
        
        Parameters
        ----------
        drainage_area : np.ndarray
            Drainage area (m^2)
        slope : np.ndarray
            Slope (m/m)
        erodibility : np.ndarray
            Spatially-varying erodibility based on material properties
        threshold_area : float
            Minimum drainage area for channel erosion (m^2).
            Below this, erosion is zero (hillslope-only).
            
        Returns
        -------
        np.ndarray
            Erosion rate (m/yr)
        """
        # Only erode where drainage area exceeds threshold
        is_channel = drainage_area >= threshold_area
        
        # Stream power erosion
        # E = K × A^m × S^n
        erosion_rate = np.zeros_like(drainage_area)
        
        # Avoid singularities
        A_safe = np.maximum(drainage_area, 1.0)
        S_safe = np.maximum(slope, 1e-6)
        
        erosion_rate[is_channel] = (
            self.K_base * 
            erodibility[is_channel] * 
            (A_safe[is_channel] ** self.m) * 
            (S_safe[is_channel] ** self.n)
        )
        
        return erosion_rate
    
    def __repr__(self):
        return f"ChannelErosion(m={self.m}, n={self.n}, K={self.K_base:.2e})"


class HillslopeDiffusion:
    """
    Hillslope processes: soil creep, mass wasting.
    
    Uses a diffusion model:
    ∂z/∂t = κ ∇²z
    
    where κ is the diffusivity coefficient.
    """
    
    def __init__(self, kappa: float = 0.01):
        """
        Initialize hillslope diffusion.
        
        Parameters
        ----------
        kappa : float
            Diffusivity coefficient (m^2/yr).
            Typical values: 0.001-0.1 m^2/yr
        """
        self.kappa = kappa
    
    def compute_elevation_change(
        self,
        surface_elev: np.ndarray,
        pixel_scale_m: float,
        dt: float
    ) -> np.ndarray:
        """
        Compute elevation change due to hillslope diffusion.
        
        Parameters
        ----------
        surface_elev : np.ndarray
            Current surface elevation (m)
        pixel_scale_m : float
            Grid spacing (m)
        dt : float
            Time step (years)
            
        Returns
        -------
        np.ndarray
            Elevation change (m)
        """
        # Compute Laplacian (curvature)
        laplacian = self._compute_laplacian(surface_elev, pixel_scale_m)
        
        # Elevation change: dz = κ × ∇²z × dt
        dz = self.kappa * laplacian * dt
        
        return dz
    
    def _compute_laplacian(
        self,
        field: np.ndarray,
        dx: float
    ) -> np.ndarray:
        """
        Compute Laplacian using 5-point stencil.
        
        ∇²f ≈ (f[i-1,j] + f[i+1,j] + f[i,j-1] + f[i,j+1] - 4f[i,j]) / dx²
        """
        kernel = np.array([
            [0,  1, 0],
            [1, -4, 1],
            [0,  1, 0]
        ], dtype=np.float32)
        
        laplacian = convolve(field, kernel, mode='nearest') / (dx ** 2)
        return laplacian
    
    def __repr__(self):
        return f"HillslopeDiffusion(κ={self.kappa} m²/yr)"


class Weathering:
    """
    Weathering process: converts bedrock to mobile regolith.
    
    This process slowly converts solid rock into loose material
    that can be transported by hillslope and channel processes.
    """
    
    def __init__(self, max_regolith_thickness: float = 2.0):
        """
        Initialize weathering model.
        
        Parameters
        ----------
        max_regolith_thickness : float
            Maximum regolith thickness (m). Weathering slows down
            as regolith approaches this thickness (transport-limited).
        """
        self.max_regolith_thickness = max_regolith_thickness
    
    def compute_weathering_rate(
        self,
        bedrock_weathering_rate: np.ndarray,
        current_regolith_thickness: np.ndarray
    ) -> np.ndarray:
        """
        Compute bedrock weathering rate.
        
        Uses exponential decay model:
        W = W0 × exp(-h / h_max)
        
        where h is current regolith thickness, h_max is maximum thickness.
        
        Parameters
        ----------
        bedrock_weathering_rate : np.ndarray
            Base weathering rate from material properties (m/yr)
        current_regolith_thickness : np.ndarray
            Current thickness of mobile regolith (m)
            
        Returns
        -------
        np.ndarray
            Actual weathering rate (m/yr)
        """
        # Exponential decay: weathering slows when regolith is thick
        decay_factor = np.exp(
            -current_regolith_thickness / self.max_regolith_thickness
        )
        
        return bedrock_weathering_rate * decay_factor
    
    def __repr__(self):
        return f"Weathering(h_max={self.max_regolith_thickness} m)"


class SedimentTransport:
    """
    Sediment transport and deposition.
    
    This is a simplified model that:
    1. Computes transport capacity based on slope and discharge
    2. Deposits sediment where capacity is exceeded
    3. Tracks mobile sediment
    """
    
    def __init__(
        self,
        transport_coefficient: float = 0.01,
        deposition_rate: float = 0.5
    ):
        """
        Initialize sediment transport.
        
        Parameters
        ----------
        transport_coefficient : float
            Controls how much sediment can be transported
        deposition_rate : float
            Fraction of excess sediment deposited per time step (0-1)
        """
        self.transport_coefficient = transport_coefficient
        self.deposition_rate = deposition_rate
    
    def compute_transport_capacity(
        self,
        discharge: np.ndarray,
        slope: np.ndarray
    ) -> np.ndarray:
        """
        Compute sediment transport capacity.
        
        Capacity ∝ discharge × slope
        
        Parameters
        ----------
        discharge : np.ndarray
            Water discharge (m^3/s)
        slope : np.ndarray
            Slope (m/m)
            
        Returns
        -------
        np.ndarray
            Transport capacity (m^3/s of sediment)
        """
        # Simple capacity law: Qs = k × Q × S
        S_safe = np.maximum(slope, 1e-6)
        capacity = self.transport_coefficient * discharge * S_safe
        
        return capacity
    
    def compute_deposition(
        self,
        erosion_rate: np.ndarray,
        transport_capacity: np.ndarray,
        mobile_sediment: np.ndarray,
        pixel_scale_m: float,
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute deposition from sediment transport.
        
        This is a simplified "local" deposition model:
        - If erosion produces more sediment than can be transported, deposit locally
        - If transport capacity drops (e.g., lower slope), deposit
        
        Parameters
        ----------
        erosion_rate : np.ndarray
            Erosion rate (m/yr)
        transport_capacity : np.ndarray
            Transport capacity (m^3/s)
        mobile_sediment : np.ndarray
            Current mobile sediment thickness (m)
        pixel_scale_m : float
            Grid spacing (m)
        dt : float
            Time step (years)
            
        Returns
        -------
        deposition_rate : np.ndarray
            Deposition rate (m/yr)
        net_elevation_change : np.ndarray
            Net elevation change from erosion - deposition (m)
        """
        # Convert transport capacity to volumetric rate (m/yr)
        # This is a simplification
        cell_area = pixel_scale_m ** 2
        seconds_per_year = 365.25 * 24 * 3600
        
        # Transport capacity in m/yr (very rough approximation)
        capacity_m_yr = transport_capacity * seconds_per_year / cell_area
        
        # Where erosion exceeds capacity, deposit excess
        excess_sediment = erosion_rate - capacity_m_yr
        deposition_rate = np.maximum(0, excess_sediment * self.deposition_rate)
        
        # Net change = erosion - deposition
        net_change = erosion_rate - deposition_rate
        
        return deposition_rate, net_change
    
    def __repr__(self):
        return (f"SedimentTransport(k={self.transport_coefficient}, "
                f"dep_rate={self.deposition_rate})")


class GeomorphicEngine:
    """
    Combines all geomorphic processes.
    
    This is the main interface for computing landscape change.
    """
    
    def __init__(
        self,
        pixel_scale_m: float,
        channel_erosion: Optional[ChannelErosion] = None,
        hillslope_diffusion: Optional[HillslopeDiffusion] = None,
        weathering: Optional[Weathering] = None,
        sediment_transport: Optional[SedimentTransport] = None
    ):
        """
        Initialize geomorphic engine.
        
        Parameters
        ----------
        pixel_scale_m : float
            Grid spacing (m)
        channel_erosion : ChannelErosion, optional
            Channel erosion model. If None, creates default.
        hillslope_diffusion : HillslopeDiffusion, optional
            Hillslope diffusion model. If None, creates default.
        weathering : Weathering, optional
            Weathering model. If None, creates default.
        sediment_transport : SedimentTransport, optional
            Sediment transport model. If None, creates default.
        """
        self.pixel_scale_m = pixel_scale_m
        
        # Initialize sub-models
        self.channel_erosion = channel_erosion or ChannelErosion()
        self.hillslope_diffusion = hillslope_diffusion or HillslopeDiffusion()
        self.weathering = weathering or Weathering()
        self.sediment_transport = sediment_transport or SedimentTransport()
    
    def compute_all_processes(
        self,
        surface_elev: np.ndarray,
        drainage_area: np.ndarray,
        slope: np.ndarray,
        erodibility: np.ndarray,
        weathering_rate: np.ndarray,
        mobile_sediment: np.ndarray,
        dt: float
    ) -> dict:
        """
        Compute all geomorphic processes for one time step.
        
        Parameters
        ----------
        surface_elev : np.ndarray
            Current surface elevation (m)
        drainage_area : np.ndarray
            Drainage area (m^2)
        slope : np.ndarray
            Slope (m/m)
        erodibility : np.ndarray
            Erodibility field from material properties
        weathering_rate : np.ndarray
            Base weathering rate field from material properties (m/yr)
        mobile_sediment : np.ndarray
            Current mobile sediment thickness (m)
        dt : float
            Time step (years)
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'channel_erosion': erosion from channels (m)
            - 'hillslope_change': elevation change from diffusion (m)
            - 'weathering': bedrock weathered to regolith (m)
            - 'deposition': sediment deposited (m)
            - 'total_change': net elevation change (m)
            - 'regolith_change': change in mobile sediment (m)
        """
        # 1. Channel erosion
        channel_erosion_rate = self.channel_erosion.compute_erosion_rate(
            drainage_area, slope, erodibility
        )
        channel_erosion_m = channel_erosion_rate * dt
        
        # 2. Hillslope diffusion
        hillslope_change_m = self.hillslope_diffusion.compute_elevation_change(
            surface_elev, self.pixel_scale_m, dt
        )
        
        # 3. Weathering
        weathering_rate_actual = self.weathering.compute_weathering_rate(
            weathering_rate, mobile_sediment
        )
        weathering_m = weathering_rate_actual * dt
        
        # 4. Combine erosion and diffusion (both remove/add material)
        # For simplicity, treat as additive
        total_erosion = channel_erosion_m
        
        # 5. Sediment transport and deposition
        # (simplified: local deposition where slope decreases)
        # Compute transport capacity
        discharge_proxy = drainage_area * 1e-3  # Rough discharge proxy
        transport_capacity = self.sediment_transport.compute_transport_capacity(
            discharge_proxy, slope
        )
        
        deposition_rate, net_change_rate = self.sediment_transport.compute_deposition(
            channel_erosion_rate,
            transport_capacity,
            mobile_sediment,
            self.pixel_scale_m,
            dt
        )
        deposition_m = deposition_rate * dt
        
        # Total elevation change = erosion + hillslope + deposition
        # Note: erosion is negative (removes material), deposition is positive
        total_change = -channel_erosion_m + hillslope_change_m + deposition_m
        
        # Update mobile sediment
        # Weathering adds to regolith, erosion removes from it
        regolith_change = weathering_m - channel_erosion_m + deposition_m
        
        return {
            'channel_erosion': channel_erosion_m,
            'hillslope_change': hillslope_change_m,
            'weathering': weathering_m,
            'deposition': deposition_m,
            'total_change': total_change,
            'regolith_change': regolith_change
        }
    
    def __repr__(self):
        return (f"GeomorphicEngine(dx={self.pixel_scale_m}m, "
                f"{self.channel_erosion}, {self.hillslope_diffusion})")
