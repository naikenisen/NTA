#!/usr/bin/env python3
"""
===============================================================================
NTA FDTD SIMULATION - Interferometric Nanoparticle Imaging
===============================================================================

This module simulates a single camera frame of an interferometric scattering
microscopy (iSCAT) system using FDTD (Finite-Difference Time-Domain) with MEEP.

Physical System:
- Single nanoparticle illuminated by coherent laser
- Rayleigh/Mie scattering computed via Maxwell equations
- INTERFEROMETRIC detection: I = |E_ref + E_scat|²
- Far-field projection with finite numerical aperture
- 2D camera intensity image generation

Interferometry Modes:
- iSCAT (interferometric Scattering): Reference from substrate reflection
- COBRI (Coherent Brightfield): Reference from transmitted beam
- Homodyne: Reference from same source, phase-locked

Signal decomposition:
  I = |E_ref|² + |E_scat|² + 2·Re(E_ref* · E_scat)
      ↑           ↑              ↑
   background   very small    INTERFERENCE TERM (linear in E_scat!)

Advantage: Sensitivity scales as r³ (not r⁶ as in dark-field)

Author: Computational Nanophotonics Research Agent
Date: February 2026

Physical Assumptions (EXPLICIT):
1. Medium: Homogeneous water (n = 1.33), dispersion neglected
2. Particle: Dielectric, size in Rayleigh to Rayleigh-Mie transition
3. Illumination: Monochromatic plane wave (coherent laser)
4. Time: Steady-state (no Brownian motion)
5. Detection: Interferometric - coherent sum of reference and scattered fields
6. Polarization: Linear (user-defined axis)
7. Boundaries: Perfect Matched Layers (PML) absorbing boundaries
8. Reference beam: Phase-locked to illumination (homodyne detection)

===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Callable
import warnings

# MEEP import with graceful fallback for development
try:
    import meep as mp
    MEEP_AVAILABLE = True
except ImportError:
    MEEP_AVAILABLE = False
    warnings.warn("MEEP not installed. Running in simulation-only mode with mock data.")

import miepython

# =============================================================================
# PHYSICAL CONSTANTS AND UNIT SYSTEM
# =============================================================================

@dataclass
class PhysicalConstants:
    """
    Physical constants in SI units.
    MEEP uses a normalized unit system where c = 1.
    We define a length scale 'a' (typically 1 μm) for unit conversion.
    """
    c_vacuum: float = 2.998e8        # Speed of light in vacuum [m/s]
    wavelength_nm: float = 488.0     # Default laser wavelength [nm]
    n_water: float = 1.33            # Refractive index of water (at ~500nm)
    
    @property
    def wavelength_um(self) -> float:
        """Wavelength in micrometers"""
        return self.wavelength_nm / 1000.0
    
    @property
    def wavelength_water(self) -> float:
        """Wavelength in water [μm]"""
        return self.wavelength_um / self.n_water
    
    @property
    def k_water(self) -> float:
        """Wavenumber in water [1/μm]"""
        return 2 * np.pi / self.wavelength_water
    
    @property
    def frequency_meep(self) -> float:
        """
        MEEP frequency in normalized units.
        If a = 1 μm, then f = a/λ = 1/λ[μm]
        """
        return 1.0 / self.wavelength_um


# =============================================================================
# NANOPARTICLE GEOMETRY MODULE
# =============================================================================

@dataclass
class ParticleParameters:
    """
    Parameters defining a single nanoparticle.
    
    Physical basis:
    - Rayleigh regime: 2πr/λ << 1 (typically r < 50nm for visible)
    - Rayleigh-Mie transition: 2πr/λ ~ 0.1-1
    - For 488nm light, transition begins around r ~ 40-80nm
    
    Surface roughness model:
    - Spherical harmonics perturbation of base radius
    - Amplitude: RMS deviation from perfect sphere [nm]
    - Correlation length: Angular scale of roughness features
    """
    # Base geometry
    radius_nm: float = 80.0                  # Base particle radius [nm]
    refractive_index: float = 1.50           # Particle refractive index
    
    # Surface roughness parameters
    roughness_amplitude_nm: float = 0.0      # RMS roughness [nm]
    roughness_lmax: int = 10                 # Max spherical harmonic order
    roughness_seed: int = 42                 # Random seed for reproducibility
    
    # Material properties
    material_name: str = "polystyrene"       # For documentation
    
    @property
    def radius_um(self) -> float:
        """Radius in micrometers (MEEP units with a=1μm)"""
        return self.radius_nm / 1000.0
    
    @property
    def roughness_amplitude_um(self) -> float:
        """Roughness amplitude in micrometers"""
        return self.roughness_amplitude_nm / 1000.0
    
    @property
    def size_parameter(self) -> float:
        """
        Mie size parameter x = 2πr/λ (in medium).
        x << 1: Rayleigh regime
        x ~ 1: Mie regime
        """
        constants = PhysicalConstants()
        return 2 * np.pi * self.radius_nm / (constants.wavelength_nm / constants.n_water)
    
    @property
    def scattering_regime(self) -> str:
        """Determine scattering regime based on size parameter"""
        x = self.size_parameter
        if x < 0.3:
            return "Rayleigh"
        elif x < 1.0:
            return "Rayleigh-Mie transition"
        else:
            return "Mie"
    
    def print_info(self):
        """Print particle physical information"""
        print("=" * 60)
        print("NANOPARTICLE PARAMETERS")
        print("=" * 60)
        print(f"Material: {self.material_name}")
        print(f"Radius: {self.radius_nm:.1f} nm ({self.radius_um:.4f} μm)")
        print(f"Refractive index: {self.refractive_index:.3f}")
        print(f"Size parameter x = 2πr/λ_medium: {self.size_parameter:.3f}")
        print(f"Scattering regime: {self.scattering_regime}")
        print(f"Surface roughness RMS: {self.roughness_amplitude_nm:.1f} nm")
        print(f"Roughness l_max: {self.roughness_lmax}")
        print("=" * 60)


class RoughParticleGeometry:
    """
    Generates rough nanoparticle geometry using spherical harmonics.
    
    Physical model:
    - Base shape: sphere of radius R
    - Perturbation: r(θ,φ) = R + δr(θ,φ)
    - δr is expressed as sum of spherical harmonics Y_lm
    - Coefficients are random with prescribed power spectrum
    
    The perturbation amplitudes follow:
        a_lm ~ N(0, σ_l²) where σ_l decreases with l
    
    This models physical roughness with controllable correlation length.
    Higher l_max = finer surface features (asperities).
    """
    
    def __init__(self, params: ParticleParameters):
        self.params = params
        self.rng = np.random.default_rng(params.roughness_seed)
        self._generate_roughness_coefficients()
    
    def _generate_roughness_coefficients(self):
        """
        Generate spherical harmonic coefficients for surface roughness.
        
        Power spectrum: P(l) ∝ exp(-l/l_c) where l_c controls correlation.
        We use l_c = l_max/3 for reasonable roughness correlation.
        """
        lmax = self.params.roughness_lmax
        amplitude = self.params.roughness_amplitude_um
        
        if amplitude == 0:
            self.alm = {}
            return
        
        # Correlation decay
        l_corr = max(lmax / 3, 1)
        
        self.alm = {}
        total_power = 0
        
        # Generate random coefficients (exclude l=0 to preserve volume, l=1 shifts center)
        for l in range(2, lmax + 1):
            # Power spectrum decay
            power_l = np.exp(-l / l_corr)
            
            for m in range(-l, l + 1):
                # Random amplitude with proper variance
                # Real and imaginary parts (for m != 0)
                if m == 0:
                    self.alm[(l, m)] = self.rng.normal(0, np.sqrt(power_l))
                else:
                    re = self.rng.normal(0, np.sqrt(power_l / 2))
                    im = self.rng.normal(0, np.sqrt(power_l / 2))
                    self.alm[(l, m)] = complex(re, im)
                
                total_power += np.abs(self.alm[(l, m)])**2
        
        # Normalize to desired RMS amplitude
        if total_power > 0:
            norm_factor = amplitude / np.sqrt(total_power)
            for key in self.alm:
                self.alm[key] *= norm_factor
    
    def radius_at_angles(self, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """
        Compute radius at given spherical angles.
        
        Args:
            theta: Polar angle [rad] (0 to π)
            phi: Azimuthal angle [rad] (0 to 2π)
        
        Returns:
            Local radius r(θ,φ) in micrometers
        """
        # scipy >= 1.15 uses sph_harm_y, older versions use sph_harm
        try:
            from scipy.special import sph_harm_y
            use_new_api = True
        except ImportError:
            from scipy.special import sph_harm
            use_new_api = False
        
        r = np.ones_like(theta) * self.params.radius_um
        
        if self.params.roughness_amplitude_nm == 0:
            return r
        
        # Add spherical harmonic perturbation
        for (l, m), coeff in self.alm.items():
            if use_new_api:
                # New scipy API: sph_harm_y(l, m, theta, phi)
                # Returns real spherical harmonics for real m
                Ylm = sph_harm_y(l, m, theta, phi)
            else:
                # Old scipy API: sph_harm(m, l, phi, theta)
                Ylm = sph_harm(m, l, phi, theta)
            r += np.real(coeff * Ylm)
        
        return r
    
    def generate_surface_mesh(self, n_theta: int = 50, n_phi: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate 3D surface mesh for visualization.
        
        Returns:
            X, Y, Z coordinate arrays for surface plotting
        """
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2 * np.pi, n_phi)
        THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
        
        R = self.radius_at_angles(THETA, PHI)
        
        X = R * np.sin(THETA) * np.cos(PHI)
        Y = R * np.sin(THETA) * np.sin(PHI)
        Z = R * np.cos(THETA)
        
        return X, Y, Z
    
    def create_meep_geometry(self, center: Tuple[float, float, float] = (0, 0, 0)) -> List:
        """
        Create MEEP geometry for the rough particle.
        
        For MEEP, we approximate the rough surface using multiple small spheres
        (voxelized representation) or use the material_function approach.
        
        Here we use a custom material function that returns the particle's
        refractive index inside the rough boundary and water outside.
        """
        if not MEEP_AVAILABLE:
            return []
        
        # For smooth sphere, use built-in geometry
        if self.params.roughness_amplitude_nm == 0:
            return [
                mp.Sphere(
                    radius=self.params.radius_um,
                    center=mp.Vector3(*center),
                    material=mp.Medium(index=self.params.refractive_index)
                )
            ]
        
        # For rough particle, we need material function approach
        # This will be handled in the simulation class
        return []
    
    def material_function(self, pos: 'mp.Vector3') -> 'mp.Medium':
        """
        Material function for rough particle geometry.
        Returns particle or background material based on position.
        
        This function is called by MEEP at each grid point.
        """
        if not MEEP_AVAILABLE:
            return None
        
        x, y, z = pos.x, pos.y, pos.z
        r = np.sqrt(x**2 + y**2 + z**2)
        
        if r < 1e-10:  # At center
            return mp.Medium(index=self.params.refractive_index)
        
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        if phi < 0:
            phi += 2 * np.pi
        
        # Local radius from roughness model
        r_local = self.radius_at_angles(np.array([theta]), np.array([phi]))[0]
        
        if r <= r_local:
            return mp.Medium(index=self.params.refractive_index)
        else:
            return mp.Medium(index=PhysicalConstants().n_water)


# =============================================================================
# ILLUMINATION MODULE
# =============================================================================

@dataclass
class IlluminationParameters:
    """
    Laser illumination parameters for NTA system.
    
    Physical basis (Videodrop-like):
    - Coherent laser source (typically 405nm, 488nm, or 532nm)
    - Collimated beam approximated as plane wave
    - Linear polarization
    - Illumination orthogonal to imaging axis
    """
    wavelength_nm: float = 488.0          # Laser wavelength
    polarization: str = "x"               # Polarization axis: "x", "y", or "z"
    propagation_direction: str = "z"      # Beam propagation: "x", "y", or "z"
    amplitude: float = 1.0                # E-field amplitude (normalized)
    
    @property
    def frequency_meep(self) -> float:
        """MEEP frequency (with a = 1 μm)"""
        return 1.0 / (self.wavelength_nm / 1000.0)
    
    @property
    def polarization_vector(self) -> Tuple[float, float, float]:
        """Electric field polarization unit vector"""
        vectors = {
            "x": (1, 0, 0),
            "y": (0, 1, 0),
            "z": (0, 0, 1)
        }
        return vectors.get(self.polarization, (1, 0, 0))
    
    @property
    def k_direction_vector(self) -> Tuple[float, float, float]:
        """Propagation direction unit vector"""
        vectors = {
            "x": (1, 0, 0),
            "y": (0, 1, 0),
            "z": (0, 0, 1)
        }
        return vectors.get(self.propagation_direction, (0, 0, 1))
    
    def print_info(self):
        """Print illumination parameters"""
        print("=" * 60)
        print("ILLUMINATION PARAMETERS")
        print("=" * 60)
        print(f"Wavelength: {self.wavelength_nm} nm")
        print(f"Propagation direction: +{self.propagation_direction}")
        print(f"Polarization: {self.polarization}-polarized")
        print(f"MEEP frequency: {self.frequency_meep:.4f}")
        print("=" * 60)


# =============================================================================
# IMAGING SYSTEM MODULE
# =============================================================================

@dataclass
class ImagingParameters:
    """
    Camera/objective imaging parameters for interferometric detection.
    
    Physical basis:
    - Finite NA objective collects scattered light
    - Only k-vectors within NA cone reach camera
    - Camera integrates intensity over pixel area
    - INTERFEROMETRIC: Reference beam interferes with scattered field
    
    Interferometry modes:
    - iSCAT: I = |E_ref + E_scat|² = |E_ref|² + |E_scat|² + 2·Re(E_ref*·E_scat)
    - The cross-term gives linear sensitivity to E_scat (∝ r³ for Rayleigh)
    - Reference amplitude controls signal-to-background ratio
    - Reference phase controls contrast (π/2 = max imaginary part)
    
    For iSCAT systems:
    - High NA objectives (0.4-1.4) for strong collection
    - Reference from glass/water interface reflection (~4% for glass)
    - Camera: CMOS with high frame rate, ~6.5 μm pixels
    """
    numerical_aperture: float = 0.3       # Objective NA
    magnification: float = 20.0           # Optical magnification
    camera_pixel_size_um: float = 6.5     # Physical pixel size [μm]
    image_pixels: int = 64                # Image size (pixels per side)
    imaging_axis: str = "y"               # Camera viewing axis
    
    # Interferometry parameters
    interferometry_enabled: bool = True   # Enable interferometric detection
    reference_amplitude: float = 1.0      # Reference field amplitude (E_ref/E_incident)
    reference_phase: float = 0.0          # Reference phase [radians] (0=real, π/2=imaginary)
    background_subtraction: bool = True   # Subtract |E_ref|² background
    
    @property
    def object_pixel_size_um(self) -> float:
        """Effective pixel size in object plane"""
        return self.camera_pixel_size_um / self.magnification
    
    @property
    def max_collection_angle_rad(self) -> float:
        """Maximum collection half-angle [rad]"""
        return np.arcsin(min(self.numerical_aperture, 0.999))
    
    @property
    def max_collection_angle_deg(self) -> float:
        """Maximum collection half-angle [degrees]"""
        return np.degrees(self.max_collection_angle_rad)
    
    @property
    def field_of_view_um(self) -> float:
        """Field of view in object plane [μm]"""
        return self.image_pixels * self.object_pixel_size_um
    
    def print_info(self):
        """Print imaging parameters"""
        print("=" * 60)
        print("IMAGING PARAMETERS")
        print("=" * 60)
        print(f"Numerical aperture: {self.numerical_aperture}")
        print(f"Maximum collection angle: {self.max_collection_angle_deg:.1f}°")
        print(f"Magnification: {self.magnification}x")
        print(f"Camera pixel size: {self.camera_pixel_size_um} μm")
        print(f"Object-plane pixel size: {self.object_pixel_size_um:.3f} μm")
        print(f"Image size: {self.image_pixels} x {self.image_pixels} pixels")
        print(f"Field of view: {self.field_of_view_um:.1f} μm")
        print(f"Imaging axis: {self.imaging_axis}")
        print("-" * 60)
        print("INTERFEROMETRY SETTINGS:")
        print(f"  Interferometry enabled: {self.interferometry_enabled}")
        if self.interferometry_enabled:
            print(f"  Reference amplitude: {self.reference_amplitude}")
            print(f"  Reference phase: {self.reference_phase:.2f} rad ({np.degrees(self.reference_phase):.1f}°)")
            print(f"  Background subtraction: {self.background_subtraction}")
            print("  Signal model: I = |E_ref|² + |E_scat|² + 2·Re(E_ref*·E_scat)")
        print("=" * 60)


# =============================================================================
# MEEP FDTD SIMULATION
# =============================================================================

@dataclass
class SimulationParameters:
    """
    FDTD simulation parameters.
    
    Resolution guidelines:
    - Minimum: 10-20 points per wavelength in highest-index material
    - For accuracy: 30-50 points per wavelength
    - For roughness: need to resolve roughness features
    
    PML guidelines:
    - Thickness: typically 0.5-1.0 wavelengths
    - Must be > several grid points
    """
    resolution: int = 50                  # Grid points per μm
    pml_thickness_um: float = 0.5         # PML thickness [μm]
    padding_um: float = 0.3               # Space between particle and PML
    simulation_time_factor: float = 200   # Runtime as multiple of source period
    
    def compute_domain_size(self, particle_radius_um: float) -> float:
        """Compute simulation domain size"""
        return 2 * (particle_radius_um + self.padding_um + self.pml_thickness_um)
    
    def print_info(self, particle_radius_um: float, wavelength_um: float):
        """Print simulation parameters"""
        n_water = PhysicalConstants().n_water
        wavelength_in_water = wavelength_um / n_water
        points_per_wavelength = self.resolution * wavelength_in_water
        
        domain_size = self.compute_domain_size(particle_radius_um)
        total_cells = int(domain_size * self.resolution)**3
        
        print("=" * 60)
        print("FDTD SIMULATION PARAMETERS")
        print("=" * 60)
        print(f"Resolution: {self.resolution} points/μm")
        print(f"Points per wavelength (in water): {points_per_wavelength:.1f}")
        print(f"PML thickness: {self.pml_thickness_um} μm")
        print(f"Simulation domain: {domain_size:.2f} μm (cubic)")
        print(f"Total grid cells: ~{total_cells/1e6:.2f} million")
        print(f"Simulation time: {self.simulation_time_factor} source periods")
        print("=" * 60)


class NTAFDTDSimulation:
    """
    Main FDTD simulation class for interferometric nanoparticle imaging.
    
    Workflow:
    1. Set up simulation domain with PML boundaries
    2. Define nanoparticle geometry (smooth or rough)
    3. Add plane wave source for laser illumination
    4. Define near-to-far-field monitors
    5. Run simulation to steady state
    6. Compute far-field scattering pattern (complex field)
    7. Apply NA filter and generate interferometric camera image
    
    Interferometry model:
    - I = |E_ref + E_scat|² = |E_ref|² + |E_scat|² + 2·Re(E_ref*·E_scat)
    - Reference beam provides phase-locked coherent reference
    - Cross-term gives linear sensitivity to scattered field
    """
    
    def __init__(
        self,
        particle: ParticleParameters,
        illumination: IlluminationParameters,
        imaging: ImagingParameters,
        simulation: SimulationParameters
    ):
        self.particle = particle
        self.illumination = illumination
        self.imaging = imaging
        self.simulation = simulation
        
        self.constants = PhysicalConstants(wavelength_nm=illumination.wavelength_nm)
        self.geometry_gen = RoughParticleGeometry(particle)
        
        self._sim = None
        self._near2far = None
        self._far_field_data = None
        self._camera_image = None
        self._interferometry_data = None  # Stores E_ref, E_scat, decomposed intensities
    
    def print_all_parameters(self):
        """Print complete simulation configuration"""
        self.particle.print_info()
        print()
        self.illumination.print_info()
        print()
        self.imaging.print_info()
        print()
        self.simulation.print_info(
            self.particle.radius_um,
            self.constants.wavelength_um
        )
    
    def _create_source(self, domain_size: float) -> List:
        """Create plane wave source for laser illumination"""
        if not MEEP_AVAILABLE:
            return []
        
        freq = self.illumination.frequency_meep
        pol = self.illumination.polarization_vector
        prop_dir = self.illumination.propagation_direction
        
        # Source position: at negative boundary of propagation axis
        half_domain = domain_size / 2 - self.simulation.pml_thickness_um - 0.05
        
        if prop_dir == "z":
            src_center = mp.Vector3(0, 0, -half_domain)
            src_size = mp.Vector3(domain_size, domain_size, 0)
            k_dir = mp.Vector3(0, 0, 1)
        elif prop_dir == "x":
            src_center = mp.Vector3(-half_domain, 0, 0)
            src_size = mp.Vector3(0, domain_size, domain_size)
            k_dir = mp.Vector3(1, 0, 0)
        else:  # y
            src_center = mp.Vector3(0, -half_domain, 0)
            src_size = mp.Vector3(domain_size, 0, domain_size)
            k_dir = mp.Vector3(0, 1, 0)
        
        # Create eigenmode source for true plane wave
        sources = [
            mp.Source(
                src=mp.ContinuousSource(frequency=freq),
                component=mp.Ex if pol[0] else (mp.Ey if pol[1] else mp.Ez),
                center=src_center,
                size=src_size,
                amplitude=self.illumination.amplitude
            )
        ]
        
        return sources
    
    def _create_near2far_monitors(self, domain_size: float) -> Optional['mp.Near2FarRegion']:
        """
        Create near-to-far-field transformation monitors.
        
        We place flux monitors on all 6 faces of a box surrounding the particle.
        This captures all scattered radiation for far-field transformation.
        """
        if not MEEP_AVAILABLE:
            return None
        
        # Monitor box: just outside particle, inside PML
        monitor_size = self.particle.radius_um + self.simulation.padding_um * 0.5
        
        # Create monitors on all 6 faces
        n2f_regions = [
            # +x face
            mp.Near2FarRegion(
                center=mp.Vector3(monitor_size, 0, 0),
                size=mp.Vector3(0, 2*monitor_size, 2*monitor_size),
                weight=+1
            ),
            # -x face
            mp.Near2FarRegion(
                center=mp.Vector3(-monitor_size, 0, 0),
                size=mp.Vector3(0, 2*monitor_size, 2*monitor_size),
                weight=-1
            ),
            # +y face
            mp.Near2FarRegion(
                center=mp.Vector3(0, monitor_size, 0),
                size=mp.Vector3(2*monitor_size, 0, 2*monitor_size),
                weight=+1
            ),
            # -y face
            mp.Near2FarRegion(
                center=mp.Vector3(0, -monitor_size, 0),
                size=mp.Vector3(2*monitor_size, 0, 2*monitor_size),
                weight=-1
            ),
            # +z face
            mp.Near2FarRegion(
                center=mp.Vector3(0, 0, monitor_size),
                size=mp.Vector3(2*monitor_size, 2*monitor_size, 0),
                weight=+1
            ),
            # -z face
            mp.Near2FarRegion(
                center=mp.Vector3(0, 0, -monitor_size),
                size=mp.Vector3(2*monitor_size, 2*monitor_size, 0),
                weight=-1
            ),
        ]
        
        return n2f_regions
    
    def setup_simulation(self):
        """Initialize MEEP simulation"""
        if not MEEP_AVAILABLE:
            print("MEEP not available. Using mock simulation.")
            return
        
        domain_size = self.simulation.compute_domain_size(self.particle.radius_um)
        
        # Define geometry
        if self.particle.roughness_amplitude_nm == 0:
            # Smooth sphere: use built-in geometry
            geometry = [
                mp.Sphere(
                    radius=self.particle.radius_um,
                    center=mp.Vector3(0, 0, 0),
                    material=mp.Medium(index=self.particle.refractive_index)
                )
            ]
            default_material = mp.Medium(index=self.constants.n_water)
        else:
            # Rough particle: use material function
            geometry = []
            
            # Create epsilon function for rough particle
            def eps_func(pos):
                x, y, z = pos.x, pos.y, pos.z
                r = np.sqrt(x**2 + y**2 + z**2)
                
                if r < 1e-10:
                    return self.particle.refractive_index**2
                
                theta = np.arccos(np.clip(z / r, -1, 1))
                phi = np.arctan2(y, x)
                if phi < 0:
                    phi += 2 * np.pi
                
                r_local = self.geometry_gen.radius_at_angles(
                    np.array([theta]), np.array([phi])
                )[0]
                
                if r <= r_local:
                    return self.particle.refractive_index**2
                else:
                    return self.constants.n_water**2
            
            default_material = mp.Medium(epsilon_func=eps_func)
        
        # Create sources
        sources = self._create_source(domain_size)
        
        # Create near-to-far monitors
        n2f_regions = self._create_near2far_monitors(domain_size)
        
        # Build simulation
        self._sim = mp.Simulation(
            cell_size=mp.Vector3(domain_size, domain_size, domain_size),
            geometry=geometry,
            sources=sources,
            resolution=self.simulation.resolution,
            boundary_layers=[mp.PML(self.simulation.pml_thickness_um)],
            default_material=default_material if self.particle.roughness_amplitude_nm > 0 
                            else mp.Medium(index=self.constants.n_water)
        )
        
        # Add near-to-far-field monitor
        self._near2far = self._sim.add_near2far(
            self.illumination.frequency_meep,
            0, 1,  # Single frequency, 1 point
            *n2f_regions
        )
        
        print("Simulation initialized successfully.")
    
    def run(self, until: Optional[float] = None):
        """
        Run FDTD simulation until steady state.
        
        The simulation runs until fields reach steady state.
        For CW source, this is determined by the field decay.
        """
        if not MEEP_AVAILABLE:
            print("MEEP not available. Generating mock far-field data.")
            self._generate_mock_far_field()
            return
        
        if self._sim is None:
            self.setup_simulation()
        
        # Compute runtime
        if until is None:
            period = 1.0 / self.illumination.frequency_meep
            until = self.simulation.simulation_time_factor * period
        
        print(f"Running FDTD simulation for {until:.1f} time units...")
        print("(This may take several minutes for 3D simulation)")
        
        # Run with progress output
        self._sim.run(
            mp.at_beginning(mp.output_epsilon),
            until=until
        )
        
        print("Simulation complete.")
    
    def _generate_mock_far_field(self):
        """
        Generate mock far-field data for testing without MEEP.
        
        For interferometry, we need the COMPLEX electric field, not just intensity.
        This allows computing the interference term 2·Re(E_ref* · E_scat).
        """
        n_theta = 90
        n_phi = 180
        
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2*np.pi, n_phi)
        THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
        
        # Mock scattering amplitude (complex field, not intensity)
        # Rayleigh scattering: E_scat ∝ α · E_inc · angular_pattern
        # For dipole: E ∝ (1 - sin²θ cos²φ)^0.5 with phase from optical path
        if self.illumination.polarization == "x":
            # For x-polarized light propagating in z:
            angular_pattern = np.sqrt(np.maximum(1.0 - np.sin(THETA)**2 * np.cos(PHI)**2, 0.01))
        else:
            angular_pattern = np.abs(np.sin(THETA))
        
        # Scattering amplitude scales with polarizability α ∝ (n²-1)/(n²+2) × V
        # For Rayleigh: E_scat ∝ k² × α ∝ r³ / λ² × (m²-1)/(m²+2)
        m = self.particle.refractive_index / self.constants.n_water
        polarizability_factor = (m**2 - 1) / (m**2 + 2)
        r = self.particle.radius_um
        k = 2 * np.pi / self.constants.wavelength_um * self.constants.n_water
        
        # Rayleigh scattering amplitude (relative units)
        E_amplitude = (k**2) * (r**3) * np.abs(polarizability_factor) * angular_pattern
        
        # Phase: geometrical phase from scattering direction
        # For a point dipole at origin, phase = k·r (but in far-field, just direction-dependent)
        # Add Gouy phase and scattering phase shift
        base_phase = PHI  # Azimuthal variation
        scattering_phase = np.angle(complex(polarizability_factor))  # Material phase
        
        # Add roughness effects: random phase perturbations (speckle)
        if self.particle.roughness_amplitude_nm > 0:
            np.random.seed(self.particle.roughness_seed)
            roughness_factor = self.particle.roughness_amplitude_nm / self.particle.radius_nm
            
            # Roughness causes amplitude AND phase variations
            amplitude_noise = np.random.randn(n_theta, n_phi) * roughness_factor * 0.3
            phase_noise = np.random.randn(n_theta, n_phi) * roughness_factor * np.pi
            
            # Smooth the noise (correlation from surface features)
            from scipy.ndimage import gaussian_filter
            amplitude_noise = gaussian_filter(amplitude_noise, sigma=2)
            phase_noise = gaussian_filter(phase_noise, sigma=2)
            
            E_amplitude = E_amplitude * (1 + amplitude_noise)
            base_phase = base_phase + phase_noise
        
        # Construct complex field
        E_theta = E_amplitude * np.exp(1j * (base_phase + scattering_phase))
        E_phi = E_amplitude * 0.1 * np.exp(1j * (base_phase + scattering_phase + np.pi/4))  # Cross-pol
        
        # Total intensity
        intensity = np.abs(E_theta)**2 + np.abs(E_phi)**2
        
        self._far_field_data = {
            'theta': THETA,
            'phi': PHI,
            'E_theta': E_theta,           # Complex scattered field (θ component)
            'E_phi': E_phi,               # Complex scattered field (φ component)
            'intensity': intensity,        # |E_scat|²
            'n_theta': n_theta,
            'n_phi': n_phi
        }
    
    def compute_far_field(self, n_theta: int = 90, n_phi: int = 180) -> dict:
        """
        Compute far-field scattering pattern from near-field data.
        
        Uses MEEP's built-in near-to-far-field transformation based on
        the equivalence principle (Love's equivalence theorem).
        
        Returns:
            Dictionary with angular coordinates and field intensities
        """
        if self._far_field_data is not None:
            return self._far_field_data
        
        if not MEEP_AVAILABLE:
            self._generate_mock_far_field()
            return self._far_field_data
        
        if self._near2far is None:
            raise RuntimeError("Simulation not run. Call run() first.")
        
        print("Computing far-field transformation...")
        
        # Define angular grid
        theta_vals = np.linspace(0, np.pi, n_theta)
        phi_vals = np.linspace(0, 2*np.pi, n_phi)
        
        # Far-field distance (should be >> wavelength)
        ff_distance = 1e6  # Effectively infinity for angles
        
        # Compute far field at each angle
        E_theta = np.zeros((n_theta, n_phi), dtype=complex)
        E_phi = np.zeros((n_theta, n_phi), dtype=complex)
        
        for i, theta in enumerate(theta_vals):
            for j, phi in enumerate(phi_vals):
                # Direction vector in Cartesian coordinates
                x = ff_distance * np.sin(theta) * np.cos(phi)
                y = ff_distance * np.sin(theta) * np.sin(phi)
                z = ff_distance * np.cos(theta)
                
                # Get far-field
                ff = self._sim.get_farfield(
                    self._near2far,
                    mp.Vector3(x, y, z)
                )
                
                # ff is [Ex, Ey, Ez, Hx, Hy, Hz]
                # Convert to spherical components
                cos_t, sin_t = np.cos(theta), np.sin(theta)
                cos_p, sin_p = np.cos(phi), np.sin(phi)
                
                # Unit vectors
                r_hat = np.array([sin_t*cos_p, sin_t*sin_p, cos_t])
                theta_hat = np.array([cos_t*cos_p, cos_t*sin_p, -sin_t])
                phi_hat = np.array([-sin_p, cos_p, 0])
                
                E_cart = np.array([ff[0], ff[1], ff[2]])
                E_theta[i, j] = np.dot(E_cart, theta_hat)
                E_phi[i, j] = np.dot(E_cart, phi_hat)
        
        # Intensity (time-averaged Poynting vector)
        intensity = np.abs(E_theta)**2 + np.abs(E_phi)**2
        
        THETA, PHI = np.meshgrid(theta_vals, phi_vals, indexing='ij')
        
        self._far_field_data = {
            'theta': THETA,
            'phi': PHI,
            'E_theta': E_theta,
            'E_phi': E_phi,
            'intensity': intensity,
            'n_theta': n_theta,
            'n_phi': n_phi
        }
        
        print("Far-field computation complete.")
        return self._far_field_data
    
    def generate_camera_image(self) -> np.ndarray:
        """
        Generate 2D camera image with INTERFEROMETRIC detection.
        
        Physical model for interferometry (iSCAT-like):
        ================================================
        
        The detected intensity is:
            I = |E_ref + E_scat|²
              = |E_ref|² + |E_scat|² + 2·Re(E_ref* · E_scat)
              
        Where:
        - |E_ref|² : Reference beam intensity (constant background)
        - |E_scat|² : Scattered intensity (∝ r⁶ for Rayleigh, very small)
        - 2·Re(E_ref* · E_scat) : INTERFERENCE TERM (∝ r³, dominant signal!)
        
        The interference term provides:
        1. Linear sensitivity to particle polarizability (∝ r³)
        2. Phase information preservation
        3. Greatly enhanced signal-to-noise for small particles
        
        Reference beam model:
        - E_ref = A_ref · exp(i·φ_ref) · PSF_ref
        - Uniform across the PSF for plane wave reference
        """
        if self._far_field_data is None:
            self.compute_far_field()
        
        ff = self._far_field_data
        theta = ff['theta']
        phi = ff['phi']
        
        # Image parameters
        N = self.imaging.image_pixels
        pixel_size = self.imaging.object_pixel_size_um
        fov = self.imaging.field_of_view_um
        NA = self.imaging.numerical_aperture
        imaging_axis = self.imaging.imaging_axis
        
        # Interferometry parameters
        interferometry = self.imaging.interferometry_enabled
        E_ref_amplitude = self.imaging.reference_amplitude
        E_ref_phase = self.imaging.reference_phase
        background_subtraction = self.imaging.background_subtraction
        
        # Maximum collection angle
        theta_max = self.imaging.max_collection_angle_rad
        
        if interferometry:
            print(f"Generating INTERFEROMETRIC camera image ({N}x{N} pixels, NA={NA})...")
            print(f"  Reference: amplitude={E_ref_amplitude}, phase={np.degrees(E_ref_phase):.1f}°")
        else:
            print(f"Generating camera image ({N}x{N} pixels, NA={NA})...")
        
        # Define image plane coordinates
        x_img = np.linspace(-fov/2, fov/2, N)
        y_img = np.linspace(-fov/2, fov/2, N)
        X_IMG, Y_IMG = np.meshgrid(x_img, y_img, indexing='ij')
        
        # Camera direction based on imaging axis
        if imaging_axis == "y":
            cam_dir = np.array([0, 1, 0])
        elif imaging_axis == "x":
            cam_dir = np.array([1, 0, 0])
        else:
            cam_dir = np.array([0, 0, 1])
        
        # Convert far-field angles to direction vectors
        dirs_x = np.sin(theta) * np.cos(phi)
        dirs_y = np.sin(theta) * np.sin(phi)
        dirs_z = np.cos(theta)
        
        # Angle from camera axis
        cos_angle_to_cam = dirs_x * cam_dir[0] + dirs_y * cam_dir[1] + dirs_z * cam_dir[2]
        angle_to_cam = np.arccos(np.clip(cos_angle_to_cam, -1, 1))
        
        # Mask for NA cone
        in_na_cone = angle_to_cam <= theta_max
        
        # PSF parameters
        wavelength_um = self.constants.wavelength_um
        sigma_psf = 0.42 * wavelength_um / NA  # Gaussian approximation to Airy
        
        # Distance from center
        R_img = np.sqrt(X_IMG**2 + Y_IMG**2)
        
        # =================================================================
        # SCATTERED FIELD (complex amplitude)
        # =================================================================
        
        # Get complex scattered field components
        if 'E_theta' in ff and 'E_phi' in ff:
            E_theta = ff['E_theta']
            E_phi = ff['E_phi']
            
            # Average complex field within NA cone (weighted by solid angle)
            E_scat_collected = np.mean(E_theta[in_na_cone]) + np.mean(E_phi[in_na_cone])
        else:
            # Fallback: use intensity only (no phase info)
            E_scat_collected = np.sqrt(np.mean(ff['intensity'][in_na_cone]))
        
        # Scattered field spatial distribution (PSF-shaped)
        # The scattered light from a point source creates a PSF pattern
        psf_amplitude = np.exp(-R_img**2 / (4 * sigma_psf**2))  # Amplitude, not intensity
        
        # Add phase curvature (Gouy phase + defocus)
        # For focused PSF: phase varies across the spot
        k = 2 * np.pi / wavelength_um * self.constants.n_water
        phase_curvature = k * R_img**2 / (4 * fov)  # Approximate quadratic phase
        
        # Roughness-induced phase variations
        if self.particle.roughness_amplitude_nm > 0:
            np.random.seed(self.particle.roughness_seed + 100)
            roughness_factor = self.particle.roughness_amplitude_nm / self.particle.radius_nm
            
            phase_var = np.random.randn(N, N) * roughness_factor * np.pi
            from scipy.ndimage import gaussian_filter
            phase_var = gaussian_filter(phase_var, sigma=N/20)
        else:
            phase_var = 0
        
        # Complex scattered field at image plane
        E_scat = E_scat_collected * psf_amplitude * np.exp(1j * (phase_curvature + phase_var))
        
        # =================================================================
        # REFERENCE FIELD (for interferometry)
        # =================================================================
        
        if interferometry:
            # Reference beam: uniform plane wave (or slightly structured)
            # E_ref = A_ref · exp(i·φ_ref)
            # La phase de référence inclut le déphasage dû à la position Z de la particule
            # Pour une particule à z=0 (centre du domaine), on utilise la phase de référence initiale
            # Le déphasage Gouy et la phase de diffusion sont déjà dans E_scat
            
            # E_ref = A_ref · exp(i·φ_ref) · PSF_ref
            E_ref = E_ref_amplitude * np.exp(1j * E_ref_phase) * psf_amplitude
            
            # =================================================================
            # INTERFEROMETRIC SIGNAL
            # =================================================================
            # I = |E_ref + E_scat|² = |E_ref|² + |E_scat|² + 2·Re(E_ref* · E_scat)
            
            E_total = E_ref + E_scat
            I_total = np.abs(E_total)**2
            
            # Decompose for analysis
            I_ref = np.abs(E_ref)**2                           # Background
            I_scat = np.abs(E_scat)**2                         # Pure scattering (weak)
            I_interference = 2 * np.real(np.conj(E_ref) * E_scat)  # Cross-term (strong!)
            
            # Store decomposition for analysis
            self._interferometry_data = {
                'I_ref': I_ref,
                'I_scat': I_scat,
                'I_interference': I_interference,
                'I_total': I_total,
                'E_ref': E_ref,
                'E_scat': E_scat
            }
            
            if background_subtraction:
                # Return interference signal (background-subtracted)
                # This is what's typically analyzed in iSCAT
                image = I_total - I_ref.mean()  # Subtract mean background
                print(f"  Background subtracted: I_ref_mean = {I_ref.mean():.4f}")
            else:
                image = I_total
            
            # Print signal analysis
            print(f"  |E_scat|² max: {np.max(I_scat):.2e}")
            print(f"  Interference term max: {np.max(np.abs(I_interference)):.2e}")
            print(f"  Signal enhancement: {np.max(np.abs(I_interference)) / (np.max(I_scat) + 1e-20):.1f}x")
            
        else:
            # =================================================================
            # NON-INTERFEROMETRIC (dark-field/scattering only)
            # =================================================================
            I_scat = np.abs(E_scat)**2
            image = I_scat
            
            self._interferometry_data = None
        
        self._camera_image = image
        
        print(f"Camera image generated. Max intensity: {np.max(image):.4e}")
        print(f"                        Min intensity: {np.min(image):.4e}")
        return self._camera_image
    
    def plot_results(self, save_path: Optional[str] = None):
        """
        Create comprehensive visualization of simulation results.
        
        Includes:
        1. Particle geometry (3D surface)
        2. Far-field scattering pattern
        3. Camera image
        4. Physical parameters summary
        """
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Particle geometry (3D)
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        X, Y, Z = self.geometry_gen.generate_surface_mesh()
        ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        ax1.set_xlabel('X [μm]')
        ax1.set_ylabel('Y [μm]')
        ax1.set_zlabel('Z [μm]')
        ax1.set_title(f'Particle Geometry\n(R={self.particle.radius_nm:.0f}nm, '
                     f'roughness={self.particle.roughness_amplitude_nm:.0f}nm)')
        
        # Set equal aspect ratio
        max_range = self.particle.radius_um * 1.5
        ax1.set_xlim([-max_range, max_range])
        ax1.set_ylim([-max_range, max_range])
        ax1.set_zlim([-max_range, max_range])
        
        # 2. Particle cross-section (shows roughness)
        ax2 = fig.add_subplot(2, 3, 2)
        theta = np.linspace(0, 2*np.pi, 200)
        phi_slice = np.zeros_like(theta)  # Equatorial plane
        r_equator = self.geometry_gen.radius_at_angles(
            np.ones_like(theta) * np.pi/2, theta
        )
        x_eq = r_equator * np.cos(theta)
        y_eq = r_equator * np.sin(theta)
        ax2.plot(x_eq * 1000, y_eq * 1000, 'b-', linewidth=1.5, label='Rough')
        # Reference circle
        r_ref = self.particle.radius_nm / 1000
        ax2.plot(r_ref * 1000 * np.cos(theta), r_ref * 1000 * np.sin(theta), 
                'r--', linewidth=1, label='Smooth ref.')
        ax2.set_xlabel('X [nm]')
        ax2.set_ylabel('Y [nm]')
        ax2.set_title('Equatorial Cross-Section')
        ax2.set_aspect('equal')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Far-field scattering pattern (polar plot)
        ax3 = fig.add_subplot(2, 3, 3, projection='polar')
        if self._far_field_data is not None:
            ff = self._far_field_data
            # Plot slice at φ = 0 (x-z plane)
            n_theta = ff['n_theta']
            theta_1d = ff['theta'][:, 0]
            intensity_slice = ff['intensity'][:, 0]
            
            ax3.plot(theta_1d, intensity_slice / np.max(intensity_slice), 'b-', linewidth=1.5)
            ax3.set_title('Far-Field Scattering\n(φ=0 plane)')
            ax3.set_theta_zero_location('N')
        else:
            ax3.text(0.5, 0.5, 'Run simulation first', ha='center')
        
        # 4. Far-field 2D pattern
        ax4 = fig.add_subplot(2, 3, 4)
        if self._far_field_data is not None:
            ff = self._far_field_data
            im = ax4.pcolormesh(
                np.degrees(ff['theta']), 
                np.degrees(ff['phi']),
                ff['intensity'].T,
                cmap='hot',
                shading='auto'
            )
            ax4.set_xlabel('θ [degrees]')
            ax4.set_ylabel('φ [degrees]')
            ax4.set_title('Far-Field Intensity Map')
            plt.colorbar(im, ax=ax4, label='Intensity [a.u.]')
        
        # 5. Camera image (with interferometry details if enabled)
        ax5 = fig.add_subplot(2, 3, 5)
        if self._camera_image is not None:
            fov = self.imaging.field_of_view_um
            extent = [-fov/2, fov/2, -fov/2, fov/2]
            
            # Choose colormap based on interferometry (can be negative with bg subtraction)
            if self.imaging.interferometry_enabled and self.imaging.background_subtraction:
                cmap = 'RdBu_r'
                vmax = np.max(np.abs(self._camera_image))
                vmin = -vmax
            else:
                cmap = 'gray'
                vmin, vmax = None, None
            
            im = ax5.imshow(
                self._camera_image.T,
                origin='lower',
                extent=extent,
                cmap=cmap,
                aspect='equal',
                vmin=vmin, vmax=vmax
            )
            ax5.set_xlabel('X [μm]')
            ax5.set_ylabel('Z [μm]')
            
            if self.imaging.interferometry_enabled:
                title = f'Interferometric Image\n(NA={self.imaging.numerical_aperture})'
                if self.imaging.background_subtraction:
                    title += ' [bg subtracted]'
            else:
                title = f'Scattering Image\n(NA={self.imaging.numerical_aperture})'
            ax5.set_title(title)
            plt.colorbar(im, ax=ax5, label='Intensity [a.u.]')
        
        # 6. Parameter summary (with interferometry info)
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.axis('off')
        
        # Build interferometry string
        if self.imaging.interferometry_enabled:
            interf_text = f"""
Interferometry:
  • Mode: ENABLED (iSCAT-like)
  • Reference amplitude: {self.imaging.reference_amplitude}
  • Reference phase: {np.degrees(self.imaging.reference_phase):.1f}°
  • Background subtraction: {self.imaging.background_subtraction}
  • Signal: I = |E_ref|² + |E_scat|² + 2·Re(E_ref*·E_scat)"""
        else:
            interf_text = """
Interferometry:
  • Mode: DISABLED (scattering only)
  • Signal: I = |E_scat|²"""
        
        params_text = f"""
PHYSICAL PARAMETERS
{'='*40}
Particle:
  • Radius: {self.particle.radius_nm:.1f} nm
  • Refractive index: {self.particle.refractive_index:.3f}
  • Size parameter: {self.particle.size_parameter:.3f}
  • Regime: {self.particle.scattering_regime}
  • Roughness RMS: {self.particle.roughness_amplitude_nm:.1f} nm
  • Roughness l_max: {self.particle.roughness_lmax}

Illumination:
  • Wavelength: {self.illumination.wavelength_nm:.0f} nm
  • Polarization: {self.illumination.polarization}
  • Propagation: +{self.illumination.propagation_direction}

Imaging:
  • NA: {self.imaging.numerical_aperture}
  • Magnification: {self.imaging.magnification}x
  • Pixel size (object): {self.imaging.object_pixel_size_um:.3f} μm
  • Field of view: {self.imaging.field_of_view_um:.1f} μm
{interf_text}

Medium:
  • Refractive index: {self.constants.n_water}
  • Wavelength in medium: {self.constants.wavelength_water*1000:.1f} nm
"""
        ax6.text(0.05, 0.95, params_text, transform=ax6.transAxes,
                fontfamily='monospace', fontsize=9,
                verticalalignment='top')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
        return fig


# =============================================================================
# COMPARISON FUNCTIONS
# =============================================================================

def compare_smooth_vs_rough(
    radius_nm: float = 80.0,
    roughness_nm: float = 5.0,
    roughness_lmax: int = 10,
    save_path: Optional[str] = None
) -> Tuple[NTAFDTDSimulation, NTAFDTDSimulation]:
    """
    Run comparison between smooth and rough particles.
    
    Args:
        radius_nm: Base particle radius
        roughness_nm: RMS roughness for rough particle
        roughness_lmax: Maximum spherical harmonic order
        save_path: Optional path to save comparison figure
    
    Returns:
        Tuple of (smooth_simulation, rough_simulation)
    """
    print("=" * 70)
    print("SMOOTH vs ROUGH PARTICLE COMPARISON")
    print("=" * 70)
    
    # Common parameters
    illumination = IlluminationParameters(
        wavelength_nm=488.0,
        polarization="x",
        propagation_direction="z"
    )
    
    imaging = ImagingParameters(
        numerical_aperture=0.3,
        magnification=20.0,
        image_pixels=64
    )
    
    simulation = SimulationParameters(
        resolution=50,
        simulation_time_factor=100
    )
    
    # Smooth particle
    print("\n--- SMOOTH PARTICLE ---")
    particle_smooth = ParticleParameters(
        radius_nm=radius_nm,
        refractive_index=1.50,
        roughness_amplitude_nm=0.0,
        material_name="polystyrene (smooth)"
    )
    
    sim_smooth = NTAFDTDSimulation(
        particle=particle_smooth,
        illumination=illumination,
        imaging=imaging,
        simulation=simulation
    )
    
    sim_smooth.run()
    sim_smooth.compute_far_field()
    sim_smooth.generate_camera_image()
    
    # Rough particle
    print("\n--- ROUGH PARTICLE ---")
    particle_rough = ParticleParameters(
        radius_nm=radius_nm,
        refractive_index=1.50,
        roughness_amplitude_nm=roughness_nm,
        roughness_lmax=roughness_lmax,
        material_name="polystyrene (rough)"
    )
    
    sim_rough = NTAFDTDSimulation(
        particle=particle_rough,
        illumination=illumination,
        imaging=imaging,
        simulation=simulation
    )
    
    sim_rough.run()
    sim_rough.compute_far_field()
    sim_rough.generate_camera_image()
    
    # Create comparison figure
    fig = plt.figure(figsize=(14, 10))
    
    # Row 1: Particle geometry
    ax1 = fig.add_subplot(2, 4, 1, projection='3d')
    X, Y, Z = sim_smooth.geometry_gen.generate_surface_mesh()
    ax1.plot_surface(X, Y, Z, cmap='Blues', alpha=0.8)
    ax1.set_title('Smooth Particle')
    ax1.set_xlabel('X [μm]')
    max_r = particle_smooth.radius_um * 1.5
    ax1.set_xlim([-max_r, max_r])
    ax1.set_ylim([-max_r, max_r])
    ax1.set_zlim([-max_r, max_r])
    
    ax2 = fig.add_subplot(2, 4, 2, projection='3d')
    X, Y, Z = sim_rough.geometry_gen.generate_surface_mesh()
    ax2.plot_surface(X, Y, Z, cmap='Reds', alpha=0.8)
    ax2.set_title(f'Rough Particle\n(σ={roughness_nm}nm, l_max={roughness_lmax})')
    ax2.set_xlabel('X [μm]')
    ax2.set_xlim([-max_r, max_r])
    ax2.set_ylim([-max_r, max_r])
    ax2.set_zlim([-max_r, max_r])
    
    # Row 1: Far-field patterns
    ax3 = fig.add_subplot(2, 4, 3, projection='polar')
    ff_smooth = sim_smooth._far_field_data
    theta_1d = ff_smooth['theta'][:, 0]
    ax3.plot(theta_1d, ff_smooth['intensity'][:, 0] / np.max(ff_smooth['intensity']), 
            'b-', label='Smooth')
    ff_rough = sim_rough._far_field_data
    ax3.plot(theta_1d, ff_rough['intensity'][:, 0] / np.max(ff_rough['intensity']), 
            'r-', label='Rough')
    ax3.set_title('Far-Field (φ=0)')
    ax3.legend(loc='upper right', fontsize=8)
    
    ax4 = fig.add_subplot(2, 4, 4, projection='polar')
    ax4.plot(theta_1d, ff_smooth['intensity'][:, ff_smooth['n_phi']//4] / np.max(ff_smooth['intensity']), 
            'b-', label='Smooth')
    ax4.plot(theta_1d, ff_rough['intensity'][:, ff_rough['n_phi']//4] / np.max(ff_rough['intensity']), 
            'r-', label='Rough')
    ax4.set_title('Far-Field (φ=π/2)')
    ax4.legend(loc='upper right', fontsize=8)
    
    # Row 2: Camera images
    ax5 = fig.add_subplot(2, 4, 5)
    fov = imaging.field_of_view_um
    extent = [-fov/2, fov/2, -fov/2, fov/2]
    im5 = ax5.imshow(sim_smooth._camera_image.T, origin='lower', extent=extent, 
                    cmap='gray', aspect='equal')
    ax5.set_xlabel('X [μm]')
    ax5.set_ylabel('Z [μm]')
    ax5.set_title('Camera Image (Smooth)')
    plt.colorbar(im5, ax=ax5, shrink=0.8)
    
    ax6 = fig.add_subplot(2, 4, 6)
    im6 = ax6.imshow(sim_rough._camera_image.T, origin='lower', extent=extent, 
                    cmap='gray', aspect='equal')
    ax6.set_xlabel('X [μm]')
    ax6.set_ylabel('Z [μm]')
    ax6.set_title('Camera Image (Rough)')
    plt.colorbar(im6, ax=ax6, shrink=0.8)
    
    # Row 2: Difference and profiles
    ax7 = fig.add_subplot(2, 4, 7)
    diff = sim_rough._camera_image - sim_smooth._camera_image
    im7 = ax7.imshow(diff.T, origin='lower', extent=extent, 
                    cmap='RdBu', aspect='equal', 
                    vmin=-np.max(np.abs(diff)), vmax=np.max(np.abs(diff)))
    ax7.set_xlabel('X [μm]')
    ax7.set_ylabel('Z [μm]')
    ax7.set_title('Difference (Rough - Smooth)')
    plt.colorbar(im7, ax=ax7, shrink=0.8)
    
    ax8 = fig.add_subplot(2, 4, 8)
    N = imaging.image_pixels
    x_coords = np.linspace(-fov/2, fov/2, N)
    ax8.plot(x_coords, sim_smooth._camera_image[N//2, :], 'b-', label='Smooth', linewidth=2)
    ax8.plot(x_coords, sim_rough._camera_image[N//2, :], 'r-', label='Rough', linewidth=2)
    ax8.set_xlabel('Position [μm]')
    ax8.set_ylabel('Intensity [a.u.]')
    ax8.set_title('Central Line Profile')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    plt.suptitle(f'Smooth vs Rough Particle Comparison\n'
                f'R={radius_nm}nm, n={particle_smooth.refractive_index}, '
                f'λ={illumination.wavelength_nm}nm, NA={imaging.numerical_aperture}',
                fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nComparison figure saved to {save_path}")
    
    plt.show()
    
    # Print analysis
    print("\n" + "=" * 70)
    print("PHYSICAL ANALYSIS: EFFECT OF SURFACE ROUGHNESS")
    print("=" * 70)
    
    # Compute statistics
    smooth_total = np.sum(sim_smooth._camera_image)
    rough_total = np.sum(sim_rough._camera_image)
    smooth_max = np.max(sim_smooth._camera_image)
    rough_max = np.max(sim_rough._camera_image)
    
    # Compute image contrast/structure
    smooth_std = np.std(sim_smooth._camera_image)
    rough_std = np.std(sim_rough._camera_image)
    
    print(f"""
IMAGE STATISTICS:
  Smooth particle:
    - Total intensity: {smooth_total:.4f}
    - Peak intensity:  {smooth_max:.4f}
    - Std deviation:   {smooth_std:.4f}
    
  Rough particle:
    - Total intensity: {rough_total:.4f}
    - Peak intensity:  {rough_max:.4f}
    - Std deviation:   {rough_std:.4f}
    
RELATIVE CHANGES:
    - Total intensity change: {(rough_total/smooth_total - 1)*100:+.1f}%
    - Peak intensity change:  {(rough_max/smooth_max - 1)*100:+.1f}%
    - Structure (std) change: {(rough_std/smooth_std - 1)*100:+.1f}%

PHYSICAL INTERPRETATION:
""")
    
    print("""
Surface roughness affects the observed camera image through several mechanisms:

1. MODIFIED SCATTERING CROSS-SECTION
   - Roughness introduces additional scattering centers (asperities)
   - For Rayleigh regime: σ ∝ |α|² where α is polarizability
   - Surface perturbations modify the effective polarizability tensor
   - Small roughness (σ << R): perturbative effect, σ_scat increases slightly
   - Large roughness: can significantly increase total scattering

2. ANGULAR REDISTRIBUTION
   - Smooth sphere: scattering pattern follows Mie theory (symmetric lobes)
   - Roughness breaks the spherical symmetry
   - Scattering is redirected to angles outside the main lobes
   - Higher-order multipoles are excited by surface features
   - l_max parameter controls the angular frequency of perturbations

3. COHERENT INTERFERENCE (SPECKLE-LIKE EFFECTS)
   - Different surface points act as coherent secondary sources
   - Random phases from irregular surface → speckle in far-field
   - This manifests as intensity fluctuations across the PSF
   - For highly coherent illumination (laser): pronounced speckle

4. DEPOLARIZATION
   - Perfect sphere maintains incident polarization in scattering
   - Roughness introduces cross-polarized components
   - Reduces contrast in polarization-sensitive detection

5. IMAGE MORPHOLOGY
   - Smooth particle: clean, symmetric Airy-like PSF
   - Rough particle: distorted PSF with asymmetric features
   - Sidelobe structure is modified
   - Central peak may be broader or narrower depending on roughness scale

For NTA applications:
- Roughness can affect size estimation from diffusion (indirectly via shape)
- Scattering intensity calibration curves must account for surface quality
- Biological particles (exosomes) often have membrane roughness ~1-5nm
""")
    
    return sim_smooth, sim_rough


def spectral_analysis_smooth_vs_rough(
    radius_nm: float = 80.0,
    roughness_nm: float = 5.0,
    roughness_lmax: int = 10,
    reference_amplitude: float = 1.0,
    save_path: Optional[str] = None
) -> dict:
    """
    Perform spectral (FFT) analysis comparing smooth vs rough particles in interferometry.
    
    Physical basis:
    ===============
    The camera image contains spatial frequencies determined by:
    
    1. PSF (Point Spread Function):
       - Cutoff frequency: f_c = NA / λ
       - For NA=0.3, λ=488nm: f_c ≈ 0.6 cycles/μm
    
    2. Scattering angular distribution:
       - Smooth particle: clean Mie/Rayleigh pattern → low frequencies
       - Rough particle: speckle from random phases → high frequencies
    
    3. Interference pattern:
       - Phase variations from roughness create fine structure
       - FFT reveals these as enhanced high-frequency components
    
    The spectral signature of roughness:
    - Increased power at high spatial frequencies
    - Broader spectral width
    - Reduced peak-to-background ratio in spectrum
    
    Args:
        radius_nm: Particle radius
        roughness_nm: RMS roughness for rough particle
        roughness_lmax: Maximum spherical harmonic order
        reference_amplitude: Reference beam amplitude
        save_path: Optional path to save figure
    
    Returns:
        Dictionary containing spectral analysis results
    """
    print("=" * 70)
    print("SPECTRAL ANALYSIS: SMOOTH vs ROUGH PARTICLE (INTERFEROMETRY)")
    print("=" * 70)
    
    # Common parameters
    illumination = IlluminationParameters(
        wavelength_nm=488.0,
        polarization="x",
        propagation_direction="z"
    )
    
    imaging = ImagingParameters(
        numerical_aperture=0.3,
        magnification=20.0,
        image_pixels=128,  # Higher resolution for better FFT
        interferometry_enabled=True,
        reference_amplitude=reference_amplitude,
        reference_phase=0.0,
        background_subtraction=True
    )
    
    simulation = SimulationParameters(
        resolution=50,
        simulation_time_factor=100
    )
    
    # --- SMOOTH PARTICLE ---
    print("\n--- SMOOTH PARTICLE (interferometry) ---")
    particle_smooth = ParticleParameters(
        radius_nm=radius_nm,
        refractive_index=1.50,
        roughness_amplitude_nm=0.0,
        material_name="polystyrene (smooth)"
    )
    
    sim_smooth = NTAFDTDSimulation(
        particle=particle_smooth,
        illumination=illumination,
        imaging=imaging,
        simulation=simulation
    )
    
    sim_smooth.run()
    sim_smooth.compute_far_field()
    sim_smooth.generate_camera_image()
    
    # --- ROUGH PARTICLE ---
    print("\n--- ROUGH PARTICLE (interferometry) ---")
    particle_rough = ParticleParameters(
        radius_nm=radius_nm,
        refractive_index=1.50,
        roughness_amplitude_nm=roughness_nm,
        roughness_lmax=roughness_lmax,
        material_name="polystyrene (rough)"
    )
    
    sim_rough = NTAFDTDSimulation(
        particle=particle_rough,
        illumination=illumination,
        imaging=imaging,
        simulation=simulation
    )
    
    sim_rough.run()
    sim_rough.compute_far_field()
    sim_rough.generate_camera_image()
    
    # =================================================================
    # SPECTRAL ANALYSIS (2D FFT)
    # =================================================================
    print("\n" + "-" * 70)
    print("COMPUTING 2D FFT SPECTRAL ANALYSIS")
    print("-" * 70)
    
    # Get images
    img_smooth = sim_smooth._camera_image
    img_rough = sim_rough._camera_image
    
    N = imaging.image_pixels
    fov = imaging.field_of_view_um
    pixel_size = fov / N  # μm per pixel
    
    # Spatial frequency axis
    freq = np.fft.fftfreq(N, d=pixel_size)  # cycles per μm
    freq_shift = np.fft.fftshift(freq)
    FX, FY = np.meshgrid(freq_shift, freq_shift, indexing='ij')
    F_radial = np.sqrt(FX**2 + FY**2)
    
    # 2D FFT (centered)
    fft_smooth = np.fft.fftshift(np.fft.fft2(img_smooth))
    fft_rough = np.fft.fftshift(np.fft.fft2(img_rough))
    
    # Power spectral density
    psd_smooth = np.abs(fft_smooth)**2
    psd_rough = np.abs(fft_rough)**2
    
    # Normalize by DC component for comparison
    psd_smooth_norm = psd_smooth / (np.max(psd_smooth) + 1e-20)
    psd_rough_norm = psd_rough / (np.max(psd_rough) + 1e-20)
    
    # =================================================================
    # RADIAL AVERAGING (azimuthally averaged power spectrum)
    # =================================================================
    
    def radial_profile(data, center=None):
        """Compute azimuthally averaged radial profile of 2D data."""
        if center is None:
            center = np.array(data.shape) // 2
        
        y, x = np.indices(data.shape)
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        r = r.astype(int)
        
        # Radial bins
        r_max = int(np.max(r))
        tbin = np.bincount(r.ravel(), weights=data.ravel())
        nr = np.bincount(r.ravel())
        radial_mean = tbin / (nr + 1e-10)
        
        return radial_mean[:r_max]
    
    radial_smooth = radial_profile(psd_smooth_norm)
    radial_rough = radial_profile(psd_rough_norm)
    
    # Frequency axis for radial profile (in cycles/μm)
    freq_max = np.max(np.abs(freq_shift))
    n_radial = len(radial_smooth)
    freq_radial = np.linspace(0, freq_max, n_radial)
    
    # =================================================================
    # SPECTRAL METRICS
    # =================================================================
    
    # 1. Spectral centroid (mean frequency weighted by power)
    def spectral_centroid(radial_psd, freq_axis):
        return np.sum(freq_axis * radial_psd) / (np.sum(radial_psd) + 1e-20)
    
    centroid_smooth = spectral_centroid(radial_smooth, freq_radial)
    centroid_rough = spectral_centroid(radial_rough, freq_radial)
    
    # 2. Spectral width (RMS bandwidth)
    def spectral_width(radial_psd, freq_axis, centroid):
        return np.sqrt(np.sum((freq_axis - centroid)**2 * radial_psd) / (np.sum(radial_psd) + 1e-20))
    
    width_smooth = spectral_width(radial_smooth, freq_radial, centroid_smooth)
    width_rough = spectral_width(radial_rough, freq_radial, centroid_rough)
    
    # 3. High-frequency power ratio (power above f_c / total power)
    f_cutoff = imaging.numerical_aperture / (illumination.wavelength_nm / 1000)  # cycles/μm
    idx_high = freq_radial > f_cutoff * 0.5  # Above half the cutoff
    
    hf_ratio_smooth = np.sum(radial_smooth[idx_high]) / (np.sum(radial_smooth) + 1e-20)
    hf_ratio_rough = np.sum(radial_rough[idx_high]) / (np.sum(radial_rough) + 1e-20)
    
    # 4. Spectral entropy (measure of spectral spread)
    def spectral_entropy(psd_norm):
        psd_prob = psd_norm / (np.sum(psd_norm) + 1e-20)
        psd_prob = psd_prob[psd_prob > 0]
        return -np.sum(psd_prob * np.log2(psd_prob + 1e-20))
    
    entropy_smooth = spectral_entropy(radial_smooth)
    entropy_rough = spectral_entropy(radial_rough)
    
    # Store results
    results = {
        'img_smooth': img_smooth,
        'img_rough': img_rough,
        'fft_smooth': fft_smooth,
        'fft_rough': fft_rough,
        'psd_smooth': psd_smooth_norm,
        'psd_rough': psd_rough_norm,
        'radial_smooth': radial_smooth,
        'radial_rough': radial_rough,
        'freq_radial': freq_radial,
        'freq_2d': (FX, FY),
        'metrics': {
            'centroid_smooth': centroid_smooth,
            'centroid_rough': centroid_rough,
            'width_smooth': width_smooth,
            'width_rough': width_rough,
            'hf_ratio_smooth': hf_ratio_smooth,
            'hf_ratio_rough': hf_ratio_rough,
            'entropy_smooth': entropy_smooth,
            'entropy_rough': entropy_rough,
            'f_cutoff': f_cutoff
        }
    }
    
    # =================================================================
    # VISUALIZATION
    # =================================================================
    
    fig = plt.figure(figsize=(18, 14))
    
    fov = imaging.field_of_view_um
    extent_space = [-fov/2, fov/2, -fov/2, fov/2]
    extent_freq = [freq_shift.min(), freq_shift.max(), freq_shift.min(), freq_shift.max()]
    
    # Row 1: Spatial domain images
    ax1 = fig.add_subplot(3, 4, 1)
    vmax = np.max(np.abs(img_smooth))
    im1 = ax1.imshow(img_smooth.T, origin='lower', extent=extent_space,
                     cmap='RdBu_r', aspect='equal', vmin=-vmax, vmax=vmax)
    ax1.set_xlabel('X [μm]')
    ax1.set_ylabel('Z [μm]')
    ax1.set_title('Smooth Particle\n(Interferometric Image)')
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    
    ax2 = fig.add_subplot(3, 4, 2)
    vmax = np.max(np.abs(img_rough))
    im2 = ax2.imshow(img_rough.T, origin='lower', extent=extent_space,
                     cmap='RdBu_r', aspect='equal', vmin=-vmax, vmax=vmax)
    ax2.set_xlabel('X [μm]')
    ax2.set_ylabel('Z [μm]')
    ax2.set_title(f'Rough Particle (σ={roughness_nm}nm)\n(Interferometric Image)')
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    
    # Difference image
    ax3 = fig.add_subplot(3, 4, 3)
    diff = img_rough - img_smooth
    vmax_diff = np.max(np.abs(diff))
    im3 = ax3.imshow(diff.T, origin='lower', extent=extent_space,
                     cmap='RdBu_r', aspect='equal', vmin=-vmax_diff, vmax=vmax_diff)
    ax3.set_xlabel('X [μm]')
    ax3.set_ylabel('Z [μm]')
    ax3.set_title('Difference\n(Rough - Smooth)')
    plt.colorbar(im3, ax=ax3, shrink=0.8)
    
    # Line profiles
    ax4 = fig.add_subplot(3, 4, 4)
    x_coords = np.linspace(-fov/2, fov/2, N)
    ax4.plot(x_coords, img_smooth[N//2, :], 'b-', linewidth=2, label='Smooth')
    ax4.plot(x_coords, img_rough[N//2, :], 'r-', linewidth=1.5, alpha=0.8, label='Rough')
    ax4.set_xlabel('Position [μm]')
    ax4.set_ylabel('Intensity')
    ax4.set_title('Central Line Profile')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Row 2: Frequency domain (2D PSD)
    ax5 = fig.add_subplot(3, 4, 5)
    im5 = ax5.imshow(np.log10(psd_smooth_norm.T + 1e-10), origin='lower', extent=extent_freq,
                     cmap='viridis', aspect='equal', vmin=-6, vmax=0)
    ax5.set_xlabel('$f_x$ [cycles/μm]')
    ax5.set_ylabel('$f_y$ [cycles/μm]')
    ax5.set_title('Power Spectrum (Smooth)\nlog₁₀(PSD)')
    plt.colorbar(im5, ax=ax5, shrink=0.8)
    # Add NA cutoff circle
    theta_circle = np.linspace(0, 2*np.pi, 100)
    ax5.plot(f_cutoff * np.cos(theta_circle), f_cutoff * np.sin(theta_circle), 
             'r--', linewidth=1.5, label=f'NA cutoff ({f_cutoff:.2f})')
    
    ax6 = fig.add_subplot(3, 4, 6)
    im6 = ax6.imshow(np.log10(psd_rough_norm.T + 1e-10), origin='lower', extent=extent_freq,
                     cmap='viridis', aspect='equal', vmin=-6, vmax=0)
    ax6.set_xlabel('$f_x$ [cycles/μm]')
    ax6.set_ylabel('$f_y$ [cycles/μm]')
    ax6.set_title('Power Spectrum (Rough)\nlog₁₀(PSD)')
    plt.colorbar(im6, ax=ax6, shrink=0.8)
    ax6.plot(f_cutoff * np.cos(theta_circle), f_cutoff * np.sin(theta_circle), 
             'r--', linewidth=1.5)
    
    # PSD difference
    ax7 = fig.add_subplot(3, 4, 7)
    psd_diff = np.log10(psd_rough_norm + 1e-10) - np.log10(psd_smooth_norm + 1e-10)
    vmax_psd = np.max(np.abs(psd_diff))
    im7 = ax7.imshow(psd_diff.T, origin='lower', extent=extent_freq,
                     cmap='RdBu_r', aspect='equal', vmin=-vmax_psd, vmax=vmax_psd)
    ax7.set_xlabel('$f_x$ [cycles/μm]')
    ax7.set_ylabel('$f_y$ [cycles/μm]')
    ax7.set_title('PSD Difference\nlog₁₀(Rough/Smooth)')
    plt.colorbar(im7, ax=ax7, shrink=0.8)
    ax7.plot(f_cutoff * np.cos(theta_circle), f_cutoff * np.sin(theta_circle), 
             'k--', linewidth=1.5)
    
    # Radial PSD comparison
    ax8 = fig.add_subplot(3, 4, 8)
    ax8.semilogy(freq_radial, radial_smooth, 'b-', linewidth=2, label='Smooth')
    ax8.semilogy(freq_radial, radial_rough, 'r-', linewidth=2, label='Rough')
    ax8.axvline(x=f_cutoff, color='k', linestyle='--', alpha=0.5, label=f'NA cutoff')
    ax8.axvline(x=centroid_smooth, color='b', linestyle=':', alpha=0.7)
    ax8.axvline(x=centroid_rough, color='r', linestyle=':', alpha=0.7)
    ax8.set_xlabel('Spatial Frequency [cycles/μm]')
    ax8.set_ylabel('Radial PSD (normalized)')
    ax8.set_title('Azimuthally Averaged\nPower Spectrum')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    ax8.set_xlim([0, freq_max])
    
    # Row 3: Detailed analysis
    ax9 = fig.add_subplot(3, 4, 9)
    ratio = radial_rough / (radial_smooth + 1e-20)
    ax9.plot(freq_radial, ratio, 'g-', linewidth=2)
    ax9.axhline(y=1, color='k', linestyle='--', alpha=0.5)
    ax9.axvline(x=f_cutoff, color='k', linestyle='--', alpha=0.5, label='NA cutoff')
    ax9.set_xlabel('Spatial Frequency [cycles/μm]')
    ax9.set_ylabel('Power Ratio (Rough/Smooth)')
    ax9.set_title('Spectral Ratio\n(Roughness Effect)')
    ax9.grid(True, alpha=0.3)
    ax9.set_xlim([0, freq_max])
    ax9.set_ylim([0, np.min([np.max(ratio), 10])])
    
    # Cumulative power
    ax10 = fig.add_subplot(3, 4, 10)
    cum_smooth = np.cumsum(radial_smooth) / np.sum(radial_smooth)
    cum_rough = np.cumsum(radial_rough) / np.sum(radial_rough)
    ax10.plot(freq_radial, cum_smooth, 'b-', linewidth=2, label='Smooth')
    ax10.plot(freq_radial, cum_rough, 'r-', linewidth=2, label='Rough')
    ax10.axvline(x=f_cutoff, color='k', linestyle='--', alpha=0.5)
    ax10.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax10.set_xlabel('Spatial Frequency [cycles/μm]')
    ax10.set_ylabel('Cumulative Power')
    ax10.set_title('Cumulative Power Spectrum')
    ax10.legend()
    ax10.grid(True, alpha=0.3)
    ax10.set_xlim([0, freq_max])
    
    # Phase spectrum comparison
    ax11 = fig.add_subplot(3, 4, 11)
    phase_smooth = np.angle(fft_smooth)
    phase_rough = np.angle(fft_rough)
    phase_diff = np.angle(fft_rough * np.conj(fft_smooth))
    im11 = ax11.imshow(phase_diff.T, origin='lower', extent=extent_freq,
                      cmap='twilight', aspect='equal', vmin=-np.pi, vmax=np.pi)
    ax11.set_xlabel('$f_x$ [cycles/μm]')
    ax11.set_ylabel('$f_y$ [cycles/μm]')
    ax11.set_title('Phase Difference\n(Rough - Smooth)')
    plt.colorbar(im11, ax=ax11, shrink=0.8, label='Phase [rad]')
    
    # Metrics summary
    ax12 = fig.add_subplot(3, 4, 12)
    ax12.axis('off')
    
    metrics_text = f"""
SPECTRAL ANALYSIS METRICS
{'='*50}

Spectral Centroid (mean frequency):
  • Smooth: {centroid_smooth:.3f} cycles/μm
  • Rough:  {centroid_rough:.3f} cycles/μm
  • Shift:  {(centroid_rough - centroid_smooth):.3f} cycles/μm
           ({(centroid_rough/centroid_smooth - 1)*100:+.1f}%)

Spectral Width (RMS bandwidth):
  • Smooth: {width_smooth:.3f} cycles/μm
  • Rough:  {width_rough:.3f} cycles/μm
  • Change: {(width_rough/width_smooth - 1)*100:+.1f}%

High-Frequency Power Ratio (f > {f_cutoff*0.5:.2f} cycles/μm):
  • Smooth: {hf_ratio_smooth*100:.2f}%
  • Rough:  {hf_ratio_rough*100:.2f}%
  • Increase: {(hf_ratio_rough/hf_ratio_smooth - 1)*100:+.1f}%

Spectral Entropy (spread measure):
  • Smooth: {entropy_smooth:.2f} bits
  • Rough:  {entropy_rough:.2f} bits

PHYSICAL INTERPRETATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Surface roughness introduces random phase variations
in the scattered field, leading to:
• Higher spectral centroid (more high-freq content)
• Broader spectral width (less concentrated power)
• Increased high-frequency power ratio
• Higher spectral entropy (more disordered spectrum)

These spectral signatures can be used to:
✓ Detect surface roughness from single images
✓ Estimate roughness parameters
✓ Distinguish smooth vs textured particles
"""
    ax12.text(0.02, 0.98, metrics_text, transform=ax12.transAxes,
              fontfamily='monospace', fontsize=8.5,
              verticalalignment='top')
    
    plt.suptitle(f'Spectral Analysis: Smooth vs Rough Particle (Interferometry)\n'
                 f'R={radius_nm}nm, σ_rough={roughness_nm}nm, l_max={roughness_lmax}, '
                 f'λ=488nm, NA={imaging.numerical_aperture}',
                 fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to {save_path}")
    
    plt.show()
    
    # Print summary
    print("\n" + "=" * 70)
    print("SPECTRAL ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"""
Particle: R = {radius_nm} nm, Roughness = {roughness_nm} nm

Key Findings:
  • Spectral centroid shift: {(centroid_rough - centroid_smooth):.3f} cycles/μm ({(centroid_rough/centroid_smooth - 1)*100:+.1f}%)
  • Spectral width increase: {(width_rough/width_smooth - 1)*100:+.1f}%
  • High-frequency power increase: {(hf_ratio_rough/hf_ratio_smooth - 1)*100:+.1f}%

Conclusion: Roughness signature is {'DETECTABLE' if abs(hf_ratio_rough/hf_ratio_smooth - 1) > 0.1 else 'WEAK'} in the spectral domain.
""")
    
    return results


def compare_interferometry_vs_scattering(
    radius_nm: float = 80.0,
    roughness_nm: float = 0.0,
    reference_amplitude: float = 1.0,
    reference_phase: float = 0.0,
    save_path: Optional[str] = None
) -> Tuple[NTAFDTDSimulation, NTAFDTDSimulation]:
    """
    Compare interferometric detection vs pure scattering detection.
    
    This demonstrates the key advantage of interferometry (iSCAT):
    - Scattering: I ∝ |E_scat|² ∝ r⁶ (Rayleigh)
    - Interferometry: I ∝ 2·Re(E_ref*·E_scat) ∝ r³ (linear in polarizability!)
    
    Args:
        radius_nm: Particle radius
        roughness_nm: RMS surface roughness
        reference_amplitude: Reference beam amplitude (controls contrast)
        reference_phase: Reference phase (0=real, π/2=imaginary component)
        save_path: Optional path to save comparison figure
    
    Returns:
        Tuple of (scattering_simulation, interferometric_simulation)
    """
    print("=" * 70)
    print("INTERFEROMETRY vs SCATTERING COMPARISON")
    print("=" * 70)
    
    # Common parameters
    particle = ParticleParameters(
        radius_nm=radius_nm,
        refractive_index=1.50,
        roughness_amplitude_nm=roughness_nm,
        material_name="polystyrene"
    )
    
    illumination = IlluminationParameters(
        wavelength_nm=488.0,
        polarization="x",
        propagation_direction="z"
    )
    
    simulation = SimulationParameters(
        resolution=50,
        simulation_time_factor=100
    )
    
    # --- SCATTERING ONLY (dark-field like) ---
    print("\n--- SCATTERING ONLY (dark-field) ---")
    imaging_scattering = ImagingParameters(
        numerical_aperture=0.3,
        magnification=20.0,
        image_pixels=64,
        interferometry_enabled=False  # No reference beam
    )
    
    sim_scattering = NTAFDTDSimulation(
        particle=particle,
        illumination=illumination,
        imaging=imaging_scattering,
        simulation=simulation
    )
    
    sim_scattering.run()
    sim_scattering.compute_far_field()
    sim_scattering.generate_camera_image()
    
    # --- INTERFEROMETRIC (iSCAT-like) ---
    print("\n--- INTERFEROMETRIC DETECTION (iSCAT) ---")
    imaging_interf = ImagingParameters(
        numerical_aperture=0.3,
        magnification=20.0,
        image_pixels=64,
        interferometry_enabled=True,
        reference_amplitude=reference_amplitude,
        reference_phase=reference_phase,
        background_subtraction=True
    )
    
    sim_interf = NTAFDTDSimulation(
        particle=particle,
        illumination=illumination,
        imaging=imaging_interf,
        simulation=simulation
    )
    
    sim_interf.run()
    sim_interf.compute_far_field()
    sim_interf.generate_camera_image()
    
    # --- Create comparison figure ---
    fig = plt.figure(figsize=(16, 12))
    
    fov = imaging_scattering.field_of_view_um
    extent = [-fov/2, fov/2, -fov/2, fov/2]
    N = imaging_scattering.image_pixels
    x_coords = np.linspace(-fov/2, fov/2, N)
    
    # Row 1: Scattering only
    ax1 = fig.add_subplot(2, 4, 1)
    im1 = ax1.imshow(sim_scattering._camera_image.T, origin='lower', extent=extent,
                     cmap='hot', aspect='equal')
    ax1.set_xlabel('X [μm]')
    ax1.set_ylabel('Z [μm]')
    ax1.set_title('Scattering Only\n(dark-field)')
    plt.colorbar(im1, ax=ax1, shrink=0.8, label='I = |E_scat|²')
    
    ax2 = fig.add_subplot(2, 4, 2)
    ax2.plot(x_coords, sim_scattering._camera_image[N//2, :], 'r-', linewidth=2)
    ax2.set_xlabel('Position [μm]')
    ax2.set_ylabel('Intensity [a.u.]')
    ax2.set_title('Scattering: Line Profile')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    ax2.set_ylim([1e-10, 1])
    
    # Row 1: Interferometry
    ax3 = fig.add_subplot(2, 4, 3)
    interf_img = sim_interf._camera_image
    vmax = np.max(np.abs(interf_img))
    im3 = ax3.imshow(interf_img.T, origin='lower', extent=extent,
                     cmap='RdBu_r', aspect='equal', vmin=-vmax, vmax=vmax)
    ax3.set_xlabel('X [μm]')
    ax3.set_ylabel('Z [μm]')
    ax3.set_title(f'Interferometric (iSCAT)\nφ_ref={np.degrees(reference_phase):.0f}°')
    plt.colorbar(im3, ax=ax3, shrink=0.8, label='ΔI (bg subtracted)')
    
    ax4 = fig.add_subplot(2, 4, 4)
    ax4.plot(x_coords, sim_interf._camera_image[N//2, :], 'b-', linewidth=2)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax4.set_xlabel('Position [μm]')
    ax4.set_ylabel('ΔIntensity [a.u.]')
    ax4.set_title('Interferometry: Line Profile')
    ax4.grid(True, alpha=0.3)
    
    # Row 2: Decomposition of interferometric signal
    if sim_interf._interferometry_data is not None:
        interf_data = sim_interf._interferometry_data
        
        ax5 = fig.add_subplot(2, 4, 5)
        im5 = ax5.imshow(interf_data['I_scat'].T, origin='lower', extent=extent,
                         cmap='hot', aspect='equal')
        ax5.set_xlabel('X [μm]')
        ax5.set_ylabel('Z [μm]')
        ax5.set_title('|E_scat|² component\n(very weak, ∝ r⁶)')
        plt.colorbar(im5, ax=ax5, shrink=0.8)
        
        ax6 = fig.add_subplot(2, 4, 6)
        I_interf = interf_data['I_interference']
        vmax_i = np.max(np.abs(I_interf))
        im6 = ax6.imshow(I_interf.T, origin='lower', extent=extent,
                         cmap='RdBu_r', aspect='equal', vmin=-vmax_i, vmax=vmax_i)
        ax6.set_xlabel('X [μm]')
        ax6.set_ylabel('Z [μm]')
        ax6.set_title('2·Re(E_ref*·E_scat)\n(strong, ∝ r³)')
        plt.colorbar(im6, ax=ax6, shrink=0.8)
        
        ax7 = fig.add_subplot(2, 4, 7)
        I_ref = interf_data['I_ref']
        im7 = ax7.imshow(I_ref.T, origin='lower', extent=extent,
                         cmap='gray', aspect='equal')
        ax7.set_xlabel('X [μm]')
        ax7.set_ylabel('Z [μm]')
        ax7.set_title('|E_ref|² (background)\n(uniform reference)')
        plt.colorbar(im7, ax=ax7, shrink=0.8)
    
    # Summary panel
    ax8 = fig.add_subplot(2, 4, 8)
    ax8.axis('off')
    
    # Compute signal comparison
    scat_max = np.max(sim_scattering._camera_image)
    interf_max = np.max(np.abs(sim_interf._camera_image))
    
    if sim_interf._interferometry_data is not None:
        I_scat_max = np.max(sim_interf._interferometry_data['I_scat'])
        I_cross_max = np.max(np.abs(sim_interf._interferometry_data['I_interference']))
        enhancement = I_cross_max / (I_scat_max + 1e-20)
    else:
        enhancement = 0
    
    summary_text = f"""
INTERFEROMETRY ADVANTAGE
{'='*45}

Signal Scaling (Rayleigh regime):
  • Scattering:      I ∝ |α|² ∝ r⁶
  • Interferometry:  I ∝ Re(α) ∝ r³

For this simulation (r = {radius_nm} nm):
  • Scattering max:     {scat_max:.2e}
  • |E_scat|² max:      {I_scat_max:.2e}
  • Interference max:   {I_cross_max:.2e}
  • Enhancement factor: {enhancement:.1f}x

Physical Interpretation:
  The interference term 2·Re(E_ref*·E_scat)
  is LINEAR in E_scat, giving r³ scaling.
  
  This enables detection of particles
  ~10-100x smaller than dark-field!

Applications:
  • iSCAT: Single protein detection
  • COBRI: Label-free imaging
  • Mass photometry: Molecular weighing
"""
    ax8.text(0.02, 0.98, summary_text, transform=ax8.transAxes,
             fontfamily='monospace', fontsize=9,
             verticalalignment='top')
    
    plt.suptitle(f'Scattering vs Interferometric Detection\n'
                 f'Particle: R={radius_nm}nm, n=1.50, λ=488nm',
                 fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nComparison figure saved to {save_path}")
    
    plt.show()
    
    return sim_scattering, sim_interf


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_single_simulation(
    radius_nm: float = 80.0,
    roughness_nm: float = 0.0,
    roughness_lmax: int = 10,
    refractive_index: float = 1.50,
    wavelength_nm: float = 488.0,
    numerical_aperture: float = 0.3,
    save_results: bool = True
):
    """
    Run a single NTA FDTD simulation with specified parameters.
    
    This is the main entry point for running individual simulations.
    """
    print("=" * 70)
    print("NTA FDTD SIMULATION - SINGLE PARTICLE")
    print("=" * 70)
    
    # Create parameter objects
    particle = ParticleParameters(
        radius_nm=radius_nm,
        refractive_index=refractive_index,
        roughness_amplitude_nm=roughness_nm,
        roughness_lmax=roughness_lmax
    )
    
    illumination = IlluminationParameters(
        wavelength_nm=wavelength_nm,
        polarization="x",
        propagation_direction="z"
    )
    
    imaging = ImagingParameters(
        numerical_aperture=numerical_aperture,
        magnification=20.0,
        image_pixels=64
    )
    
    simulation = SimulationParameters(
        resolution=50,
        simulation_time_factor=100
    )
    
    # Create and run simulation
    sim = NTAFDTDSimulation(
        particle=particle,
        illumination=illumination,
        imaging=imaging,
        simulation=simulation
    )
    
    sim.print_all_parameters()
    
    print("\n" + "-" * 70)
    print("RUNNING SIMULATION")
    print("-" * 70)
    
    sim.run()
    sim.compute_far_field()
    sim.generate_camera_image()
    
    # Plot results
    save_path = "nta_simulation_results.png" if save_results else None
    sim.plot_results(save_path=save_path)
    
    return sim


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Interferometric FDTD Simulation for Nanoparticle Imaging (iSCAT-like)"
    )
    parser.add_argument("--mode", choices=["single", "compare", "interferometry", "spectral"], 
                       default="interferometry",
                       help="Simulation mode: 'single', 'compare' (smooth/rough), 'interferometry' (iSCAT vs scattering), or 'spectral' (FFT analysis)")
    parser.add_argument("--radius", type=float, default=80.0,
                       help="Particle radius in nm")
    parser.add_argument("--roughness", type=float, default=5.0,
                       help="Surface roughness RMS in nm")
    parser.add_argument("--lmax", type=int, default=10,
                       help="Maximum spherical harmonic order for roughness")
    parser.add_argument("--n-particle", type=float, default=1.50,
                       help="Particle refractive index")
    parser.add_argument("--wavelength", type=float, default=488.0,
                       help="Laser wavelength in nm")
    parser.add_argument("--na", type=float, default=0.3,
                       help="Objective numerical aperture")
    parser.add_argument("--ref-amplitude", type=float, default=1.0,
                       help="Reference beam amplitude for interferometry")
    parser.add_argument("--ref-phase", type=float, default=0.0,
                       help="Reference beam phase in degrees for interferometry")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save output figures")
    
    args = parser.parse_args()
    
    if args.mode == "single":
        sim = run_single_simulation(
            radius_nm=args.radius,
            roughness_nm=args.roughness,
            roughness_lmax=args.lmax,
            refractive_index=args.n_particle,
            wavelength_nm=args.wavelength,
            numerical_aperture=args.na,
            save_results=not args.no_save
        )
    elif args.mode == "compare":
        sim_smooth, sim_rough = compare_smooth_vs_rough(
            radius_nm=args.radius,
            roughness_nm=args.roughness,
            roughness_lmax=args.lmax,
            save_path="nta_comparison.png" if not args.no_save else None
        )
    elif args.mode == "spectral":
        results = spectral_analysis_smooth_vs_rough(
            radius_nm=args.radius,
            roughness_nm=args.roughness,
            roughness_lmax=args.lmax,
            reference_amplitude=args.ref_amplitude,
            save_path="spectral_analysis.png" if not args.no_save else None
        )
    else:  # interferometry mode
        sim_scat, sim_interf = compare_interferometry_vs_scattering(
            radius_nm=args.radius,
            roughness_nm=args.roughness,
            reference_amplitude=args.ref_amplitude,
            reference_phase=np.radians(args.ref_phase),
            save_path="interferometry_comparison.png" if not args.no_save else None
        )
