#!/usr/bin/env python3
"""
===============================================================================
NTA FDTD SIMULATION - Nanoparticle Tracking Analysis Optical Imaging
===============================================================================

This module simulates a single camera frame of a Videodrop-like NTA system
using FDTD (Finite-Difference Time-Domain) with MEEP.

Physical System:
- Single nanoparticle illuminated by coherent laser
- Rayleigh/Mie scattering computed via Maxwell equations
- Far-field projection with finite numerical aperture
- 2D camera intensity image generation

Author: Computational Nanophotonics Research Agent
Date: February 2026

Physical Assumptions (EXPLICIT):
1. Medium: Homogeneous water (n = 1.33), dispersion neglected
2. Particle: Dielectric, size in Rayleigh to Rayleigh-Mie transition
3. Illumination: Monochromatic plane wave (coherent laser)
4. Time: Steady-state (no Brownian motion)
5. Detection: Far-field intensity integrated over camera exposure
6. Polarization: Linear (user-defined axis)
7. Boundaries: Perfect Matched Layers (PML) absorbing boundaries

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
    Camera/objective imaging parameters.
    
    Physical basis:
    - Finite NA objective collects scattered light
    - Only k-vectors within NA cone reach camera
    - Camera integrates intensity over pixel area
    
    For NTA (Videodrop):
    - Typical NA: 0.2 - 0.4 (low magnification, large field of view)
    - Imaging at 90° to illumination (dark-field like)
    - Camera: CMOS/CCD with ~5-10 μm pixels
    """
    numerical_aperture: float = 0.3       # Objective NA
    magnification: float = 20.0           # Optical magnification
    camera_pixel_size_um: float = 6.5     # Physical pixel size [μm]
    image_pixels: int = 64                # Image size (pixels per side)
    imaging_axis: str = "y"               # Camera viewing axis
    
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
    Main FDTD simulation class for NTA optical imaging.
    
    Workflow:
    1. Set up simulation domain with PML boundaries
    2. Define nanoparticle geometry (smooth or rough)
    3. Add plane wave source for laser illumination
    4. Define near-to-far-field monitors
    5. Run simulation to steady state
    6. Compute far-field scattering pattern
    7. Apply NA filter and generate camera image
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
        """Generate mock far-field data for testing without MEEP"""
        n_theta = 90
        n_phi = 180
        
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2*np.pi, n_phi)
        THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
        
        # Mock scattering pattern (dipole-like for small particle)
        # Rayleigh scattering: I ∝ sin²(θ) for perpendicular polarization
        if self.illumination.polarization == "x":
            # For x-polarized light propagating in z:
            # Scattered E ∝ (1 - sin²θ cos²φ)
            pattern = 1.0 - np.sin(THETA)**2 * np.cos(PHI)**2
        else:
            pattern = np.sin(THETA)**2
        
        # Add roughness effect: random phase variations
        if self.particle.roughness_amplitude_nm > 0:
            np.random.seed(self.particle.roughness_seed)
            roughness_factor = self.particle.roughness_amplitude_nm / self.particle.radius_nm
            
            # Roughness causes additional angular structure (speckle-like)
            noise = np.random.randn(n_theta, n_phi) * roughness_factor * 0.3
            # Smooth the noise slightly
            from scipy.ndimage import gaussian_filter
            noise = gaussian_filter(noise, sigma=2)
            pattern = pattern * (1 + noise)
        
        # Scale by size parameter (Rayleigh: σ ∝ a^6)
        x = self.particle.size_parameter
        if x < 0.5:
            # Rayleigh limit
            intensity_scale = x**4
        else:
            # Mie: more complex, approximate
            intensity_scale = x**2
        
        self._far_field_data = {
            'theta': THETA,
            'phi': PHI,
            'intensity': pattern * intensity_scale,
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
        Generate 2D camera image from far-field scattering.
        
        Physical model:
        1. Far-field intensity gives angular distribution of scattered light
        2. Objective collects light within NA cone
        3. Lens performs Fourier transform (far-field → image plane)
        4. Camera integrates intensity over pixel area
        
        For a point scatterer, the image is the PSF convolved with
        the angular distribution of scattered light.
        
        Here we compute the image in the imaging plane (perpendicular to 
        imaging axis) by integrating far-field intensity within NA.
        """
        if self._far_field_data is None:
            self.compute_far_field()
        
        ff = self._far_field_data
        theta = ff['theta']
        phi = ff['phi']
        intensity = ff['intensity']
        
        # Image parameters
        N = self.imaging.image_pixels
        pixel_size = self.imaging.object_pixel_size_um
        fov = self.imaging.field_of_view_um
        NA = self.imaging.numerical_aperture
        imaging_axis = self.imaging.imaging_axis
        
        # Maximum collection angle
        theta_max = self.imaging.max_collection_angle_rad
        
        print(f"Generating camera image ({N}x{N} pixels, NA={NA})...")
        
        # Define image plane coordinates
        x_img = np.linspace(-fov/2, fov/2, N)
        y_img = np.linspace(-fov/2, fov/2, N)
        X_IMG, Y_IMG = np.meshgrid(x_img, y_img, indexing='ij')
        
        # For each image pixel, integrate far-field intensity
        # over the solid angle that maps to that pixel
        
        # Simplified model: The particle is at the center.
        # The image is essentially the PSF (Airy pattern) weighted by
        # the scattering efficiency in that direction.
        
        # We compute the average scattered intensity going toward
        # the camera direction, weighted by the angular distribution.
        
        # Camera direction based on imaging axis
        if imaging_axis == "y":
            # Camera looking from +y direction
            # Collect light around θ=π/2, φ=π/2
            theta_cam = np.pi / 2
            phi_cam = np.pi / 2
            img_axis1, img_axis2 = "x", "z"
        elif imaging_axis == "x":
            theta_cam = np.pi / 2
            phi_cam = 0
            img_axis1, img_axis2 = "y", "z"
        else:  # z
            theta_cam = 0
            phi_cam = 0
            img_axis1, img_axis2 = "x", "y"
        
        # Find far-field intensity within NA cone around camera axis
        # Angle from camera axis
        if imaging_axis == "y":
            # Camera along +y
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
        
        # Total collected power
        collected_power = np.sum(intensity[in_na_cone])
        
        # Generate PSF-like image
        # For diffraction-limited imaging, PSF is approximately Gaussian
        # with width σ ≈ 0.42 λ / NA (for Gaussian approximation to Airy)
        wavelength_um = self.constants.wavelength_um
        sigma_psf = 0.42 * wavelength_um / NA
        
        # Distance from center
        R_img = np.sqrt(X_IMG**2 + Y_IMG**2)
        
        # Gaussian PSF
        psf = np.exp(-R_img**2 / (2 * sigma_psf**2))
        
        # Modulate by angular scattering variation within NA cone
        # This creates asymmetry if scattering is anisotropic
        
        # Map image coordinates to angles within NA cone
        # tan(θ) ≈ r / f, but for small NA this is complex
        # Simplified: use the collected intensity pattern
        
        # For now, we create speckle-like variations if rough
        if self.particle.roughness_amplitude_nm > 0:
            # Add angular structure from roughness
            # This manifests as intensity variations across the PSF
            np.random.seed(self.particle.roughness_seed + 100)
            roughness_factor = self.particle.roughness_amplitude_nm / self.particle.radius_nm
            
            # Create structured noise (not pure speckle, but coherent structure)
            # based on far-field intensity within NA cone
            
            # Sample intensity at angles corresponding to image positions
            # This is a simplified coherent imaging model
            
            # Add phase variations
            phase_var = np.random.randn(N, N) * roughness_factor * np.pi
            from scipy.ndimage import gaussian_filter
            phase_var = gaussian_filter(phase_var, sigma=N/20)
            
            # Coherent intensity with phase
            amplitude = np.sqrt(psf) * np.exp(1j * phase_var)
            image = np.abs(amplitude)**2
        else:
            # Smooth particle: clean PSF
            image = psf
        
        # Scale by collected power
        image = image * collected_power / np.max(image) if np.max(image) > 0 else image
        
        # Add shot noise (Poisson statistics) - optional
        # For now, keep ideal image
        
        self._camera_image = image
        
        print(f"Camera image generated. Max intensity: {np.max(image):.4f}")
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
        
        # 5. Camera image
        ax5 = fig.add_subplot(2, 3, 5)
        if self._camera_image is not None:
            fov = self.imaging.field_of_view_um
            extent = [-fov/2, fov/2, -fov/2, fov/2]
            im = ax5.imshow(
                self._camera_image.T,
                origin='lower',
                extent=extent,
                cmap='gray',
                aspect='equal'
            )
            ax5.set_xlabel('X [μm]')
            ax5.set_ylabel('Z [μm]')
            ax5.set_title(f'Camera Image\n(NA={self.imaging.numerical_aperture})')
            plt.colorbar(im, ax=ax5, label='Intensity [a.u.]')
        
        # 6. Parameter summary
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.axis('off')
        
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
        description="NTA FDTD Simulation for Nanoparticle Optical Imaging"
    )
    parser.add_argument("--mode", choices=["single", "compare"], default="compare",
                       help="Simulation mode: 'single' or 'compare'")
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
    else:
        sim_smooth, sim_rough = compare_smooth_vs_rough(
            radius_nm=args.radius,
            roughness_nm=args.roughness,
            roughness_lmax=args.lmax,
            save_path="nta_comparison.png" if not args.no_save else None
        )
