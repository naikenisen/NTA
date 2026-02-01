#!/usr/bin/env python3
"""
===============================================================================
NTA FDTD DEMONSTRATION - Quick Start Example
===============================================================================

This script demonstrates the NTA FDTD simulation without requiring MEEP.
It uses the mock mode to show the workflow and expected outputs.

Run this to verify the code works before installing MEEP.
"""

# First, check if scipy is available (needed for spherical harmonics)
try:
    from scipy.special import sph_harm
    from scipy.ndimage import gaussian_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("WARNING: scipy not installed. Some features will be limited.")
    print("Install with: pip install scipy")

import numpy as np
import matplotlib.pyplot as plt

# Import our simulation module
from nta_fdtd_simulation import (
    ParticleParameters,
    IlluminationParameters,
    ImagingParameters,
    SimulationParameters,
    NTAFDTDSimulation,
    RoughParticleGeometry,
    compare_smooth_vs_rough
)


def demo_particle_geometry():
    """
    Demonstrate particle geometry generation with different roughness levels.
    """
    print("=" * 70)
    print("DEMO 1: PARTICLE GEOMETRY WITH SURFACE ROUGHNESS")
    print("=" * 70)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), 
                             subplot_kw={'projection': '3d'})
    
    roughness_values = [0, 2, 5, 10]  # nm
    lmax_values = [5, 15]  # low and high angular resolution
    
    for i, lmax in enumerate(lmax_values):
        for j, roughness in enumerate(roughness_values):
            ax = axes[i, j]
            
            params = ParticleParameters(
                radius_nm=80.0,
                refractive_index=1.50,
                roughness_amplitude_nm=roughness,
                roughness_lmax=lmax,
                roughness_seed=42
            )
            
            geom = RoughParticleGeometry(params)
            X, Y, Z = geom.generate_surface_mesh(n_theta=40, n_phi=80)
            
            ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9)
            ax.set_title(f'σ={roughness}nm, l_max={lmax}')
            
            max_r = params.radius_um * 1.3
            ax.set_xlim([-max_r, max_r])
            ax.set_ylim([-max_r, max_r])
            ax.set_zlim([-max_r, max_r])
    
    fig.suptitle('Effect of Roughness Amplitude (σ) and Angular Resolution (l_max)\n'
                'on Nanoparticle Surface Morphology', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('demo_particle_geometry.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Figure saved: demo_particle_geometry.png")


def demo_size_parameter():
    """
    Demonstrate the size parameter and scattering regimes.
    """
    print("\n" + "=" * 70)
    print("DEMO 2: SIZE PARAMETER AND SCATTERING REGIMES")
    print("=" * 70)
    
    # Range of particle sizes
    radii = np.array([20, 40, 60, 80, 100, 150, 200, 300])  # nm
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Compute size parameters for different wavelengths
    wavelengths = [405, 488, 532, 638]  # nm
    colors = ['violet', 'blue', 'green', 'red']
    n_water = 1.33
    
    for wavelength, color in zip(wavelengths, colors):
        size_params = 2 * np.pi * radii / (wavelength / n_water)
        ax1.plot(radii, size_params, 'o-', color=color, label=f'λ = {wavelength} nm')
    
    # Mark regime boundaries
    ax1.axhline(y=0.3, color='gray', linestyle='--', alpha=0.5)
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax1.fill_between([0, 350], [0, 0], [0.3, 0.3], alpha=0.1, color='green', label='Rayleigh')
    ax1.fill_between([0, 350], [0.3, 0.3], [1.0, 1.0], alpha=0.1, color='yellow', label='Transition')
    ax1.fill_between([0, 350], [1.0, 1.0], [3.0, 3.0], alpha=0.1, color='red', label='Mie')
    
    ax1.set_xlabel('Particle Radius [nm]')
    ax1.set_ylabel('Size Parameter x = 2πr/λ_medium')
    ax1.set_title('Scattering Regimes')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 320])
    ax1.set_ylim([0, 3])
    
    # Scattering cross-section scaling
    # Rayleigh: σ ∝ r^6 / λ^4
    # Mie: σ ∝ r^2 (geometric limit)
    
    r_rayleigh = np.linspace(10, 60, 100)
    r_mie = np.linspace(60, 300, 100)
    
    sigma_rayleigh = (r_rayleigh / 50)**6  # Normalized
    sigma_mie = (r_mie / 50)**2 * (50/60)**4  # Match at boundary
    
    ax2.loglog(r_rayleigh, sigma_rayleigh, 'b-', linewidth=2, label='Rayleigh (∝ r⁶)')
    ax2.loglog(r_mie, sigma_mie, 'r-', linewidth=2, label='Mie/Geometric (∝ r²)')
    ax2.axvline(x=60, color='gray', linestyle='--', alpha=0.5, label='Transition')
    
    ax2.set_xlabel('Particle Radius [nm]')
    ax2.set_ylabel('Scattering Cross-Section [a.u.]')
    ax2.set_title('Scattering Cross-Section Scaling')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('demo_size_parameter.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Figure saved: demo_size_parameter.png")
    
    # Print table
    print("\nSIZE PARAMETER TABLE (λ = 488 nm in water):")
    print("-" * 50)
    print(f"{'Radius [nm]':>12} | {'Size Parameter x':>16} | {'Regime':>15}")
    print("-" * 50)
    for r in radii:
        params = ParticleParameters(radius_nm=r)
        print(f"{r:>12.0f} | {params.size_parameter:>16.3f} | {params.scattering_regime:>15}")


def demo_far_field_patterns():
    """
    Demonstrate far-field scattering patterns for different particles.
    """
    print("\n" + "=" * 70)
    print("DEMO 3: FAR-FIELD SCATTERING PATTERNS")
    print("=" * 70)
    
    # Create simulations for different particle sizes
    radii = [40, 80, 150]  # nm
    roughness_values = [0, 5]  # nm
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), 
                             subplot_kw={'projection': 'polar'})
    
    for i, roughness in enumerate(roughness_values):
        for j, radius in enumerate(radii):
            ax = axes[i, j]
            
            particle = ParticleParameters(
                radius_nm=radius,
                roughness_amplitude_nm=roughness,
                roughness_lmax=10
            )
            
            illumination = IlluminationParameters(wavelength_nm=488.0)
            imaging = ImagingParameters(numerical_aperture=0.3)
            simulation = SimulationParameters()
            
            sim = NTAFDTDSimulation(particle, illumination, imaging, simulation)
            sim.run()  # Uses mock data
            sim.compute_far_field()
            
            ff = sim._far_field_data
            theta = ff['theta'][:, 0]
            intensity = ff['intensity'][:, 0]
            intensity_norm = intensity / np.max(intensity)
            
            ax.plot(theta, intensity_norm, 'b-', linewidth=2)
            ax.set_title(f'R={radius}nm, σ={roughness}nm\n({particle.scattering_regime})')
            ax.set_theta_zero_location('N')
            
            # Mark NA collection cone
            na_angle = imaging.max_collection_angle_rad
            ax.axvline(x=np.pi/2 - na_angle, color='r', linestyle='--', alpha=0.5)
            ax.axvline(x=np.pi/2 + na_angle, color='r', linestyle='--', alpha=0.5)
    
    fig.suptitle('Far-Field Scattering Patterns\n'
                '(Red dashed lines: NA collection cone for side-imaging)',
                fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('demo_far_field.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Figure saved: demo_far_field.png")


def demo_camera_images():
    """
    Demonstrate camera image formation with different parameters.
    """
    print("\n" + "=" * 70)
    print("DEMO 4: CAMERA IMAGE FORMATION")
    print("=" * 70)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Row 1: Effect of NA
    na_values = [0.15, 0.25, 0.35, 0.45]
    
    for j, na in enumerate(na_values):
        ax = axes[0, j]
        
        particle = ParticleParameters(radius_nm=80, roughness_amplitude_nm=0)
        illumination = IlluminationParameters(wavelength_nm=488.0)
        imaging = ImagingParameters(numerical_aperture=na, image_pixels=64)
        simulation = SimulationParameters()
        
        sim = NTAFDTDSimulation(particle, illumination, imaging, simulation)
        sim.run()
        sim.compute_far_field()
        image = sim.generate_camera_image()
        
        fov = imaging.field_of_view_um
        extent = [-fov/2, fov/2, -fov/2, fov/2]
        
        im = ax.imshow(image.T, origin='lower', extent=extent, cmap='gray')
        ax.set_title(f'NA = {na}')
        ax.set_xlabel('X [μm]')
        if j == 0:
            ax.set_ylabel('Z [μm]')
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    # Row 2: Effect of roughness
    roughness_values = [0, 3, 6, 10]  # nm
    
    for j, roughness in enumerate(roughness_values):
        ax = axes[1, j]
        
        particle = ParticleParameters(
            radius_nm=80, 
            roughness_amplitude_nm=roughness,
            roughness_lmax=12
        )
        illumination = IlluminationParameters(wavelength_nm=488.0)
        imaging = ImagingParameters(numerical_aperture=0.3, image_pixels=64)
        simulation = SimulationParameters()
        
        sim = NTAFDTDSimulation(particle, illumination, imaging, simulation)
        sim.run()
        sim.compute_far_field()
        image = sim.generate_camera_image()
        
        fov = imaging.field_of_view_um
        extent = [-fov/2, fov/2, -fov/2, fov/2]
        
        im = ax.imshow(image.T, origin='lower', extent=extent, cmap='gray')
        ax.set_title(f'Roughness = {roughness} nm')
        ax.set_xlabel('X [μm]')
        if j == 0:
            ax.set_ylabel('Z [μm]')
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    axes[0, 0].set_title(f'NA = {na_values[0]}\n(R=80nm, smooth)', fontsize=10)
    axes[1, 0].set_title(f'Roughness = {roughness_values[0]} nm\n(NA=0.3)', fontsize=10)
    
    fig.suptitle('Camera Images: Effect of Numerical Aperture (top) and Surface Roughness (bottom)',
                fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('demo_camera_images.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Figure saved: demo_camera_images.png")


def demo_roughness_analysis():
    """
    Detailed analysis of how roughness affects the observed image.
    """
    print("\n" + "=" * 70)
    print("DEMO 5: ROUGHNESS EFFECT ANALYSIS")
    print("=" * 70)
    
    # Compare particles with increasing roughness
    roughness_values = np.array([0, 1, 2, 3, 5, 7, 10])
    
    total_intensities = []
    peak_intensities = []
    image_widths = []  # FWHM
    asymmetry_metrics = []
    
    for roughness in roughness_values:
        particle = ParticleParameters(
            radius_nm=80,
            roughness_amplitude_nm=roughness,
            roughness_lmax=10
        )
        illumination = IlluminationParameters(wavelength_nm=488.0)
        imaging = ImagingParameters(numerical_aperture=0.3, image_pixels=64)
        simulation = SimulationParameters()
        
        sim = NTAFDTDSimulation(particle, illumination, imaging, simulation)
        sim.run()
        sim.compute_far_field()
        image = sim.generate_camera_image()
        
        # Metrics
        total_intensities.append(np.sum(image))
        peak_intensities.append(np.max(image))
        
        # FWHM estimation
        N = image.shape[0]
        profile = image[N//2, :]
        half_max = np.max(profile) / 2
        above_half = profile > half_max
        fwhm = np.sum(above_half) * imaging.object_pixel_size_um
        image_widths.append(fwhm)
        
        # Asymmetry: difference between quadrants
        q1 = np.sum(image[:N//2, :N//2])
        q2 = np.sum(image[:N//2, N//2:])
        q3 = np.sum(image[N//2:, :N//2])
        q4 = np.sum(image[N//2:, N//2:])
        asymmetry = np.std([q1, q2, q3, q4]) / np.mean([q1, q2, q3, q4])
        asymmetry_metrics.append(asymmetry)
    
    # Plot analysis
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    ax1, ax2, ax3, ax4 = axes.flatten()
    
    # Total intensity vs roughness
    ax1.plot(roughness_values, total_intensities, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Surface Roughness RMS [nm]')
    ax1.set_ylabel('Total Scattered Intensity [a.u.]')
    ax1.set_title('Total Scattering vs Roughness')
    ax1.grid(True, alpha=0.3)
    
    # Peak intensity vs roughness
    ax2.plot(roughness_values, peak_intensities, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Surface Roughness RMS [nm]')
    ax2.set_ylabel('Peak Intensity [a.u.]')
    ax2.set_title('Peak Intensity vs Roughness')
    ax2.grid(True, alpha=0.3)
    
    # FWHM vs roughness
    ax3.plot(roughness_values, image_widths, 'go-', linewidth=2, markersize=8)
    ax3.set_xlabel('Surface Roughness RMS [nm]')
    ax3.set_ylabel('Image FWHM [μm]')
    ax3.set_title('Image Width vs Roughness')
    ax3.grid(True, alpha=0.3)
    
    # Asymmetry vs roughness
    ax4.plot(roughness_values, asymmetry_metrics, 'mo-', linewidth=2, markersize=8)
    ax4.set_xlabel('Surface Roughness RMS [nm]')
    ax4.set_ylabel('Asymmetry Metric')
    ax4.set_title('Image Asymmetry vs Roughness')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Quantitative Analysis of Surface Roughness Effects\n'
                '(80nm particle, NA=0.3, λ=488nm)',
                fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('demo_roughness_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Figure saved: demo_roughness_analysis.png")
    
    # Print table
    print("\nQUANTITATIVE RESULTS:")
    print("-" * 80)
    print(f"{'Roughness [nm]':>14} | {'Total Intensity':>15} | {'Peak':>8} | {'FWHM [μm]':>10} | {'Asymmetry':>10}")
    print("-" * 80)
    for i, r in enumerate(roughness_values):
        print(f"{r:>14.1f} | {total_intensities[i]:>15.4f} | {peak_intensities[i]:>8.4f} | {image_widths[i]:>10.3f} | {asymmetry_metrics[i]:>10.4f}")


def run_full_demo():
    """
    Run all demonstration examples.
    """
    print("\n" + "=" * 70)
    print("NTA FDTD SIMULATION - FULL DEMONSTRATION")
    print("=" * 70)
    print("\nThis demo will generate 5 figures showing different aspects of the simulation.\n")
    
    # Demo 1: Particle geometry
    demo_particle_geometry()
    
    # Demo 2: Size parameter
    demo_size_parameter()
    
    # Demo 3: Far-field patterns
    demo_far_field_patterns()
    
    # Demo 4: Camera images
    demo_camera_images()
    
    # Demo 5: Roughness analysis
    demo_roughness_analysis()
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - demo_particle_geometry.png")
    print("  - demo_size_parameter.png")
    print("  - demo_far_field.png")
    print("  - demo_camera_images.png")
    print("  - demo_roughness_analysis.png")
    
    print("\n" + "=" * 70)
    print("RUNNING SMOOTH vs ROUGH COMPARISON")
    print("=" * 70)
    
    # Run the comparison
    compare_smooth_vs_rough(
        radius_nm=80.0,
        roughness_nm=5.0,
        roughness_lmax=10,
        save_path="nta_comparison.png"
    )


if __name__ == "__main__":
    run_full_demo()
