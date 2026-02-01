#!/usr/bin/env python3
"""
===============================================================================
VIDEODROP SIMULATION - Interferometric Brightfield Nanoparticle Imaging
===============================================================================

Simulation d'une image de caméra pour la microscopie interférométrique
en champ clair (type Videodrop / Myriade).

Principe physique :
- Illumination en transmission (LED)
- Interférence entre l'onde directe et l'onde diffusée par la particule
- Détection : I = |E_ref + E_scat|²

Signal :
  I = |E_ref|² + |E_scat|² + 2·Re(E_ref* · E_scat)
      ↑           ↑              ↑
   fond clair   très petit   TERME D'INTERFÉRENCE (dominant!)

Avantage : Sensibilité en r³ (pas r⁶ comme en dark-field)
Résultat : Tache sombre ou claire sur fond clair selon la phase

===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Optional
import warnings


# =============================================================================
# CONSTANTES PHYSIQUES
# =============================================================================

@dataclass
class PhysicalConstants:
    """Constantes physiques pour la simulation."""
    wavelength_nm: float = 488.0     # Longueur d'onde LED [nm]
    n_water: float = 1.33            # Indice de réfraction de l'eau
    
    @property
    def wavelength_um(self) -> float:
        return self.wavelength_nm / 1000.0
    
    @property
    def wavelength_water(self) -> float:
        return self.wavelength_um / self.n_water
    
    @property
    def k_water(self) -> float:
        return 2 * np.pi / self.wavelength_water


# =============================================================================
# PARAMÈTRES DE LA NANOPARTICULE
# =============================================================================

@dataclass
class ParticleParameters:
    """Paramètres de la nanoparticule."""
    radius_nm: float = 80.0              # Rayon [nm]
    refractive_index: float = 1.50       # Indice de réfraction
    material_name: str = "polystyrene"   # Nom du matériau
    
    @property
    def radius_um(self) -> float:
        return self.radius_nm / 1000.0
    
    @property
    def size_parameter(self) -> float:
        """Paramètre de taille de Mie x = 2πr/λ"""
        constants = PhysicalConstants()
        return 2 * np.pi * self.radius_nm / (constants.wavelength_nm / constants.n_water)
    
    @property
    def scattering_regime(self) -> str:
        x = self.size_parameter
        if x < 0.3:
            return "Rayleigh"
        elif x < 1.0:
            return "Rayleigh-Mie transition"
        else:
            return "Mie"


# =============================================================================
# PARAMÈTRES D'ILLUMINATION
# =============================================================================

@dataclass
class IlluminationParameters:
    """Paramètres de l'illumination LED."""
    wavelength_nm: float = 488.0
    polarization: str = "x"
    
    @property
    def frequency_meep(self) -> float:
        return 1.0 / (self.wavelength_nm / 1000.0)


# =============================================================================
# PARAMÈTRES D'IMAGERIE
# =============================================================================

@dataclass
class ImagingParameters:
    """Paramètres du système d'imagerie."""
    numerical_aperture: float = 0.3
    magnification: float = 20.0
    camera_pixel_size_um: float = 6.5
    image_pixels: int = 64
    
    # Paramètres d'interférométrie (Videodrop)
    reference_amplitude: float = 1.0      # Amplitude du faisceau de référence
    reference_phase: float = 0.0          # Phase de référence [rad]
    background_subtraction: bool = True   # Soustraire |E_ref|²
    
    @property
    def object_pixel_size_um(self) -> float:
        return self.camera_pixel_size_um / self.magnification
    
    @property
    def max_collection_angle_rad(self) -> float:
        return np.arcsin(min(self.numerical_aperture, 0.999))
    
    @property
    def field_of_view_um(self) -> float:
        return self.image_pixels * self.object_pixel_size_um


# =============================================================================
# SIMULATION VIDEODROP
# =============================================================================

class VideodropSimulation:
    """
    Simulation d'imagerie interférométrique type Videodrop.
    
    Workflow :
    1. Calculer la diffusion de la nanoparticule (Rayleigh/Mie)
    2. Appliquer le filtre NA de l'objectif
    3. Calculer l'interférence avec le faisceau de référence
    4. Générer l'image caméra
    """
    
    def __init__(
        self,
        particle: ParticleParameters,
        illumination: IlluminationParameters,
        imaging: ImagingParameters
    ):
        self.particle = particle
        self.illumination = illumination
        self.imaging = imaging
        self.constants = PhysicalConstants(wavelength_nm=illumination.wavelength_nm)
        
        # Données de simulation
        self._far_field_data = None
        self._camera_image = None
        self._interferometry_data = None
    
    def _compute_scattering(self):
        """
        Calcule le champ diffusé (modèle Rayleigh/Mie simplifié).
        """
        n_theta = 90
        n_phi = 180
        
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2*np.pi, n_phi)
        THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
        
        # Pattern angulaire dipôle (polarisation x)
        if self.illumination.polarization == "x":
            angular_pattern = np.sqrt(np.maximum(1.0 - np.sin(THETA)**2 * np.cos(PHI)**2, 0.01))
        else:
            angular_pattern = np.abs(np.sin(THETA))
        
        # Amplitude de diffusion Rayleigh
        m = self.particle.refractive_index / self.constants.n_water
        polarizability_factor = (m**2 - 1) / (m**2 + 2)
        r = self.particle.radius_um
        k = 2 * np.pi / self.constants.wavelength_um * self.constants.n_water
        
        E_amplitude = (k**2) * (r**3) * np.abs(polarizability_factor) * angular_pattern
        
        # Phase de diffusion
        base_phase = PHI
        scattering_phase = np.angle(complex(polarizability_factor))
        
        # Champ complexe
        E_theta = E_amplitude * np.exp(1j * (base_phase + scattering_phase))
        E_phi = E_amplitude * 0.1 * np.exp(1j * (base_phase + scattering_phase + np.pi/4))
        
        intensity = np.abs(E_theta)**2 + np.abs(E_phi)**2
        
        self._far_field_data = {
            'theta': THETA,
            'phi': PHI,
            'E_theta': E_theta,
            'E_phi': E_phi,
            'intensity': intensity,
            'n_theta': n_theta,
            'n_phi': n_phi
        }
    
    def generate_camera_image(self) -> np.ndarray:
        """
        Génère l'image caméra avec détection interférométrique.
        
        I = |E_ref + E_scat|² = |E_ref|² + |E_scat|² + 2·Re(E_ref* · E_scat)
        """
        if self._far_field_data is None:
            self._compute_scattering()
        
        ff = self._far_field_data
        theta = ff['theta']
        phi = ff['phi']
        
        N = self.imaging.image_pixels
        fov = self.imaging.field_of_view_um
        NA = self.imaging.numerical_aperture
        theta_max = self.imaging.max_collection_angle_rad
        
        E_ref_amplitude = self.imaging.reference_amplitude
        E_ref_phase = self.imaging.reference_phase
        background_subtraction = self.imaging.background_subtraction
        
        print(f"Génération image Videodrop ({N}x{N} pixels, NA={NA})...")
        print(f"  Référence: amplitude={E_ref_amplitude}, phase={np.degrees(E_ref_phase):.1f}°")
        
        # Grille image
        x_img = np.linspace(-fov/2, fov/2, N)
        y_img = np.linspace(-fov/2, fov/2, N)
        X_IMG, Y_IMG = np.meshgrid(x_img, y_img, indexing='ij')
        
        # Direction caméra (axe y)
        cam_dir = np.array([0, 1, 0])
        
        # Directions de diffusion
        dirs_x = np.sin(theta) * np.cos(phi)
        dirs_y = np.sin(theta) * np.sin(phi)
        dirs_z = np.cos(theta)
        
        # Angle par rapport à l'axe caméra
        cos_angle_to_cam = dirs_x * cam_dir[0] + dirs_y * cam_dir[1] + dirs_z * cam_dir[2]
        angle_to_cam = np.arccos(np.clip(cos_angle_to_cam, -1, 1))
        
        # Masque NA
        in_na_cone = angle_to_cam <= theta_max
        
        # Champ diffusé collecté
        E_scat_collected = np.sqrt(ff['intensity']) * in_na_cone
        E_scat_total = np.mean(E_scat_collected)
        
        # Champ diffusé complexe moyen dans le cône NA
        E_theta = ff['E_theta']
        E_phi = ff['E_phi']
        E_scat_collected = np.mean(E_theta[in_na_cone]) + np.mean(E_phi[in_na_cone])
        
        # PSF (Point Spread Function) - approximation gaussienne
        wavelength = self.constants.wavelength_um
        sigma_psf = 0.42 * wavelength / NA  # Largeur gaussienne de la PSF
        R_IMG = np.sqrt(X_IMG**2 + Y_IMG**2)
        
        # Amplitude de la PSF (gaussienne)
        psf_amplitude = np.exp(-R_IMG**2 / (4 * sigma_psf**2))
        
        # Courbure de phase (Gouy + défocus)
        k = 2 * np.pi / wavelength * self.constants.n_water
        phase_curvature = k * R_IMG**2 / (4 * fov)
        
        # Champ diffusé sur l'image
        E_scat = E_scat_collected * psf_amplitude * np.exp(1j * phase_curvature)
        
        # Champ de référence (Videodrop : lumière directe transmise)
        E_ref = E_ref_amplitude * np.exp(1j * E_ref_phase) * psf_amplitude
        
        # =================================================================
        # SIGNAL INTERFÉROMÉTRIQUE (cœur du Videodrop)
        # =================================================================
        # I = |E_ref + E_scat|² = |E_ref|² + |E_scat|² + 2·Re(E_ref* · E_scat)
        
        E_total = E_ref + E_scat
        I_total = np.abs(E_total)**2
        
        # Décomposition
        I_ref = np.abs(E_ref)**2                              # Fond clair
        I_scat = np.abs(E_scat)**2                            # Très faible
        I_interference = 2 * np.real(np.conj(E_ref) * E_scat) # Signal dominant!
        
        # Stockage pour analyse
        self._interferometry_data = {
            'E_ref': E_ref,
            'E_scat': E_scat,
            'I_ref': I_ref,
            'I_scat': I_scat,
            'I_interference': I_interference,
            'I_total': I_total
        }
        
        # Soustraction du fond
        if background_subtraction:
            I_ref_mean = np.mean(I_ref)
            camera_signal = I_total - I_ref_mean
            print(f"  Fond soustrait: I_ref_mean = {I_ref_mean:.4f}")
        else:
            camera_signal = I_total
        
        self._camera_image = camera_signal
        
        # Statistiques
        print(f"  |E_scat|² max: {np.max(I_scat):.2e}")
        print(f"  Terme d'interférence max: {np.max(np.abs(I_interference)):.2e}")
        print(f"  Enhancement: {np.max(np.abs(I_interference))/np.max(I_scat):.1f}x")
        print(f"Image générée. Max: {np.max(camera_signal):.4e}, Min: {np.min(camera_signal):.4e}")
        
        return self._camera_image
    
    def plot_results(self, save_path: Optional[str] = None):
        """Visualisation des résultats."""
        if self._camera_image is None:
            self.generate_camera_image()
        
        fig = plt.figure(figsize=(14, 10))
        
        fov = self.imaging.field_of_view_um
        extent = [-fov/2, fov/2, -fov/2, fov/2]
        N = self.imaging.image_pixels
        x_coords = np.linspace(-fov/2, fov/2, N)
        
        # 1. Image Videodrop (interférométrique)
        ax1 = fig.add_subplot(2, 3, 1)
        img = self._camera_image
        vmax = np.max(np.abs(img))
        im1 = ax1.imshow(img.T, origin='lower', extent=extent,
                        cmap='RdBu_r', aspect='equal', vmin=-vmax, vmax=vmax)
        ax1.set_xlabel('X [μm]')
        ax1.set_ylabel('Z [μm]')
        ax1.set_title('Image Videodrop\n(fond soustrait)')
        plt.colorbar(im1, ax=ax1, label='ΔI')
        
        # 2. Profil central
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.plot(x_coords, img[N//2, :], 'b-', linewidth=2)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax2.set_xlabel('Position [μm]')
        ax2.set_ylabel('ΔIntensité')
        ax2.set_title('Profil central')
        ax2.grid(True, alpha=0.3)
        
        # 3. Décomposition du signal
        if self._interferometry_data is not None:
            ax3 = fig.add_subplot(2, 3, 3)
            I_scat = self._interferometry_data['I_scat'][N//2, :]
            I_interf = self._interferometry_data['I_interference'][N//2, :]
            
            ax3.plot(x_coords, I_scat, 'r-', linewidth=2, label='|E_scat|²')
            ax3.plot(x_coords, I_interf, 'b-', linewidth=2, label='2·Re(E_ref*·E_scat)')
            ax3.set_xlabel('Position [μm]')
            ax3.set_ylabel('Intensité')
            ax3.set_title('Décomposition du signal')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. |E_scat|² seul (ce qu'on verrait en dark-field)
        ax4 = fig.add_subplot(2, 3, 4)
        I_scat_img = self._interferometry_data['I_scat']
        im4 = ax4.imshow(I_scat_img.T, origin='lower', extent=extent,
                        cmap='hot', aspect='equal')
        ax4.set_xlabel('X [μm]')
        ax4.set_ylabel('Z [μm]')
        ax4.set_title('Scattering seul (dark-field)\n|E_scat|²')
        plt.colorbar(im4, ax=ax4, label='I')
        
        # 5. Terme d'interférence
        ax5 = fig.add_subplot(2, 3, 5)
        I_interf_img = self._interferometry_data['I_interference']
        vmax_i = np.max(np.abs(I_interf_img))
        im5 = ax5.imshow(I_interf_img.T, origin='lower', extent=extent,
                        cmap='RdBu_r', aspect='equal', vmin=-vmax_i, vmax=vmax_i)
        ax5.set_xlabel('X [μm]')
        ax5.set_ylabel('Z [μm]')
        ax5.set_title('Terme d\'interférence\n2·Re(E_ref*·E_scat)')
        plt.colorbar(im5, ax=ax5, label='ΔI')
        
        # 6. Résumé des paramètres
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.axis('off')
        
        enhancement = np.max(np.abs(I_interf_img)) / np.max(I_scat_img)
        
        summary = f"""
PARAMÈTRES VIDEODROP
{'='*40}

Particule:
  • Rayon: {self.particle.radius_nm:.0f} nm
  • Indice: {self.particle.refractive_index:.2f}
  • Régime: {self.particle.scattering_regime}

Illumination:
  • λ = {self.illumination.wavelength_nm:.0f} nm

Imagerie:
  • NA = {self.imaging.numerical_aperture}
  • Champ de vue: {fov:.1f} μm

SIGNAL INTERFÉROMÉTRIQUE
{'='*40}

Principe:
  I = |E_ref + E_scat|²
    = |E_ref|² + |E_scat|² + 2·Re(E_ref*·E_scat)
          ↑          ↑              ↑
       fond      très petit    SIGNAL DOMINANT

Enhancement: {enhancement:.1f}x

Le terme d'interférence donne une
sensibilité en r³ (pas r⁶ comme dark-field)
→ Détection de particules beaucoup 
  plus petites possible!
"""
        ax6.text(0.02, 0.98, summary, transform=ax6.transAxes,
                fontfamily='monospace', fontsize=9,
                verticalalignment='top')
        
        plt.suptitle(f'Simulation Videodrop - R={self.particle.radius_nm:.0f}nm, '
                    f'n={self.particle.refractive_index}, λ={self.illumination.wavelength_nm:.0f}nm',
                    fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure sauvegardée: {save_path}")
        
        plt.show()
        return fig


# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================

def run_videodrop_simulation(
    radius_nm: float = 80.0,
    refractive_index: float = 1.50,
    wavelength_nm: float = 488.0,
    numerical_aperture: float = 0.3,
    reference_amplitude: float = 1.0,
    image_pixels: int = 64,
    save_path: Optional[str] = None
) -> VideodropSimulation:
    """
    Lance une simulation Videodrop complète.
    
    Args:
        radius_nm: Rayon de la particule [nm]
        refractive_index: Indice de réfraction de la particule
        wavelength_nm: Longueur d'onde de la LED [nm]
        numerical_aperture: Ouverture numérique de l'objectif
        reference_amplitude: Amplitude du faisceau de référence
        image_pixels: Taille de l'image en pixels
        save_path: Chemin pour sauvegarder la figure
    
    Returns:
        Instance de VideodropSimulation avec les résultats
    """
    print("=" * 60)
    print("SIMULATION VIDEODROP")
    print("=" * 60)
    
    particle = ParticleParameters(
        radius_nm=radius_nm,
        refractive_index=refractive_index
    )
    
    illumination = IlluminationParameters(
        wavelength_nm=wavelength_nm
    )
    
    imaging = ImagingParameters(
        numerical_aperture=numerical_aperture,
        image_pixels=image_pixels,
        reference_amplitude=reference_amplitude
    )
    
    print(f"Particule: R={radius_nm}nm, n={refractive_index}")
    print(f"Régime: {particle.scattering_regime}")
    print(f"λ={wavelength_nm}nm, NA={numerical_aperture}")
    print("-" * 60)
    
    sim = VideodropSimulation(
        particle=particle,
        illumination=illumination,
        imaging=imaging
    )
    
    sim.generate_camera_image()
    sim.plot_results(save_path=save_path)
    
    return sim


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Simulation Videodrop - Imagerie interférométrique de nanoparticules"
    )
    parser.add_argument("--radius", type=float, default=80.0,
                       help="Rayon de la particule [nm]")
    parser.add_argument("--n-particle", type=float, default=1.50,
                       help="Indice de réfraction de la particule")
    parser.add_argument("--wavelength", type=float, default=488.0,
                       help="Longueur d'onde LED [nm]")
    parser.add_argument("--na", type=float, default=0.3,
                       help="Ouverture numérique")
    parser.add_argument("--pixels", type=int, default=64,
                       help="Taille de l'image [pixels]")
    parser.add_argument("--ref-amplitude", type=float, default=1.0,
                       help="Amplitude du faisceau de référence")
    parser.add_argument("--save", type=str, default="videodrop_result.png",
                       help="Fichier de sortie")
    parser.add_argument("--no-save", action="store_true",
                       help="Ne pas sauvegarder la figure")
    
    args = parser.parse_args()
    
    sim = run_videodrop_simulation(
        radius_nm=args.radius,
        refractive_index=args.n_particle,
        wavelength_nm=args.wavelength,
        numerical_aperture=args.na,
        image_pixels=args.pixels,
        reference_amplitude=args.ref_amplitude,
        save_path=None if args.no_save else args.save
    )
