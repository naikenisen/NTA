#!/usr/bin/env python3
"""
===============================================================================
NTA SIMULATION - Nanoparticle Tracking Analysis (Dark Field Scattering)
===============================================================================

Simulation d'une image de caméra pour la microscopie en fond noir (Dark Field)
utilisée en NTA (Nanoparticle Tracking Analysis).

Comparaison : Particule lisse vs Particule rugueuse

Principe physique :
- Illumination LASER latérale (90° par rapport à la caméra)
- Détection de la lumière diffusée (Scattering) - PAS d'interférence
- Fond noir absolu (0 photon sans particule)
- Intensité ∝ d^6 (Loi de Rayleigh pour petites particules)
- Particules apparaissent comme des taches d'Airy (PSF limitée par diffraction)

Différences avec Videodrop :
- Videodrop : LED, interférométrie, I ∝ d³, contraste +/-
- NTA : LASER, diffusion, I ∝ d⁶, contraste positif sur fond noir

===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from dataclasses import dataclass
from typing import Tuple, Optional, List

# Pour les harmoniques sphériques (rugosité)
try:
    from scipy.special import sph_harm_y
    USE_NEW_SCIPY = True
except ImportError:
    from scipy.special import sph_harm
    USE_NEW_SCIPY = False

from scipy.special import j1


# =============================================================================
# CONSTANTES PHYSIQUES
# =============================================================================

@dataclass
class PhysicalConstants:
    """Constantes physiques pour la simulation NTA."""
    wavelength_nm: float = 532.0  # Laser vert typique pour NTA
    n_water: float = 1.33
    temperature_K: float = 298.0  # 25°C
    viscosity_Pa_s: float = 0.001  # Viscosité de l'eau à 25°C
    k_B: float = 1.380649e-23  # Constante de Boltzmann [J/K]
    
    @property
    def wavelength_um(self) -> float:
        return self.wavelength_nm / 1000.0
    
    @property
    def k_wave(self) -> float:
        """Nombre d'onde dans le milieu [rad/µm]."""
        return 2 * np.pi * self.n_water / self.wavelength_um


# =============================================================================
# PARAMÈTRES DE LA NANOPARTICULE
# =============================================================================

@dataclass
class ParticleParameters:
    """Paramètres de la nanoparticule."""
    radius_nm: float = 100.0  # Rayon typique pour NTA
    refractive_index: float = 1.50  # Polystyrène
    roughness_amplitude_nm: float = 0.0   # Amplitude RMS de la rugosité
    roughness_lmax: int = 10              # Ordre max des harmoniques sphériques
    roughness_seed: int = 42              # Seed pour reproductibilité
    material_name: str = "polystyrene"
    
    @property
    def radius_um(self) -> float:
        return self.radius_nm / 1000.0
    
    @property
    def diameter_nm(self) -> float:
        return 2 * self.radius_nm
    
    @property
    def diameter_um(self) -> float:
        return 2 * self.radius_um
    
    @property
    def roughness_amplitude_um(self) -> float:
        return self.roughness_amplitude_nm / 1000.0


# =============================================================================
# GÉOMÉTRIE DE PARTICULE RUGUEUSE
# =============================================================================

class RoughParticleGeometry:
    """
    Génère une géométrie de particule rugueuse avec harmoniques sphériques.
    
    r(θ,φ) = R + δr(θ,φ)
    δr = Σ a_lm · Y_lm(θ,φ)
    """
    
    def __init__(self, params: ParticleParameters):
        self.params = params
        self.rng = np.random.default_rng(params.roughness_seed)
        self._generate_roughness_coefficients()
    
    def _generate_roughness_coefficients(self):
        """Génère les coefficients des harmoniques sphériques."""
        lmax = self.params.roughness_lmax
        amplitude = self.params.roughness_amplitude_um
        
        if amplitude == 0:
            self.alm = {}
            return
        
        l_corr = max(lmax / 3, 1)
        self.alm = {}
        total_power = 0
        
        for l in range(2, lmax + 1):
            power_l = np.exp(-l / l_corr)
            for m in range(-l, l + 1):
                a_real = self.rng.normal(0, 1)
                a_imag = self.rng.normal(0, 1) if m != 0 else 0
                self.alm[(l, m)] = (a_real + 1j * a_imag) * np.sqrt(power_l)
                total_power += np.abs(self.alm[(l, m)])**2
        
        if total_power > 0:
            norm_factor = amplitude / np.sqrt(total_power)
            for key in self.alm:
                self.alm[key] *= norm_factor
    
    def radius_at_angles(self, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """Calcule le rayon local r(θ,φ)."""
        r = np.ones_like(theta) * self.params.radius_um
        
        if self.params.roughness_amplitude_nm == 0:
            return r
        
        for (l, m), coeff in self.alm.items():
            if USE_NEW_SCIPY:
                Y_lm = sph_harm_y(l, m, theta, phi)
            else:
                Y_lm = sph_harm(m, l, phi, theta)
            r = r + np.real(coeff * Y_lm)
        
        return r
    
    def generate_surface_mesh(self, n_theta: int = 50, n_phi: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Génère le maillage 3D de la surface."""
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2 * np.pi, n_phi)
        THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
        
        R = self.radius_at_angles(THETA, PHI)
        
        X = R * np.sin(THETA) * np.cos(PHI)
        Y = R * np.sin(THETA) * np.sin(PHI)
        Z = R * np.cos(THETA)
        
        return X, Y, Z
    
    def get_effective_radius(self) -> float:
        """Calcule le rayon effectif moyen incluant la rugosité."""
        if self.params.roughness_amplitude_nm == 0:
            return self.params.radius_um
        
        # Échantillonnage sur la sphère
        n_samples = 1000
        theta = np.random.uniform(0, np.pi, n_samples)
        phi = np.random.uniform(0, 2 * np.pi, n_samples)
        r_samples = self.radius_at_angles(theta, phi)
        
        return np.mean(r_samples)


# =============================================================================
# PARAMÈTRES D'ILLUMINATION LASER
# =============================================================================

@dataclass
class LaserParameters:
    """Paramètres du laser pour NTA."""
    wavelength_nm: float = 532.0      # Laser vert (typique NTA)
    power_mW: float = 50.0            # Puissance du laser
    beam_waist_um: float = 50.0       # Largeur du faisceau laser
    polarization: str = "x"           # Polarisation linéaire


# =============================================================================
# PARAMÈTRES D'IMAGERIE (DARK FIELD)
# =============================================================================

@dataclass
class ImagingParameters:
    """Paramètres d'imagerie en fond noir pour NTA."""
    numerical_aperture: float = 0.4       # NA typique pour NTA
    magnification: float = 20.0           # Grossissement
    camera_pixel_size_um: float = 6.5     # Taille pixel CMOS
    image_pixels: int = 256               # Résolution image
    exposure_time_ms: float = 30.0        # Temps d'exposition
    camera_gain: float = 1.0              # Gain de la caméra
    camera_noise_level: float = 0.01      # Bruit de lecture (faible en fond noir)
    background_level: int = 0             # Fond NOIR pour dark field
    
    @property
    def pixel_size_um(self) -> float:
        """Taille du pixel dans l'espace objet [µm]."""
        return self.camera_pixel_size_um / self.magnification
    
    @property
    def field_of_view_um(self) -> float:
        """Champ de vue total [µm]."""
        return self.image_pixels * self.pixel_size_um
    
    @property
    def airy_radius_um(self, wavelength_um: float = 0.532) -> float:
        """Rayon du disque d'Airy (résolution) [µm]."""
        return 0.61 * wavelength_um / self.numerical_aperture


# =============================================================================
# MOUVEMENT BROWNIEN (STOKES-EINSTEIN)
# =============================================================================

class BrownianMotion:
    """
    Simulation du mouvement Brownien basée sur l'équation de Stokes-Einstein.
    
    D = k_B * T / (6 * π * η * r)
    
    où:
    - D : coefficient de diffusion [m²/s]
    - k_B : constante de Boltzmann
    - T : température [K]
    - η : viscosité dynamique [Pa·s]
    - r : rayon hydrodynamique [m]
    """
    
    def __init__(self, particle: ParticleParameters, constants: PhysicalConstants):
        self.particle = particle
        self.constants = constants
        self._compute_diffusion_coefficient()
    
    def _compute_diffusion_coefficient(self):
        """Calcule le coefficient de diffusion D."""
        r_m = self.particle.radius_nm * 1e-9  # Convertir en mètres
        
        self.D = (self.constants.k_B * self.constants.temperature_K / 
                  (6 * np.pi * self.constants.viscosity_Pa_s * r_m))
        
        # Convertir en µm²/s pour la simulation
        self.D_um2_s = self.D * 1e12
    
    def generate_trajectory(
        self, 
        n_frames: int, 
        dt_s: float, 
        initial_position: Tuple[float, float] = (0.0, 0.0)
    ) -> np.ndarray:
        """
        Génère une trajectoire 2D de mouvement Brownien.
        
        Args:
            n_frames: Nombre de frames
            dt_s: Intervalle de temps entre frames [s]
            initial_position: Position initiale (x, y) [µm]
        
        Returns:
            Tableau (n_frames, 2) des positions [µm]
        """
        # Écart-type du déplacement par step
        sigma = np.sqrt(2 * self.D_um2_s * dt_s)
        
        # Déplacements aléatoires gaussiens
        displacements = np.random.randn(n_frames, 2) * sigma
        
        # Trajectoire cumulative
        trajectory = np.cumsum(displacements, axis=0)
        trajectory[0] = initial_position
        
        return trajectory
    
    @property
    def theoretical_msd_slope(self) -> float:
        """Pente théorique du MSD : 4D pour diffusion 2D [µm²/s]."""
        return 4 * self.D_um2_s


# =============================================================================
# SIMULATION NTA (DARK FIELD SCATTERING)
# =============================================================================

class NTASimulation:
    """
    Simulation d'imagerie NTA en fond noir (Dark Field).
    
    Physique implémentée :
    1. Illumination laser latérale (90° de la caméra)
    2. Diffusion Rayleigh : I ∝ d⁶ (pour particules << λ)
    3. Fond noir absolu (pas de lumière directe vers la caméra)
    4. PSF = Disque d'Airy (diffraction)
    5. Mouvement Brownien (optionnel)
    """
    
    def __init__(
        self,
        particle: ParticleParameters,
        laser: LaserParameters,
        imaging: ImagingParameters
    ):
        self.particle = particle
        self.laser = laser
        self.imaging = imaging
        self.constants = PhysicalConstants(wavelength_nm=laser.wavelength_nm)
        self.geometry = RoughParticleGeometry(particle)
        self.brownian = BrownianMotion(particle, self.constants)
        
        self._camera_image = None
        self._scattering_data = None
    
    def _compute_rayleigh_intensity(self) -> float:
        """
        Calcule l'intensité de diffusion Rayleigh.
        
        Formule de Rayleigh pour la section efficace de diffusion :
        σ_s = (2π⁵/3) * (d⁶/λ⁴) * ((m²-1)/(m²+2))²
        
        L'intensité diffusée est proportionnelle à d⁶.
        
        Returns:
            Intensité relative de diffusion (normalisée)
        """
        # Diamètre en µm
        d = self.particle.diameter_um
        
        # Longueur d'onde en µm
        wavelength = self.constants.wavelength_um
        
        # Indice relatif m = n_particle / n_medium
        m = self.particle.refractive_index / self.constants.n_water
        
        # Facteur de Clausius-Mossotti (polarisabilité)
        polarizability_factor = (m**2 - 1) / (m**2 + 2)
        
        # Intensité de diffusion Rayleigh ∝ d⁶ / λ⁴ * |α|²
        # Normalisation pour une particule de référence (100nm)
        d_ref = 0.1  # 100 nm en µm
        
        I_rayleigh = (d / d_ref)**6 * (wavelength / 0.532)**(-4) * polarizability_factor**2
        
        return I_rayleigh
    
    def _compute_roughness_scattering_factor(self) -> float:
        """
        Calcule le facteur de modification de diffusion dû à la rugosité.
        
        La rugosité augmente la section efficace de diffusion car :
        - Augmentation de la surface effective
        - Création de dipôles supplémentaires
        - Diffusion multiple locale
        
        Returns:
            Facteur multiplicatif (>1 pour surfaces rugueuses)
        """
        if self.particle.roughness_amplitude_nm == 0:
            return 1.0
        
        # Modèle simplifié : augmentation proportionnelle à (σ/R)²
        sigma_over_r = (self.particle.roughness_amplitude_nm / 
                        self.particle.radius_nm)
        
        # Augmentation typique de 10-50% pour rugosités significatives
        enhancement_factor = 1.0 + 5.0 * sigma_over_r**2
        
        return enhancement_factor
    
    def _generate_airy_psf(self, x: np.ndarray, y: np.ndarray, 
                            center_x: float, center_y: float) -> np.ndarray:
        """
        Génère la PSF d'Airy (Fonction d'Étalement du Point).
        
        PSF(r) = [2 * J₁(π*r/r_Airy) / (π*r/r_Airy)]²
        
        où r_Airy = 0.61 * λ / NA
        
        Args:
            x, y: Grilles de coordonnées [µm]
            center_x, center_y: Centre de la PSF [µm]
        
        Returns:
            Image 2D de la PSF normalisée
        """
        # Distance au centre
        R = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Rayon d'Airy
        wavelength = self.constants.wavelength_um
        NA = self.imaging.numerical_aperture
        r_airy = 0.61 * wavelength / NA
        
        # Variable réduite
        rho = np.pi * R / r_airy
        rho_safe = np.maximum(rho, 1e-10)
        
        # Fonction d'Airy : [2*J1(x)/x]²
        airy = (2 * j1(rho_safe) / rho_safe)**2
        
        # Correction au centre (limite x→0 : J1(x)/x → 1/2)
        airy = np.where(rho < 1e-6, 1.0, airy)
        
        return airy
    
    def generate_camera_image(
        self, 
        particle_positions: Optional[List[Tuple[float, float]]] = None,
        particle_radii: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Génère l'image caméra en fond noir (Dark Field NTA).
        
        Physique :
        1. Fond noir absolu (pas de lumière incidente directe)
        2. Chaque particule diffuse la lumière (Rayleigh : I ∝ d⁶)
        3. La lumière diffusée forme un disque d'Airy sur la caméra
        4. Superposition linéaire des contributions
        
        Args:
            particle_positions: Liste de positions (x, y) en µm.
                               Si None, une seule particule au centre.
            particle_radii: Liste des rayons en nm pour chaque particule.
                           Si None, utilise le rayon de self.particle.
        
        Returns:
            Image 2D (valeurs 0-255)
        """
        N = self.imaging.image_pixels
        fov = self.imaging.field_of_view_um
        
        # Grille image
        pixel_size = fov / N
        x = np.linspace(-fov/2, fov/2, N)
        y = np.linspace(-fov/2, fov/2, N)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # =====================================================================
        # 1. FOND NOIR ABSOLU (Dark Field)
        # =====================================================================
        image = np.zeros((N, N), dtype=float)
        
        # =====================================================================
        # 2. PARTICULE(S) DIFFUSANTE(S)
        # =====================================================================
        if particle_positions is None:
            # Une seule particule au centre
            particle_positions = [(0.0, 0.0)]
        
        if particle_radii is None:
            particle_radii = [self.particle.radius_nm] * len(particle_positions)
        
        # Stocker les données de diffusion pour analyse
        scattering_intensities = []
        
        for i, (px, py) in enumerate(particle_positions):
            # Rayon de cette particule
            r_nm = particle_radii[i] if i < len(particle_radii) else self.particle.radius_nm
            
            # Créer une particule temporaire pour le calcul
            temp_particle = ParticleParameters(
                radius_nm=r_nm,
                refractive_index=self.particle.refractive_index,
                roughness_amplitude_nm=self.particle.roughness_amplitude_nm,
                roughness_lmax=self.particle.roughness_lmax
            )
            
            # Calcul de l'intensité Rayleigh (∝ d⁶)
            d = 2 * r_nm / 1000.0  # Diamètre en µm
            m = temp_particle.refractive_index / self.constants.n_water
            polarizability_factor = (m**2 - 1) / (m**2 + 2)
            
            # Intensité normalisée (référence: particule de 100nm)
            d_ref = 0.1  # µm
            I_scatter = (d / d_ref)**6 * polarizability_factor**2
            
            # Facteur de rugosité
            if self.particle.roughness_amplitude_nm > 0:
                roughness_factor = self._compute_roughness_scattering_factor()
                I_scatter *= roughness_factor
            
            scattering_intensities.append(I_scatter)
            
            # =====================================================================
            # 3. PSF : DISQUE D'AIRY
            # =====================================================================
            psf = self._generate_airy_psf(X, Y, px, py)
            
            # Contribution de cette particule
            # Échelle pour visibilité (ajuster selon le contexte)
            intensity_scale = 200.0  # Pour image 8-bit
            
            image += I_scatter * psf * intensity_scale
        
        # =====================================================================
        # 4. BRUIT DU CAPTEUR
        # =====================================================================
        # Shot noise (bruit poissonien) - dominant pour les signaux faibles
        image_with_shot = np.random.poisson(np.maximum(image, 0).astype(int)).astype(float)
        
        # Read noise (bruit de lecture gaussien)
        read_noise = np.random.randn(N, N) * self.imaging.camera_noise_level * 10
        
        image_noisy = image_with_shot + read_noise
        
        # =====================================================================
        # 5. CLAMP ET CONVERSION
        # =====================================================================
        image_final = np.clip(image_noisy, 0, 255)
        
        self._camera_image = image_final
        self._scattering_data = {
            'positions': particle_positions,
            'radii': particle_radii,
            'intensities': scattering_intensities,
            'total_particles': len(particle_positions),
            'mean_intensity': np.mean(scattering_intensities) if scattering_intensities else 0,
            'image_max': np.max(image_final),
            'image_mean': np.mean(image_final),
            'snr': np.max(image_final) / (np.std(image_final) + 1e-10)
        }
        
        return image_final
    
    def generate_video_frames(
        self, 
        n_frames: int = 100,
        dt_s: float = 0.033,  # ~30 fps
        initial_position: Tuple[float, float] = (0.0, 0.0)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Génère une séquence vidéo d'une particule en mouvement Brownien.
        
        Args:
            n_frames: Nombre de frames
            dt_s: Intervalle de temps entre frames [s]
            initial_position: Position initiale [µm]
        
        Returns:
            (video_frames, trajectory): Stack d'images et trajectoire
        """
        # Générer la trajectoire Brownienne
        trajectory = self.brownian.generate_trajectory(n_frames, dt_s, initial_position)
        
        N = self.imaging.image_pixels
        video = np.zeros((n_frames, N, N))
        
        for i in range(n_frames):
            pos = [(trajectory[i, 0], trajectory[i, 1])]
            video[i] = self.generate_camera_image(particle_positions=pos)
        
        return video, trajectory
    
    def get_surface_mesh(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Retourne le maillage 3D de la surface de la particule."""
        return self.geometry.generate_surface_mesh()
    
    def get_diffusion_coefficient(self) -> float:
        """Retourne le coefficient de diffusion D [µm²/s]."""
        return self.brownian.D_um2_s


# =============================================================================
# ANALYSE SPECTRALE (FFT) - Adaptée pour Dark Field
# =============================================================================

def compute_spectral_analysis(image: np.ndarray, fov_um: float) -> dict:
    """
    Calcule l'analyse spectrale 2D (FFT) de l'image NTA.
    
    Pour le dark field, le fond est déjà noir, donc pas besoin de
    soustraire un fond moyen important.
    
    Args:
        image: Image 2D
        fov_um: Champ de vue en microns
    
    Returns:
        Dict avec PSD 2D, profil radial, fréquences, etc.
    """
    N = image.shape[0]
    pixel_size = fov_um / N
    
    # =========================================================================
    # PRÉTRAITEMENT (minimal pour dark field)
    # =========================================================================
    image_centered = image - np.mean(image)
    
    # Fenêtrage 2D de Hann
    hann_1d = np.hanning(N)
    hann_2d = np.outer(hann_1d, hann_1d)
    image_windowed = image_centered * hann_2d
    
    # =========================================================================
    # FFT 2D
    # =========================================================================
    freq = np.fft.fftfreq(N, d=pixel_size)
    freq_shift = np.fft.fftshift(freq)
    FX, FY = np.meshgrid(freq_shift, freq_shift, indexing='ij')
    
    fft_img = np.fft.fftshift(np.fft.fft2(image_windowed))
    psd = np.abs(fft_img)**2
    psd_norm = psd / (np.sum(psd) + 1e-20) * N * N
    
    # =========================================================================
    # PROFIL RADIAL
    # =========================================================================
    center = N // 2
    y_idx, x_idx = np.indices(psd.shape)
    r = np.sqrt((x_idx - center)**2 + (y_idx - center)**2)
    r_int = r.astype(int)
    r_max = min(center, N - center)
    
    radial_profile = np.zeros(r_max)
    radial_std = np.zeros(r_max)
    
    for i in range(r_max):
        mask = r_int == i
        if np.sum(mask) > 0:
            radial_profile[i] = np.mean(psd_norm[mask])
            radial_std[i] = np.std(psd_norm[mask])
    
    freq_step = freq_shift[1] - freq_shift[0] if len(freq_shift) > 1 else 1.0
    freq_radial = np.arange(r_max) * freq_step
    
    # =========================================================================
    # MÉTRIQUES
    # =========================================================================
    cumsum = np.cumsum(radial_profile)
    if cumsum[-1] > 0:
        idx_median = np.searchsorted(cumsum, cumsum[-1] / 2)
        freq_median = freq_radial[idx_median] if idx_median < len(freq_radial) else 0
    else:
        freq_median = 0
    
    if np.sum(radial_profile) > 0:
        freq_mean = np.sum(freq_radial * radial_profile) / np.sum(radial_profile)
        freq_width = np.sqrt(np.sum((freq_radial - freq_mean)**2 * radial_profile) / np.sum(radial_profile))
    else:
        freq_mean = 0
        freq_width = 0
    
    return {
        'psd_2d': psd_norm,
        'psd_2d_log': np.log10(psd_norm + 1e-10),
        'radial_profile': radial_profile,
        'radial_std': radial_std,
        'freq_radial': freq_radial,
        'freq_2d': (FX, FY),
        'extent': [freq_shift.min(), freq_shift.max(), freq_shift.min(), freq_shift.max()],
        'freq_median': freq_median,
        'freq_mean': freq_mean,
        'freq_width': freq_width,
        'total_power': np.sum(psd)
    }


# =============================================================================
# COMPARAISON LISSE vs RUGUEUSE (NTA)
# =============================================================================

def run_comparison(
    radius_nm: float = 100.0,
    roughness_nm: float = 5.0,
    roughness_lmax: int = 10,
    refractive_index: float = 1.50,
    wavelength_nm: float = 532.0,
    numerical_aperture: float = 0.4,
    image_pixels: int = 256,
    save_path: Optional[str] = None
):
    """
    Lance une comparaison NTA entre particule lisse et rugueuse.
    
    Génère une figure avec 2 colonnes (lisse / rugueuse) et 4 lignes :
    1. Représentation 3D de la particule
    2. Image NTA (fond noir, particule brillante)
    3. Analyse spectrale (FFT)
    4. Comparaison spectrale (profils radiaux)
    """
    print("=" * 70)
    print("SIMULATION NTA : COMPARAISON LISSE vs RUGUEUSE")
    print("=" * 70)
    print(f"Physique: Dark Field Scattering (I ∝ d⁶)")
    print(f"Laser: λ = {wavelength_nm} nm")
    print(f"Particule: R = {radius_nm} nm, n = {refractive_index}")
    print("=" * 70)
    
    laser = LaserParameters(wavelength_nm=wavelength_nm)
    imaging = ImagingParameters(
        numerical_aperture=numerical_aperture,
        image_pixels=image_pixels
    )
    
    # --- PARTICULE LISSE ---
    print("\n[1/2] Simulation particule LISSE...")
    particle_smooth = ParticleParameters(
        radius_nm=radius_nm,
        refractive_index=refractive_index,
        roughness_amplitude_nm=0.0,
        material_name="polystyrene (lisse)"
    )
    sim_smooth = NTASimulation(particle_smooth, laser, imaging)
    sim_smooth.generate_camera_image()
    X_s, Y_s, Z_s = sim_smooth.get_surface_mesh()
    
    # --- PARTICULE RUGUEUSE ---
    print("[2/2] Simulation particule RUGUEUSE...")
    particle_rough = ParticleParameters(
        radius_nm=radius_nm,
        refractive_index=refractive_index,
        roughness_amplitude_nm=roughness_nm,
        roughness_lmax=roughness_lmax,
        material_name="polystyrene (rugueuse)"
    )
    sim_rough = NTASimulation(particle_rough, laser, imaging)
    sim_rough.generate_camera_image()
    X_r, Y_r, Z_r = sim_rough.get_surface_mesh()
    
    # --- ANALYSE SPECTRALE ---
    print("\nAnalyse spectrale...")
    fov = imaging.field_of_view_um
    spectral_smooth = compute_spectral_analysis(sim_smooth._camera_image, fov)
    spectral_rough = compute_spectral_analysis(sim_rough._camera_image, fov)
    
    # =================================================================
    # CRÉATION DE LA FIGURE
    # =================================================================
    print("\nGénération de la figure...")
    
    fig = plt.figure(figsize=(14, 18))
    
    # Paramètres communs
    extent_img = [-fov/2, fov/2, -fov/2, fov/2]
    
    # =====================================================================
    # LIGNE 1 : GÉOMÉTRIE 3D
    # =====================================================================
    
    # Colonne 1 : Particule lisse
    ax1 = fig.add_subplot(4, 2, 1, projection='3d')
    ax1.plot_surface(X_s * 1000, Y_s * 1000, Z_s * 1000, 
                     cmap='Blues', alpha=0.9, edgecolor='none')
    ax1.set_xlabel('X [nm]')
    ax1.set_ylabel('Y [nm]')
    ax1.set_zlabel('Z [nm]')
    ax1.set_title('Particule LISSE\n(sphère parfaite)', fontweight='bold')
    lim = radius_nm * 1.3
    ax1.set_xlim([-lim, lim])
    ax1.set_ylim([-lim, lim])
    ax1.set_zlim([-lim, lim])
    ax1.set_box_aspect([1, 1, 1])
    
    # Colonne 2 : Particule rugueuse
    ax2 = fig.add_subplot(4, 2, 2, projection='3d')
    ax2.plot_surface(X_r * 1000, Y_r * 1000, Z_r * 1000, 
                     cmap='Oranges', alpha=0.9, edgecolor='none')
    ax2.set_xlabel('X [nm]')
    ax2.set_ylabel('Y [nm]')
    ax2.set_zlabel('Z [nm]')
    ax2.set_title(f'Particule RUGUEUSE\n(σ={roughness_nm}nm, l_max={roughness_lmax})', fontweight='bold')
    ax2.set_xlim([-lim, lim])
    ax2.set_ylim([-lim, lim])
    ax2.set_zlim([-lim, lim])
    ax2.set_box_aspect([1, 1, 1])
    
    # =====================================================================
    # LIGNE 2 : IMAGES NTA (FOND NOIR - DARK FIELD)
    # =====================================================================
    
    img_smooth = sim_smooth._camera_image
    img_rough = sim_rough._camera_image
    
    # Pour NTA : fond noir, donc vmin=0, vmax basé sur le max
    vmax = max(np.max(img_smooth), np.max(img_rough)) * 1.1
    
    # Colonne 1 : Image lisse
    ax3 = fig.add_subplot(4, 2, 3)
    im3 = ax3.imshow(img_smooth.T, origin='lower', extent=extent_img,
                     cmap='gray', aspect='equal', vmin=0, vmax=vmax)
    ax3.set_xlabel('X [μm]')
    ax3.set_ylabel('Y [μm]')
    ax3.set_title('Image NTA (Dark Field) - LISSE\n(Fond noir, particule brillante)', fontweight='bold')
    plt.colorbar(im3, ax=ax3, label='Intensité', shrink=0.8)
    
    # Ajouter cercle pour montrer la taille de l'Airy disk
    airy_r = 0.61 * (wavelength_nm/1000) / numerical_aperture
    circle = plt.Circle((0, 0), airy_r, fill=False, color='cyan', 
                        linestyle='--', linewidth=1.5, label=f'Airy (r={airy_r:.2f}µm)')
    ax3.add_patch(circle)
    ax3.legend(loc='upper right', fontsize=8)
    
    # Colonne 2 : Image rugueuse
    ax4 = fig.add_subplot(4, 2, 4)
    im4 = ax4.imshow(img_rough.T, origin='lower', extent=extent_img,
                     cmap='gray', aspect='equal', vmin=0, vmax=vmax)
    ax4.set_xlabel('X [μm]')
    ax4.set_ylabel('Y [μm]')
    ax4.set_title('Image NTA (Dark Field) - RUGUEUSE\n(Fond noir, particule brillante)', fontweight='bold')
    plt.colorbar(im4, ax=ax4, label='Intensité', shrink=0.8)
    circle2 = plt.Circle((0, 0), airy_r, fill=False, color='cyan', 
                         linestyle='--', linewidth=1.5)
    ax4.add_patch(circle2)
    
    # =====================================================================
    # LIGNE 3 : ANALYSE SPECTRALE (FFT)
    # =====================================================================
    
    psd_smooth_log = spectral_smooth['psd_2d_log']
    psd_rough_log = spectral_rough['psd_2d_log']
    vmin_spec = min(np.percentile(psd_smooth_log, 5), np.percentile(psd_rough_log, 5))
    vmax_spec = max(np.percentile(psd_smooth_log, 99), np.percentile(psd_rough_log, 99))
    
    # Colonne 1 : Spectre lisse
    ax5 = fig.add_subplot(4, 2, 5)
    extent_freq = spectral_smooth['extent']
    im5 = ax5.imshow(psd_smooth_log.T, 
                     origin='lower', extent=extent_freq,
                     cmap='inferno', aspect='equal', vmin=vmin_spec, vmax=vmax_spec)
    ax5.set_xlabel('$f_x$ [cycles/μm]')
    ax5.set_ylabel('$f_y$ [cycles/μm]')
    ax5.set_title(f'Spectre FFT - LISSE\n(f_moy={spectral_smooth["freq_mean"]:.2f} cyc/μm)', fontweight='bold')
    plt.colorbar(im5, ax=ax5, label='log₁₀(PSD)', shrink=0.8)
    
    # Cercle de coupure NA
    f_cutoff = numerical_aperture / (wavelength_nm / 1000)
    theta_c = np.linspace(0, 2*np.pi, 100)
    ax5.plot(f_cutoff * np.cos(theta_c), f_cutoff * np.sin(theta_c), 
             'w--', linewidth=1.5, label=f'NA cutoff')
    
    # Colonne 2 : Spectre rugueux
    ax6 = fig.add_subplot(4, 2, 6)
    im6 = ax6.imshow(psd_rough_log.T, 
                     origin='lower', extent=extent_freq,
                     cmap='inferno', aspect='equal', vmin=vmin_spec, vmax=vmax_spec)
    ax6.set_xlabel('$f_x$ [cycles/μm]')
    ax6.set_ylabel('$f_y$ [cycles/μm]')
    ax6.set_title(f'Spectre FFT - RUGUEUSE\n(f_moy={spectral_rough["freq_mean"]:.2f} cyc/μm)', fontweight='bold')
    plt.colorbar(im6, ax=ax6, label='log₁₀(PSD)', shrink=0.8)
    ax6.plot(f_cutoff * np.cos(theta_c), f_cutoff * np.sin(theta_c), 
             'w--', linewidth=1.5)
    
    # =====================================================================
    # LIGNE 4 : COMPARAISON SPECTRALE (PROFILS RADIAUX)
    # =====================================================================
    
    ax7 = fig.add_subplot(4, 1, 4)
    
    freq_s = spectral_smooth['freq_radial']
    radial_s = spectral_smooth['radial_profile']
    radial_s_std = spectral_smooth['radial_std']
    freq_r = spectral_rough['freq_radial']
    radial_r = spectral_rough['radial_profile']
    radial_r_std = spectral_rough['radial_std']
    
    ax7.semilogy(freq_s, radial_s + 1e-10, 'b-', linewidth=2, label='Lisse')
    ax7.fill_between(freq_s, 
                     np.maximum(radial_s - radial_s_std, 1e-10), 
                     radial_s + radial_s_std + 1e-10,
                     alpha=0.2, color='blue')
    
    ax7.semilogy(freq_r, radial_r + 1e-10, 'r-', linewidth=2, label='Rugueuse')
    ax7.fill_between(freq_r, 
                     np.maximum(radial_r - radial_r_std, 1e-10), 
                     radial_r + radial_r_std + 1e-10,
                     alpha=0.2, color='red')
    
    ax7.axvline(x=f_cutoff, color='gray', linestyle='--', linewidth=1.5, 
                label=f'Coupure NA ({f_cutoff:.2f} cyc/μm)')
    
    ax7.axvline(x=spectral_smooth['freq_mean'], color='blue', linestyle=':', linewidth=1.5, alpha=0.7)
    ax7.axvline(x=spectral_rough['freq_mean'], color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    
    ax7.set_xlabel('Fréquence spatiale [cycles/μm]', fontsize=11)
    ax7.set_ylabel('PSD (normalisée)', fontsize=11)
    ax7.set_title('Comparaison des Spectres : Profil Radial (Moyenne Azimutale)\n'
                  f'Largeur spectrale: Lisse={spectral_smooth["freq_width"]:.2f}, '
                  f'Rugueuse={spectral_rough["freq_width"]:.2f} cyc/μm', 
                  fontweight='bold', fontsize=11)
    ax7.legend(loc='upper right', fontsize=10)
    ax7.grid(True, alpha=0.3)
    ax7.set_xlim([0, min(freq_s[-1], f_cutoff * 2)])
    
    all_radial = np.concatenate([radial_s, radial_r])
    ymin = max(np.min(all_radial[all_radial > 0]) * 0.1, 1e-8)
    ymax = np.max(all_radial) * 2
    ax7.set_ylim([ymin, ymax])
    
    # =====================================================================
    # TITRE GLOBAL
    # =====================================================================
    
    plt.suptitle(
        f'Simulation NTA (Dark Field Scattering) : Comparaison Lisse vs Rugueuse\n'
        f'R={radius_nm}nm, n={refractive_index}, λ={wavelength_nm}nm, NA={numerical_aperture}\n'
        f'Loi de Rayleigh: I ∝ d⁶',
        fontsize=14, fontweight='bold', y=1.02
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"\nFigure sauvegardée : {save_path}")
    
    plt.show()
    
    # =================================================================
    # STATISTIQUES
    # =================================================================
    print("\n" + "=" * 70)
    print("STATISTIQUES NTA")
    print("=" * 70)
    
    D_smooth = sim_smooth.get_diffusion_coefficient()
    D_rough = sim_rough.get_diffusion_coefficient()
    
    I_smooth = sim_smooth._scattering_data.get('mean_intensity', 0)
    I_rough = sim_rough._scattering_data.get('mean_intensity', 0)
    
    print(f"""
PHYSIQUE NTA (Dark Field Scattering):
  • Loi de diffusion: I ∝ d⁶ (Rayleigh)
  • Fond: Noir absolu (0 photon sans particule)
  • PSF: Disque d'Airy (r_Airy = {0.61 * wavelength_nm/1000 / numerical_aperture:.3f} µm)

PARTICULE LISSE:
  • Rayon: {radius_nm} nm
  • Intensité de diffusion relative: {I_smooth:.4f}
  • Coefficient de diffusion D: {D_smooth:.4f} µm²/s
  • MSD théorique (pente): {4*D_smooth:.4f} µm²/s
  • Niveau max image: {np.max(img_smooth):.1f} (sur 255)
  • SNR: {sim_smooth._scattering_data['snr']:.1f}
  • Fréquence spectrale moyenne: {spectral_smooth['freq_mean']:.3f} cyc/μm

PARTICULE RUGUEUSE:
  • Rayon: {radius_nm} nm (σ_rugosité = {roughness_nm} nm)
  • Intensité de diffusion relative: {I_rough:.4f}
  • Ratio intensité (rugueux/lisse): {I_rough/max(I_smooth, 1e-10):.2f}
  • Coefficient de diffusion D: {D_rough:.4f} µm²/s
  • Niveau max image: {np.max(img_rough):.1f} (sur 255)
  • SNR: {sim_rough._scattering_data['snr']:.1f}
  • Fréquence spectrale moyenne: {spectral_rough['freq_mean']:.3f} cyc/μm

COMPARAISON:
  • Ratio fréquence moyenne (rugueux/lisse): {spectral_rough['freq_mean']/max(spectral_smooth['freq_mean'], 1e-10):.2f}
  • Fréquence de coupure NA: {f_cutoff:.2f} cyc/μm

EFFET D⁶ (Rayleigh) - Comparaison de tailles:
  • Particule 50nm  → I ∝ (50)⁶  = {(50/100)**6:.4f} × I_100nm
  • Particule 100nm → I ∝ (100)⁶ = 1.0000 × I_100nm (référence)
  • Particule 200nm → I ∝ (200)⁶ = {(200/100)**6:.1f} × I_100nm
""")
    
    return {
        'sim_smooth': sim_smooth,
        'sim_rough': sim_rough,
        'spectral_smooth': spectral_smooth,
        'spectral_rough': spectral_rough
    }


# =============================================================================
# DÉMONSTRATION DE L'EFFET D⁶
# =============================================================================

def demo_d6_effect(
    radii_nm: List[float] = [30, 50, 80, 100, 150, 200],
    wavelength_nm: float = 532.0,
    numerical_aperture: float = 0.4,
    save_path: Optional[str] = None
):
    """
    Démonstration visuelle de l'effet d⁶ en NTA.
    
    Montre comment l'intensité chute drastiquement pour les petites particules.
    """
    print("=" * 70)
    print("DÉMONSTRATION EFFET D⁶ (Loi de Rayleigh)")
    print("=" * 70)
    
    laser = LaserParameters(wavelength_nm=wavelength_nm)
    imaging = ImagingParameters(
        numerical_aperture=numerical_aperture,
        image_pixels=128
    )
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    intensities = []
    
    for i, r_nm in enumerate(radii_nm):
        print(f"Simulation particule R = {r_nm} nm...")
        
        particle = ParticleParameters(radius_nm=r_nm)
        sim = NTASimulation(particle, laser, imaging)
        img = sim.generate_camera_image()
        
        intensities.append(sim._scattering_data['mean_intensity'])
        
        fov = imaging.field_of_view_um
        extent = [-fov/2, fov/2, -fov/2, fov/2]
        
        # Même échelle pour tous
        vmax = 255
        
        ax = axes[i]
        im = ax.imshow(img.T, origin='lower', extent=extent,
                      cmap='hot', aspect='equal', vmin=0, vmax=vmax)
        ax.set_title(f'd = {2*r_nm} nm\nI_rel = {intensities[-1]:.4f}', 
                    fontweight='bold', fontsize=11)
        ax.set_xlabel('X [μm]')
        ax.set_ylabel('Y [μm]')
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.suptitle(
        f'Effet de la Loi de Rayleigh : I ∝ d⁶\n'
        f'λ = {wavelength_nm} nm, NA = {numerical_aperture}',
        fontsize=14, fontweight='bold'
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"\nFigure sauvegardée : {save_path}")
    
    plt.show()
    
    # Graphique log-log de l'intensité vs diamètre
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    
    diameters = [2*r for r in radii_nm]
    
    ax2.loglog(diameters, intensities, 'bo-', markersize=10, linewidth=2, label='Simulation')
    
    # Courbe théorique d⁶
    d_theory = np.linspace(min(diameters)*0.8, max(diameters)*1.2, 100)
    d_ref = 200  # Référence
    I_ref = intensities[diameters.index(d_ref)] if d_ref in diameters else intensities[-1]
    I_theory = I_ref * (d_theory / d_ref)**6
    ax2.loglog(d_theory, I_theory, 'r--', linewidth=2, label='Théorie: I ∝ d⁶')
    
    ax2.set_xlabel('Diamètre [nm]', fontsize=12)
    ax2.set_ylabel('Intensité relative', fontsize=12)
    ax2.set_title('Vérification de la Loi de Rayleigh : I ∝ d⁶', fontweight='bold', fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.show()
    
    print("\nIntensités relatives:")
    for d, I in zip(diameters, intensities):
        print(f"  d = {d:3d} nm → I_rel = {I:.6f}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys
    
    parser = argparse.ArgumentParser(
        description="Simulation NTA - Nanoparticle Tracking Analysis (Dark Field Scattering)"
    )
    parser.add_argument("--radius", type=float, default=100.0,
                       help="Rayon de la particule [nm]")
    parser.add_argument("--roughness", type=float, default=5.0,
                       help="Amplitude RMS de la rugosité [nm]")
    parser.add_argument("--lmax", type=int, default=10,
                       help="Ordre max des harmoniques sphériques")
    parser.add_argument("--n-particle", type=float, default=1.50,
                       help="Indice de réfraction")
    parser.add_argument("--wavelength", type=float, default=532.0,
                       help="Longueur d'onde du laser [nm]")
    parser.add_argument("--na", type=float, default=0.4,
                       help="Ouverture numérique")
    parser.add_argument("--pixels", type=int, default=256,
                       help="Taille de l'image [pixels]")
    parser.add_argument("--save", type=str, default="nta_comparison.png",
                       help="Fichier de sortie")
    parser.add_argument("--no-save", action="store_true",
                       help="Ne pas sauvegarder")
    parser.add_argument("--demo-d6", action="store_true",
                       help="Lancer la démonstration de l'effet d⁶")
    
    args = parser.parse_args()
    
    if args.demo_d6:
        demo_d6_effect(
            wavelength_nm=args.wavelength,
            numerical_aperture=args.na,
            save_path=None if args.no_save else "nta_d6_demo.png"
        )
    else:
        run_comparison(
            radius_nm=args.radius,
            roughness_nm=args.roughness,
            roughness_lmax=args.lmax,
            refractive_index=args.n_particle,
            wavelength_nm=args.wavelength,
            numerical_aperture=args.na,
            image_pixels=args.pixels,
            save_path=None if args.no_save else args.save
        )
