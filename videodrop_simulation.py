#!/usr/bin/env python3
"""
===============================================================================
VIDEODROP SIMULATION - Interferometric Brightfield Nanoparticle Imaging
===============================================================================

Simulation d'une image de caméra pour la microscopie interférométrique
en champ clair (type Videodrop / Myriade).

Comparaison : Particule lisse vs Particule rugueuse

Principe physique :
- Illumination en transmission (LED)
- Interférence entre l'onde directe et l'onde diffusée par la particule
- Détection : I = |E_ref + E_scat|²

===============================================================================
"""

# =============================================================================
# HYPERPARAMÈTRES DE SIMULATION
# =============================================================================
# Modifier ces valeurs pour changer les paramètres de simulation

HYPERPARAMS = {
    # Particule
    'radius_nm': 60.0,              # Rayon de la particule [nm]
    'roughness_nm': 10.0,           # Amplitude RMS de la rugosité [nm]
    'roughness_lmax': 10,           # Ordre max des harmoniques sphériques
    'refractive_index': 1.50,       # Indice de réfraction (polystyrène)
    
    # Optique
    'wavelength_nm': 488.0,         # Longueur d'onde LED [nm]
    'numerical_aperture': 0.3,      # Ouverture numérique
    
    # Image
    'image_pixels': 256,            # Résolution de l'image [pixels]
    
    # Sauvegarde
    'save_path': 'videodrop_comparison.png',  # Fichier de sortie (None = pas de sauvegarde)
}

# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Optional
import warnings

# Pour les harmoniques sphériques (rugosité)
try:
    from scipy.special import sph_harm_y
    USE_NEW_SCIPY = True
except ImportError:
    from scipy.special import sph_harm
    USE_NEW_SCIPY = False


# =============================================================================
# CONSTANTES PHYSIQUES
# =============================================================================

@dataclass
class PhysicalConstants:
    """Constantes physiques pour la simulation."""
    wavelength_nm: float = 488.0
    n_water: float = 1.33
    
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
    radius_nm: float = 80.0
    refractive_index: float = 1.50
    roughness_amplitude_nm: float = 0.0   # Amplitude RMS de la rugosité
    roughness_lmax: int = 10              # Ordre max des harmoniques sphériques
    roughness_seed: int = 42              # Seed pour reproductibilité
    material_name: str = "polystyrene"
    
    @property
    def radius_um(self) -> float:
        return self.radius_nm / 1000.0
    
    @property
    def roughness_amplitude_um(self) -> float:
        return self.roughness_amplitude_nm / 1000.0
    
    @property
    def size_parameter(self) -> float:
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
                if m == 0:
                    self.alm[(l, m)] = self.rng.normal(0, np.sqrt(power_l))
                else:
                    re = self.rng.normal(0, np.sqrt(power_l / 2))
                    im = self.rng.normal(0, np.sqrt(power_l / 2))
                    self.alm[(l, m)] = complex(re, im)
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
                Ylm = sph_harm_y(l, m, theta, phi)
            else:
                Ylm = sph_harm(m, l, phi, theta)
            r += np.real(coeff * Ylm)
        
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


# =============================================================================
# PARAMÈTRES D'ILLUMINATION ET D'IMAGERIE
# =============================================================================

@dataclass
class IlluminationParameters:
    wavelength_nm: float = 488.0
    polarization: str = "x"


@dataclass
class ImagingParameters:
    numerical_aperture: float = 0.3
    magnification: float = 50.0             # Dézoom ×4 (était 200)
    camera_pixel_size_um: float = 6.5
    image_pixels: int = 256                  # Résolution augmentée
    reference_amplitude: float = 1.0
    defocus_um: float = 1.0                  # Plus de défocus pour étaler le halo
    camera_noise_level: float = 0.005        # Moins de bruit
    background_level: int = 180
    
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
    """Simulation d'imagerie interférométrique type Videodrop."""
    
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
        self.geometry = RoughParticleGeometry(particle)
        
        self._far_field_data = None
        self._camera_image = None
        self._interferometry_data = None
    
    def _compute_scattering(self):
        """Calcule le champ diffusé."""
        n_theta = 90
        n_phi = 180
        
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2*np.pi, n_phi)
        THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
        
        # Pattern angulaire dipôle
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
        
        # Effets de rugosité : perturbations de phase et amplitude
        if self.particle.roughness_amplitude_nm > 0:
            np.random.seed(self.particle.roughness_seed)
            roughness_factor = self.particle.roughness_amplitude_nm / self.particle.radius_nm
            
            amplitude_noise = np.random.randn(n_theta, n_phi) * roughness_factor * 0.3
            phase_noise = np.random.randn(n_theta, n_phi) * roughness_factor * np.pi
            
            from scipy.ndimage import gaussian_filter
            amplitude_noise = gaussian_filter(amplitude_noise, sigma=2)
            phase_noise = gaussian_filter(phase_noise, sigma=2)
            
            E_amplitude = E_amplitude * (1 + amplitude_noise)
            base_phase = base_phase + phase_noise
        
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
        Génère l'image caméra avec détection interférométrique type Videodrop.
        
        Physique implémentée :
        1. Onde incidente Ei : onde plane uniforme (LED collimatée)
        2. Onde diffusée Es : diffusion Rayleigh, onde sphérique avec déphasage
        3. Déphasage : déterminé par la polarisabilité complexe α = (m²-1)/(m²+2)
        4. Interférence : I = |Ei + Es|² = |Ei|² + |Es|² + 2·Re(Ei·Es*)
        5. Amplification homodyne : le terme croisé 2·Re(Ei·Es*) domine
        6. Imagerie : propagation vers le plan image avec PSF du système
        """
        if self._far_field_data is None:
            self._compute_scattering()
        
        N = self.imaging.image_pixels
        fov = self.imaging.field_of_view_um
        NA = self.imaging.numerical_aperture
        defocus = self.imaging.defocus_um
        background_level = self.imaging.background_level
        noise_level = self.imaging.camera_noise_level
        
        # Paramètres optiques
        wavelength = self.constants.wavelength_um
        n_medium = self.constants.n_water
        k = 2 * np.pi * n_medium / wavelength  # Nombre d'onde dans le milieu
        
        # Grille image (plan du capteur, coordonnées dans l'espace objet)
        pixel_size = fov / N
        x = np.linspace(-fov/2, fov/2, N)
        y = np.linspace(-fov/2, fov/2, N)
        X, Y = np.meshgrid(x, y, indexing='ij')
        R = np.sqrt(X**2 + Y**2)  # Distance radiale au centre (position particule)
        
        # =====================================================================
        # 1. ONDE INCIDENTE (Référence) : Onde plane uniforme
        # =====================================================================
        # LED collimatée → front d'onde plan, amplitude constante
        E_i_amplitude = self.imaging.reference_amplitude
        E_i = E_i_amplitude * np.ones((N, N), dtype=complex)
        
        # =====================================================================
        # 2. POLARISABILITÉ ET DÉPHASAGE (Physique de la diffusion)
        # =====================================================================
        # Indice relatif m = n_particule / n_milieu
        m = self.particle.refractive_index / n_medium
        
        # Polarisabilité complexe (Clausius-Mossotti)
        # α = (m² - 1) / (m² + 2)
        # Pour polystyrène (n=1.50) dans eau (n=1.33): m ≈ 1.128
        # → α est RÉEL et POSITIF pour les diélectriques
        polarizability = (m**2 - 1) / (m**2 + 2)
        
        # Le déphasage intrinsèque de la diffusion Rayleigh
        # Pour un dipôle oscillant : déphasage de π/2 par rapport au champ incident
        # PLUS la phase de la polarisabilité
        intrinsic_phase = np.angle(polarizability) + np.pi/2
        
        # =====================================================================
        # 3. AMPLITUDE DE DIFFUSION RAYLEIGH
        # =====================================================================
        r_particle = self.particle.radius_um
        
        # Amplitude de diffusion Rayleigh : E_s ∝ k² · r³ · α
        # Le facteur k² vient de la radiation du dipôle
        scattering_strength = (k**2) * (r_particle**3) * np.abs(polarizability)
        
        # =====================================================================
        # 4. PROPAGATION DE L'ONDE DIFFUSÉE VERS LE CAPTEUR
        # =====================================================================
        # L'onde diffusée est sphérique, mais après passage par l'objectif,
        # elle devient une onde convergente/divergente selon le défocus
        
        # Rayon d'Airy (résolution du système optique)
        airy_radius = 0.61 * wavelength / NA
        
        # Paramètre de défocus (phase de Gouy incluse)
        # z > 0 : plan image au-dessus du focus (particule floue, halo externe clair)
        # z < 0 : plan image en-dessous du focus (particule floue, halo externe sombre)
        z = defocus
        
        # Rayleigh range (profondeur de champ)
        z_R = np.pi * airy_radius**2 / wavelength
        
        # Phase de Gouy : change le signe du contraste selon le défocus
        gouy_phase = np.arctan(z / z_R) if z_R > 0 else 0
        
        # =====================================================================
        # 5. PSF COMPLEXE AVEC DÉFOCUS (Modèle de Fresnel)
        # =====================================================================
        # La PSF défocalisée inclut des anneaux de Fresnel
        
        # Paramètre réduit radial
        rho = R / airy_radius
        rho_safe = np.maximum(rho, 1e-8)
        
        # PSF d'Airy (en focus)
        from scipy.special import j1
        airy_amplitude = 2 * j1(np.pi * rho_safe) / (np.pi * rho_safe)
        airy_amplitude = np.where(rho < 1e-6, 1.0, airy_amplitude)
        
        # Phase de défocus quadratique (approximation paraxiale)
        # Phase ∝ k · z · (r/f)² où f est liée à NA
        defocus_phase = k * z * (R / (fov/2))**2 * (NA**2) / 2
        
        # Fonction pupille avec défocus
        # Ceci crée les anneaux de Fresnel caractéristiques
        psf_complex = airy_amplitude * np.exp(1j * defocus_phase)
        
        # =====================================================================
        # 6. CHAMP DIFFUSÉ AU NIVEAU DU CAPTEUR
        # =====================================================================
        # E_s = amplitude × exp(i·phase_totale) × PSF
        
        # Phase totale de l'onde diffusée
        total_phase = intrinsic_phase + gouy_phase
        
        # Perturbations de phase dues à la rugosité
        if self.particle.roughness_amplitude_nm > 0:
            np.random.seed(self.particle.roughness_seed)
            roughness_factor = self.particle.roughness_amplitude_nm / self.particle.radius_nm
            from scipy.ndimage import gaussian_filter
            # La rugosité crée des fluctuations de chemin optique
            phase_roughness = gaussian_filter(
                np.random.randn(N, N) * roughness_factor * k * (self.particle.roughness_amplitude_nm/1000), 
                sigma=3
            )
            total_phase = total_phase + phase_roughness
        
        # Champ diffusé complet
        E_s = scattering_strength * psf_complex * np.exp(1j * total_phase)
        
        # Facteur d'échelle pour contraste visible (ajustement pour visualisation)
        # En réalité le contraste serait de ~0.1-1% pour des NP de 80nm
        visibility_factor = 50.0  # Amplifie pour visualisation
        E_s = E_s * visibility_factor
        
        # =====================================================================
        # 7. INTERFÉRENCE : I = |Ei + Es|²
        # =====================================================================
        E_total = E_i + E_s
        I_total = np.abs(E_total)**2
        
        # Décomposition des termes
        I_i = np.abs(E_i)**2           # Fond (|Ei|²) - terme dominant
        I_s = np.abs(E_s)**2           # Intensité diffusée seule (|Es|²) - très faible
        I_interference = 2 * np.real(np.conj(E_i) * E_s)  # Terme croisé - AMPLIFICATION HOMODYNE
        
        # Vérification : I_total ≈ I_i + I_interference (car I_s << I_i)
        
        # =====================================================================
        # 8. CONVERSION EN IMAGE CAPTEUR
        # =====================================================================
        # Normalisation : le fond (sans particule) = background_level
        I_background = np.mean(I_i)
        I_normalized = I_total / I_background
        
        # Image en niveaux de gris
        I_image = I_normalized * background_level
        
        # Bruit du capteur CMOS (shot noise + read noise)
        np.random.seed(None)
        shot_noise = np.sqrt(I_image) * np.random.randn(N, N) * 0.02
        read_noise = np.random.randn(N, N) * noise_level * background_level
        I_noisy = I_image + shot_noise + read_noise
        
        # Clamp aux valeurs valides [0, 255]
        I_noisy = np.clip(I_noisy, 0, 255)
        
        # =====================================================================
        # STOCKAGE DES DONNÉES D'INTERFÉROMÉTRIE
        # =====================================================================
        contrast = np.abs(I_interference).max() / I_background
        
        self._interferometry_data = {
            'E_i': E_i,              # Champ incident
            'E_s': E_s,              # Champ diffusé
            'I_i': I_i,              # Intensité incidente (fond)
            'I_s': I_s,              # Intensité diffusée
            'I_interference': I_interference,  # Terme d'interférence (signal)
            'I_total': I_total,      # Intensité totale
            'I_ref': I_i,            # Alias pour compatibilité
            'I_scat': I_s,           # Alias pour compatibilité
            'contrast': contrast,    # Contraste de la particule
            'gouy_phase': gouy_phase,
            'polarizability': polarizability
        }
        
        self._camera_image = I_noisy
        return self._camera_image
    
    def get_surface_mesh(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Retourne le maillage 3D de la particule."""
        return self.geometry.generate_surface_mesh()


# =============================================================================
# ANALYSE SPECTRALE (FFT)
# =============================================================================

def compute_spectral_analysis(image: np.ndarray, fov_um: float, subtract_background: bool = True) -> dict:
    """
    Calcule l'analyse spectrale 2D (FFT) de l'image.
    
    Améliorations :
    - Soustraction du fond (DC) pour voir les hautes fréquences
    - Fenêtrage de Hann pour éviter les artefacts de bord
    - Normalisation appropriée
    
    Args:
        image: Image 2D
        fov_um: Champ de vue en microns
        subtract_background: Si True, soustrait le fond moyen (recommandé)
    
    Returns:
        Dict avec PSD 2D, profil radial, fréquences, etc.
    """
    N = image.shape[0]
    pixel_size = fov_um / N
    
    # =========================================================================
    # PRÉTRAITEMENT
    # =========================================================================
    
    # Soustraction du fond (enlève le pic DC)
    if subtract_background:
        image_centered = image - np.mean(image)
    else:
        image_centered = image.copy()
    
    # Fenêtrage 2D de Hann (réduit les artefacts de bord)
    hann_1d = np.hanning(N)
    hann_2d = np.outer(hann_1d, hann_1d)
    image_windowed = image_centered * hann_2d
    
    # =========================================================================
    # FFT 2D
    # =========================================================================
    
    # Fréquences spatiales
    freq = np.fft.fftfreq(N, d=pixel_size)
    freq_shift = np.fft.fftshift(freq)
    FX, FY = np.meshgrid(freq_shift, freq_shift, indexing='ij')
    
    # FFT 2D avec shift pour centrer le DC
    fft_img = np.fft.fftshift(np.fft.fft2(image_windowed))
    
    # Power Spectral Density
    psd = np.abs(fft_img)**2
    
    # Normalisation : par rapport à l'énergie totale (pas le max)
    psd_norm = psd / (np.sum(psd) + 1e-20) * N * N
    
    # =========================================================================
    # PROFIL RADIAL (moyenne azimutale)
    # =========================================================================
    
    center = N // 2
    y_idx, x_idx = np.indices(psd.shape)
    r = np.sqrt((x_idx - center)**2 + (y_idx - center)**2)
    r_int = r.astype(int)
    r_max = min(center, N - center)
    
    # Moyenne azimutale plus robuste
    radial_profile = np.zeros(r_max)
    radial_std = np.zeros(r_max)
    
    for i in range(r_max):
        mask = (r_int == i)
        if np.any(mask):
            values = psd_norm[mask]
            radial_profile[i] = np.mean(values)
            radial_std[i] = np.std(values)
    
    # Fréquence radiale correspondante
    freq_step = freq_shift[1] - freq_shift[0] if len(freq_shift) > 1 else 1.0
    freq_radial = np.arange(r_max) * freq_step
    
    # =========================================================================
    # MÉTRIQUES SPECTRALES
    # =========================================================================
    
    # Fréquence médiane (où se trouve 50% de l'énergie)
    cumsum = np.cumsum(radial_profile)
    if cumsum[-1] > 0:
        freq_median_idx = np.searchsorted(cumsum, cumsum[-1] / 2)
        freq_median = freq_radial[min(freq_median_idx, len(freq_radial)-1)]
    else:
        freq_median = 0
    
    # Largeur spectrale (écart-type pondéré)
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
# COMPARAISON LISSE vs RUGUEUSE
# =============================================================================

def run_comparison(
    radius_nm: float = 80.0,
    roughness_nm: float = 5.0,
    roughness_lmax: int = 10,
    refractive_index: float = 1.50,
    wavelength_nm: float = 488.0,
    numerical_aperture: float = 0.3,
    image_pixels: int = 64,
    save_path: Optional[str] = None
):
    """
    Lance une comparaison entre particule lisse et rugueuse.
    
    Génère une figure avec 2 colonnes (lisse / rugueuse) et 3 lignes :
    1. Représentation 3D de la particule
    2. Image Videodrop (noir et blanc)
    3. Analyse spectrale (FFT)
    """
    print("=" * 70)
    print("SIMULATION VIDEODROP : COMPARAISON LISSE vs RUGUEUSE")
    print("=" * 70)
    
    illumination = IlluminationParameters(wavelength_nm=wavelength_nm)
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
    sim_smooth = VideodropSimulation(particle_smooth, illumination, imaging)
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
    sim_rough = VideodropSimulation(particle_rough, illumination, imaging)
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
    
    fig = plt.figure(figsize=(12, 18))
    
    # Paramètres communs
    extent_img = [-fov/2, fov/2, -fov/2, fov/2]
    max_r = particle_smooth.radius_um * 1.3
    
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
    # LIGNE 2 : IMAGES VIDEODROP (NOIR ET BLANC) - RENDU RÉALISTE
    # =====================================================================
    
    img_smooth = sim_smooth._camera_image
    img_rough = sim_rough._camera_image
    
    # Affichage avec contraste ajusté pour voir les détails
    # On centre sur le fond et on étend pour voir la particule
    all_pixels = np.concatenate([img_smooth.ravel(), img_rough.ravel()])
    vmin = np.min(all_pixels) - 5
    vmax = np.max(all_pixels) + 5
    
    # Colonne 1 : Image lisse
    ax3 = fig.add_subplot(4, 2, 3)
    im3 = ax3.imshow(img_smooth.T, origin='lower', extent=extent_img,
                     cmap='gray', aspect='equal', vmin=vmin, vmax=vmax)
    ax3.set_xlabel('X [μm]')
    ax3.set_ylabel('Z [μm]')
    ax3.set_title('Image Videodrop - LISSE', fontweight='bold')
    plt.colorbar(im3, ax=ax3, label='Intensité', shrink=0.8)
    
    # Colonne 2 : Image rugueuse
    ax4 = fig.add_subplot(4, 2, 4)
    im4 = ax4.imshow(img_rough.T, origin='lower', extent=extent_img,
                     cmap='gray', aspect='equal', vmin=vmin, vmax=vmax)
    ax4.set_xlabel('X [μm]')
    ax4.set_ylabel('Z [μm]')
    ax4.set_title('Image Videodrop - RUGUEUSE', fontweight='bold')
    plt.colorbar(im4, ax=ax4, label='Intensité', shrink=0.8)
    
    # =====================================================================
    # LIGNE 3 : ANALYSE SPECTRALE (FFT)
    # =====================================================================
    
    # Limites dynamiques pour l'affichage du spectre
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
    
    # Plot des profils radiaux en échelle log avec bandes d'erreur
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
    
    # Ligne de coupure NA
    ax7.axvline(x=f_cutoff, color='gray', linestyle='--', linewidth=1.5, 
                label=f'Coupure NA ({f_cutoff:.2f} cyc/μm)')
    
    # Fréquences moyennes
    ax7.axvline(x=spectral_smooth['freq_mean'], color='blue', linestyle=':', linewidth=1.5, alpha=0.7)
    ax7.axvline(x=spectral_rough['freq_mean'], color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    
    ax7.set_xlabel('Fréquence spatiale [cycles/μm]', fontsize=11)
    ax7.set_ylabel('PSD (normalisée)', fontsize=11)
    ax7.set_title('Comparaison des Spectres : Profil Radial (Moyenne Azimutale)\n'
                  f'Largeur spectrale: Lisse={spectral_smooth["freq_width"]:.2f}, Rugueuse={spectral_rough["freq_width"]:.2f} cyc/μm', 
                  fontweight='bold', fontsize=11)
    ax7.legend(loc='upper right', fontsize=10)
    ax7.grid(True, alpha=0.3)
    ax7.set_xlim([0, min(freq_s[-1], f_cutoff * 2)])
    
    # Limites Y dynamiques
    all_radial = np.concatenate([radial_s, radial_r])
    ymin = max(np.min(all_radial[all_radial > 0]) * 0.1, 1e-8)
    ymax = np.max(all_radial) * 2
    ax7.set_ylim([ymin, ymax])
    
    # =====================================================================
    # TITRE GLOBAL
    # =====================================================================
    
    plt.suptitle(
        f'Simulation Videodrop : Comparaison Lisse vs Rugueuse\n'
        f'R={radius_nm}nm, n={refractive_index}, λ={wavelength_nm}nm, NA={numerical_aperture}',
        fontsize=14, fontweight='bold', y=1.02
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure sauvegardée: {save_path}")
    
    plt.show()
    
    # =================================================================
    # STATISTIQUES
    # =================================================================
    print("\n" + "=" * 70)
    print("STATISTIQUES")
    print("=" * 70)
    
    contrast_smooth = sim_smooth._interferometry_data.get('contrast', 0)
    contrast_rough = sim_rough._interferometry_data.get('contrast', 0)
    
    print(f"""
PARTICULE LISSE:
  • Contraste optique: {contrast_smooth*100:.1f}%
  • Niveau moyen image: {np.mean(img_smooth):.1f} (sur 255)
  • Min/Max pixels: {np.min(img_smooth):.1f} / {np.max(img_smooth):.1f}
  • Fréquence spectrale moyenne: {spectral_smooth['freq_mean']:.3f} cyc/μm
  • Largeur spectrale: {spectral_smooth['freq_width']:.3f} cyc/μm

PARTICULE RUGUEUSE:
  • Contraste optique: {contrast_rough*100:.1f}%
  • Niveau moyen image: {np.mean(img_rough):.1f} (sur 255)
  • Min/Max pixels: {np.min(img_rough):.1f} / {np.max(img_rough):.1f}
  • Fréquence spectrale moyenne: {spectral_rough['freq_mean']:.3f} cyc/μm
  • Largeur spectrale: {spectral_rough['freq_width']:.3f} cyc/μm

COMPARAISON SPECTRALE:
  • Ratio fréquence moyenne (rugueux/lisse): {spectral_rough['freq_mean']/max(spectral_smooth['freq_mean'], 1e-10):.2f}
  • Ratio largeur spectrale (rugueux/lisse): {spectral_rough['freq_width']/max(spectral_smooth['freq_width'], 1e-10):.2f}
  • Fréquence de coupure NA: {f_cutoff:.2f} cyc/μm
""")
    
    return sim_smooth, sim_rough


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Utilise les hyperparamètres définis en début de fichier
    run_comparison(
        radius_nm=HYPERPARAMS['radius_nm'],
        roughness_nm=HYPERPARAMS['roughness_nm'],
        roughness_lmax=HYPERPARAMS['roughness_lmax'],
        refractive_index=HYPERPARAMS['refractive_index'],
        wavelength_nm=HYPERPARAMS['wavelength_nm'],
        numerical_aperture=HYPERPARAMS['numerical_aperture'],
        image_pixels=HYPERPARAMS['image_pixels'],
        save_path=HYPERPARAMS['save_path']
    )
