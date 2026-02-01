import bpy
import os

# ============================================
# SCRIPT DE RENDU VIDÉO H.264
# Lance le rendu de la simulation NTA
# ============================================

# Fichier source
BLEND_FILE = "NTA_realistic_simulation.blend"
OUTPUT_VIDEO = "//NTA_simulation.mp4"

# Paramètres de rendu (optionnel - à ajuster selon besoin)
USE_GPU = True          # Utiliser le GPU si disponible
SAMPLES = 64            # Nombre de samples Cycles
RESOLUTION_PERCENT = 100  # 100% = résolution native (640x480)

def setup_video_output():
    """Configure la sortie vidéo en H.264"""
    scene = bpy.context.scene
    
    # Format de sortie: FFmpeg vidéo
    scene.render.image_settings.file_format = 'FFMPEG'
    
    # Conteneur MP4
    scene.render.ffmpeg.format = 'MPEG4'
    
    # Codec H.264
    scene.render.ffmpeg.codec = 'H264'
    
    # Qualité (bitrate constant)
    scene.render.ffmpeg.constant_rate_factor = 'MEDIUM'  # Bon équilibre qualité/taille
    
    # Bitrate vidéo (kbit/s) - optionnel si on utilise CRF
    scene.render.ffmpeg.video_bitrate = 6000
    
    # GOP size (intervalle entre keyframes)
    scene.render.ffmpeg.gopsize = 15
    
    # Pas d'audio
    scene.render.ffmpeg.audio_codec = 'NONE'
    
    # Chemin de sortie
    scene.render.filepath = OUTPUT_VIDEO
    
    print("Configuration vidéo H.264:")
    print(f"  - Format: MP4 / H.264")
    print(f"  - Qualité: MEDIUM (CRF)")
    print(f"  - Bitrate: 6000 kbit/s")
    print(f"  - Sortie: {OUTPUT_VIDEO}")


def setup_render_device():
    """Configure le device de rendu (GPU si disponible)"""
    scene = bpy.context.scene
    
    # Désactiver le denoiser (non supporté sur certaines builds)
    scene.cycles.use_denoising = False
    print("  Denoiser: désactivé")
    
    if USE_GPU:
        # Essayer d'activer le GPU
        cycles_prefs = bpy.context.preferences.addons.get('cycles')
        if cycles_prefs:
            prefs = cycles_prefs.preferences
            
            # Détecter les devices disponibles
            prefs.get_devices()
            
            # Chercher CUDA, OptiX, ou HIP
            for compute_type in ['OPTIX', 'CUDA', 'HIP', 'ONEAPI', 'METAL']:
                try:
                    prefs.compute_device_type = compute_type
                    # Activer tous les GPU disponibles
                    for device in prefs.devices:
                        if device.type != 'CPU':
                            device.use = True
                            print(f"  GPU activé: {device.name}")
                    scene.cycles.device = 'GPU'
                    print(f"  Type de calcul: {compute_type}")
                    return
                except:
                    continue
    
    # Fallback CPU
    scene.cycles.device = 'CPU'
    print("  Rendu sur CPU")


def render_animation():
    """Lance le rendu de l'animation"""
    scene = bpy.context.scene
    
    print("\n" + "=" * 60)
    print("DÉMARRAGE DU RENDU")
    print("=" * 60)
    print(f"Frames: {scene.frame_start} à {scene.frame_end}")
    print(f"Résolution: {scene.render.resolution_x}x{scene.render.resolution_y}")
    print(f"FPS: {scene.render.fps}")
    print(f"Samples: {scene.cycles.samples}")
    print("=" * 60 + "\n")
    
    # Lancer le rendu
    bpy.ops.render.render(animation=True)
    
    print("\n" + "=" * 60)
    print("RENDU TERMINÉ!")
    print(f"Vidéo sauvegardée: {bpy.path.abspath(OUTPUT_VIDEO)}")
    print("=" * 60)


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("RENDU VIDÉO H.264 - Simulation NTA")
    print("=" * 60)
    
    # Vérifier que le fichier .blend existe
    script_dir = os.path.dirname(os.path.abspath(__file__))
    blend_path = os.path.join(script_dir, BLEND_FILE)
    
    if not os.path.exists(blend_path):
        print(f"ERREUR: Fichier '{BLEND_FILE}' non trouvé!")
        print("Exécutez d'abord: blender --background --python main.py")
        exit(1)
    
    # Ouvrir le fichier .blend
    print(f"\nOuverture de {BLEND_FILE}...")
    bpy.ops.wm.open_mainfile(filepath=blend_path)
    
    # Configuration
    print("\nConfiguration du rendu...")
    setup_render_device()
    
    # Ajuster les samples si spécifié
    bpy.context.scene.cycles.samples = SAMPLES
    bpy.context.scene.render.resolution_percentage = RESOLUTION_PERCENT
    
    # Configurer la sortie H.264
    setup_video_output()
    
    # Lancer le rendu
    render_animation()
