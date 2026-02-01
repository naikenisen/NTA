import bpy
import random
import math
import mathutils

# ============================================
# SIMULATION NTA RÉALISTE - Pour Deep Learning
# Reproduction fidèle du rendu NanoSight NS300
# ============================================

# --- Paramètres de simulation ---
NUM_PARTICLES = 100         # Nombre d'EVs individuelles
NUM_AGGREGATES = 12         # Nombre d'agrégats
TOTAL_FRAMES = 500          # Durée de l'animation

# Tailles des exosomes (en unités Blender, représentant ~30-150nm)
MIN_EXOSOME_SIZE = 0.02     # ~30nm - à peine visible
MAX_EXOSOME_SIZE = 0.12     # ~150nm
AGGREGATE_SIZE_FACTOR = 3.0 # Les agrégats sont bien plus gros

# Zone de simulation (volume d'observation NTA)
VOLUME_X = 3.0
VOLUME_Y = 2.0
VOLUME_Z = 1.5  # Profondeur de champ réduite

# Profondeur de champ (plan focal à z=0) - TRÈS AGRESSIVE
FOCAL_PLANE_Z = 0.0
DOF_SHARP_RANGE = 0.15      # Zone vraiment nette (très mince)
DOF_VISIBLE_RANGE = 0.6     # Zone où on voit encore quelque chose

# Paramètres visuels NTA
SENSOR_NOISE_INTENSITY = 0.15   # Bruit de capteur
LASER_WAVELENGTH_NM = 488       # Laser bleu typique NTA
AIRY_DISK_BASE_SIZE = 0.08      # Taille de base du disque d'Airy


def clear_scene():
    """Nettoie la scène Blender"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    for mat in bpy.data.materials:
        if mat.users == 0:
            bpy.data.materials.remove(mat)
    
    for img in bpy.data.images:
        if img.users == 0:
            bpy.data.images.remove(img)


def calculate_airy_size(particle_radius, z_position):
    """
    Calcule la taille du disque d'Airy en fonction de:
    - La taille de la particule (diffraction)
    - La position Z (défocalisation -> élargissement)
    """
    # Taille de base proportionnelle à la racine de la taille
    base_size = AIRY_DISK_BASE_SIZE * (1 + particle_radius / MAX_EXOSOME_SIZE)
    
    # Défocalisation: élargissement rapide hors du plan focal
    z_distance = abs(z_position - FOCAL_PLANE_Z)
    
    if z_distance < DOF_SHARP_RANGE:
        defocus_factor = 1.0
    elif z_distance < DOF_VISIBLE_RANGE:
        # Élargissement progressif -> effet "donut"
        defocus_factor = 1.0 + ((z_distance - DOF_SHARP_RANGE) / DOF_SHARP_RANGE) ** 1.5 * 3
    else:
        defocus_factor = 5.0  # Très flou, presque invisible
    
    return base_size * defocus_factor


def calculate_intensity(particle_radius, z_position, is_aggregate=False):
    """
    Calcule l'intensité de la particule (Rayleigh + défocalisation + saturation)
    """
    # Intensité Rayleigh de base (∝ r^6)
    normalized = particle_radius / MAX_EXOSOME_SIZE
    rayleigh_intensity = normalized ** 6
    
    # Les agrégats saturent le capteur
    if is_aggregate:
        rayleigh_intensity = min(rayleigh_intensity * 8, 1.0)  # Saturation
        base_emission = 80  # Très brillant
    else:
        base_emission = 25  # EVs plus subtiles
    
    # Atténuation par défocalisation
    z_distance = abs(z_position - FOCAL_PLANE_Z)
    
    if z_distance < DOF_SHARP_RANGE:
        dof_factor = 1.0
    elif z_distance < DOF_VISIBLE_RANGE:
        dof_factor = max(0.05, 1.0 - (z_distance - DOF_SHARP_RANGE) / DOF_VISIBLE_RANGE * 1.5)
    else:
        dof_factor = 0.02  # Presque invisible
    
    # Scintillement aléatoire (flicker)
    flicker = random.uniform(0.85, 1.15)
    
    return base_emission * rayleigh_intensity * dof_factor * flicker


def create_airy_disk_material(name, particle_radius, z_position, is_aggregate=False):
    """
    Crée un matériau simulant un disque d'Airy (pattern de diffraction)
    avec halo central et anneaux pour les grosses particules
    """
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    mat.blend_method = 'BLEND'
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    # Coordonnées de texture
    tex_coord = nodes.new('ShaderNodeTexCoord')
    tex_coord.location = (-800, 0)
    
    # Mapping centré
    mapping = nodes.new('ShaderNodeMapping')
    mapping.location = (-600, 0)
    mapping.inputs['Location'].default_value = (0.5, 0.5, 0)
    links.new(tex_coord.outputs['UV'], mapping.inputs['Vector'])
    
    # Gradient radial pour le profil d'Airy
    gradient = nodes.new('ShaderNodeTexGradient')
    gradient.gradient_type = 'SPHERICAL'
    gradient.location = (-400, 0)
    links.new(mapping.outputs['Vector'], gradient.inputs['Vector'])
    
    # Profil gaussien central (pic d'Airy)
    math_pow = nodes.new('ShaderNodeMath')
    math_pow.operation = 'POWER'
    math_pow.location = (-200, 100)
    math_pow.inputs[1].default_value = 2.0
    links.new(gradient.outputs['Fac'], math_pow.inputs[0])
    
    # Inversion pour avoir le max au centre
    math_sub = nodes.new('ShaderNodeMath')
    math_sub.operation = 'SUBTRACT'
    math_sub.location = (-200, -50)
    math_sub.inputs[0].default_value = 1.0
    links.new(math_pow.outputs['Value'], math_sub.inputs[1])
    
    # Renforcement du pic central (plus pointu)
    math_pow2 = nodes.new('ShaderNodeMath')
    math_pow2.operation = 'POWER'
    math_pow2.location = (0, 100)
    
    # Les petites particules ont un profil plus étalé
    if particle_radius < MAX_EXOSOME_SIZE * 0.5:
        math_pow2.inputs[1].default_value = 1.5  # Plus flou
    else:
        math_pow2.inputs[1].default_value = 3.0  # Plus net
    
    links.new(math_sub.outputs['Value'], math_pow2.inputs[0])
    
    # Pour les agrégats: ajouter des anneaux de diffraction
    if is_aggregate:
        # Onde sinusoïdale pour les anneaux
        math_sin = nodes.new('ShaderNodeMath')
        math_sin.operation = 'SINE'
        math_sin.location = (-200, -200)
        
        # Multiplier le gradient pour créer plusieurs anneaux
        math_mult_rings = nodes.new('ShaderNodeMath')
        math_mult_rings.operation = 'MULTIPLY'
        math_mult_rings.location = (-350, -200)
        math_mult_rings.inputs[1].default_value = 15.0  # Fréquence des anneaux
        links.new(gradient.outputs['Fac'], math_mult_rings.inputs[0])
        links.new(math_mult_rings.outputs['Value'], math_sin.inputs[0])
        
        # Atténuer les anneaux avec la distance
        math_mult_atten = nodes.new('ShaderNodeMath')
        math_mult_atten.operation = 'MULTIPLY'
        math_mult_atten.location = (0, -200)
        links.new(math_sin.outputs['Value'], math_mult_atten.inputs[0])
        links.new(math_sub.outputs['Value'], math_mult_atten.inputs[1])
        
        # Combiner pic central + anneaux
        math_add = nodes.new('ShaderNodeMath')
        math_add.operation = 'ADD'
        math_add.location = (200, 0)
        links.new(math_pow2.outputs['Value'], math_add.inputs[0])
        math_add.inputs[1].default_value = 0.0
        links.new(math_mult_atten.outputs['Value'], math_add.inputs[1])
        
        intensity_output = math_add
    else:
        intensity_output = math_pow2
    
    # Couleur de la particule
    if is_aggregate:
        # Agrégats: saturation vers le blanc (halo)
        color_ramp = nodes.new('ShaderNodeValToRGB')
        color_ramp.location = (200, 200)
        color_ramp.color_ramp.elements[0].position = 0.0
        color_ramp.color_ramp.elements[0].color = (0.1, 0.2, 0.3, 1)  # Bleu sombre
        color_ramp.color_ramp.elements[1].position = 0.5
        color_ramp.color_ramp.elements[1].color = (0.8, 0.9, 1.0, 1)  # Blanc-bleu
        # Ajouter un point pour la saturation
        color_ramp.color_ramp.elements.new(0.8)
        color_ramp.color_ramp.elements[2].color = (1.0, 1.0, 1.0, 1)  # Blanc pur (saturation)
        links.new(intensity_output.outputs['Value'], color_ramp.inputs['Fac'])
        color_output = color_ramp.outputs['Color']
    else:
        # EVs: couleur bleu-cyan typique NTA
        color_ramp = nodes.new('ShaderNodeValToRGB')
        color_ramp.location = (200, 200)
        color_ramp.color_ramp.elements[0].position = 0.0
        color_ramp.color_ramp.elements[0].color = (0.0, 0.1, 0.2, 1)
        color_ramp.color_ramp.elements[1].position = 1.0
        color_ramp.color_ramp.elements[1].color = (0.3, 0.7, 1.0, 1)  # Cyan
        links.new(intensity_output.outputs['Value'], color_ramp.inputs['Fac'])
        color_output = color_ramp.outputs['Color']
    
    # Émission
    emission = nodes.new('ShaderNodeEmission')
    emission.location = (400, 100)
    emission.inputs['Strength'].default_value = calculate_intensity(particle_radius, z_position, is_aggregate)
    links.new(color_output, emission.inputs['Color'])
    
    # Transparence basée sur le profil
    transparent = nodes.new('ShaderNodeBsdfTransparent')
    transparent.location = (400, -100)
    
    # Mix shader
    mix = nodes.new('ShaderNodeMixShader')
    mix.location = (600, 0)
    links.new(intensity_output.outputs['Value'], mix.inputs['Fac'])
    links.new(transparent.outputs['BSDF'], mix.inputs[1])
    links.new(emission.outputs['Emission'], mix.inputs[2])
    
    # Output
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (800, 0)
    links.new(mix.outputs['Shader'], output.inputs['Surface'])
    
    return mat


def create_ev_particle(name, radius, location):
    """
    Crée une EV comme un disque orienté vers la caméra (billboard)
    avec un matériau de disque d'Airy
    """
    # Taille visuelle du disque d'Airy
    airy_size = calculate_airy_size(radius, location[2])
    
    # Créer un cercle (disque) au lieu d'un plan carré
    bpy.ops.mesh.primitive_circle_add(radius=airy_size/2, fill_type='NGON', location=location)
    obj = bpy.context.active_object
    obj.name = name
    
    # Le plan fait toujours face à la caméra (constraint)
    constraint = obj.constraints.new('TRACK_TO')
    constraint.target = bpy.data.objects.get("NTA_Camera")
    constraint.track_axis = 'TRACK_Z'
    constraint.up_axis = 'UP_Y'
    
    # Matériau
    mat = create_airy_disk_material(f"mat_{name}", radius, location[2], is_aggregate=False)
    obj.data.materials.append(mat)
    
    # Stocker le rayon réel comme propriété
    obj["particle_radius"] = radius
    obj["is_aggregate"] = False
    
    return obj


def create_aggregate(name, radius, location):
    """
    Crée un agrégat avec forme irrégulière et motifs de diffraction complexes
    """
    airy_size = calculate_airy_size(radius, location[2]) * 1.5  # Plus gros visuellement
    
    # Créer un cercle (disque) au lieu d'un plan carré
    bpy.ops.mesh.primitive_circle_add(radius=airy_size/2, fill_type='NGON', location=location)
    obj = bpy.context.active_object
    obj.name = name
    
    # Déformer légèrement le disque pour forme irrégulière
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.transform.resize(value=(
        random.uniform(0.85, 1.15),
        random.uniform(0.85, 1.15),
        1.0
    ))
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Track caméra
    constraint = obj.constraints.new('TRACK_TO')
    constraint.target = bpy.data.objects.get("NTA_Camera")
    constraint.track_axis = 'TRACK_Z'
    constraint.up_axis = 'UP_Y'
    
    # Matériau avec diffraction complexe
    mat = create_airy_disk_material(f"mat_{name}", radius, location[2], is_aggregate=True)
    obj.data.materials.append(mat)
    
    obj["particle_radius"] = radius
    obj["is_aggregate"] = True
    
    return obj


def brownian_step(radius, is_aggregate=False):
    """
    Génère un pas de mouvement brownien en 3D
    Les petites particules bougent plus vite (∝ 1/√r)
    """
    base_amplitude = 0.025
    
    if is_aggregate:
        # Agrégats bougent beaucoup moins
        size_factor = 0.3
    else:
        # Petites EVs bougent plus
        size_factor = (MAX_EXOSOME_SIZE / radius) ** 0.5
    
    sigma = base_amplitude * size_factor
    
    dx = random.gauss(0, sigma)
    dy = random.gauss(0, sigma)
    dz = random.gauss(0, sigma * 0.5)  # Mouvement Z pour entrer/sortir du focus
    
    return dx, dy, dz


def update_particle_appearance(obj, new_z):
    """
    Met à jour la taille et l'intensité du disque d'Airy selon la position Z
    """
    radius = obj.get("particle_radius", MAX_EXOSOME_SIZE)
    is_aggregate = obj.get("is_aggregate", False)
    
    # Nouvelle taille du disque d'Airy (élargissement hors focus)
    new_airy_size = calculate_airy_size(radius, new_z)
    if is_aggregate:
        new_airy_size *= 1.5
    
    # Mettre à jour l'échelle du plan
    base_size = AIRY_DISK_BASE_SIZE * (1 + radius / MAX_EXOSOME_SIZE)
    if is_aggregate:
        base_size *= 1.5
    scale_factor = new_airy_size / base_size
    obj.scale = (scale_factor, scale_factor, 1.0)
    
    # Mettre à jour l'intensité du matériau
    if obj.data.materials:
        mat = obj.data.materials[0]
        if mat.use_nodes:
            for node in mat.node_tree.nodes:
                if node.type == 'EMISSION':
                    new_intensity = calculate_intensity(radius, new_z, is_aggregate)
                    node.inputs['Strength'].default_value = new_intensity


def setup_camera_and_lighting():
    """
    Configure la caméra et l'éclairage pour le look NTA réaliste
    """
    # Caméra orthographique pour look microscopie
    bpy.ops.object.camera_add(location=(0, 0, 5))
    camera = bpy.context.active_object
    camera.name = "NTA_Camera"
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = 4.0
    
    # DOF très agressive
    camera.data.dof.use_dof = True
    camera.data.dof.focus_distance = 5.0
    camera.data.dof.aperture_fstop = 1.4  # Très ouvert = DOF très faible
    
    bpy.context.scene.camera = camera
    
    # Pas de lumière externe - les particules sont auto-émissives (diffusion laser)
    
    # Fond noir
    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get('Background')
    if bg_node:
        bg_node.inputs['Color'].default_value = (0, 0, 0, 1)
        bg_node.inputs['Strength'].default_value = 0


def setup_compositing():
    """
    Configure le compositing pour les effets post-process:
    - Bruit de capteur
    - Bloom/Glare
    - Motion blur
    """
    scene = bpy.context.scene
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    
    # === Render Layers ===
    render_layers = tree.nodes.new('CompositorNodeRLayers')
    render_layers.location = (0, 400)
    
    # === Glare (bloom pour les particules brillantes) ===
    glare = tree.nodes.new('CompositorNodeGlare')
    glare.location = (200, 400)
    glare.glare_type = 'FOG_GLOW'
    glare.quality = 'HIGH'
    glare.threshold = 0.3
    glare.size = 7
    
    tree.links.new(render_layers.outputs['Image'], glare.inputs['Image'])
    
    # === Glare secondaire pour halo des agrégats ===
    glare2 = tree.nodes.new('CompositorNodeGlare')
    glare2.location = (400, 400)
    glare2.glare_type = 'BLOOM'
    glare2.quality = 'MEDIUM'
    glare2.threshold = 0.8  # Seulement les très brillants (agrégats saturés)
    glare2.size = 5
    
    tree.links.new(glare.outputs['Image'], glare2.inputs['Image'])
    
    # === Bruit de capteur (via noise shader en compositing) ===
    # Utiliser le noeud de bruit intégré au compositing
    
    # Coordonnées pour le bruit
    # On va créer un bruit procédural directement
    
    # Mix pour ajouter du bruit
    mix_noise = tree.nodes.new('CompositorNodeMixRGB')
    mix_noise.location = (600, 400)
    mix_noise.blend_type = 'ADD'
    mix_noise.inputs['Fac'].default_value = SENSOR_NOISE_INTENSITY
    
    # Le bruit sera ajouté via un léger offset RGB aléatoire par pixel
    # On utilise le noeud "Brightness/Contrast" avec des valeurs pour simuler le grain
    bright_contrast = tree.nodes.new('CompositorNodeBrightContrast')
    bright_contrast.location = (600, 200)
    bright_contrast.inputs['Bright'].default_value = 0.02
    bright_contrast.inputs['Contrast'].default_value = 1.15  # Augmente le contraste
    
    tree.links.new(glare2.outputs['Image'], bright_contrast.inputs['Image'])
    
    # === Légère désaturation (capteurs monochromes) ===
    hue_sat = tree.nodes.new('CompositorNodeHueSat')
    hue_sat.location = (800, 300)
    hue_sat.inputs['Saturation'].default_value = 0.7  # Légèrement désaturé
    hue_sat.inputs['Value'].default_value = 1.1  # Légèrement plus lumineux
    
    tree.links.new(bright_contrast.outputs['Image'], hue_sat.inputs['Image'])
    
    # === Contraste et niveaux ===
    curves = tree.nodes.new('CompositorNodeCurveRGB')
    curves.location = (1000, 300)
    # Augmenter légèrement le contraste
    curves.mapping.curves[3].points[0].location = (0.05, 0)  # Noir plus profond
    curves.mapping.curves[3].points[1].location = (0.95, 1)  # Blanc légèrement clippé
    curves.mapping.update()
    
    tree.links.new(hue_sat.outputs['Image'], curves.inputs['Image'])
    
    # === Output ===
    composite = tree.nodes.new('CompositorNodeComposite')
    composite.location = (1200, 300)
    
    tree.links.new(curves.outputs['Image'], composite.inputs['Image'])
    
    # Viewer pour prévisualisation
    viewer = tree.nodes.new('CompositorNodeViewer')
    viewer.location = (1200, 100)
    tree.links.new(curves.outputs['Image'], viewer.inputs['Image'])


def setup_render_settings():
    """
    Configure les paramètres de rendu pour NTA réaliste
    """
    scene = bpy.context.scene
    
    # Utiliser Cycles pour meilleur motion blur et DOF
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'  # ou 'GPU' si disponible
    scene.cycles.samples = 64    # Suffisant avec denoiser
    scene.cycles.use_denoising = True
    
    # Motion blur
    scene.render.use_motion_blur = True
    scene.render.motion_blur_shutter = 0.3  # Effet subtil mais visible
    
    # Résolution typique NTA
    scene.render.resolution_x = 640   # Résolution typique caméra NTA
    scene.render.resolution_y = 480
    scene.render.fps = 30
    scene.frame_start = 1
    scene.frame_end = TOTAL_FRAMES
    
    # Format de sortie
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = "//frames/"
    
    # Transparence pour composition
    scene.render.film_transparent = False  # Fond noir opaque


# ============================================
# MAIN - Exécution de la simulation
# ============================================

print("=" * 60)
print("SIMULATION NTA RÉALISTE - Deep Learning Training Data")
print("=" * 60)

# Seed pour reproductibilité (commenter pour variation)
# random.seed(42)

# Nettoie la scène
clear_scene()

# Configure la scène
setup_camera_and_lighting()
setup_render_settings()
setup_compositing()

particles = []

# --- Création des EVs individuelles ---
print(f"Création de {NUM_PARTICLES} vésicules extracellulaires...")
for i in range(NUM_PARTICLES):
    # Distribution log-normale réaliste pour les exosomes
    radius = random.lognormvariate(
        math.log(MIN_EXOSOME_SIZE * 2),  # Médiane
        0.5  # Dispersion
    )
    radius = max(MIN_EXOSOME_SIZE, min(MAX_EXOSOME_SIZE, radius))
    
    # Position initiale
    x = random.uniform(-VOLUME_X / 2, VOLUME_X / 2)
    y = random.uniform(-VOLUME_Y / 2, VOLUME_Y / 2)
    z = random.uniform(-VOLUME_Z / 2, VOLUME_Z / 2)
    
    obj = create_ev_particle(f"EV_{i:03d}", radius, (x, y, z))
    
    particles.append({
        'object': obj,
        'radius': radius,
        'is_aggregate': False
    })

# --- Création des agrégats ---
print(f"Création de {NUM_AGGREGATES} agrégats...")
for i in range(NUM_AGGREGATES):
    # Agrégats beaucoup plus gros
    radius = random.uniform(
        MAX_EXOSOME_SIZE * 1.5,
        MAX_EXOSOME_SIZE * AGGREGATE_SIZE_FACTOR
    )
    
    x = random.uniform(-VOLUME_X / 2, VOLUME_X / 2)
    y = random.uniform(-VOLUME_Y / 2, VOLUME_Y / 2)
    z = random.uniform(-VOLUME_Z / 2, VOLUME_Z / 2)
    
    obj = create_aggregate(f"Aggregate_{i:03d}", radius, (x, y, z))
    
    particles.append({
        'object': obj,
        'radius': radius,
        'is_aggregate': True
    })

# --- Animation du mouvement brownien ---
print(f"Animation sur {TOTAL_FRAMES} frames...")
for frame in range(1, TOTAL_FRAMES + 1):
    if frame % 50 == 0:
        print(f"  Frame {frame}/{TOTAL_FRAMES}")
    
    bpy.context.scene.frame_set(frame)
    
    for p in particles:
        obj = p['object']
        
        # Pas brownien
        dx, dy, dz = brownian_step(p['radius'], p['is_aggregate'])
        
        # Nouvelle position
        new_x = obj.location.x + dx
        new_y = obj.location.y + dy
        new_z = obj.location.z + dz
        
        # Limites du volume avec rebond doux
        if abs(new_x) > VOLUME_X / 2:
            new_x = obj.location.x - dx * 0.5
        if abs(new_y) > VOLUME_Y / 2:
            new_y = obj.location.y - dy * 0.5
        if abs(new_z) > VOLUME_Z / 2:
            new_z = obj.location.z - dz * 0.5
        
        obj.location = (new_x, new_y, new_z)
        obj.keyframe_insert(data_path="location")
        
        # Mise à jour apparence (taille/intensité selon Z)
        update_particle_appearance(obj, new_z)
        obj.keyframe_insert(data_path="scale")
        
        # Keyframe intensité matériau
        if obj.data.materials:
            mat = obj.data.materials[0]
            if mat.use_nodes:
                for node in mat.node_tree.nodes:
                    if node.type == 'EMISSION':
                        node.inputs['Strength'].keyframe_insert(
                            data_path="default_value",
                            frame=frame
                        )

print("Animation terminée!")

# Sauvegarde
output_file = "NTA_realistic_simulation.blend"
bpy.ops.wm.save_as_mainfile(filepath=output_file)
print(f"\nFichier sauvegardé: {output_file}")
print("=" * 60)
print("Pour rendre la vidéo: Render > Render Animation (Ctrl+F12)")
print("Les frames seront sauvées dans le dossier 'frames/'")
print("=" * 60)
