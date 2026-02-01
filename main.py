import bpy
import random
import math
import mathutils

# ============================================
# SIMULATION NTA - Nanoparticle Tracking Analysis
# Exosomes / Vésicules Extracellulaires
# to train : blender --background --python main.py
# to view ; blender NTA_exosome_simulation.blend
# ============================================

# --- Paramètres de simulation ---
NUM_PARTICLES = 80          # Nombre de particules individuelles (exosomes)
NUM_GRANULAR_PARTICLES = 20 # Nombre de particules granuleuses
TOTAL_FRAMES = 500          # Durée de l'animation

# Tailles des exosomes (en unités Blender, représentant ~100-200nm)
MIN_EXOSOME_SIZE = 0.10     # ~100nm
MAX_EXOSOME_SIZE = 0.20     # ~200nm

# Zone de simulation (volume d'observation NTA)
VOLUME_X = 8.0
VOLUME_Y = 8.0
VOLUME_Z = 10.0  # Profondeur de champ

# Paramètres physiques
TEMPERATURE = 298           # Kelvin (25°C)
VISCOSITY = 0.001           # Pa.s (eau)
BOLTZMANN = 1.38e-23        # Constante de Boltzmann

# Profondeur de champ (plan focal à z=0)
FOCAL_PLANE_Z = 0.0
DOF_RANGE = 0.8             # Zone nette autour du plan focal


def clear_scene():
    """Nettoie la scène Blender"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    # Nettoie les matériaux orphelins
    for mat in bpy.data.materials:
        if mat.users == 0:
            bpy.data.materials.remove(mat)


def calculate_diffusion_coefficient(radius_nm):
    """
    Calcule le coefficient de diffusion selon Stokes-Einstein
    D = kT / (6πηr)
    """
    radius_m = radius_nm * 1e-9
    D = BOLTZMANN * TEMPERATURE / (6 * math.pi * VISCOSITY * radius_m)
    # Conversion en unités Blender (facteur d'échelle)
    return D * 1e10


def calculate_rayleigh_intensity(radius):
    """
    Intensité de diffusion Rayleigh ∝ r^6
    Normalisé pour des valeurs d'émission raisonnables
    """
    # Normalisation par rapport à la taille max
    normalized = radius / MAX_EXOSOME_SIZE
    intensity = (normalized ** 6) * 50  # Facteur d'échelle pour visibilité
    return max(intensity, 0.5)  # Minimum pour que les petites soient visibles


def calculate_dof_opacity(z_position):
    """
    Calcule l'opacité/visibilité basée sur la profondeur de champ
    Les particules hors du plan focal sont plus transparentes
    """
    distance_from_focal = abs(z_position - FOCAL_PLANE_Z)
    
    if distance_from_focal < DOF_RANGE:
        # Dans la zone nette
        opacity = 1.0
    else:
        # Atténuation progressive hors zone nette
        opacity = max(0.1, 1.0 - (distance_from_focal - DOF_RANGE) * 0.8)
    
    return opacity


def create_emission_material(name, radius, base_color=(0.4, 0.8, 1.0)):
    """
    Crée un matériau émissif pour simuler la diffusion lumineuse
    L'intensité dépend de la taille (Rayleigh)
    """
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    
    # Noeud Emission
    emission = nodes.new('ShaderNodeEmission')
    emission.inputs['Color'].default_value = (*base_color, 1.0)
    emission.inputs['Strength'].default_value = calculate_rayleigh_intensity(radius)
    
    # Noeud de sortie
    output = nodes.new('ShaderNodeOutputMaterial')
    
    # Connexion
    mat.node_tree.links.new(emission.outputs['Emission'], output.inputs['Surface'])
    
    return mat


def create_granular_material(name, radius, base_color=(1.0, 0.6, 0.2)):
    """
    Crée un matériau émissif granuleux pour simuler des particules avec texture
    Utilise un shader de bruit pour créer l'effet de granularité
    """
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    # Coordonnées de texture
    tex_coord = nodes.new('ShaderNodeTexCoord')
    tex_coord.location = (-600, 0)
    
    # Noeud de bruit pour la granularité
    noise = nodes.new('ShaderNodeTexNoise')
    noise.inputs['Scale'].default_value = random.uniform(15.0, 40.0)  # Granularité variable
    noise.inputs['Detail'].default_value = random.uniform(8.0, 16.0)
    noise.inputs['Roughness'].default_value = random.uniform(0.6, 0.9)
    noise.location = (-400, 0)
    
    # Color Ramp pour contrôler le contraste de la granularité
    color_ramp = nodes.new('ShaderNodeValToRGB')
    color_ramp.color_ramp.elements[0].position = 0.3
    color_ramp.color_ramp.elements[0].color = (*[c * 0.5 for c in base_color], 1.0)
    color_ramp.color_ramp.elements[1].position = 0.7
    color_ramp.color_ramp.elements[1].color = (*base_color, 1.0)
    color_ramp.location = (-200, 0)
    
    # Noeud Emission
    emission = nodes.new('ShaderNodeEmission')
    emission.inputs['Strength'].default_value = calculate_rayleigh_intensity(radius) * 1.2
    emission.location = (0, 0)
    
    # Noeud de sortie
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (200, 0)
    
    # Connexions
    links.new(tex_coord.outputs['Object'], noise.inputs['Vector'])
    links.new(noise.outputs['Fac'], color_ramp.inputs['Fac'])
    links.new(color_ramp.outputs['Color'], emission.inputs['Color'])
    links.new(emission.outputs['Emission'], output.inputs['Surface'])
    
    return mat


def create_particle(name, radius, location):
    """
    Crée une particule (sphère) avec les propriétés appropriées
    """
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=radius,
        segments=16,
        ring_count=8,
        location=location
    )
    obj = bpy.context.active_object
    obj.name = name
    
    color = (0.3, 0.7, 1.0)  # Bleu pour exosomes
    
    # Applique le matériau
    mat = create_emission_material(f"mat_{name}", radius, color)
    obj.data.materials.append(mat)
    
    return obj


def create_granular_particle(name, radius, location):
    """
    Crée une particule granuleuse avec une texture de surface variable
    et un vrai relief 3D (displacement)
    """
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=radius,
        segments=32,  # Plus de segments pour le displacement
        ring_count=24,
        location=location
    )
    obj = bpy.context.active_object
    obj.name = name
    
    # --- Ajout du relief 3D avec displacement ---
    # Subdiviser pour avoir plus de vertices à déplacer
    subsurf = obj.modifiers.new(name="Subsurf", type='SUBSURF')
    subsurf.levels = 2
    subsurf.render_levels = 2
    
    # Créer une texture de bruit pour le displacement
    displace = obj.modifiers.new(name="Displace", type='DISPLACE')
    
    # Créer une nouvelle texture
    tex_name = f"tex_granular_{name}"
    tex = bpy.data.textures.new(tex_name, type='CLOUDS')
    tex.noise_scale = random.uniform(0.3, 0.8)
    tex.noise_depth = random.randint(2, 6)
    tex.noise_type = 'HARD_NOISE'
    
    displace.texture = tex
    displace.texture_coords = 'LOCAL'
    displace.strength = radius * random.uniform(0.15, 0.35)  # Force proportionnelle à la taille
    displace.mid_level = 0.5
    
    # Appliquer les modificateurs pour figer la géométrie
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier="Subsurf")
    bpy.ops.object.modifier_apply(modifier="Displace")
    
    # Lisser légèrement les normales
    bpy.ops.object.shade_smooth()
    
    # Couleur orangée/dorée pour les particules granuleuses
    color = (
        random.uniform(0.8, 1.0),   # Rouge
        random.uniform(0.4, 0.7),   # Vert
        random.uniform(0.1, 0.3)    # Bleu
    )
    
    # Applique le matériau granuleux
    mat = create_granular_material(f"mat_granular_{name}", radius, color)
    obj.data.materials.append(mat)
    
    return obj


def brownian_step(radius, dt=1.0):
    """
    Génère un pas de mouvement brownien en 3D
    Les petites particules bougent plus vite (∝ 1/r)
    """
    # Amplitude de base inversement proportionnelle à la taille
    base_amplitude = 0.03  # Amplitude visible
    size_factor = (MAX_EXOSOME_SIZE / radius) ** 0.5  # Petites = plus rapides
    
    sigma = base_amplitude * size_factor
    
    dx = random.gauss(0, sigma)
    dy = random.gauss(0, sigma)
    dz = random.gauss(0, sigma * 0.3)  # Moins de mouvement en Z
    return dx, dy, dz


def update_material_for_dof(obj, z_position):
    """
    Met à jour l'intensité du matériau selon la profondeur de champ
    """
    if obj.data.materials:
        mat = obj.data.materials[0]
        if mat.use_nodes:
            for node in mat.node_tree.nodes:
                if node.type == 'EMISSION':
                    base_intensity = calculate_rayleigh_intensity(obj.dimensions[0] / 2)
                    dof_factor = calculate_dof_opacity(z_position)
                    node.inputs['Strength'].default_value = base_intensity * dof_factor


def setup_camera_and_lighting():
    """
    Configure la caméra et l'éclairage pour la simulation NTA
    """
    # Caméra
    bpy.ops.object.camera_add(location=(0, 0, 8))
    camera = bpy.context.active_object
    camera.name = "NTA_Camera"
    camera.data.lens = 50
    camera.data.dof.use_dof = True
    camera.data.dof.focus_distance = 8.0
    camera.data.dof.aperture_fstop = 2.8
    bpy.context.scene.camera = camera
    
    # Lumière laser (simule l'illumination NTA)
    bpy.ops.object.light_add(type='AREA', location=(0, 0, 5))
    light = bpy.context.active_object
    light.name = "Laser_Light"
    light.data.energy = 100
    light.data.color = (0.4, 0.6, 1.0)
    light.data.size = 10
    
    # Fond noir
    bpy.context.scene.world.use_nodes = True
    bg_node = bpy.context.scene.world.node_tree.nodes.get('Background')
    if bg_node:
        bg_node.inputs['Color'].default_value = (0, 0, 0, 1)


# ============================================
# MAIN - Exécution de la simulation
# ============================================

print("=" * 50)
print("SIMULATION NTA - Nanoparticle Tracking Analysis")
print("=" * 50)

# Nettoie la scène
clear_scene()

# Configure la scène
setup_camera_and_lighting()

particles = []

# --- Création des particules individuelles ---
print(f"Création de {NUM_PARTICLES} exosomes individuels...")
for i in range(NUM_PARTICLES):
    # Taille aléatoire selon distribution log-normale (réaliste pour exosomes)
    radius = random.lognormvariate(
        math.log((MIN_EXOSOME_SIZE + MAX_EXOSOME_SIZE) / 2),
        0.4
    )
    radius = max(MIN_EXOSOME_SIZE, min(MAX_EXOSOME_SIZE, radius))
    
    # Position initiale aléatoire dans le volume
    x = random.uniform(-VOLUME_X / 2, VOLUME_X / 2)
    y = random.uniform(-VOLUME_Y / 2, VOLUME_Y / 2)
    z = random.uniform(-VOLUME_Z / 2, VOLUME_Z / 2)
    
    obj = create_particle(f"exosome_{i:03d}", radius, (x, y, z))
    
    particles.append({
        'object': obj,
        'radius': radius
    })

# --- Création des particules granuleuses ---
print(f"Création de {NUM_GRANULAR_PARTICLES} particules granuleuses...")
for i in range(NUM_GRANULAR_PARTICLES):
    # Taille légèrement plus grande pour les particules granuleuses
    radius = random.lognormvariate(
        math.log((MIN_EXOSOME_SIZE + MAX_EXOSOME_SIZE) / 2 * 1.3),
        0.5
    )
    radius = max(MIN_EXOSOME_SIZE * 0.8, min(MAX_EXOSOME_SIZE * 1.5, radius))
    
    # Position initiale aléatoire dans le volume
    x = random.uniform(-VOLUME_X / 2, VOLUME_X / 2)
    y = random.uniform(-VOLUME_Y / 2, VOLUME_Y / 2)
    z = random.uniform(-VOLUME_Z / 2, VOLUME_Z / 2)
    
    obj = create_granular_particle(f"granular_{i:03d}", radius, (x, y, z))
    
    particles.append({
        'object': obj,
        'radius': radius
    })

# --- Animation du mouvement brownien ---
print(f"Animation sur {TOTAL_FRAMES} frames...")
for frame in range(1, TOTAL_FRAMES + 1):
    if frame % 100 == 0:
        print(f"  Frame {frame}/{TOTAL_FRAMES}")
    
    bpy.context.scene.frame_set(frame)
    
    for p in particles:
        obj = p['object']
        
        # Calcul du pas brownien (basé sur la taille)
        dx, dy, dz = brownian_step(p['radius'])
        
        # Mise à jour de la position
        new_x = obj.location.x + dx
        new_y = obj.location.y + dy
        new_z = obj.location.z + dz
        
        # Limites du volume (rebond élastique)
        if abs(new_x) > VOLUME_X / 2:
            new_x = obj.location.x - dx
        if abs(new_y) > VOLUME_Y / 2:
            new_y = obj.location.y - dy
        if abs(new_z) > VOLUME_Z / 2:
            new_z = obj.location.z - dz
        
        obj.location = (new_x, new_y, new_z)
        obj.keyframe_insert(data_path="location")
        
        # Mise à jour de l'intensité selon la profondeur de champ
        if frame == 1 or frame % 10 == 0:
            update_material_for_dof(obj, new_z)
            # Keyframe pour l'intensité du matériau
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
output_file = "NTA_exosome_simulation.blend"
bpy.ops.wm.save_as_mainfile(filepath=output_file)
print(f"Fichier sauvegardé: {output_file}")
print("=" * 50)
