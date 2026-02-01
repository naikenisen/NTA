import bpy, random

obj = bpy.data.objects["particle"]

for frame in range(1, 500):
    bpy.context.scene.frame_set(frame)
    obj.location.x += random.gauss(0, 0.01)
    obj.location.y += random.gauss(0, 0.01)
    obj.keyframe_insert(data_path="location")
