import bpy
import mathutils
import bpy_extras
from mathutils import *
import math
import numpy as np
import csv
import os
import shutil

def add_lamp(lamp_location,num):
		# deselect all
	bpy.ops.object.select_all(action='DESELECT')

	try:
			# selection
		bpy.data.objects['Lamp_' +num].select = True

		# remove it
		bpy.ops.object.delete()
	except:
		print('Lamp not added yet')
	scene = bpy.context.scene

	# Create new lamp datablock
	lamp_data = bpy.data.lamps.new(name='Lamp_' +num, type='POINT')

	# Create new object with our lamp datablock
	lamp_object = bpy.data.objects.new(name='Lamp_' +num, object_data=lamp_data)

	# Link lamp object to the scene so it'll appear in this scene
	scene.objects.link(lamp_object)

	# Place lamp to a specified location
	lamp_object.location = lamp_location

	# And finally select it make active
	lamp_object.select = True
	scene.objects.active = lamp_object
	
def seed_position(CamerPoint,Origin):
    
    cam = bpy.data.objects['Camera']
    random_focal = np.random.randint(10)-5
    cam.data.sensor_width=25+random_focal
    cam.data.sensor_height=25+random_focal
    #cam.data.angle = 30*(3.1415/180.0)
    cam.location.x = CamerPoint[0]
    cam.location.y = CamerPoint[1]
    cam.location.z = CamerPoint[2]
    obj_lamp = bpy.data.objects["Lamp"]
    obj_lamp.location = cam.location
    obj_lamp.location.x = -obj_lamp.location.x
    print(obj_lamp.location)
    loc_camera = cam.location
    add_lamp(loc_camera,'1')
    add_lamp(-loc_camera,'2')
    print(loc_camera)
    vec = Vector((Origin[0], Origin[1], Origin[2]))
    direction = vec - loc_camera
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot_quat.to_euler()
    #bpy.ops.object.camera_add(location = CamerPoint,rotation = rot_quat.to_euler())
    #cam.rotation_euler[2] = 1
   #look_at(cam, obj_other.matrix_world.to_translation())

def look_at(obj_camera, point):
    loc_camera = obj_camera.matrix_world.to_translation()
    direction = point - loc_camera
    # point the cameras '-Z' and use its 'Y' as up
    rot_quat = direction.to_track_quat('-Z', 'Y')
    # assume we're using euler rotation
    obj_camera.rotation_euler = rot_quat.to_euler()
    
def rotateCameraAroundOrigin(rx=0):
    cam = bpy.data.objects['Camera']
    old_x = cam.location.x
    old_y = cam.location.y
    old_z = cam.location.z
    radius = math.sqrt((math.pow(old_x,2) + math.pow(old_y,2)))
    current_angle = math.degrees(math.atan2( old_y, old_x))
    print("CUR:\t%+04d degrees" % (current_angle))
    new_angle = current_angle + rx
    print("FROM:\t%+04d, %+04d, %+04d" % (old_x,old_y,old_z))
    new_x = radius * math.cos(math.radians(new_angle))
    new_y = radius * math.sin(math.radians(new_angle))
    cam.location.x = new_x
    cam.location.y = new_y
    cam.location.z = old_z
    # Set camera rotation in euler angles
    obj_lamp = bpy.data.objects["Lamp"]
    obj_lamp.location = cam.location
 #   obj_other = bpy.data.objects["Cube"]
    loc_camera = cam.location
    vec = Vector((0.0, 0.0, 0.0))
    direction = vec - loc_camera
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot_quat.to_euler()
    #cam.rotation_mode = 'XYZ'
    #cam.rotation_euler[2] = cam.rotation_euler[2] + math.radians(rx)
 #   moveCamTo(new_x,new_y,old_z)
    print("TO:\t%+04d, %+04d, %+04d\n" % (new_x,new_y,old_z))
 
def Clear_Blender():
	candidate_list = [item.name for item in bpy.data.objects if item.type == "MESH"]
	# select them only.
	for object_name in candidate_list:
		bpy.data.objects[object_name].select = True
		bpy.ops.object.delete()
	for item in bpy.data.meshes:
		bpy.data.meshes.remove(item)

def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        ((1, 0,  0),
         (0, -1, 0),
         (0, 0, -1)))

    # Transpose since the rotation is object rotation, 
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam * location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*cam.location
    # Use location from matrix_world to account for constraints:     
    T_world2bcam = -1*R_world2bcam * location

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = R_bcam2cv*R_world2bcam
    T_world2cv = R_bcam2cv*T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],)
         ))
    return RT

def get_calibration_matrix_K_from_blender(camd):
    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if (camd.sensor_fit == 'VERTICAL'):
        # the sensor height is fixed (sensor fit is horizontal), 
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio 
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else: # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal), 
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm


    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0 # only use rectangular pixels

    K = Matrix(
        ((alpha_u, skew,    u_0),
        (    0  , alpha_v, v_0),
        (    0  , 0,        1 )))
    return K


def get_3x4_P_matrix_from_blender(cam):
    K = get_calibration_matrix_K_from_blender(cam.data)
    RT = get_3x4_RT_matrix_from_blender(cam)
    return K,RT


def project_by_object_utils(cam, point):
    scene = bpy.context.scene
    co_2d = bpy_extras.object_utils.world_to_camera_view(scene, cam, point)
    render_scale = scene.render.resolution_percentage / 100
    render_size = (
            int(scene.render.resolution_x * render_scale),
            int(scene.render.resolution_y * render_scale),
            )
    return Vector((co_2d.x * render_size[0], render_size[1] - co_2d.y * render_size[1]))

def ensure_dir(f):
    #d = os.path.dirname(f)
    print(f)
    if not os.path.exists(f):
        os.makedirs(f)

def makeMaterial(name, diffuse, specular, alpha):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = diffuse
    mat.diffuse_shader = 'LAMBERT' 
    mat.diffuse_intensity = 1.0 
    mat.specular_color = specular
    mat.specular_shader = 'COOKTORR'
    mat.specular_intensity = 0.5
    mat.alpha = alpha
    mat.ambient = 1
    return mat

def createMaterial(imagePath,opengl):    
    # Create image texture from image. Change here if the snippet 
    # folder is not located in you home directory.
    realpath = os.path.expanduser(imagePath)
    
    tex = bpy.data.textures.new('ColorTex', type = 'IMAGE')
    tex.image = bpy.data.images.load(realpath)
    tex.use_alpha = True
 
    # Create shadeless material and MTex
    mat = bpy.data.materials.new('TexMat')
    mat.use_shadeless = True
    mtex = mat.texture_slots.add()
    mtex.texture = tex
    mtex.texture_coords = 'UV'
    if opengl == True:
        mtex.texture_coords = 'ORCO'
    else:
        mtex.texture_coords = 'UV'
    
    mtex.use_map_color_diffuse = True 
    mtex.diffuse_color_factor = 1.0
    return mat
 
def setMaterial(ob, mat):
    me = ob.data
    me.materials.append(mat)
    ob.select = False
 

