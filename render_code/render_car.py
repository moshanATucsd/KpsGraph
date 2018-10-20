
import glob

from optparse import OptionParser
from mathutils import Vector
import bpy_extras
from random import randint
import glob
import random
import math 
import sys
sys.path.append('.')
from CameraParameters import *
from icosahedron import icosahedron_level
from bpy_extras.object_utils import world_to_camera_view
import OpenEXR
import Imath
from PIL import Image, ImageDraw
import array

opengl = False
limit = 0.001

def second_largest(numbers):
    count = 0
    m1 = m2 = float('-inf')
    for x in numbers:
        count += 1
        if x > m2:
            if x >= m1:
                m1, m2 = x, m1            
            else:
                m2 = x
    return m2 if count >= 2 else None
    
def ProcessEXR(path):
	file = OpenEXR.InputFile(path)

	# Compute the size
	dw = file.header()['dataWindow']
	sz = ((dw.max.x - dw.min.x + 1)*1, (dw.max.y - dw.min.y + 1)*1)

	# Read the three color channels as 32-bit floats
	FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
	#print(Chan)
	(R,G,B,Z) = [array.array('f', file.channel(Chan, FLOAT)).tolist() for Chan in ("Image.R", "Image.G", "Image.B","Depth.V") ]
	print(np.size(Z))
    
	# Normalize so that brightest sample is 1
	Z =np.array(Z)
	Z[Z > 10000] =0
	farthest = max(Z)
	Z_image = [[ i / farthest for i in Z ]]


	depthArray = np.zeros((sz[0],sz[1],3), 'uint8')
	depthArray[..., 0] =  np.reshape(Z_image, (sz[0],sz[1]))*256
	depthArray[..., 1] = np.reshape(Z_image, (sz[0],sz[1]))*256
	depthArray[..., 2] = np.reshape(Z_image, (sz[0],sz[1]))*256



	img_depth = Image.fromarray(depthArray)
	img_depth.save(path.replace('Raw','Depth').replace('exr','png'))
	return((depthArray/256)*farthest,path.replace('Raw','Depth').replace('exr','png'))


def Render_Camera_positions(CameraPosition,Folder_path,vec):

	f=open(Folder_path + '/Extrinsics/frame_par.txt','w')
	f.write(str(len(CameraPosition))+'\n')
	bpy.context.scene.render.use_compositing = True
	bpy.context.scene.use_nodes = True
	scene_tree = bpy.context.scene.node_tree
	renderlayers_node = scene_tree.nodes.new('CompositorNodeRLayers')
	outputfile_node = scene_tree.nodes.new('CompositorNodeOutputFile')
	outputfile_node.format.file_format = 'OPEN_EXR_MULTILAYER'  # 'OPEN_EXR'

	for idx,key in enumerate(CameraPosition):
	
			#print(key[1],key[2],key[0])
			#print(vertice(1))
			cam = bpy.data.objects['Camera']
			seed_position(key,vec) 
		
			for area in bpy.context.screen.areas:
				if area.type == 'VIEW_3D':
					area.spaces[0].region_3d.view_perspective = 'CAMERA'
					for space in area.spaces:
						if space.type == 'VIEW_3D':
							space.viewport_shade = 'MATERIAL'
							
			outputfile_node.base_path = Folder_path
			outputfile_node.file_slots.new('Z')
			scene_tree.links.new(renderlayers_node.outputs['Image'], outputfile_node.inputs['Image'])
			#scene_tree.links.new(renderlayers_node.outputs['Z'], outputfile_node.inputs['Z'])
			bpy.data.scenes['Scene'].render.filepath = Folder_path + "/images/" + str(idx).zfill(5) +'.png'
			#bpy.context.user_preferences.system.compute_device_type = 'CUDA'
			bpy.context.scene.render.resolution_x = 512*4 #perhaps set resolution in code
			bpy.context.scene.render.resolution_y = 512*4
			if opengl==True:
				bpy.ops.render.opengl(write_still=True)
			else:
				bpy.ops.render.render(write_still=True)
				
			shutil.move(Folder_path + "/0001.exr", Folder_path + "/Raw/" + str(idx).zfill(5) +'.exr')
			#asas
			
			cam = bpy.data.objects['Camera']
			K, RT = get_3x4_P_matrix_from_blender(cam)
			K_reshape=np.around(np.reshape(K, (1, 9)), decimals=6)
			RT_reshape=np.around(np.reshape(RT, (1, 12)), decimals=6)
			f.write(str(idx).zfill(5) + '.png ' + str(K_reshape[0][0])+ ' ' + str(K_reshape[0][1])  +  ' ' + str(K_reshape[0][2])  + ' ' + str(K_reshape[0][3])  +' ' + str(K_reshape[0][4])  +' ' + str(K_reshape[0][5])  +' ' + str(K_reshape[0][6])  +' ' + str(K_reshape[0][7])  + ' ' + str(K_reshape[0][8])  
					+ ' ' + str(RT_reshape[0][0])  +' ' + str(RT_reshape[0][1])  + ' ' + str(RT_reshape[0][2])  + ' ' + str(RT_reshape[0][4])  +' ' + str(RT_reshape[0][5])  +' ' + str(RT_reshape[0][6])  +' ' + str(RT_reshape[0][8])  +' ' + str(RT_reshape[0][9])  +' ' + str(RT_reshape[0][10])  +' ' + str(RT_reshape[0][3])  +' ' + str(RT_reshape[0][7])  +' ' + str(RT_reshape[0][11])  +'\n')
	
	f.close()

def Render_scene_icosahedron(level, Distance, Folder_path,RandomPoint):
		vertice = icosahedron_level(level)
		#ensure_dir(Folder_path + "/Depth")
		ensure_dir(Folder_path + "/Extrinsics")
		ensure_dir(Folder_path + "/Image")
		#ensure_dir(Folder_path + "/Flow")
		ensure_dir(Folder_path + "/Raw")
		CameraPosition = []
		for idx,key in enumerate(vertice):
			if key[2] > 0:
					Value = []
					Value.append(key[0]*Distance + RandomPoint[0])
					Value.append(key[1]*Distance + RandomPoint[1])
					Value.append(key[2]*Distance + RandomPoint[2])
					CameraPosition.append(Value)
		
		Render_Camera_positions(CameraPosition,Folder_path,RandomPoint)
		
# Create a BVH tree and return bvh and vertices in world coordinates 
def BVHTreeAndVerticesInWorldFromObj( obj ):
    mWorld = obj.matrix_world
    vertsInWorld = [mWorld * v.co for v in obj.data.vertices]

    bvh = BVHTree.FromPolygons( vertsInWorld, [p.vertices for p in obj.data.polygons] )

    return bvh, vertsInWorld
    	
# Deselect mesh polygons and vertices
def DeselectEdgesAndPolygons( obj ):
    for p in obj.data.polygons:
        p.select = False
    for e in obj.data.edges:
        e.select = False
            		
def visibity(anchor_points,K,RT,depth_map,path):
			scene = bpy.context.scene
			image = Image.open(path.replace('Depth','images'))
			#image=image.rotate(90) 
			draw = ImageDraw.Draw(image)
			scene = bpy.context.scene
			cam = bpy.data.objects['Camera']
			point_save = np.zeros((len(anchor_points),5))
			print(point_save)
			for a,b in enumerate(anchor_points):
				mat_trns = [[1,0,0],[0,0,-1],[0,1,0]]
				b = np.matmul(mat_trns,b)

				point_3d = np.matmul(RT,np.append(b,1))
				point_2d = np.matmul(np.array(K),point_3d[0:3])
				point_2d = point_2d/point_2d[2]
				point_save[a,0]=point_2d[0]
				point_save[a,1]=point_2d[1]
				point_save[a,2]=a
				point_save[a,3]=0
				#print(depth_map[int(point_2d[1]),int(point_2d[0]),1],point_3d)
				try:
					depth = depth_map[int(point_2d[1]),int(point_2d[0]),1]
				except:
					continue
				if np.abs(depth_map[int(point_2d[1]),int(point_2d[0]),1]-point_3d[2])<0.01:
					print(np.abs(depth_map[int(point_2d[1]),int(point_2d[0]),1]-point_3d[2]))
					draw.ellipse((int(point_2d[0])-5, int(point_2d[1])-5,int(point_2d[0])+5, int(point_2d[1])+5), fill =(255,0,0,255))
					point_save[a,4]=1
				else:
					draw.ellipse((int(point_2d[0])-5, int(point_2d[1])-5,int(point_2d[0])+5, int(point_2d[1])+5), fill =(0,255,0,255))
					point_save[a,4]=2
					
				#point_2d[2] =  visibilty
			image.save(path.replace('Depth','Projections'))
			np.savetxt(path.replace('Depth','gt').replace('.png','.txt'),point_save, delimiter=',', fmt='%d')
			#asas
			#print(point_2d)
			
			return(point_save)
	
def Render_Camera_positions_keypoints(CameraPosition,Folder_path,vec,anchor_points):

	f=open(Folder_path + '/Extrinsics/frame_par.txt','w')
	f.write(str(len(CameraPosition))+'\n')
	bpy.context.scene.render.use_compositing = True
	bpy.context.scene.use_nodes = True
	scene_tree = bpy.context.scene.node_tree
	renderlayers_node = scene_tree.nodes.new('CompositorNodeRLayers')
	outputfile_node = scene_tree.nodes.new('CompositorNodeOutputFile')
	outputfile_node.format.file_format = 'OPEN_EXR_MULTILAYER'  # 'OPEN_EXR'

	for idx,key in enumerate(CameraPosition):
		
			#if idx<30:
				#asas
			#	continue#cam = bpy.data.objects['Camera']
	
			#print(key[1],key[2],key[0])
			#print(vertice(1))
			cam = bpy.data.objects['Camera']
			seed_position(key,vec) 
		
			for area in bpy.context.screen.areas:
				if area.type == 'VIEW_3D':
					area.spaces[0].region_3d.view_perspective = 'CAMERA'
					for space in area.spaces:
						if space.type == 'VIEW_3D':
							space.viewport_shade = 'MATERIAL'
							
			outputfile_node.base_path = Folder_path
			outputfile_node.file_slots.new('Depth')
			scene_tree.links.new(renderlayers_node.outputs['Image'], outputfile_node.inputs['Image'])
			scene_tree.links.new(renderlayers_node.outputs['Depth'], outputfile_node.inputs['Depth'])
			bpy.data.scenes['Scene'].render.filepath = Folder_path + "/images/" + str(idx).zfill(5) +'.png'
			#bpy.context.user_preferences.system.compute_device_type = 'CUDA'
			bpy.context.scene.render.resolution_x = 512*4 #perhaps set resolution in code
			bpy.context.scene.render.resolution_y = 512*4
			if opengl==True:
				bpy.ops.render.opengl(write_still=True)
			else:
				bpy.ops.render.render(write_still=True)
				
			shutil.move(Folder_path + "/0001.exr", Folder_path + "/Raw/" + str(idx).zfill(5) +'.exr')
			depth_map,path=ProcessEXR( Folder_path + "/Raw/" + str(idx).zfill(5) +'.exr')
			
			cam = bpy.data.objects['Camera']
			K, RT = get_3x4_P_matrix_from_blender(cam)
			visibity(anchor_points,K, RT,depth_map,path)
			K_reshape=np.around(np.reshape(K, (1, 9)), decimals=6)
			RT_reshape=np.around(np.reshape(RT, (1, 12)), decimals=6)
			f.write(str(idx).zfill(5) + '.png ' + str(K_reshape[0][0])+ ' ' + str(K_reshape[0][1])  +  ' ' + str(K_reshape[0][2])  + ' ' + str(K_reshape[0][3])  +' ' + str(K_reshape[0][4])  +' ' + str(K_reshape[0][5])  +' ' + str(K_reshape[0][6])  +' ' + str(K_reshape[0][7])  + ' ' + str(K_reshape[0][8])  
					+ ' ' + str(RT_reshape[0][0])  +' ' + str(RT_reshape[0][1])  + ' ' + str(RT_reshape[0][2])  + ' ' + str(RT_reshape[0][4])  +' ' + str(RT_reshape[0][5])  +' ' + str(RT_reshape[0][6])  +' ' + str(RT_reshape[0][8])  +' ' + str(RT_reshape[0][9])  +' ' + str(RT_reshape[0][10])  +' ' + str(RT_reshape[0][3])  +' ' + str(RT_reshape[0][7])  +' ' + str(RT_reshape[0][11])  +'\n')
	
	f.close()


def Render_scene_icosahedron_keypoints(level, Dist, Folder_path,RandomPoint,anchor_points):
		vertice = icosahedron_level(level)
		ensure_dir(Folder_path + "/Depth")
		ensure_dir(Folder_path + "/Extrinsics")
		ensure_dir(Folder_path + "/images")
		ensure_dir(Folder_path + "/Projections")
		ensure_dir(Folder_path + "/Raw")
		ensure_dir(Folder_path + "/gt")
		CameraPosition = []
		for idx,key in enumerate(vertice):
			if key[2] > 0:
					Value = []
					Distance = Dist + np.random.randint(100, size=1)[0]/100
					Value.append(key[0]*Distance + RandomPoint[0])
					Value.append(key[1]*Distance + RandomPoint[1])
					Value.append(key[2]*Distance + RandomPoint[2])
					CameraPosition.append(Value)
		
		Render_Camera_positions_keypoints(CameraPosition,Folder_path,RandomPoint,anchor_points)
		
def ScaleWorld(MaxSize):
	maxValue = 0
	scene = bpy.context.scene
	for ob in scene.objects:
		centre = sum((Vector(b) for b in ob.bound_box), Vector())
		centre /= 8
		check = ob.dimensions + centre
		if check[0] > maxValue:
			maxValue = check[0]
		if check[1] > maxValue:
			maxValue = check[1]
		if check[2] > maxValue:
			maxValue = check[2]
				
	scale = MaxSize/maxValue
	for ob in scene.objects:
		ob.select = True; 
		
	bpy.ops.transform.resize(value=(scale, scale, scale), constraint_axis=(False, False, False),
									constraint_orientation='GLOBAL', mirror=False, proportional='DISABLED', 
									proportional_edit_falloff='SMOOTH', proportional_size=1)	#save scene to file 
	for ob in scene.objects:
		#ob.location = (-5,-5,-5)
		ob.select = False; 


def LoadData(full_path_to_file,rand1):
		image_list = glob.glob('/is/ps2/dreddy/Project/SyntheticCube/train2014/COCO_train2014_00000000*.jpg')
		red = createMaterial(image_list[rand1],opengl)#makeMaterial('Red', (1,0,0), (1,1,1), 1)
		red1 = createMaterial(image_list[rand1+1],opengl)
		bpy.ops.mesh.primitive_cube_add(location=(0,0,0),rotation = (math.pi/2,0,0))
		ob = bpy.context.object
		setMaterial(ob, red)
		bpy.ops.mesh.primitive_cube_add(location=(random.randint(200, 300)/100,random.randint(200, 300)/100,random.randint(200, 300)/100))
		ob = bpy.context.object
		setMaterial(ob, red1)
		
	

def main(num):
	
	level=5
	Folder = "/home/dinesh/Research/CADs/car/"
	Output = "/home/dinesh/Research/car_render/"
	list_folders = os.listdir(Folder)
	for index,name in enumerate(list_folders):
		#if index <342:
		#   continue
		print(name)
		#asas
		Folder_save = Output+ str(index) +'/'
		Folder_path = Folder + name 
		print(Folder_path)
		#asas 
		full_path_to_file = Folder_path + '/model.obj'
		distance = 1.5
		RandomPoints = 1
		
		if not os.path.exists(Folder_path):
			os.makedirs(Folder_path)
		
		Clear_Blender()
		
		#LoadData(full_path_to_file,rand1)
		#bpy.ops.import_scene.obj(filepath=full_path_to_file)
		#ScaleWorld(10)
		

		#Clear_Blender()
		#bpy.ops.import_scene.obj(filepath = Folder_save + "/world.obj")
	#	for RenderPoints in range(0,RandomPoints):
	#		obj = bpy.context.object #Gets the object
	#		print(obj)
	#		num = randint(0,len(obj.data.vertices)-1)
	#		Point = obj.data.vertices[num].co
		anchor_points=np.loadtxt(Folder_path + '/anchors.txt')
		print(Folder_path)
		#asas
		Point = np.sum(anchor_points,axis=0)/len(anchor_points)
		#asas
		#Point=[-0.427268,-0.0220649,0.0606231]
		#print(Point)
#		Render_scene_icosahedron(level,distance,Folder_save,Point)
		#Render_scene_icosahedron_keypoints(level,distance,Folder_save,Point,anchor_points)
		ensure_dir(Folder_save)
		#bpy.ops.export_scene.obj(filepath = Folder_save + "/world.obj", axis_forward='Y', axis_up='Z')
		np.savetxt(Folder_save + '/anchors.txt',anchor_points,delimiter=',',fmt="%.5f")
		#asas
		#asas
	

	
	
if __name__ == '__main__':
	for num in range(1,2500):
		main(num)

