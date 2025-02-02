import numpy as np
import os
from vapory import *
from geometry import normals_surfaces
import nibabel as nib
import trimesh
import slam.io as sio
import meshio
from numba import njit, prange

from geometry import volume_from_coordinates, surface_edge_length
from normalisation import coordinates_denormalisation
from time_management import t_to_GA_XWang

# Writes POV-Ray source files and then output in .png files
def writePov(PATH_DIR, initial_geometry, step, Ut, faces, nodal_idx, nodal_idx_b, n_surface_nodes, zoom, zoom_pos):

  png_name = str(initial_geometry) + "_step%d.png"%(step)
  foldname = "%s/"%(PATH_DIR)

  #Create output folder if not already 
  try:
    if not os.path.exists(foldname):
      os.makedirs(foldname)
  except OSError:
    print ('Error: Creating directory. ' + foldname) 

  # Normals in deformed state
  N_init = np.zeros((n_surface_nodes,3), dtype = float)
  N = normals_surfaces(Ut, faces, nodal_idx_b, len(faces), n_surface_nodes, N_init)

  #os.path.dirname(png_name)

  camera = Camera('location', [-3*zoom, 3*zoom, -3*zoom], 'look_at', [0, 0, 0], 'sky', [0, 0, -1], 'focal_point', [-0.55, 0.55, -0.55], 'aperture', 0.055, 'blur_samples', 10)
  light = LightSource([-14, 3, -14], 'color', [1, 1, 1])
  background = Background('color', [1, 1, 1])

  vertices = np.zeros((n_surface_nodes,3), dtype = float)
  normals = np.zeros((n_surface_nodes,3), dtype = float)
  f_indices = np.zeros((len(faces),3), dtype = int)
  vertices[:,:] = Ut[nodal_idx[:],:]*zoom_pos
  normals[:,:] = N[:,:]
  f_indices[:,0] = nodal_idx_b[faces[:,0]]
  f_indices[:,1] = nodal_idx_b[faces[:,1]]
  f_indices[:,2] = nodal_idx_b[faces[:,2]]

  """vertices = []
  normals = []
  f_indices = []
  for i in range(n_surface_nodes):
    vertex = [Ut[SN[i]][0]*zoom_pos, Ut[SN[i]][1]*zoom_pos, Ut[SN[i]][2]*zoom_pos]
    vertices.append(vertex)
    normal = [N[i][0], N[i][1], N[i][2]]
    normals.append(normal)
  for i in range(len(faces)):
    f_indice = [nodal_idx_b[faces[i][0]], nodal_idx_b[faces[i][1]], nodal_idx_b[faces[i][2]]]
    f_indices.append(f_indice)"""

  Mesh = Mesh2(VertexVectors(n_surface_nodes, *vertices), NormalVectors(n_surface_nodes, *normals), FaceIndices(len(faces), *f_indices), 'inside_vector', [0,1,0])
  box = Box([-100, -100, -100], [100, 100, 100])
  pigment = Pigment('color', [1, 1, 0.5])
  normal = Normal('bumps', 0.05, 'scale', 0.005)
  finish = Finish('phong', 1, 'reflection', 0.05, 'ambient', 0, 'diffuse', 0.9)

  intersection = Intersection(Mesh, box, Texture(pigment, normal, finish))

  scene = Scene(camera, objects=[light, background, intersection], included=["colors.inc"])
  #scene.render(png_name, width=400, height=300, quality = 9, antialiasing = 1e-5 )
  scene.render(png_name, width=800, height=600, quality=9)
  
# Writes POV-Ray source files
def writePov2(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE, step, Ut, faces, nodal_idx, nodal_idx_b, n_surface_nodes, zoom, zoom_pos):

  povname = "B%d.pov"%(step)
  foldname = "%s/pov_H%fAT%f/"%(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE)

  try:
    if not os.path.exists(foldname):
      os.makedirs(foldname)
  except OSError:
    print ('Error: Creating directory. ' + foldname)

  completeName = os.path.join(foldname, povname)
  filepov = open(completeName, "w")

  # Normals in deformed state
  N_init = np.zeros((n_surface_nodes,3), dtype = float)
  N = normals_surfaces(Ut, faces, nodal_idx_b, len(faces), n_surface_nodes, N_init)

  #for i in range(len(faces)):
  #  Ntmp = np.cross(Ut[faces[i][1]] - Ut[faces[i][0]], Ut[faces[i][2]] - Ut[faces[i][0]])
  #  N[nodal_idx_b[faces[i][0]]] += Ntmp
  #  N[nodal_idx_b[faces[i][1]]] += Ntmp
  #  N[nodal_idx_b[faces[i][2]]] += Ntmp
  #for i in range(n_surface_nodes):
  #  N_norm = np.linalg.norm(N[i])
  #  N[i] *= 1.0/N_norm

  filepov.write("#include \"colors.inc\"\n")
  filepov.write("background { color rgb <1,1,1> }\n")
  filepov.write("camera { location <" + str(-3*zoom) + ", " + str(3*zoom) + ", " + str(-3*zoom) + "> look_at <0, 0, 0> sky <0, 0, -1> focal_point <-0.55, 0.55, -0.55> aperture 0.055 blur_samples 10 }\n")
  filepov.write("light_source { <-14, 3, -14> color rgb <1, 1, 1> }\n")

  filepov.write("intersection {\n")
  filepov.write("mesh2 { \n")
  filepov.write("vertex_vectors { " + str(n_surface_nodes) + ",\n")
  for i in range(n_surface_nodes):
    filepov.write("<" + "{0:.5f}".format(Ut[nodal_idx[i]][0]*zoom_pos) + "," + "{0:.5f}".format(Ut[nodal_idx[i]][1]*zoom_pos) + "," + "{0:.5f}".format(Ut[nodal_idx[i]][2]*zoom_pos) + ">,\n")
  filepov.write("} normal_vectors { " + str(n_surface_nodes) + ",\n")
  for i in range(n_surface_nodes):
    filepov.write("<" + "{0:.5f}".format(N[i][0]) + "," + "{0:.5f}".format(N[i][1]) + "," + "{0:.5f}".format(N[i][2]) + ">,\n")
  filepov.write("} face_indices { " + str(len(faces)) + ",\n")
  for i in range(len(faces)):
    filepov.write("<" + str(nodal_idx_b[faces[i][0]]) + "," + str(nodal_idx_b[faces[i][1]]) + "," + str(nodal_idx_b[faces[i][2]]) + ">,\n")
  filepov.write("} inside_vector<0,1,0> }\n")
  filepov.write("box { <-100, -100, -100>, <100, 100, 100> }\n")
  filepov.write("pigment { rgb<1,1,0.5> } normal { bumps 0.05 scale 0.005 } finish { phong 1 reflection 0.05 ambient 0 diffuse 0.9 } }\n")

  filepov.close()

# Write surface mesh in .txt files
def writeTXT(PATH_DIR, initial_geometry, step, Ut, faces, nodal_idx, nodal_idx_b, n_surface_nodes, zoom_pos, center_of_gravity, maxd, miny, halforwholebrain):

  txtname = str(initial_geometry) + "_step%d.txt"%(step)
  foldname = "%s/"%(PATH_DIR)

  try:
    if not os.path.exists(foldname):
      os.makedirs(foldname)
  except OSError:
    print ('Error: Creating directory. ' + foldname)
  
  vertices = np.zeros((n_surface_nodes,3), dtype = float)
  vertices_seg = np.zeros((n_surface_nodes,3), dtype = float)

  # mesh coordinates denormalisation
  vertices[:,:] = Ut[nodal_idx[:],:]*zoom_pos
  vertices_seg[:,1] = center_of_gravity[0] - vertices[:,0]*maxd
  if halforwholebrain.__eq__("half"):
    vertices_seg[:,0] = vertices[:,1]*maxd + miny
  else:
    vertices_seg[:,0] = vertices[:,1]*maxd + center_of_gravity[1]
  vertices_seg[:,2] = center_of_gravity[2] - vertices[:,2]*maxd

  completeName = os.path.join(foldname, txtname)
  filetxt = open(completeName, "w")
  filetxt.write(str(n_surface_nodes) + "\n")
  for i in range(n_surface_nodes):
    filetxt.write(str(vertices_seg[i,0]) + " " + str(vertices_seg[i,1]) + " " + str(vertices_seg[i,2]) + "\n")
  filetxt.write(str(len(faces)) + "\n")
  for i in range(len(faces)):
    filetxt.write(str(nodal_idx_b[faces[i][0]]+1) + " " + str(nodal_idx_b[faces[i][1]]+1) + " " + str(nodal_idx_b[faces[i][2]]+1) + "\n")
  filetxt.close()



# Convert surface mesh structure (from simulations) to .stl format file
def mesh_to_stl_no_denorm(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE, step, Ut, nodal_idx, n_surface_nodes, faces, node_deformation):

  stlname = "B%d_pr.ply"%(step)

  foldname = "%s/pov_H%fAT%f/"%(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE)

  save_path = os.path.join(foldname, stlname)

  # Transform coordinates (because the coordinates are normalized at the beginning)
  vertices = np.zeros((n_surface_nodes,3), dtype = float)

  vertices[:,:] = Ut[nodal_idx[:],:] #= np.max(faces) +1

  # Create the .stl mesh par Trimesh and save it
  mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
  texture = node_deformation.copy()
  texture = texture / np.max(texture) * 255

  for i in range(len(mesh.vertices)):
    mesh.visual.vertex_colors[i] = [texture[i], texture[i], texture[i], 255]
  mesh.export(save_path)

# Convert surface mesh structure (from simulations) to .stl format file
def mesh_to_stl(PATH_DIR, initial_geometry, step, Ut, nodal_idx, zoom_pos, center_of_gravity, maxd, n_surface_nodes, faces, nodal_idx_b, miny, halforwholebrain):

  stlname = str(initial_geometry) + "_step%d.stl"%(step)
  foldname = "%s/"%(PATH_DIR)

  save_path = os.path.join(foldname, stlname)

  # Transform coordinates (because the coordinates are normalized at the beginning)
  vertices = np.zeros((n_surface_nodes,3), dtype = float)
  f_indices = np.zeros((len(faces),3), dtype = int)
  vertices_seg = np.zeros((n_surface_nodes,3), dtype = float)

  # mesh coordinates denormalisation
  vertices[:,:] = Ut[nodal_idx[:],:]*zoom_pos
  vertices_seg[:,1] = center_of_gravity[0] - vertices[:,0]*maxd
  #vertices_seg[:,1] = vertices[:,0]*maxd + center_of_gravity[0]
  if halforwholebrain.__eq__("half"):
    vertices_seg[:,0] = vertices[:,1]*maxd + miny
  else:
    vertices_seg[:,0] = vertices[:,1]*maxd + center_of_gravity[1]
  #vertices_seg[:,0] = center_of_gravity[1] - vertices[:,1]*maxd
  vertices_seg[:,2] = center_of_gravity[2] - vertices[:,2]*maxd
  #vertices_seg[:,2] = vertices[:,2]*maxd + center_of_gravity[2]

  f_indices[:,0] = nodal_idx_b[faces[:,0]] #f_indices = faces
  f_indices[:,1] = nodal_idx_b[faces[:,1]]
  f_indices[:,2] = nodal_idx_b[faces[:,2]]

  # Create the .stl mesh par Trimesh and save it
  mesh = trimesh.Trimesh(vertices=vertices_seg, faces=f_indices, process=False)
  mesh.export(save_path, file_type='stl_ascii')

  """# Create the .stl mesh
  brain = mesh.Mesh(np.zeros(f_indices.shape[0], dtype=mesh.Mesh.dtype))
  for i, f in enumerate(f_indices):
    for j in range(3):
        brain.vectors[i][j] = vertices_seg[f[j],:]
  # Write the mesh to file ".stl"
  brain.save(save_path, mode=Mode.ASCII)"""
  
# Convert surface mesh structure (from simulations) to .gii format file
def mesh_to_gifti(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE, step, Ut, nodal_idx, zoom_pos, center_of_gravity, maxd, n_surface_nodes, faces, nodal_idx_b, miny, halforwholebrain):

  stlname = "B%d.gii"%(step)

  foldname = "%s/pov_H%fAT%f/"%(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE)

  save_path = os.path.join(foldname, stlname)

  # Transform coordinates (because the coordinates are normalized at the beginning)
  vertices = np.zeros((n_surface_nodes,3), dtype = float)
  f_indices = np.zeros((len(faces),3), dtype = int)
  vertices_seg = np.zeros((n_surface_nodes,3), dtype = float)

  vertices[:,:] = Ut[nodal_idx[:],:]*zoom_pos
  vertices_seg[:,1] = center_of_gravity[0] - vertices[:,0]*maxd
  if halforwholebrain.__eq__("half"):
    vertices_seg[:,0] = vertices[:,1]*maxd + miny
  else:
    vertices_seg[:,0] = vertices[:,1]*maxd + center_of_gravity[1]
  vertices_seg[:,2] = center_of_gravity[2] - vertices[:,2]*maxd

  f_indices[:,0] = nodal_idx_b[faces[:,0]]
  f_indices[:,1] = nodal_idx_b[faces[:,1]]
  f_indices[:,2] = nodal_idx_b[faces[:,2]]

  # Create the .stl mesh par Trimesh and save it
  mesh = trimesh.Trimesh(vertices=vertices_seg, faces=f_indices, process=False)
  sio.write_mesh(mesh, save_path)

# Convert mesh .stl to image .nii.gz
def point3d_to_voxel(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE, step, filename_nii_reso, Ut, zoom_pos, maxd, center_of_gravity, n_nodes, miny):

  """stlname = "B%d.stl"%(step)
  foldname = "%s/pov_H%fAT%f/"%(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE)
  file_stl_path = os.path.join(foldname, stlname)
  
  # Load stl mesh
  m = trimesh.load(file_stl_path)
  # Voxelize mesh with the specific edge length of a single voxel
  v = m.voxelized(pitch=0.25)
  # Fill surface mesh
  v = v.fill(method='holes')"""

  foldname = "%s/pov_H%fAT%f/"%(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE)
  
  # Transform coordinates (because the coordinates are normalized at the beginning)
  vertices = np.zeros((n_nodes,3), dtype = float)
  vertices_seg = np.zeros((n_nodes,3), dtype = float)

  vertices[:,:] = Ut[:,:]*zoom_pos
  vertices_seg[:,1] = (center_of_gravity[0] - Ut[:,0]*maxd)  #/1.396
  #vertices_seg[:,1] = vertices[:,0]*maxd + center_of_gravity[0]
  vertices_seg[:,0] = (Ut[:,1]*maxd + miny)  #/1.564
  #vertices_seg[:,0] = center_of_gravity[1] - vertices[:,1]*maxd
  vertices_seg[:,2] = (center_of_gravity[2] - Ut[:,2]*maxd)  #/1.122
  #vertices_seg[:,2] = vertices[:,2]*maxd + center_of_gravity[2]

  # Convert to binary image
  img = nib.load(filename_nii_reso)
  data = img.get_fdata()
  matrix_image_to_world = img.affine[:3, :3]
  abc = img.affine[:3, 3]
  image = np.zeros((n_nodes,3), dtype = np.float32)
  for i in range(n_nodes):
    image[i] = np.linalg.inv(matrix_image_to_world).dot(np.transpose(vertices_seg[i]) - abc)
  """array_index = np.transpose(np.asarray(np.where(data==1)))
  tree = spatial.KDTree(array_index)
  pp = tree.query(image)"""
  outimage = np.zeros((int(np.round(np.max(image[:,0])))+1, int(np.round(np.max(image[:,0])))+1, int(np.round(np.max(image[:,0])))+1), dtype = np.float32)
  for i in range(n_nodes):
    #outimage[array_index[pp[1][i],0], array_index[pp[1][i],1], array_index[pp[1][i],2]] = 1.0
    outimage[int(np.round(image[i,0])), int(np.round(image[i,1])), int(np.round(image[i,2]))] = 1.0

  # Save binary image in a nifti file  
  niiname = "B%d.nii.gz"%(step)
  file_nii_path = os.path.join(foldname, niiname)
  """aff = np.eye(4)
  aff[0,0] = reso
  aff[1,1] = reso
  aff[2,2] = reso
  img = nib.Nifti1Image(outimage, aff)"""
  nib.save(outimage, file_nii_path)

# Convert mesh .stl to image .nii.gz
def stl_to_image(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE, step, filename_nii_reso, reso):

  stlname = "B%d.stl"%(step)

  foldname = "%s/pov_H%fAT%f/"%(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE)

  file_stl_path = os.path.join(foldname, stlname)
  
  # Load stl mesh
  m = trimesh.load(file_stl_path)

  # Voxelize mesh with the specific edge length of a single voxel
  v = m.voxelized(pitch=0.25)

  # Fill surface mesh
  v = v.fill(method='holes')
  
  # Convert to binary image
  arr_reso = nib.load(filename_nii_reso).get_data()
  outimage = np.zeros(arr_reso.shape)
  for i in range(np.size(v.points, axis=0)):
    outimage[int(np.round(v.points[i,0]/reso)), int(np.round(v.points[i,1]/reso)), int(np.round(v.points[i,2]/reso))] = 1

  # Save binary image in a nifti file  
  niiname = "B%d_1.nii.gz"%(step)
  file_nii_path = os.path.join(foldname, niiname)
  aff = np.eye(4)
  aff[0,0] = reso
  aff[1,1] = reso
  aff[2,2] = reso
  img = nib.Nifti1Image(outimage, aff)
  nib.save(img, file_nii_path)

# Convert volumetric mesh structure (from simulations) to image .nii.gz of a specific resolution
def mesh_to_image(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE, step, filename_nii_reso, reso, coordinates, zoom_pos, center_of_gravity, maxd, n_nodes, faces, tets, miny):

  niiname = "B%d_2.nii.gz"%(step)

  foldname = "%s/pov_H%fAT%f/"%(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE)

  # Transform coordinates (because the coordinates are normalized at the beginning)
  vertices = np.zeros((n_nodes,3), dtype = float)
  vertices_seg = np.zeros((n_nodes,3), dtype = float)

  vertices[:,:] = coordinates[:,:]*zoom_pos
  vertices_seg[:,1] = center_of_gravity[0] - coordinates[:,0]*maxd
  #vertices_seg[:,1] = vertices[:,0]*maxd + center_of_gravity[0]
  vertices_seg[:,0] = coordinates[:,1]*maxd + miny
  #vertices_seg[:,0] = center_of_gravity[1] - vertices[:,1]*maxd
  vertices_seg[:,2] = center_of_gravity[2] - coordinates[:,2]*maxd
  #vertices_seg[:,2] = vertices[:,2]*maxd + center_of_gravity[2]

  """mesh = pymesh.form_mesh(vertices_seg, faces, tets)
  grid = pymesh.VoxelGrid(cell_size, mesh.dim)
  grid.insert_mesh(mesh)
  grid.create_grid()
  out_mesh = grid.mesh"""
  #mayavi.mlab.points3d(vertices_seg, mode="cube", scale_factor=1)

  # Convert to binary image
  arr_reso = nib.load(filename_nii_reso).get_data()
  outimage = np.zeros(arr_reso.shape)
  for i in range(np.size(vertices_seg, axis=0)):
    outimage[int(np.round(vertices_seg[i,0]/reso)), int(np.round(vertices_seg[i,1]/reso)), int(np.round(vertices_seg[i,2]/reso))] = 1

  # Save binary image in a nifti file  
  file_nii_path = os.path.join(foldname, niiname)
  aff = np.eye(4)
  aff[0,0] = reso
  aff[1,1] = reso
  aff[2,2] = reso
  img = nib.Nifti1Image(outimage, aff)
  nib.save(img, file_nii_path)

# Write growth texture in .gii files
def writeTex(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE, step, bt):

  giiname = "B%d_texture.gii"%(step)

  foldname = "%s/pov_H%fAT%f/"%(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE)
  
  try:
    if not os.path.exists(foldname):
      os.makedirs(foldname)
  except OSError:
    print ('Error: Creating directory. ' + foldname)

  file_gii_path = os.path.join(foldname, giiname)

  sio.write_texture(bt, file_gii_path) 

def mesh_to_vtk(PATH_DIR, coordinates, faces, tets, center_of_gravity, step, maxd, miny, node_textures, halforwholebrain, initial_geometry):
  vtk_name = str(initial_geometry) + "_step%d.vtk"%(step)
  foldname = "%s/vtk/"%(PATH_DIR)

  try:
    if not os.path.exists(foldname):
      os.makedirs(foldname)
  except OSError:
    print ('Error: Creating directory. ' + foldname)
  save_path = os.path.join(foldname, vtk_name)
  
  #coordinates denormalisation
  coordinates_denorm = coordinates_denormalisation(coordinates, len(coordinates), center_of_gravity, maxd, miny, halforwholebrain) 
  
  #tetra_mesh = meshio.Mesh(points=coordinates_denorm, cells = [ ("vertex", np.array([[i,] for i in range(len(coordinates_denorm))]) ) ] , )  
  #triangle_mesh = meshio.Mesh(points=coordinates_denorm, cells=[('triangle', faces)],)
  tetra_mesh = meshio.Mesh(points=coordinates_denorm, cells=[("tetra", tets)],)
  
  #add point data
  for key in node_textures:
    tetra_mesh.point_data[key] = node_textures[key]
    
  tetra_mesh.write(save_path)

  return 

  # Write advanced .txt file (computational time, normalized cortical area and volume, and "real" (denormalized) volume and area)
def writeTXT_advanced_XWangtimeconversion(PATH_DIR, initial_geometry, step, t, Area, Volume, step_duration, coordinates, n_nodes, zoom_pos, center_of_gravity, maxd, miny, halforwholebrain, n_tets, tets, faces, n_faces):

  txtname = "folding_{}_step{}.txt".format(initial_geometry, step)
  foldname = "%s/txt/"%(PATH_DIR)

  try:
    if not os.path.exists(foldname):
      os.makedirs(foldname)
  except OSError:
    print ('Error: Creating directory. ' + foldname)  
  
  # Compute spacings on normalized cortical surface (information to assess Bounding Box)
  norm_min_surf_edge, norm_max_surf_edge, norm_av_surf_edge = surface_edge_length(coordinates, faces, n_faces)
  
  # zoom + denormalization to compute denormalized volume, area and cortical surface spacings 
  realistic_volume_normalized_coordinates = np.zeros((n_nodes,3), dtype = float)
  realistic_volume_normalized_coordinates = coordinates * zoom_pos
  real_vol_coordinates_denorm = coordinates_denormalisation(realistic_volume_normalized_coordinates, len(realistic_volume_normalized_coordinates), center_of_gravity, maxd, miny, halforwholebrain)   
  
  maxx_norm = max(coordinates[:,0])
  minx_norm = min(coordinates[:,0])
  maxy_norm = max(coordinates[:,1])
  miny_norm = min(coordinates[:,1])
  maxz_norm = max(coordinates[:,2])
  minz_norm = min(coordinates[:,2])

  maxx_real = max(real_vol_coordinates_denorm[:,0])
  minx_real = min(real_vol_coordinates_denorm[:,0])
  maxy_real = max(real_vol_coordinates_denorm[:,1])
  miny_real = min(real_vol_coordinates_denorm[:,1])
  maxz_real = max(real_vol_coordinates_denorm[:,2])
  minz_real = min(real_vol_coordinates_denorm[:,2])

  real_volume = volume_from_coordinates(n_nodes, n_tets, tets, real_vol_coordinates_denorm)

  real_area = 0.0
  for i in range(len(faces)):
    Ntmp = np.cross(real_vol_coordinates_denorm[faces[i,1]] - real_vol_coordinates_denorm[faces[i,0]], real_vol_coordinates_denorm[faces[i,2]] - real_vol_coordinates_denorm[faces[i,0]])
    real_area += 0.5*np.linalg.norm(Ntmp)

  min_surf_edge, max_surf_edge, av_surf_edge = surface_edge_length(real_vol_coordinates_denorm, faces, n_faces)

  # write .txt file
  completeName = os.path.join(foldname, txtname)
  filetxt = open(completeName, "w")

  filetxt.write('At step ' + str(step) + ', the folding-brain-mesh characteristics are: \n')
  filetxt.write('\n')

  if t == 0:
    filetxt.write('>> Simulation time t = {:.5f} / Real time tGA approx.= 2 weeks \n'.format(t))
  elif t > 0.001 and t < 0.986: # limits for t with X.Wang time relationship 
    filetxt.write('>> Simulation time t = {:.5f} / Real time tGA = {:.3f} weeks \n'.format(t, t_to_GA_XWang(t)))
  filetxt.write('>> Computing time required for this step was: {:.2f} s \n'.format(step_duration))
  filetxt.write('\n')

  # Normalized mesh geometry  
  filetxt.write('NORMALIZED MESH GEOMETRY: \n')

  filetxt.write('>> {:.5f} mm < x_norm < {:.5f} mm \n'.format(minx_norm, maxx_norm))
  filetxt.write('>> {:.5f} mm < y_norm < {:.5f} mm \n'.format(miny_norm, maxy_norm))
  filetxt.write('>> {:.5f} mm < z_norm < {:.5f} mm \n'.format(minz_norm, maxz_norm))
  filetxt.write('\n')

  filetxt.write('>> min surface spacing in normalized mesh = {:.2f} mm2 \n'.format(norm_min_surf_edge))
  filetxt.write('>> max surface spacing in normalized mesh = {:.2f} mm2 \n'.format(norm_max_surf_edge))
  filetxt.write('>> mean surface spacing in normalized mesh = {:.2f} mm2 \n'.format(norm_av_surf_edge))
  filetxt.write('\n')

  filetxt.write('>> Normalized cortical area = {:.2f} mm2 \n'.format(Area))
  filetxt.write('\n')

  filetxt.write('>> Normalized volume = {:.2f} mm3 \n'.format(Volume))
  filetxt.write('\n')

  # 'Real' mesh geometry 
  filetxt.write('REAL MESH GEOMETRY: \n')

  filetxt.write('>> {:.5f} mm < x_real < {:.5f} mm \n'.format(minx_real, maxx_real))
  filetxt.write('>> {:.5f} mm < y_real < {:.5f} mm \n'.format(miny_real, maxy_real))
  filetxt.write('>> {:.5f} mm < z_real < {:.5f} mm \n'.format(minz_real, maxz_real))
  filetxt.write('\n')

  filetxt.write('>> min surface spacing in real mesh = {:.2f} mm2 \n'.format(min_surf_edge))
  filetxt.write('>> max surface spacing in real mesh = {:.2f} mm2 \n'.format(max_surf_edge)) 
  filetxt.write('>> mean surface spacing in real mesh = {:.2f} mm2 \n'.format(av_surf_edge))  
  filetxt.write('\n')

  filetxt.write('>> Real cortical area = {:.2f} mm2 \n'.format(real_area))
  filetxt.write('\n')

  filetxt.write('>> Real volume = {:.2f} mm3 \n'.format(real_volume))
  filetxt.write('\n')

  filetxt.close()

  return

'''# Convert mesh to binary .nii.gz image
def mesh_to_image(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE, step, Ut, SN, zoom_pos, center_of_gravity, maxd, nn):
  nifname = "B%d.nii.gz"%(step)
  foldname = "%s/pov_H%fAT%f/"%(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE)
  # Transform coordinates (because the coordinates are normalized at the beginning)
  vertices = np.zeros((nn,3), dtype = float)
  vertices_seg = np.zeros((nn,3), dtype = float)
  vertices[:,:] = Ut[:,:]*zoom_pos
  vertices_seg[:,1] = center_of_gravity[0] - vertices[:,0]*maxd
  vertices_seg[:,0] = vertices[:,1]*maxd + center_of_gravity[1]
  vertices_seg[:,2] = center_of_gravity[2] - vertices[:,2]*maxd
  # Calculate the center coordinate(x,y,z) of the mesh to define the binary image size
  center_of_gravity = np.sum(vertices_seg, axis=0)
  center_of_gravity /= nn 
  # Convert mesh to binary image
  outimage = np.zeros((2*int(round(center_of_gravity[0]))+1, 2*int(round(center_of_gravity[1]))+1, 2*int(round(center_of_gravity[2]))+1), dtype=np.int16)
  for i in range(nn):
    outimage[int(round(vertices_seg[i,0])), int(round(vertices_seg[i,1])), int(round(vertices_seg[i,2]))] = 1
  # Save binary image in a nifti file
  try:
    if not os.path.exists(foldname):
      os.makedirs(foldname)
  except OSError:
    print ('Error: Creating directory. ' + foldname)
  nii = nib.load('/home/x17wang/Exp/London/London-23weeks/brain_crisp_2_refilled.nii.gz')
  #out_inter = ndimage.morphology.binary_fill_holes(out1).astype(out1.dtype)
  #out_inter = ndimage.morphology.binary_dilation(out1, iterations=2).astype(out1.dtype)
  img = nib.Nifti1Image(outimage, nii.affine)
  save_path = os.path.join(foldname, nifname)
  nib.save(img, save_path)'''
