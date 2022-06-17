# -*- coding: utf-8 -*-
"""
python simulation.py -i './data/ellipsoid/sphere5.mesh' -ig 'ellispoid' -o './res/ellipsoid' -hc 'whole' -t 0.042 -g 2.136 -gme 'global' -ga0 '22' -gaf '44'
"""

#global modules
import argparse
import numpy as np
from numba.typed import List
import time
import slam.io as sio
from scipy.spatial import cKDTree
import os

#global modules for tracking
import cProfile
import pstats
import io
import sys

#local modules
from geometry import netgen_to_array, tetra_normals, get_nodes, get_tetra_indices, get_face_indices, get_nb_surface_nodes, surface_edge_length, volume_from_coordinates, volume_from_Vn, area_mesh_surface, mark_nogrowth, config_refer, config_deform, normals_surfaces, calc_vol_nodal, calc_mid_plane, calc_longi_length, paraZoom, tetra_labels_surface_half, tetra_labels_volume_half, curve_fitting_half, tetra_labels_surface_whole, tetra_labels_volume_whole, curve_fitting_whole
from growth import growth_rate_global, growth_rate_regional_half, growth_rate_regional_whole, regional_at_nodal, differential_term_tetrahedron, differential_term_nodal, tangential_growth_coefficient_nodal, growth_directions, growth_tensor
from normalisation import normalise_coord
from collision_Tallinen import contact_process
from mechanics import shear_modulus, tetra_elasticity, move
from output import mesh_to_vtk, writeTXT_advanced_XWangtimeconversion, writePov, writeTXT, mesh_to_stl
from normalisation import coordinates_denormalisation
from regional_growth_computation import parcel_half_mesh, parcel_whole_mesh, braingrowth_surf_parcellation_file_writer, braingrowth_vol_parcellation_file_writer
from inputmesh_automatized_parameters import input_mesh_spacing, nogrowth_starting_barrier, mark_nogrowth_generic, compute_initial_BLL_ratio, compute_expected_normalized_BLL, corrective_output_zoom_factor
from time_management import GA_to_t_XWang, t_to_GA_XWang

if __name__ == '__main__':
  start_time_initialization = time.time ()
  parser = argparse.ArgumentParser(description='Dynamic simulations')
  parser.add_argument('-i', '--input', help='Input mesh', type=str, default='./data/ellipsoid/sphere5.mesh', required=False) # TO UPDATE
  parser.add_argument('-ig', '--initialgeometry', help='Initial geometry', type=str, default="ellipsoid", required=False) # TO UPDATE
  parser.add_argument('-o', '--output', help='Output folder', type=str, default='./res/ellipsoid', required=False) # TO UPDATE
  parser.add_argument('-hc', '--halforwholebrain', help=' "half" or "whole" brain', type=str, default='whole', required=False)
  parser.add_argument('-t', '--thickness', help='Normalized cortical thickness at 22GW', type=float, default=0.042, required=False)
  parser.add_argument('-g', '--growth', help='Normalized relative growth rate', type=float, default=1.829, required=False) # 2.136. alpha growth coefficient obtained with g**2 = 8 at GA=44GW (t_XWang=0.856) [T.Tallinen et al. 2014]
  parser.add_argument('-gme', '--growthmethod', help=' "global" or "regional" growth', type=str, default='global', required=False) 
  parser.add_argument('-mr', '--registermeshright', help='Mesh of right brain after registration', type=str, default= './data/garcia/register_meshes/rh.gii', required=False)
  parser.add_argument('-ml', '--registermeshleft', help='Mesh of left brain after registration', type=str, default= './data/garcia/register_meshes/lh.gii', required=False)
  parser.add_argument('-tr', '--textureright', help='Texture of template of right brain', type=str, default='./data/garcia/textures/covariateinteraction2.R.noivh.GGnorm.func.gii', required=False)
  parser.add_argument('-tl', '--textureleft', help='Texture of template of left brain', type=str, default='./data/garcia/textures/covariateinteraction2.L.noivh.GGnorm.func.gii', required=False)
  parser.add_argument('-lr', '--lobesright', help='User-defined lobes of right brain', type=str, default='./data/garcia/lobes/ATLAS30.R.Fiducial.surf.fineregions.gii', required=False)
  parser.add_argument('-ll', '--lobesleft', help='User-defined lobes of left brain', type=str, default='./data/garcia/lobes/ATLAS30.L.Fiducial.surf.fineregions.gii', required=False)
  parser.add_argument('-sc', '--stepcontrol', help='Step length regulation', type=float, default=0.01, required=False) 
  #parser.add_argument('-ms', '--meshspacing', help='Average spacing in the mesh', type=float, default=0.01, required=False) 
  parser.add_argument('-md', '--massdensity', help='Mass density of brain mesh', type=float, default=0.01, required=False) 
  parser.add_argument('-ga0', '--initialga', help='Gestational Age of the input brain mesh', type=float, default=22, required=False) # TO UPDATE
  parser.add_argument('-gaf', '--endga', help='Gestational Age at which to stop the simulation', type=float, default=44, required=False) # TO UPDATE
  args = parser.parse_args()

  #####################
  ## Main parameters ##
  #####################

  GA_0 = args.initialga
  t0 = GA_to_t_XWang(GA_0) #Current simulation time
  t = t0
  step = 0 #Current time step

  GA_end = args.endga
  tf = GA_to_t_XWang(GA_end)

  # Main parameters influencing folding process 
  GROWTH_RELATIVE = args.growth

  #THICKNESS_CORTEX = args.thickness
  #cortex_thickness = THICKNESS_CORTEX #Cortical plate thickness

  mug = 1.0 #65.0 Shear modulus of gray matter
  muw = 1.167 #75.86 Shear modulus of white matter
  bulk_modulus = 5.0 # "for numerical convenience we now adopt modest compressibility with K = 5μ, corresponding to Poisson’s ratio ν ≈ 0.4." [Tallinen et al. 2014]
  
  mass_density = args.massdensity # adjust to run the simulation faster or slower

  # Contact forces parameters
  bounding_box = 3.2 #3.2 #Width of a bounding box, centered at origin, that encloses the whole geometry even after growth ***** TOMODIFY
  contact_stiffness = 10.0 * bulk_modulus #100.0*K Contact stiffness
  
  # Elastic force parameters
  eps = 0.1 #Epsilon
  k_param = 0.0

  # Inter-hemisphere stability parameter
  midplane_pos = -0.004 #Midplane position

  # Viscous force
  damping_coef = 0.5 #0.1 Damping coefficent

  # Output parameters
  di = 500 #Output data once every di steps
  zoom = 1.0 #Zoom variable for visualization

  ##########################
  ## Read input mesh file ##
  ##########################

  PATH_DIR = args.output

  # Import mesh, each line as a list
  mesh = netgen_to_array(args.input)
  mesh = List(mesh) # added to avoid "reflected list" for "mesh" argument issue with numba

  # Read nodes, get undeformed coordinates0 and initialize deformed coordinates for all nodes. # X-Y switch at this point
  coordinates, n_nodes = get_nodes(mesh) 
  coordinates0 = coordinates.copy()

  # Read element indices. Handness switch at this point
  tets, n_tets = get_tetra_indices(mesh, n_nodes)

  # Read surface triangle indices 
  faces, n_faces = get_face_indices(mesh, n_nodes, n_tets)

  # Determine surface nodes and index maps 
  n_surface_nodes, nodal_idx, nodal_idx_b = get_nb_surface_nodes(faces, n_nodes)

  # Check minimum, maximum and average edge lengths (average mesh spacing) at the surface
  mine, maxe, ave = surface_edge_length(coordinates, faces, n_faces)
  print ('\ninitial minimum edge lengths: ' + str(mine) + ', initial maximum edge lengths: ' + str(maxe) + ', initial average value of edge length: ' + str(ave))

  # Calculate the total volume of the tetrahedral mesh
  V_initial = volume_from_coordinates(n_nodes, n_tets, tets, coordinates)
  print('initial mesh volume is ' + str(V_initial))

  # Calculate the total surface area of the tetrahedral mesh
  Area_initial = area_mesh_surface(coordinates0, faces)
  print('initial mesh area is ' + str(Area_initial))

  BLL_ratio_GA_0 = compute_initial_BLL_ratio(GA_0, coordinates, n_surface_nodes, nodal_idx)

  ####################################
  ## Physical arrays initialization ##
  ####################################
  nearest_surf_node = np.zeros(n_nodes, dtype = np.int64)  #Nearest surface nodes for all nodes
  dist_2_surf = np.zeros(n_nodes, dtype = np.float64)  #Distances to nearest surface nodes for all nodes
  surf_node_norms = np.zeros((n_surface_nodes,3), dtype = np.float64)  #Normals of surface nodes
  Vt = np.zeros((n_nodes,3), dtype = np.float64)  #Velocities
  Ft = np.zeros((n_nodes,3), dtype = np.float64)  #Forces
  growth_filter = np.ones(n_tets, dtype = np.float64)
  NNLt = [[] for _ in range (n_surface_nodes)] #Triangle-proximity lists for surface nodes
  coordinates_old = np.zeros((n_surface_nodes,3), dtype = np.float64)  #Stores positions when proximity list is updated
  tan_growth_tensor = np.ones((n_tets,3,3), dtype = np.float64) * np.identity(3)  # Initial tangential growth tensor

  ###################################################################
  ## Normalize mesh and compute associated initial physical values ##
  ###################################################################

  # Normalize initial mesh coordinates, change mesh information by values normalized
  coordinates0, coordinates, center_of_gravity, maxd, miny = normalise_coord(coordinates0, coordinates, n_nodes, args.halforwholebrain)

  # Compute input mesh spacing 
  min_or_ave = 'min' # 'min' or 'average'
  mesh_spacing = input_mesh_spacing(coordinates0, n_nodes, min_or_ave)

  # Compute simulation time step (CFL condition)
  dt = args.stepcontrol*np.sqrt(mass_density * mesh_spacing * mesh_spacing / bulk_modulus) #0.05*np.sqrt(rho*a*a/K) in X.Wang. 
  print('dt is ' + str(dt))

  # Compute bounding box and contact forces characteristics 
  cell_width = 8 * mesh_spacing #Width of a cell in the linked cell algorithm for proximity detection
  prox_skin = 0.6 * mesh_spacing #Thickness of proximity skin
  repuls_skin = 0.2 * mesh_spacing #Thickness of repulsive skin

  # Find the nearest surface nodes (nearest_surf_node) to nodes and distances to them (dist_2_surf)
  tree = cKDTree(coordinates0[nodal_idx])
  dist_2_surf, nearest_surf_node = tree.query(coordinates0)

  # Configuration of tetrahedra at reference state (ref_state_tets)
  ref_state_tets = config_refer(coordinates0, tets, n_tets)

  # Mark non-growing areas
  #gr = mark_nogrowth(coordinates0, n_nodes)
  dist_surf_2_nogrowthbarrier = nogrowth_starting_barrier(coordinates0, nodal_idx)
  gr = mark_nogrowth_generic(n_nodes, dist_2_surf, nearest_surf_node, dist_surf_2_nogrowthbarrier)

  # Calculate normals of each surface triangle at each node
  surf_node_norms = normals_surfaces(coordinates0, faces, nodal_idx_b, n_faces, n_surface_nodes, surf_node_norms)

  # Calculate reference normals of each deformed tetrahedron 
  tet_norms0 = tetra_normals(surf_node_norms, nearest_surf_node, tets, n_tets)

  # Initialize growth tensor
  A, B = growth_directions(tet_norms0, n_tets)

  ########################################
  ## Compute regional growth parameters ##
  ########################################

  if args.growthmethod.__eq__("regional"):
    n_clusters = 10   #Number of wished lobes
    method = 'User-defined lobes' #Method of parcellation in lobes

    # Half brain
    ############
    if args.halforwholebrain.__eq__("half"):
      # Parcel brain in lobes: define the label for each surface node and tets of the BrainGrowth half mesh
      mesh_file = args.registermeshright
      lobes_file = args.lobesright
      lobes, labels_reference_allnodes, labels_surfacenodes, labels_tets = parcel_half_mesh(n_clusters, method, mesh_file, lobes_file, coordinates0, nodal_idx, tets)
      
      # Compute regional growth model parameters
      texture_file = args.textureright
      peak, amplitude, latency = curve_fitting_half(texture_file, labels_reference_allnodes, n_clusters, lobes)

    # Whole brain
    #############
    else:
      #Parcel brain in lobes: define the label for each surface node and tets of the BrainGrowth whole mesh
      mesh_file = args.registermeshright
      mesh_file_2 = args.registermeshleft
      lobes_file = args.lobesright
      lobes_file_2 = args.lobesleft
      labels_reference_allnodes, labels_reference_allnodes_2, labels_surfacenodes, labels_surfacenodes_2, labels_tets, labels_tets_2, lobes, lobes_2, indices_a, indices_b, indices_c, indices_d = \
        parcel_whole_mesh(n_clusters, method, mesh_file, mesh_file_2, lobes_file, lobes_file_2, coordinates0, nodal_idx, tets)

      # Check BrainGrowth mesh parcellation 
      #braingrowth_surf_parcellation_file_writer(args.initialgeometry, coordinates0, nodal_idx, indices_a, indices_b, labels_surfacenodes, labels_surfacenodes_2) # Surface nodes label
      #braingrowth_vol_parcellation_file_writer(args.initialgeometry, coordinates0, tets, indices_c, indices_d, labels_tets, labels_tets_2) # Volume nodes label

      # Curve-fit of temporal growth for each label
      texture_file = args.textureright
      texture_file_2 = args.textureleft
      peak, amplitude, latency, peak_2, amplitude_2, latency_2 = curve_fitting_whole(texture_file, texture_file_2, labels_reference_allnodes, labels_reference_allnodes_2, n_clusters, lobes, lobes_2)


  end_time_initialization = time.time () - start_time_initialization
  print ('\ntime required for initialization was {} seconds\n'.format(end_time_initialization))

  ##############################
  ## Simulate folding process ##
  ##############################

  #times_tetraelasticity = 0.
  start_time_simulation = time.time ()
  while t < tf: 

    # Calculate the relative growth rate //bt not used
    if args.growthmethod.__eq__("regional"):
      if args.halforwholebrain.__eq__("half"):
        at, bt = growth_rate_regional_half(t, n_tets, n_surface_nodes, labels_surfacenodes, labels_tets, peak, amplitude, latency, lobes)
      else:
        at, bt = growth_rate_regional_whole(t, n_tets, n_surface_nodes, labels_surfacenodes, labels_surfacenodes_2, labels_tets, labels_tets_2, peak, amplitude, latency, peak_2, amplitude_2, latency_2, lobes, lobes_2, indices_a, indices_b, indices_c, indices_d)
    else:
      at = growth_rate_global(n_tets, GROWTH_RELATIVE, t, t0)
      
    #growth_filter = calc_growth_filter(growth_filter, dist_2_surf, n_tets, tets, cortex_thickness)

    #update cortex thickness
    cortex_thickness = 0.046 - 0.028*t # cortical thickness evolution law obtained from: fitting (in X.Wang time) fetal data [Xu et al. 2022] [Liu et al. 2021] + BLL(t) expression [X.Wang et al. 2019] 
    #cortex_thickness = THICKNESS_CORTEX + 0.017*(t - 0.0658) # cortical thickness evolution law obtained by transfering to X.Wang time relationship
    #cortex_thickness = THICKNESS_CORTEX + 0.01*t # [22GW THICKNESS_CORTEX reference provided by X.Wang et al.]

    # Deformed configuration of tetrahedra (At)
    material_tets = config_deform(coordinates, tets, n_tets)

    # Calculate stress-free (after growth) nodal volume (Vn0) and deformed nodal volume (Vn) 
    Vn0, Vn = calc_vol_nodal(tan_growth_tensor, ref_state_tets, material_tets, tets, n_tets, n_nodes)

    # Calculate contact forces (Reference: Real Time Detection Collision, C. Ericson)
    Ft, NNLt = contact_process(coordinates, Ft, nodal_idx, coordinates_old, n_surface_nodes, NNLt, faces, n_faces, bounding_box, cell_width, prox_skin, repuls_skin, contact_stiffness, mesh_spacing, gr) 
    
    # Calculate gray and white matter shear modulus (gm and wm) for a tetrahedron, calculate the global shear modulus
    gm = differential_term_tetrahedron(dist_2_surf, cortex_thickness, tets, n_tets, gr)
    mu = shear_modulus(n_tets, muw, mug, gm)

    # Calculate elastic forces
    #t1 = time.time ()
    Ft = tetra_elasticity(material_tets, ref_state_tets, Ft, tan_growth_tensor, bulk_modulus, k_param, mu, tets, Vn, Vn0, n_tets, eps) 
    #t2 = time.time ()
    #times_tetraelasticity += t2 - t1

    #Seperate tetraelasticity initialization and calculatin, useful for optimization purposes. 
    #left_cauchy_grad, rel_vol_chg, rel_vol_chg1, rel_vol_chg2, rel_vol_chg3, rel_vol_chg4, rel_vol_chg_av, deformation_grad, ref_state_growth = tetra1(tets, tan_growth_tensor, ref_state_tets, ref_state_growth, material_tets, Vn, Vn0)
    #Ft = tetra2(n_tets, tets, Ft, left_cauchy_grad, mu, eps, rel_vol_chg, bulk_modulus,rel_vol_chg_av, deformation_grad, rel_vol_chg1, rel_vol_chg2, rel_vol_chg3, rel_vol_chg4, k_param, ref_state_growth)

    # Calculate relative tangential growth factor G
    tan_growth_tensor = growth_tensor(n_tets, tan_growth_tensor, B, gm, at)

    # Midplane
    Ft = calc_mid_plane(coordinates, coordinates0, Ft, nodal_idx, n_surface_nodes, midplane_pos, mesh_spacing, repuls_skin, bulk_modulus) 
    
    ############
    ## Output ##
    ############
    if step % di == 0:

      print('step{}:'.format(step))

      # Calculate mesh surface area and volume
      Area_step = area_mesh_surface(coordinates, faces)
      Volume_step = volume_from_Vn(Vn) 
      print('normalized area is {} mm2, normalized volume is {} mm3'.format(Area_step, Volume_step))

      # Obtain output zoom factor by checking the normative longitudinal length of the brain model and applying to the normalized input mesh
      expected_normalized_mesh_BLL = compute_expected_normalized_BLL(t, BLL_ratio_GA_0, maxd)      
      corrective_zoom_factor = corrective_output_zoom_factor(coordinates, n_surface_nodes, nodal_idx, expected_normalized_mesh_BLL) # Factor to obtain 'realistic' normalized coordinates 
      
      #timestamp for step duration export 
      step_duration = time.time() - start_time_simulation
      
      # Write surface mesh output files in .txt file
      writeTXT_advanced_XWangtimeconversion(PATH_DIR, args.initialgeometry, step, t, Area_step, Volume_step, step_duration, coordinates, n_nodes, corrective_zoom_factor, center_of_gravity, maxd, miny, args.halforwholebrain, n_tets, tets, faces, n_faces)
      
      # Write volume nodal physical values computed by simulation to .vtk file
      #Compute nodal physical values to display
      nodal_displacement = np.zeros((n_nodes), dtype=np.float64)
      nodal_displacement = np.linalg.norm(coordinates_denormalisation(coordinates, n_nodes, center_of_gravity, maxd, miny, args.halforwholebrain) - coordinates_denormalisation(coordinates0, n_nodes, center_of_gravity, maxd, miny, args.halforwholebrain), axis=1)
      
      gm_nodal = differential_term_nodal(dist_2_surf, cortex_thickness, n_nodes, gr)
      
      if args.growthmethod.__eq__("regional"):
        at_nodal = regional_at_nodal(n_nodes, tets, at)
      else:
        at_nodal = np.zeros(n_tets, dtype=np.float64)
        at_nodal[:] = at[0]
      g_nodal = tangential_growth_coefficient_nodal(n_nodes, at_nodal, gm_nodal)

      node_textures = {} 
      # Metrics that need to be denormalized 
      node_textures['Displacement'] = nodal_displacement 
      node_textures['Distance_to_surface'] = dist_2_surf*maxd # since dist_2_surf computed for normalized mesh 
      #node_textures['Constraint'] = tex_tets_to_nodes(n_nodes, tets, np.linalg.norm(P_vec, axis=(1, 2)))

      # Ratio metrics, independant from coordinates
      node_textures['Growth_ponderation'] = gr 
      node_textures['Differential_term'] = gm_nodal 
      node_textures['Tangential_growth_coefficient'] = g_nodal 

      mesh_to_vtk(PATH_DIR, coordinates, faces, tets, center_of_gravity, step, maxd, miny, node_textures, args.halforwholebrain, args.initialgeometry)

      #timestamp for simulation loop
      end_time_simulation = time.time() - start_time_simulation
      print('time required for step{} was {} seconds\n'.format(step, end_time_simulation))
      start_time_simulation = time.time()
 
    # Newton dynamics: compute coordinates t+dt and reinitialize Ft
    Ft, coordinates, Vt = move(n_nodes, Ft, Vt, coordinates, damping_coef, Vn0, mass_density, dt) 

    t += dt
    step += 1


"""
########################
### ARCHIVED OUTPUTS ###
########################

# Write texture of growth in .gii files
writeTex(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE, step, bt)

# Write .pov files and output mesh in .png file
writePov(PATH_DIR, args.initialgeometry, step, coordinates, faces, nodal_idx, nodal_idx_b, n_surface_nodes, zoom, corrective_zoom_factor)

# Convert surface mesh structure (from simulations) to .stl format file
mesh_to_stl(PATH_DIR, args.initialgeometry, step, coordinates, nodal_idx, corrective_zoom_factor, center_of_gravity, maxd, n_surface_nodes, faces, nodal_idx_b, miny, args.halforwholebrain)

# Convert surface mesh structure (from simulations) to .gii format file
mesh_to_gifti(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE, step, coordinates, nodal_idx, corrective_zoom_factor, center_of_gravity, maxd, n_surface_nodes, faces, nodal_idx_b, miny, args.halforwholebrain)

# Convert mesh .stl to image .nii.gz
stl_to_image(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE, step, filename_nii_reso, reso)

# Convert 3d points to image voxel
point3d_to_voxel(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE, step, filename_nii_reso, Ut, corrective_zoom_factor, maxd, center_of_gravity, nn, miny)

# Convert volumetric mesh structure (from simulations) to image .nii.gz of a specific resolution
mesh_to_image(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE, step, filename_nii_reso, reso, Ut, corrective_zoom_factor, center_of_gravity, maxd, nn, faces, tets, miny)

"""