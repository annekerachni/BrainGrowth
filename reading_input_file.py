

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
mine, maxe, ave = edge_length(coordinates, faces, n_faces)
print ('\ninitial minimum edge lengths: ' + str(mine) + ', initial maximum edge lengths: ' + str(maxe) + ', initial average value of edge length: ' + str(ave))

# Calculate the total volume of the tetrahedral mesh
Vm = volume_mesh(n_nodes, n_tets, tets, coordinates)
print('initial mesh volume is ' + str(Vm))

# Calculate the total surface area of the tetrahedral mesh
Area = 0.0
for i in range(len(faces)):
Ntmp = np.cross(coordinates0[faces[i,1]] - coordinates0[faces[i,0]], coordinates0[faces[i,2]] - coordinates0[faces[i,0]])
Area += 0.5*np.linalg.norm(Ntmp)
print('initial mesh area is ' + str(Area))