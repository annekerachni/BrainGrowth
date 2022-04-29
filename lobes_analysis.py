
import os
import meshio
import numpy as np

# Export .vtk file of the reference registered mesh with lobes tag
def inputmeshlobes_to_vtk(initial_geometry, topology, path_dir, vertices, lobes):

  vtk_name = "regional_ref_mesh_lobes_projection_{}{}.vtk".format(initial_geometry, topology)
  foldname = "%s/lobes_analysis/"%(path_dir)
  
  try:
    if not os.path.exists(foldname):
      os.makedirs(foldname)
  except OSError:
    print ('Error: Creating directory. ' + foldname)
  save_path = os.path.join(foldname, vtk_name)
   
  meshio.write(save_path, meshio.Mesh(points=vertices, point_data={"Lobes": lobes}, \
    cells={"vertex": np.array([[i,] for i in range(len(vertices))])} ) )

  return 



