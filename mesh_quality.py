import os
from pyvista.core.filters.data_set import * 
import meshio
import vtk

from geometry import netgen_to_array, get_nodes, get_tetra_indices

# Main code from:
# https://docs.pyvista.org/examples/01-filter/mesh-quality.html#mesh-quality-example
# https://docs.pyvista.org/examples/01-filter/compute-volume.html

#from geometry import netgen_to_array, get_nodes, get_tetra_indices, get_face_indices
input_data = './data/tallinen/week23-3M-tets' # TOMODIFY: './data/sphere5'; './data/tallinen/week23-3M-tets'
input_vtk_path = input_data + '.vtk' 

###############################################################################
# Create .vtk file from input netgen .mesh file
try:
    if not os.path.exists(input_vtk_path):
        print('create .vtk')
        # input netgen mesh
        input_mesh_path = input_data + '.mesh' # './data/sphere5.mesh'; './data/tallinen/week23-3M-tets.mesh'
        mesh = netgen_to_array(input_mesh_path)
        coordinates, n_nodes = get_nodes(mesh) 
        tets, n_tets = get_tetra_indices(mesh, n_nodes)
        # write .vtk mesh file
        tetra_mesh = meshio.Mesh(points=coordinates, cells=[("tetra", tets)],)

        tetra_mesh.write(input_data + '.vtk')
except OSError:
    print ('Error: Creating directory. ' + input_vtk_path)

###############################################################################
# Get pyvista dataset 
print('run and display quality metrics of the mesh')
pyvista_dataset = pyvista.read(input_vtk_path)  # './data/sphere5.vtk'; './data/tallinen/week23-3M-tets.vtk'

###############################################################################
# Surface analysis:
# Compute the cell quality. Note that there are many different quality measures
qual = pyvista_dataset.compute_cell_quality(quality_measure='aspect_ratio')
"""
Options for cell quality measure:
- ``'area'`` 
- ``'aspect_beta'``
- ``'aspect_frobenius'``
- ``'aspect_gamma'``
- ``'aspect_ratio'`` ***** 
- ``'collapse_ratio'``
- ``'condition'``
- ``'diagonal'``
- ``'dimension'``
- ``'distortion'``
- ``'jacobian'`` ***** (negative for tallinenbrain)
- ``'max_angle'``
- ``'max_aspect_frobenius'``
- ``'max_edge_ratio'``
- ``'med_aspect_frobenius'``
- ``'min_angle'``
- ``'oddy'``
- ``'radius_ratio'`` *****
- ``'relative_size_squared'``
- ``'scaled_jacobian'`` ***** (negative for tallinenbrain)
- ``'shape'``
- ``'shape_and_size'``
- ``'shear'``
- ``'shear_and_size'``
- ``'skew'``
- ``'stretch'``
- ``'taper'``
- ``'volume'`` ***** (negative for tallinenbrain)
- ``'warpage'``
"""
# Plot the mesh faces quality value
#qual.plot(show_edges=True, scalars='CellQuality')

###############################################################################
# 3D analysis (1):
# Plot part of the mesh tets depending on their computed quality value  
threshed = qual.threshold_percent(percent=0.85, invert=False)
"""
threshold_percent():
Parameters 
        ----------
        percent : float or tuple(float), optional
            The percentage (0,1) to threshold. If value is out of 0 to 1 range,
            then it will be divided by 100 and checked to be in that range.

        scalars : str, optional
            Name of scalars to threshold on. Defaults to currently active scalars.

        invert : bool, optional
            When invert is ``True`` cells are kept when their values are
            below the percentage of the range.  When invert is
            ``False``, cells are kept when their value is above the
            percentage of the range. Default is ``False``: yielding
            above the threshold ``"value"``.
            [0,8, 0,9], invert=False --> points above 0.9?
            [0,8, 0,9], invert=True --> points below 0.8?
            For jacobian, scaled_jacobians, volume => as negative, need to invert: threshed = qual.threshold_percent(percent=0.3, invert=True) (|J|>0.7range)

        continuous : bool, optional
            When ``True``, the continuous interval [minimum cell scalar,
            maximum cell scalar] will be used to intersect the threshold
            bound, rather than the set of discrete scalar values from
            the vertices.

        preference : str, optional
            When ``scalars`` is specified, this is the preferred array
            type to search for in the dataset.  Must be either
            ``'point'`` or ``'cell'``.

        progress_bar : bool, optional
            Display a progress bar to indicate progress.
"""
threshed.plot(show_edges=True, show_grid=True)

##############################################################################
# 3D analysis (2): 
#pyvista_dataset.set_active_scalars("Spatial Cell Data")

# Compute volumes and areas of every cell of the array
sized = pyvista_dataset.compute_cell_sizes()

# Grab volumes for all cells in the mesh (to print, not to plot)
cell_volumes = sized.cell_data["Volume"]
cell_areas = sized.cell_data["Area"]

# Compute the total volume of the mesh
#volume = pyvista_dataset.volume




