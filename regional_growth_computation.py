#global modules
from ast import Return
import numpy as np
import slam.io as sio

#local modules
from geometry import tetra_labels_surface_half, tetra_labels_volume_half, tetra_labels_surface_whole, tetra_labels_volume_whole
from lobes_analysis import inputmeshlobes_to_vtk


def parcel_half_mesh(n_clusters, method, reference_mesh_file_for_parcellation, lobes_file, coordinates0, nodal_idx, tets):

    # Attribute a label to each surface node of the BrainGrowth mesh
    lobes = sio.load_texture(lobes_file)
    lobes = np.round(lobes.darray[0])   #extract texture info from siam object
    labels_surfacenodes, labels_reference_allnodes = tetra_labels_surface_half(reference_mesh_file_for_parcellation, method, n_clusters, coordinates0, nodal_idx, tets, lobes)   

    # Define the label for each tetrahedron of the BrainGrowth mesh
    labels_tets = tetra_labels_volume_half(coordinates0, nodal_idx, tets, labels_surfacenodes)

    """
    texture_file_27 = '/home/x17wang/Data/GarciaPNAS2018_K65Z/PMA28to30/noninjured_ab.L.configincaltrelaxaverage.GGnorm.func.gii'
    texture_file_31 ='/home/x17wang/Data/GarciaPNAS2018_K65Z/PMA30to34/noninjured_bc.L.configincaltrelaxaverage.GGnorm.func.gii'
    texture_file_33 = '/home/x17wang/Data/GarciaPNAS2018_K65Z/PMA34to38/noninjured_cd.L.configincaltrelaxaverage.GGnorm.func.gii'
    texture_file_37 ='/home/x17wang/Data/GarciaPNAS2018_K65Z/PMA30to38/noninjured_bd.L.configincaltrelaxaverage.GGnorm.func.gii'
    """

    return lobes, labels_reference_allnodes, labels_surfacenodes, labels_tets


def parcel_whole_mesh(n_clusters, method, reference_mesh_file_for_parcellation, reference_mesh_file_for_parcellation_2, lobes_file, lobes_file_2, coordinates0, nodal_idx, tets):

    # Define the label for each surface node
    lobes = sio.load_texture(lobes_file)
    lobes = np.round(lobes.darray[0])

    lobes_2 = sio.load_texture(lobes_file_2)
    lobes_2 = np.round(lobes_2.darray[0])

    indices_a = np.where(coordinates0[nodal_idx[:],1] >= (max(coordinates0[:,1]) + min(coordinates0[:,1]))/2.0)[0]  #right part surface node indices
    indices_b = np.where(coordinates0[nodal_idx[:],1] < (max(coordinates0[:,1]) + min(coordinates0[:,1]))/2.0)[0]  #left part surface node indices

    indices_c = np.where((coordinates0[tets[:,0],1]+coordinates0[tets[:,1],1]+coordinates0[tets[:,2],1]+coordinates0[tets[:,3],1])/4 >= \
        (max(coordinates0[:,1]) + min(coordinates0[:,1]))/2.0)[0]  #right part tetrahedral indices
    indices_d = np.where((coordinates0[tets[:,0],1]+coordinates0[tets[:,1],1]+coordinates0[tets[:,2],1]+coordinates0[tets[:,3],1])/4 < \
        (max(coordinates0[:,1]) + min(coordinates0[:,1]))/2.0)[0]  #left part tetrahedral indices
        
    labels_surfacenodes, labels_surfacenodes_2, labels_reference_allnodes, labels_reference_allnodes_2 = \
        tetra_labels_surface_whole(reference_mesh_file_for_parcellation, reference_mesh_file_for_parcellation_2, method, n_clusters, coordinates0, nodal_idx, tets, indices_a, indices_b, lobes, lobes_2)

    # Define the label for each tetrahedron
    labels_tets, labels_tets_2 = tetra_labels_volume_whole(coordinates0, nodal_idx, tets, indices_a, indices_b, indices_c, indices_d, labels_surfacenodes, labels_surfacenodes_2)


    return labels_reference_allnodes, labels_reference_allnodes_2, labels_surfacenodes, labels_surfacenodes_2, labels_tets, labels_tets_2, lobes, lobes_2, indices_a, indices_b, indices_c, indices_d


def braingrowth_surf_parcellation_file_writer(initial_geometry, coordinates0, nodal_idx, indices_right_surfacenodes, indices_left_surfacenodes, labels_surfacenodes_right, labels_surfacenodes_left):

    # Compute surface nodes and faces element of the right hemisphere of the input mesh
    surface_verticescoord_right = coordinates0[nodal_idx[indices_right_surfacenodes]]
    # Compute surface nodes and faces element of the left hemisphere of the input mesh
    surface_verticescoord_left = coordinates0[nodal_idx[indices_left_surfacenodes]]
    # Join right and left geometry and associated lobes 
    surface_verticescoord_joined = np.concatenate((surface_verticescoord_right, surface_verticescoord_left), axis=0)
    labels_surface_joined = np.concatenate((labels_surfacenodes_right, labels_surfacenodes_left), axis=None)
    # Export projected lobes on the whole input mesh 
    inputmeshlobes_to_vtk(initial_geometry, 'surf', '.', surface_verticescoord_joined, labels_surface_joined)

    return 

def braingrowth_vol_parcellation_file_writer(initial_geometry, coordinates0, tets, indices_right_tets, indices_left_tets, labels_volume_right_pernode, labels_volume_left_pernode):
    volume_verticescoord_right = coordinates0[tets[indices_right_tets, :]]
    volume_verticescoord_left = coordinates0[tets[indices_left_tets, :]]
    volume_verticescoord_joined = np.concatenate((volume_verticescoord_right, volume_verticescoord_left), axis=0)
    labels_volume_joined = np.concatenate((labels_volume_right_pernode, labels_volume_left_pernode), axis=0)

    shape_volvertjoined = np.shape(volume_verticescoord_joined)
    volume_verticescoord_joined2 = volume_verticescoord_joined.reshape((shape_volvertjoined[0]*shape_volvertjoined[1],shape_volvertjoined[2]),)
    labels_volume_joined2 = labels_volume_joined.flatten()

    inputmeshlobes_to_vtk(initial_geometry, 'vol', '.', volume_verticescoord_joined2, labels_volume_joined2)

    return