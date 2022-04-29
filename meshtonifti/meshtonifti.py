import numpy as np
import argparse
import nibabel as nib
from scipy.interpolate import griddata
import meshio
from scipy.spatial import cKDTree 
import itk
import time

from meshtonifti.spatialorientationadapter_to_ras import apply_lps_ras_transformation


# If reference MRI nifti and associated brain input mesh
def transfer_niftivalues_to_correspondingmeshnodes(nii_path: str, mesh_coordinates: np.array):
    """ 
    Interpolation of the input nifti values (mri intensity, segmentation label, etc.) over all mesh nodes. 

    Parameters:
    mesh_coordinates (numpy array: (number_nodes, 3)): Sparse 3D nodes coordinates
    reference_nii_path (string): pathway to the reference nifti image

    Returns: 
    numpy array (number of nodes): Interpolated values from nifti for each corresponding mesh node.
    """
    nii_img_itk = itk.imread(nii_path)

    # Reorient the MRI reference nifti from LPS+ coordinate system (itk convention) to RAS+ if coordinates are in RAS+ 
    nii_img_itk_ras = apply_lps_ras_transformation(nii_img_itk)

    # 
    niivalues_interpolator = itk.BSplineInterpolateImageFunction.New(nii_img_itk) # NearestNeighborInterpolateImageFunction; LinearInterpolateImageFunction; BSplineInterpolateImageFunction'

    # Interpolate the initial mri values to the coordinates (ijk space)
    values_loaded_to_mesh = np.zeros(len(mesh_coordinates))
    for i in range(len(mesh_coordinates)):
        coordinates_in_image_system = nii_img_itk_ras.TransformPhysicalPointToContinuousIndex(mesh_coordinates[i]) #find the closest pixel to the vertex[i] (continuous index)
        values_loaded_to_mesh[i] = niivalues_interpolator.EvaluateAtContinuousIndex(coordinates_in_image_system) #interpolates values of grey around the index et attribute the interpolation value to the associated mesh node

    return values_loaded_to_mesh  

def generate_mriref_nifti_from_meshvalues(mesh_coordinates: np.array, values: np.array, reference_nii_path: str, interpolation_method: str):
    """
    Interpolate the input mesh nodal values onto a 3d reference-image-shape grid. And generate a nifti image of interpolated sparse values.
    A reference nifti image is used to build the grid.
    
    Parameters:
    mesh_coordinates (numpy array: (number_nodes, 3)): Sparse 3D nodes coordinates
    reference_nii_path (string): pathway to the reference nifti image.

    Returns: 
    nib.Nifti: Nifti image of the sparse value interpolated.
    """
    # Get data from reference MRI nifti 
    reference_img = nib.load(reference_nii_path)  
    reference_affine = reference_img.affine     
    shape_img = np.shape(reference_img) 
    shape_i, shape_j, shape_k = shape_img[0], shape_img[1], shape_img[2] 

    # Get the specific mesh coordinates array to use as input of inverse affine matrix https://nipy.org/nibabel/coordinate_systems.html (world space)
    mesh_coordinates = np.concatenate((mesh_coordinates, np.ones((mesh_coordinates.shape[0], 1))), axis=1)

    # Transform mesh world coordinates to image coordinates: apply inverse affine matrix (image space)
    matrix_wc_2_img = np.linalg.inv(reference_affine)
    mesh_coordinates_in_image_space = (matrix_wc_2_img.dot(mesh_coordinates.T)).T

    # Generate an interpolation grid based on the size than the MRI reference nifti (image space)
    grid_X, grid_Y, grid_Z = np.mgrid[0:shape_i, 0:shape_j, 0:shape_k]

    # Apply interpolation on the sparse coordinates (image space)
    interpolation_array = griddata(mesh_coordinates_in_image_space[:,0:3], values, (grid_X, grid_Y, grid_Z), method=interpolation_method) 

    # Generate the output nifti of mesh 'values' with the same size than the reference MRI nifti
    reg_values_img = nib.Nifti1Image(interpolation_array, affine=reference_affine)
    #nib.save(reg_values_img, output_path)

    return reg_values_img

# If generic input mesh with no associated reference nifti (e.g. case of sphere, ellipsoid meshes)
def generate_generic_nifti_from_meshvalues(mesh_coordinates: np.array, values: np.array, interpolation_method: str, reso: float, margin: int):
    """
    Interpolate the input mesh nodal values onto a 3d grid generated based on the sparse data geometry. And generate a nifti image of interpolated sparse values.
    No reference nifti image is used to build the grid.
    
    Parameters:
    mesh_coordinates (numpy array: (number_nodes, 3)): Sparse 3D nodes coordinates
    reso (float): Resolution wished for the output nifti image. e.g. 0.01 for mesh length of 2mm; 0.1 for mesh length of 20mm. If np.mgrid array size is too huge for memory, decrease the resolution.
    margin (int): Image total margin (indicates the number of voxels on both sides of the nifti) 

    Returns: 
    nib.Nifti: Nifti image of the interpolated sparse value.
    
    """

    # Calculate the dimension of the values mesh
    length_x = max(mesh_coordinates[:,0]) - min(mesh_coordinates[:,0])
    length_y = max(mesh_coordinates[:,1]) - min(mesh_coordinates[:,1])
    length_z = max(mesh_coordinates[:,2]) - min(mesh_coordinates[:,2])

    # Generate the shape of the anisotropic interpolation grid
    shape_x, shape_y, shape_z = length_x/reso + margin, length_y/reso + margin, length_z/reso + margin

    # Localize the geometrical center of the values mesh
    mesh_center_x, mesh_center_y, mesh_center_z = 0.5*(min(mesh_coordinates[:,0]) + max(mesh_coordinates[:,0])), \
        0.5*(min(mesh_coordinates[:,1]) + max(mesh_coordinates[:,1])), 0.5*(min(mesh_coordinates[:,2]) + max(mesh_coordinates[:,2]))

    # Calculate the translation vector -B between the mesh coordinates and the image space
    coord_img_center_x, coord_img_center_y, coord_img_center_z = 0.5*shape_x*reso, 0.5*shape_y*reso, 0.5*shape_z*reso
    mesh_to_img_vect = [coord_img_center_x - mesh_center_x, coord_img_center_y - mesh_center_y, coord_img_center_z - mesh_center_z]

    # Apply translation vector to mesh coordinates ( (x,y,z) = M@(i,j,k) + B ) https://nipy.org/nibabel/coordinate_systems.html
    mesh_coord_in_img_space = mesh_coordinates.copy() 
    mesh_coord_in_img_space[:,0] += mesh_to_img_vect[0] # -B[0]
    mesh_coord_in_img_space[:,1] += mesh_to_img_vect[1] # -B[1]
    mesh_coord_in_img_space[:,2] += mesh_to_img_vect[2] # -B[2]
    mesh_coord_in_img_space /= reso # inv(M)

    # Build the output nifti affine 
    generated_affine = np.zeros((4,4), dtype=np.float64)
    generated_affine[0,0] = generated_affine[1,1] = generated_affine[2,2] = reso
    generated_affine[0,3] = - mesh_to_img_vect[0]
    generated_affine[1,3] = - mesh_to_img_vect[1]
    generated_affine[2,3] = - mesh_to_img_vect[2]
    generated_affine[3,3] = 1

    # Generate an interpolation grid (image space)
    grid_X, grid_Y, grid_Z = np.mgrid[0:shape_x, 0:shape_y, 0:shape_z]

    # Apply interpolation on the sparse coordinates (image space)
    interpolation_array = griddata(mesh_coord_in_img_space, values, (grid_X, grid_Y, grid_Z), method=interpolation_method) 

    # Generate the output nifti of mesh 'values'
    values_img = nib.Nifti1Image(interpolation_array, affine=generated_affine) #Calculates the affine between image and coordinates : translation center of gravity
    #nib.save(values_img, output_path)

    return values_img

if __name__ == '__main__':
    start_time_initialization = time.time ()
    parser = argparse.ArgumentParser(description='Visualisation of brain values (nifti)')
    parser.add_argument('-i', '--input', help='Path to input vtk file (step, coordinates, brain values)', \
        default='../res/17fev/sphere5_meshspacauto_K0-05_gr_hibeta/vtk/sphere5_tetramesh_step500.vtk', type=str, required=False)
    parser.add_argument('-r', '--reference', help='Reference nifti', type=str, default='../data/data_anne/dhcp/dhcp.nii', required=False)
    parser.add_argument('-o', '--output', help='Path to output nifti file', type=str, \
        default='../res/17fev/sphere5_meshspacauto_K0-05_gr_hibeta/vtk/sphere5_tetramesh_step500_tg.nii.gz', required=False)
    parser.add_argument('-m', '--method', help='griddata interpolation method: nearest; linear; cubic', type=str, default='linear', required=False)
    parser.add_argument('-mnv', '--meshnodalvalue', help='mesh nodal value to display in nifti: \
        Displacement; Distance_to_surface; Growth_ponderation ; Tangential_growth_wg_term; Tangential_growth', type=str, default='Tangential_growth', required=False)
    args = parser.parse_args()

    # DATA COLLECION FROM .VTK MESH
    mesh = meshio.read(args.input)
    mesh_coordinates = mesh.points # list of nodes coordinates
    mesh_nodal_values = mesh.point_data[args.meshnodalvalue] # 'Displacement'; 'Distance_to_surface'; 'Growth_ponderation' (gr) ; 'Tangential_growth' (g(y,t)) 

    ## NIFTI GENERATION
    # Reference MRI nifti and associated brain input mesh
    # Generate nifti from reference mri values
    """ mesh0 = meshio.read('../data/data_anne/dhcp/dhcp.nii')
    mesh_coordinates0 = mesh0.points
    reference_mri_values = get_reference_mrivalues_array(mesh_coordinates0, args.reference)
    generate_mriref_nifti_from_meshvalues(mesh_coordinates0, reference_mri_values, args.reference, args.method, '../res/dhcpbrain_fine_vtk/dhcpbrain_fine_tetra_0_mrivalues.nii.gz')  
    print('\n The nifti dhcpbrain_fine_mrivalues0.nii.gz has been generated. \n')
    end_time_initialization = time.time () - start_time_initialization
    print ('Time required for mri values interpolation : ' + str (end_time_initialization) )

    # Generate nifti from other mesh nodal values.  
    generate_mriref_nifti_from_meshvalues(mesh_coordinates, mesh_nodal_values, args.reference, args.method, args.output)  
    print('\n The nifti ' + str(args.output) + ' has been generated. \n')
    end_time_initialization = time.time () - start_time_initialization
    print ('Time required for displacement interpolation : ' + str (end_time_initialization) ) """

    # Generic input mesh with no associated reference nifti 
    meshvalues_generic_img = generate_generic_nifti_from_meshvalues(mesh_coordinates, mesh_nodal_values, args.method)
    nib.save(meshvalues_generic_img, args.output)
    print('\n The nifti ' + str(args.output) + ' has been generated. \n') 
    