from scipy.spatial import cKDTree
import numpy as np
# local modules
from geometry import netgen_to_array, get_nodes, get_tetra_indices, get_face_indices, get_nb_surface_nodes
from normalisation import normalise_coord
from numba import jit, njit, prange

from time_management import t_to_GA_XWang  
from geometry import compute_mesh_BLL


@jit(forceobj=True, parallel=True)
def input_mesh_spacing(coordinates0_normalized, n_nodes, min_or_ave):
    """ 
    Compute input mesh spacing from nodes distances
    Arguments:
        normalized nodes coordinates: np.array(n_nodes,3)
        n_nodes: int
        min_or_ave: str. "min" or "average"
    Returns:
        min or average mesh spacing
    """

    # For each node, calculate the closest other node and distance
    tree = cKDTree(coordinates0_normalized)
    distance, idex_of_node_in_mesh = tree.query(coordinates0_normalized, k=2) # 2 closest neighbours (including the parsed node itself)
    distance_2 = np.zeros((n_nodes), dtype=np.float64)

    for i in prange(n_nodes):
        distance_2[i] = distance[i][1]

    if min_or_ave == 'min': 
        mesh_spacing = np.min(distance_2)
    elif min_or_ave == 'average':
        mesh_spacing = np.mean(distance_2)

    print("\n{} mesh spacing value for normalized mesh is {:.3f} mm\n".format(min_or_ave, mesh_spacing)) 

    return mesh_spacing

@jit(forceobj=True, parallel=True)
def nogrowth_starting_barrier(coordinates0, nodal_idx):
  """
  Args: surface node coordinates
  Returns: numpy array((n_surface_nodes), dtype=np.float64)
  >> Returns inner coordinates that have the same shape than the mesh surface nodes after homothety *0.6. 
  >> These new coordinates will provide a local barrier for marking non growing nodes. 
  """
  nogrowth_starting_barrier = coordinates0[nodal_idx] * 0.6
  nogrowthbarrier_tree = cKDTree(nogrowth_starting_barrier)
  dist_surf_2_nogrowthbarrier, indx_surf_2_nogrowthbarrier = nogrowthbarrier_tree.query(coordinates0[nodal_idx])

  return dist_surf_2_nogrowthbarrier

@njit(parallel=True)
def mark_nogrowth_generic(n_nodes, dist_2_surf, nearest_surf_node, dist_surf_2_nogrowthbarrier):
  '''
  Mark non-growing nodes
  Args:
  coordinates0 (numpy array): initial cartesian coordinates of vertices
  n_nodes (int): number of nodes
  Returns:
  gr (numpy array): growth factors that control the magnitude of growth of each region
  '''
  gr = np.zeros(n_nodes, dtype = np.float64)
  for i in prange(n_nodes):
    if dist_2_surf[i] > dist_surf_2_nogrowthbarrier[nearest_surf_node[i]]: # (0.4*np.max(dist_2_surf))
      gr[i] = max(1.0 - 10.0*(dist_2_surf[i] - dist_surf_2_nogrowthbarrier[nearest_surf_node[i]]), 0.0) # from distance = 0.4*local_radius[i] to distance = 0.5*local_radius[i], gr decreases from 1.0 to 0.0 (at distance = 0.45*local_radius[i] --> gr=0.5, at distance = 0.455*local_radius[i] --> gr=0.45)
    else:
      gr[i] = 1.0

  return gr

@jit(forceobj=True)
def compute_initial_BLL_ratio(GA_0, coordinates, n_surface_nodes, nodal_idx):
    """
    The ratio enables to adapt BLL(GA) normative expression to the input mesh morphology. ratio = input_mesh_BLL(tGA=GA0) / normative_BLL(tGA=GA0). 
    BLL(GA) normative expression from T.Tallinen, provided in [X.Wang et al. 2019].
    """

    normative_BLL_GA_0 = -0.067*GA_0**2 + 7.16*GA_0 - 66.05 
    mesh_BLL_GA_0, BLL_nodes_indices_0 = compute_mesh_BLL(coordinates, n_surface_nodes, nodal_idx) 
    BLL_ratio_GA_0 = mesh_BLL_GA_0 / normative_BLL_GA_0

    return BLL_ratio_GA_0

@njit
def compute_expected_normalized_BLL(t, BLL_ratio_GA_0, maxd):
  
  normative_BLL = -0.067 * t_to_GA_XWang(t)**2 + 7.16 * t_to_GA_XWang(t) - 66.05 # [X.Wang et al. 2019]
  expected_mesh_BLL = normative_BLL * BLL_ratio_GA_0
  expected_normalized_mesh_BLL = expected_mesh_BLL/maxd

  # expected_normalized_mesh_BLL = -0.94560*t**2 + 3.05177*t + 1.79574 # This polynomial fit is valid if input mesh is the ellipsoid 'sphere5'

  return expected_normalized_mesh_BLL

@jit(forceobj=True)
def corrective_output_zoom_factor(normalized_coordinates, n_surface_nodes, nodal_idx, expected_normalized_mesh_BLL):
    """ Compute zoom factor to apply to normalized coordinates in order to get new normalized with realistic longitudinal length"""

    normalized_mesh_BLL, BLL_nodes_indices = compute_mesh_BLL(normalized_coordinates, n_surface_nodes, nodal_idx) 
    corrective_zoom_factor = expected_normalized_mesh_BLL/normalized_mesh_BLL  

    return corrective_zoom_factor
