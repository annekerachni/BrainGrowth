import numpy as np
import math
from numba import jit, njit, prange

########################
## GLOBAL GROWTH RATE ##
########################

@jit(nopython=True)
def growth_rate_global(n_tets, GROWTH_RELATIVE, t, t0):
  """
  Calculates global relative growth rate for half or whole brain
  Args:
  GROWTH_RELATIVE (float): Constante set at simulation level
  t (float): current time of the simulation
  n_tets(int): number of tetrahedrons in the model
  Returns:
  at (array): growth rate per tetrahedre
  filter (array): Optional factor for smoothing
  """
  at = np.zeros(n_tets, dtype=np.float64)
  at[:] = GROWTH_RELATIVE*(t - t0)/(0.0856 - t0) # at(GA=44GW) = at(t_TTallinen=1.0) = = 1 + GROWTH_RELATIVE = at(t_XWang=0.856) 

  return at

"""
##############################################
###LEGACY code for reference, safely ignore###
##############################################
#Smoothing using slam
    if step == 0:
        #create filtering for each surface node
        stl_path = "./res/sphere5/pov_H0.042000AT1.829000/B0.stl"
        stl_mesh = trimesh.load (stl_path)
        filtering = np.ones (len(stl_mesh.vertices))
        filtering = laplacian_texture_smoothing (stl_mesh, filtering, 10, dt)
        #Expand filtering via nearest surface node (vectorisable)
        gauss = np.ones (n_nodes, dtype = np.float64)
        for i in range (len(gauss)):
          gauss[i] = filtering[nearest_surf_node[i]]
        
        #conversion from node to tet length, take the average from nodes
        gauss_tets = np.ones (n_tets, dtype = np.float64)
        for i in range (len(tets)):
          x = 0
          for j in tets[i]:
            x += gauss [j]
          gauss_tets [i] = x/4
    
    #apply filter (why not before ? Because stl not available at this point)
    at *= gauss_tets

def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)
    return wrapped

#optional gaussian filtering on at
  # centroids = calc_tets_center(coordinates, tets)
  # gauss = gaussian_filter (centroids)
"""

@jit(nopython=True, parallel=True)
def calc_growth_filter(growth_filter, dist_2_surf, n_tets, tets, cortex_thickness):
  for i in prange(n_tets):
    if float(dist_2_surf[tets[i][0]]) < cortex_thickness:
      growth_filter[i] = 1.0 #is the first somit of the tet deeper in the brain than cortical tickness?
    else:
      growth_filter[i] = 1.2
    return growth_filter

##########################
## REGIONAL GROWTH RATE ##
##########################

@jit
def growth_rate_regional_half(t, n_tets, n_surface_nodes, labels_surface, labels_volume, peak, amplitude, latency, lobes):
  """
  TODO: test and finish function
  Calculates the regional relative growth rate for half brain
  Args:
  t (float): current time of simulation
  n_tets (int): number of tetrahedrons in model
  n_surface_nodes (int): number of surface nodes
  labels_surface: lobar labels of surafces nodes
  labels_volume: lobar labels of tetrahedrons
  peak: parameter of Gompertz function
  aplitude: parameter of Gompertz function
  latency: parameter of Gompertz function
  lobes: lobar labels of all nodes of surface mesh
  """
  at = np.zeros(n_tets, dtype=np.float64)
  bt = np.zeros(n_surface_nodes, dtype=np.float64)
  m = 0
  for i in np.unique(lobes):
    at[np.where(labels_volume == i)[0]] = amplitude[m]*np.exp(-np.exp(-peak[m]*(t-latency[m])))
    bt[np.where(labels_surface == i)[0]] = amplitude[m]*np.exp(-np.exp(-peak[m]*(t-latency[m])))
    m += 1
  at = np.where(at > 0.0, at, 0.0)
  bt = np.where(bt > 0.0, bt, 0.0)

  return at, bt

# Calculate the regional relative growth rate for whole brain
@jit
def growth_rate_regional_whole(t, n_tets, n_surface_nodes, labels_surface, labels_surface_2, labels_volume, labels_volume_2, peak, amplitude, latency, peak_2, amplitude_2, latency_2, lobes, lobes_2, indices_a, indices_b, indices_c, indices_d):
  at = np.zeros(n_tets, dtype=np.float64)
  bt = np.zeros(n_surface_nodes, dtype=np.float64)
  #for i in range(n_clusters):
  m = 0
  for i in np.unique(lobes):
    at[indices_c[np.where(labels_volume == i)[0]]] = amplitude[m]*np.exp(-np.exp(-peak[m]*(t-latency[m])))
    bt[indices_a[np.where(labels_surface == i)[0]]] = amplitude[m]*np.exp(-np.exp(-peak[m]*(t-latency[m])))
    m += 1
  m_2 = 0
  for i in np.unique(lobes_2):
    at[indices_d[np.where(labels_volume_2 == i)[0]]] = amplitude_2[m_2]*np.exp(-np.exp(-peak_2[m_2]*(t-latency_2[m_2])))
    bt[indices_b[np.where(labels_surface_2 == i)[0]]] = amplitude_2[m_2]*np.exp(-np.exp(-peak_2[m_2]*(t-latency_2[m_2])))
    m_2 += 1
  at = np.where(at > 0.0, at, 0.0)
  bt = np.where(bt > 0.0, bt, 0.0)

  return at, bt

# Get nodal at growth term in case of regional growth
@jit(nopython=True)
def regional_at_nodal(n_nodes, tets, at):
  """ For each node, mean the 'at' biological growth term of all proximal tets. """

  at_nodal = np.zeros(n_nodes, dtype=np.float64)
  proximal_tets = np.zeros(n_nodes, dtype=np.float64) 

  for i in prange(len(tets)):
    at_nodal[tets[i,0]] += at[i] #biological growth is a time-dependant but no spatial-dependant term, so only need to mean the term over all proximal tetrahedrons around the node.
    at_nodal[tets[i,1]] += at[i]
    at_nodal[tets[i,2]] += at[i]
    at_nodal[tets[i,3]] += at[i]
    proximal_tets[tets[i,0]] += 1 #proximal_tets[node indice 0 for tet i]
    proximal_tets[tets[i,1]] += 1
    proximal_tets[tets[i,2]] += 1
    proximal_tets[tets[i,3]] += 1

  for j in prange(n_nodes): 
    at_nodal[j] /= proximal_tets[j]
    
  return at_nodal

##################################################
## DIFFERENTIAL AND TANGENTIAL GROWTH FUNCTIONS ##
##################################################c

@njit(parallel=True)
def differential_term_tetrahedron(dist_2_surf, cortex_thickness, tets, n_tets, gr):
  """
  Calculates global shear modulus for white and gray matter for each tetrahedron

  Args:
  dist_2_surf (array): distance to surface for each node
  cortex_thickness (float): thickness of growing layer
  tets (array): tetrahedrons index
  n_tets (int): number of tetrahedron
  gr (array): yes/no growth mask for each node.

  Returns:
  gm (array): differential growth term depending on tetrahedron initial position
  """
  gm = np.zeros(n_tets, dtype=np.float64)

  for i in prange(n_tets):
    gm[i] = 1.0/(1.0 + math.exp(10.0*(0.25*(dist_2_surf[tets[i,0]] + dist_2_surf[tets[i,1]] + dist_2_surf[tets[i,2]] + dist_2_surf[tets[i,3]])/cortex_thickness - 1.0)))*0.25*(gr[tets[i,0]] + gr[tets[i,1]] + gr[tets[i,2]] + gr[tets[i,3]])

  return gm

@njit(parallel=True)
def differential_term_nodal(dist_2_surf, cortex_thickness, n_nodes, gr):
  """
  Calculates global shear modulus for white and gray matter for each tetrahedron

  Args:
  dist_2_surf (array): distance to surface for each node
  cortex_thickness (float): thickness of growing layer
  tets (array): tetrahedrons index
  n_tets (int): number of tetrahedron
  gr (array): yes/no growth mask for each node.

  Returns:
  gm (array): differential growth term depending on tetrahedron initial position
  """
  gm_nodal = np.zeros(n_nodes, dtype=np.float64)

  for i in prange(n_nodes):
    gm_nodal[i] = gr[i]/(1.0 + math.exp(10.0*(dist_2_surf[i]/cortex_thickness - 1.0))) # grey/white matter ponderation term

  return gm_nodal

@njit(parallel=True)
def tangential_growth_coefficient_nodal(n_nodes, at_nodal, gm_nodal): 
  """ 
  Calculate the nodal (differential) tangential growth term g(y,t) for all nodes. 
  For visualization. 
  """
  g_nodal = np.zeros(n_nodes, dtype=np.float64)

  for i in prange(n_nodes):
    g_nodal[i] = 1 + at_nodal[i] * gm_nodal[i] # tangential expansion gradient term
  
  return g_nodal

@jit(nopython=True)
def growth_tensor_tangen(tet_norms, gm, at, tan_growth_tensor, n_tets):
    '''
    Calculate relative (relates to dist_2_surf) tangential growth factor G
    '''
    A = np.zeros((n_tets,3,3), dtype=np.float64)
    A[:,0,0] = tet_norms[:,0]*tet_norms[:,0]
    A[:,0,1] = tet_norms[:,0]*tet_norms[:,1]
    A[:,0,2] = tet_norms[:,0]*tet_norms[:,2]
    A[:,1,0] = tet_norms[:,0]*tet_norms[:,1]
    A[:,1,1] = tet_norms[:,1]*tet_norms[:,1]
    A[:,1,2] = tet_norms[:,1]*tet_norms[:,2]
    A[:,2,0] = tet_norms[:,0]*tet_norms[:,2]
    A[:,2,1] = tet_norms[:,1]*tet_norms[:,2]
    A[:,2,2] = tet_norms[:,2]*tet_norms[:,2]

    gm = np.reshape(np.repeat(gm, 9), (n_tets, 3, 3))
    at = np.reshape(np.repeat(at, 9), (n_tets, 3, 3))
    #identity = np.resize(identity, (n_tets, 3, 3)) // not compatible numba, but apparently broacasting similar
    tan_growth_tensor = np.identity(3) + (np.identity(3) - A) * gm * at
    #tan_growth_tensor = (1 + gm * at) * np.identity(3) + (- gm * at) * A # g * Id + (1 - g) * N0XN0
    #tan_growth_tensor = (1 + gm * at) * (np.identity(3) - A) + A # g * (Id - N0XN0) + N0XN0 --> G is expressed in terms of tangential (g) and radial growth (1) coefficients.

    return tan_growth_tensor

@jit(nopython=True, parallel=True)
def growth_directions(tet_norms, n_tets):
    
    A = np.zeros((n_tets,3,3), dtype=np.float64)
    B = np.zeros((n_tets,3,3), dtype=np.float64)
    
    for tet in prange(n_tets):
        A[tet,0,0] = tet_norms[tet,0]*tet_norms[tet,0]
        A[tet,0,1] = tet_norms[tet,0]*tet_norms[tet,1]
        A[tet,0,2] = tet_norms[tet,0]*tet_norms[tet,2]
        A[tet,1,0] = tet_norms[tet,0]*tet_norms[tet,1]
        A[tet,1,1] = tet_norms[tet,1]*tet_norms[tet,1]
        A[tet,1,2] = tet_norms[tet,1]*tet_norms[tet,2]
        A[tet,2,0] = tet_norms[tet,0]*tet_norms[tet,2]
        A[tet,2,1] = tet_norms[tet,1]*tet_norms[tet,2]
        A[tet,2,2] = tet_norms[tet,2]*tet_norms[tet,2] 
        
        B[tet] = np.identity(3) - A[tet]
    
    return A, B

@njit(parallel=True)   
def growth_tensor(n_tets, tan_growth_tensor, B, gm, at): 
    
    gm = np.reshape(np.repeat(gm, 9), (n_tets, 3, 3))
    at = np.reshape(np.repeat(at, 9), (n_tets, 3, 3))
    
    tan_growth_tensor = np.identity(3) + B * gm * at
    
    return tan_growth_tensor

""" 
################
### ARCHIVED ###
################

@jit
def growthTensor_homo(G, n_tets, GROWTH_RELATIVE, t):
 
  #Calculates homogenous growth factor G
  #Args:
  #G (array): normals for each tetrahedron
  #n_tets (int): number of tetrahedrons
  #GROWTH_RELATIVE (float): constante set at simulation level
  #t (float): current time of simulation
  #Returns:
  #G (array): homogenous growth factor G

  for i in prange(n_tets):
    G[i] = 1.0 + GROWTH_RELATIVE*t

  return G

# Calculate homogeneous growth factor G (2nd version)
@jit
def growthTensor_homo_2(G, n_tets, GROWTH_RELATIVE):
  for i in prange(n_tets):
    G[i] = GROWTH_RELATIVE

  return G

@jit
def growthTensor_relahomo(gm, G, n_tets, GROWTH_RELATIVE, t):

  #Calculates cortical layer (related to dist_2_surf) homogenous growth factor G

  for i in prange(n_tets):
  #G[i] = np.full((3, 3), gm*GROWTH_RELATIVE)
    G[i] = 1.0 + GROWTH_RELATIVE*t*gm

  return G 
  
  """