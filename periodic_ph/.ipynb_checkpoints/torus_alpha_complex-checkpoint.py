# libraries
import numpy as np
import sympy as sp
import gudhi as gd
from .utils import simplex_classes as sc
from .utils.create_simplex_objects import build_torus_complex



eps = 1e-5 # for identification purposes (rounding errors)


# Duplicating points to neighbouring cells
def torus_copy(points, a=1,b=1,c=1):
    """
    Takes numpy array of N points on axbxc cube
    and creates 26 copies surrounding it.
    
    Returns new_points, containing original points and 9+8+9=26 offset copies,
    so Nx27 points in total.
    """
    
    N = np.shape(points)[0]
    # first index  = which axbxc cuboid we are working on
    # second index = index of corresponding original point
    # third index  = x, y, z coordinates of point
    new_points = np.zeros((27,N,3)) 

    i = 0 # to index the cuboid
    for x in [-a,0,a]:
        for y in [-b,0,b]:
            for z in [-c,0,c]:
                transl = np.array([[x,y,z] for dummy in range(N)])
                new_points[i,:,:] = points + transl
                i += 1
            
    return new_points.reshape(-1,3)




# setup complex with extra points from which we calculate the torus complex


def create_auxiliary_complex(points, a=1, b=1, c=1):
    """
    Setup for periodic_fitration() where the gudhi package is used
    to create an alpha filtration on the duplicated points. 
    """
    points_3x3x3 = torus_copy(points, a, b, c)

    # alpha-complex gets generation by gudhi
    alpha_complex = gd.AlphaComplex(points_3x3x3)
    simplex_tree = alpha_complex.create_simplex_tree(max_alpha_square=float("inf"))
    filtration = simplex_tree.get_filtration()   
    coords_cuboid = [alpha_complex.get_point(i) for i in range(len(points_3x3x3))]

    # SWITCH: axbxc --> 1x1x1
    coords_unit = [[x/a, y/b, z/c] for [x,y,z] in coords_cuboid]

    return coords_unit, filtration



### Other functions


def dim_split(filt):
    """
    Takes list of simplices 'filt' with filtration values, 
    i.e. with elements ([3,2], 0.123)
    Splits filt by dimension of elements.
    Outputs the 4 seperate lists, one for each dimension 0 to 3.
    """
    simps_split = [[] for i in range(4)]
    
    for (simp, val) in filt:
        p = len(simp)-1
        simps_split[p].append((simp,val))
        
    return simps_split[0], simps_split[1], simps_split[2], simps_split[3]




### Functions for generating the (correct) torus filtration




def generate_pfilt(simplex_objects):
    """
    Takes the lists of simplex objects Si and generates the 
    mixed-dimension periodic filtration, so with all the simplices, but only saved
    through their vertices and their continuous filtration value.
    """
    
    """
    ! 
    This needs to be optimised / made prettier
    !
    """
    
    periodic_filt = []
    
    for simplices in simplex_objects:
        periodic_filt += [(simplex.verts, simplex.index_cont) for simplex in simplices]

    periodic_filt.sort(key = lambda some_tuple: some_tuple[1])

    return periodic_filt



def reorder_by_cont(simplex_objects, periodic_filt):
    """
    Since the integer filtration value is wrong (first storted by dimension, 
    only then by actuall filtration value), this needs to be reordered.
    This function takes the Simplex, looks where it is in the periodic filtration,
    and reassignes this placement. 
    """
    # if there are some simplices, which combinatorially are equal and which are born at the same time,
    # we need to pick an arbitrary order
    # so we replace in dummy_filt already picked elements with None
    dummy_filt = periodic_filt.copy()
    
    for simplices in simplex_objects:
        update_index(simplices, dummy_filt)



def update_index(simplices, periodic_filt):
    for simplex in simplices:
        int_filt = periodic_filt.index((simplex.verts, simplex.index_cont))
        periodic_filt[int_filt] = None
        
        simplex.update_index_total(int_filt)
        
        
def calc_cc(simplex_objects):
    """
    Input:
        simplex_objects ... list of lists [S0,S1,S2,S3,S4] of all Simplex objects in periodic filtration
        
    No output. Changes the connected component attribute of all Simplex0D objects in S. 
    Hence if v is in S0, then v.component gets updated to the correct list of connected components at each time step.
    """
    N = sum([len(simplex_objects[i]) for i in range(4)]) # length of whole filtration
    N0 = len(simplex_objects[0])
    
    component_evolution = np.zeros((N0,N), dtype = np.int64)

    # from t=0 to t=N0-1 the vertices get born
    vertex_creation = [[-1 for t in range(vert)] + [vert for t in range(N0-vert)] for vert in range(N0)]
    # the -1 component is instead of None, because we are using a numpy array this time
    component_evolution[:,:N0] = vertex_creation
    
    last_t = N0-1
    for edge in simplex_objects[1]:
        t = edge.index_total
        v0_comp  = component_evolution[edge.vert0.index_total, last_t]
        v1_comp  = component_evolution[edge.vert1.index_total, last_t]
        dead_comp = max(v0_comp, v1_comp)
        life_comp = min(v0_comp, v1_comp)
        
        # fill up evolution up to and including t with most recent information
        for s in range(last_t+1,t+1):
            component_evolution[:,s] = component_evolution[:,s-1]
        
        if dead_comp != life_comp: # merger has happened
            for comp in range(N0):
                if component_evolution[comp, t] == dead_comp:
                    component_evolution[comp, t] = life_comp
                else:
                    component_evolution[comp, t] = component_evolution[comp, t-1]
        last_t = t
        
    for s in range(last_t+1,N):
        component_evolution[:,s] = component_evolution[:,s-1]
        
    for n in range(0,N0):
        comp_list = [None if x == -1 else x for x in component_evolution[n,:]]
        simplex_objects[0][n].create_cc(comp_list) # assign cc lists to vertices
        

        
        
        

def int2cont(filtration):
    """
    Constructs dict from filtration list containing with key 
    being integer time steps and values being corresponding continuous times.
    """
    return {i: filtration[i][1] for i in range(len(filtration))}
    




class TorusComplex:
    def __init__(self, filtration, coordinates):
        self.auxiliary_filtration = list(dim_split(filtration))
        self.simplex_objects = [[] for dim in range(4)]
        self.identification_list = None
        self.coordinates = coordinates
        self.torus_filtration = None
        
     
    def build_complex(self):
        build_torus_complex(self)
        
    def generate_torus_filtration(self):
        self.torus_filtration = generate_pfilt(self.simplex_objects)
        
    def reorder_by_continuous_times(self):
        # re-assign correct integer filtration value
        reorder_by_cont(self.simplex_objects, self.torus_filtration)
        
    def rescale_cell(self, a, b, c):
        # rescaling to axbxc cuboid
        for simplex in self.simplex_objects[0]:
            simplex.transf_coord(a, b, c)
            
    def calculate_components(self):
        calc_cc(self.simplex_objects)


        
        

## The final torus_filtration()


def create_torus_complex(points, a=1, b=1, c=1):

    """
    Input:
        a, b, c            ... side lengths of cell [0,a) x [0,b) x [0,c)
        points             ... numpy array of size (N,3) containing N points in cell
        
    Output:
        list of filtration elements of the form ([1,2,38], index_cont)
        list of lists, each sub-list containing all simplex objects of fixed dimension
    """
    coords_unit, filtration = create_auxiliary_complex(points, a, b, c)
    
    torus_complex = TorusComplex(filtration, coords_unit)
    torus_complex.build_complex()
    torus_complex.generate_torus_filtration()
    torus_complex.reorder_by_continuous_times()
    torus_complex.rescale_cell(a, b, c)
    torus_complex.calculate_components()
    
    return torus_complex.torus_filtration, torus_complex.simplex_objects
    
