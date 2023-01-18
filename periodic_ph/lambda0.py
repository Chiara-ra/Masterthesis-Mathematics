# libraries
import numpy as np
import numpy.linalg as la
import sympy as sp
from .lattice import reduce_spanning_set_3d as rss3

# Calculating the evolution of $\Lambda_0$

## Auxiliary functions for Lambda_0_evolution()

def vol_0d():
    return 1

def vol_1d(vector):
    return sp.Matrix(vector).norm()

def vol_2d(vector1, vector2):
    return sp.Matrix(vector1).cross(sp.Matrix(vector2)).norm()

def vol_3d(vector1, vector2, vector3):
    return sp.Matrix([vector1, vector2, vector3]).det()


def volume_pd(vector_matrix):
    r""" Takes list of p linearly independant vectors 
    and calculates the p-dimensional volume they span.
    
    """
    p = la.matrix_rank(vector_matrix)
    
    if p == 0:
        det_p = vol_0d()
    if p == 1:
        det_p = vol_1d(vector_matrix[:,0])
    if p == 2:
        det_p = vol_2d(vector_matrix[:,0], vector_matrix[:,1])
    if p == 3:
        det_p = vol_3d(vector_matrix[:,0], vector_matrix[:,1], vector_matrix[:,2])
    
    return abs(det_p), p


## Event Types (Objects)


# these are the final object types that are outputted

class Merger:
    def __init__(self, timestep, old_comp, new_comp, cv_list, dim, det_abs, det_rel):
        self.time_index = timestep
        self.old_component = old_comp
        self.new_component = new_comp
        self.dim = dim
        self.det_abs = det_abs
        self.det_rel = det_rel 
        self.crossing_vector_list = cv_list
    
    
class StaticSublattice:
    def __init__(self, time, component, cv_list, basis_matrix, dim, det_abs, det_rel):
        self.time_index = time
        self.component = component
        self.crossing_vector_list = cv_list
        self.basis_matrix = basis_matrix
        self.dim = dim
        self.det_abs = det_abs
        self.det_rel = det_rel    

        
## Pair Addition

class PairAddition:
    def __init__(self, pair, time, trafo_matrix):
        self.pair = pair
        self.time_index = time
        self.trafo_matrix = trafo_matrix
        
        # dim 0
        self.dead_component = None
        self.surviving_component = None
        
        # dim 1
        self.component = None
        self.crossing_vector = None
        
        
        self.basis_matrix = None
        self.det_abs = None
        self.det_rel = None
        self.vector_dim = None
    
    
    def check_birth(self):
        """
        Returns True if persistence pair is born at self.time_index, otherwise returns False. 
        """
        return ((self.pair.dim == 1) and 
                (self.pair.lifespan_int[0] == self.time_index)
               )
    
    def check_death(self):
        """
        Returns True if persistence pair is born at time t, otherwise returns False. 
        """
        return ((self.pair.dim == 0) and 
                (self.pair.lifespan_int[1] == self.time_index)
               )
    def check_cv_nontrivial(self):
        """
        Returns True if (integer) vector is non-zero.
        """
        return abs(la.norm(self.crossing_vector)) > 0.9
    
    def generate_basis_matrix(self, crossing_vector_list):
        
        self.basis_matrix = np.zeros((3,3), dtype=np.int32)
        for i, vector in enumerate(crossing_vector_list):
            self.basis_matrix[:,i] = vector
            
        self.basis_matrix = np.dot(self.trafo_matrix, self.basis_matrix)
     
    
    def check_sublattice_change(self, last_event):
        """
        ! ! !
        here we might get floating point errors when compaing the determinants
        make this more foolproof
        """
        return (last_event.det_abs != self.det_abs or 
                last_event.dim != self.vector_dim
               )
    
    def cycle_birth(self, past_total_evolution):
        self.component = self.pair.component[self.time_index]
        self.crossing_vector = self.pair.calc_cv()

        if self.check_cv_nontrivial(): 
            old_crossing_vectors = past_total_evolution[self.component][-1].crossing_vector_list
            cv_list = rss3(old_crossing_vectors, self.crossing_vector)
            self.generate_basis_matrix(cv_list)
            self.det_abs, self.vector_dim = volume_pd(self.basis_matrix)
            self.det_rel = self.det_abs / la.det(self.trafo_matrix) 

            self.conditional_new_sublattice(past_total_evolution[self.component], cv_list)
            
            
    def component_death(self, past_total_evolution):
        self.dead_component = self.pair.component[self.time_index-1]
        self.surviving_component = self.pair.component[self.time_index]


        dead_comp_cvs = past_total_evolution[self.dead_component][-1].crossing_vector_list
        surviving_comp_cvs = past_total_evolution[self.surviving_component][-1].crossing_vector_list

        cv_list = surviving_comp_cvs.copy()

        for vec in dead_comp_cvs:
            cv_list = rss3(cv_list, vec)

        self.generate_basis_matrix(cv_list)
        self.det_abs, self.vector_dim = volume_pd(self.basis_matrix)
        self.det_rel = self.det_abs / la.det(self.trafo_matrix) 

        merger = Merger(self.time_index, 
                        self.dead_component, 
                        self.surviving_component, 
                        cv_list, 
                        self.vector_dim, 
                        self.det_abs, 
                        self.det_rel)

        past_total_evolution[self.dead_component].append(merger)
        past_total_evolution[self.surviving_component].append(merger)

        self.conditional_new_sublattice(past_total_evolution[self.surviving_component], cv_list)

        
    def add_pair(self, Lambda0_list):
        if self.check_birth():
            self.cycle_birth(Lambda0_list)

        
        elif self.check_death(): # 0-homology (connected component) dies
            self.component_death(Lambda0_list)

            
    def conditional_new_sublattice(self, past_comp_evolution, cv_list):
        if self.check_sublattice_change(past_comp_evolution[-1]):
            past_comp_evolution.append(StaticSublattice(self.time_index, 
                                                        self.component, 
                                                        cv_list, 
                                                        self.basis_matrix, 
                                                        self.vector_dim, 
                                                        self.det_abs, 
                                                        self.det_rel))



            
## The final Lambda_0_evolution()

    
    
def Lambda_0_evolution(p_filt, N, persistence_pairs, trafo_matrix):
    """
    Takes filtration data (p_filt of torus filtration data, N number of initial vertices, 
    persistence_pairs and trafo_matrix dependent on initial cell) and returns Lambda0_list,
    which contains one sublist for each initial vertex (component), made up of Merger and 
    StaticSublattice objects, where each one corresponds to an event in the lifespan of that component. 
    """
    Lambda0_list = [[StaticSublattice(0, 
                                      component, 
                                      [], 
                                      np.zeros((3,3),dtype=np.int32), 
                                      0, 
                                      1, 
                                      1/la.det(trafo_matrix))] 
                    for component in range(N)]
    
                        
    for t in range(len(p_filt)):
        # search for pair that acts at time t
        for pair in persistence_pairs:
            pair_addition = PairAddition(pair, t, trafo_matrix)
            pair_addition.add_pair(Lambda0_list)
            
    return Lambda0_list

