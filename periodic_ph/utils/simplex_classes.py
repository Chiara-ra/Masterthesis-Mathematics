# libraries
import numpy as np
from dataclasses import dataclass, field


# CHANGE SIMPLEX CLASSES TO DATACLASSES


@dataclass
class Simplex:
    verts: list
    index_dim: int
    index_cont: int
    
    index_total: int = field(init=False, repr=True)
    dim: int = field(init=False, repr=True)
    
    def __post_init__(self):
        self.index_total = self.index_dim
        self.dim = len(self.verts)-1

    def update_index_total(self, new_val):
        self.index_total = new_val

        
        
        
class Simplex0D(Simplex):
    def __init__(self, vertices, int_filt, cont_filt, coordinates):
        super().__init__(vertices, int_filt, cont_filt)
        self.coords    = coordinates
        self.component = None
        
        # self.component is List of 0-simplices in form of int_filt values
        # Make .component a method, not something that is initialised right at the beginning

    def create_cc(self, cc_list):
        self.component = cc_list
        
    def transf_coord(self, a,b,c):
        self.coords = [self.coords[0]*a,self.coords[1]*b,self.coords[2]*c]

        
        
        
# --------------
"""
Is it really necessary to have .ordered_vertices?
With the names of the vertices (.vertices) we already have all the information we need to extract these 2-simplices ...

Also, wouldn't it be more robust if, instead of saving one SimplexnD within the other, to save their .index_dim ?
This way we can easily extract this information ...
"""
class Simplex1D(Simplex):
    def __init__(self, vertices, int_filt, cont_filt, crossing_v, ordered_vertices):
        super().__init__(vertices, int_filt, cont_filt)

        self.cv = crossing_v.astype(np.int32)
        self.ord_verts = ordered_vertices # list of Simplex0D elements
        self.vert0 = ordered_vertices[0]  # Simplex0D
        self.vert1 = ordered_vertices[1]  # Simplex0D
        
        
        
# --------------


class Simplex2D(Simplex):
    def __init__(self, vertices, int_filt, cont_filt, ordered_vertices, boundary):
        super().__init__(vertices, int_filt, cont_filt)
        self.ord_verts = ordered_vertices
        self.boundary = boundary # boundary is a list of 3 objects of the class Simplex1D
        
        
        
        
# --------------


class Simplex3D(Simplex):
    def __init__(self, vertices, int_filt, cont_filt, ordered_vertices, boundary):
        super().__init__(vertices, int_filt, cont_filt)
        self.ord_verts = ordered_vertices
        self.boundary = boundary # boundary is a list of 4 objects of the class Simplex2D       
        

# --------------

class Persistence_Pair:
    def __init__(self, dimension, interval_int):
        self.dim        = dimension
        self.lifespan_int   = interval_int # life span of class in integer steps
        
        self.lifespan_cont  = None
        self.rep       = None # list of Simplex-objects whose sum gives representative
        self.component = None # we first have to call self.calc_cc
        #self.cv         = None # we first have to call self.calc_cc
    

    def int2cont_interval(self, torus_filtration):
        birth = self.lifespan_int[0]
        death = self.lifespan_int[1]
        a_cont = torus_filtration[birth][1]

        if death != float("inf"):
            b_cont = torus_filtration[death][1]
        else:
            b_cont = float("inf")
        self.lifespan_cont  = (a_cont, b_cont)
        
        
        
    def calc_birth_rep(self, V, simplex_objects):
        if self.dim == 0:
            self.rep = [simplex_objects[0][self.lifespan_int[0]]]
        
        elif self.dim == 1:
            # The j-th column of V encodes the columns in ∂ 
            # that add up to give the j-th column in R. 

            j = self.find_representative_column(simplex_objects)
            self.collect_simplices_from_column(V[:,j], simplex_objects)
 

    def find_representative_column(self, simplex_objects):
        # this for-loop searches for the simplex in simplex_objects 
        # that triggers the birth of the given peristence pair
        for simp in simplex_objects[self.dim]:
            if simp.index_total == self.lifespan_int[0]:
                # index within the dimension + size of previous dimensions = total index in V
                j = simplex_objects[self.dim].index(simp) + sum([len(simplex_objects[k]) for k in range(self.dim)])
                break
        return j

    
    
    def collect_simplices_from_column(self, jcolumn, simplex_objects):
        # We want to take the birth representative,
        # meaning we look at the column of the cycle being born

        rep = []
        simplex_objects_flat =  [simp for sublist in simplex_objects for simp in sublist]

        for i in range(len(simplex_objects_flat)): 
            if jcolumn[i] == 1: 
                simp = simplex_objects_flat[i]
                rep.append(simp)
        self.rep = rep
    
    
    def calc_cc(self): # right now this only works for 1-dim pps. Generalise this for 0 to 3. 
        if self.dim == 0:
            self.component = self.rep[0].component
            
        elif self.dim == 1:
            first_rep = self.rep[0]
            self.component = first_vert = first_rep.vert0.component

    def calc_cv(self):
        V = np.zeros(3,dtype=np.int32)
        
        edge = self.rep[0] # Simplex1D object
        V += edge.cv       # add crossing vector of edge to V
        end_vert = edge.vert1
        pp_red = self.rep[1:]

        while len(pp_red) != 0:
            # look for correct representative that has end_vert as one of its vertices
            for i in range(len(pp_red)):
                edge = pp_red[i]

                # check if we are looking at the correct edge
                if end_vert in edge.ord_verts:
                    # check if orientations are lining up or not
                    if end_vert == edge.ord_verts[0]:
                        ori = +1
                        end_vert = edge.vert1
                    else:
                        ori = -1
                        end_vert = edge.vert0

                    V += ori * edge.cv
                    pp_red.pop(i)

                    break # breaks for loop, but while loop continues

                if i == len(pp_red)-1:
                    """
                    ! ! !
                    create a real error message here!
                    """
                    print("Could not find next edge.") 
                    pp_red = []
                    break
        return V


