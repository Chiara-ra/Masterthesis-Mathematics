# libraries
import numpy as np
from . import simplex_classes as sc


# Calculating torus PH

# After the calculation of the periodic filtration, next comes calculating the periodic persistent homology. The algorithm here should not differ significantly from the standard ones. The only reason we are writing this from scratch is that pre-existing software is not made to handle our data, especialy in case of degenerate simplices. 

## Auxiliary functions for PH()

### Classical PH algorithms

def boundary_matrix(simplex_objects):
    """
    Takes list of subslists of Simplix type objects.
    Outputs a boundary matrix as numpy matrix.
    
    Note that the boundary matrix is sorted by dimension (their index in S[i]), not as given by .index_total
    """
    S0, S1, S2, S3 = simplex_objects
    
    # Initialise matrix of dimension s0+s1+s2(boundaries) x s1+s2+s3
    D = np.zeros((len(S0+S1+S2),len(S0+S1+S2+S3)))
    
    # The 0-Simplices all appear at the beginning and are already ordered by their usual index
 
    
    d = [len(simplex_objects[i]) for i in range(4)]
    
    for j in range(d[1]):
        for i in S1[j].verts:
            D[i,j+d[0]] += 1
    
    for j in range(d[2]):
        for i in range(d[1]):
            if S1[i] in S2[j].boundary:
                D[i+d[0],j+d[0]+d[1]] += 1
                
    for j in range(d[3]):
        for i in range(d[2]):
            if S2[i] in S3[j].boundary:
                D[i+d[0]+d[1],j+d[0]+d[1]+d[2]] += 1
    
   
    
    return np.mod(D,2)


def check_low(column):
    """ 
    Takes a column i.e. numpy 1dim array. 
    Outputs the index of the lowest non-zero element of the given column.
    If the column is all zero, it returns -1.
    """
    low = -1
    m = len(column)
    for i in range(0,m):
        if column[m-1-i]==1:
            low = m-1-i
            break
    return low


def column_reduction(D, exhaustive=False): 
    """
    Takes a binary boundary matrix and performs column reduction on it.
    Returns the reduced matrix "R", as well as a list of its pivot elements "pivots". 
    Zero columns are encoded in pivots as -1. 
    """
    
    
    """
    !!!!
    We want to implement exhaustive reduction at some point. 
    """
    # D is numpy array of size mxn
    (m,n) =  D.shape # rows, column
    R = D.copy()
    V = np.identity(n)
    
    pivots = []
    
    for i in range(0,n):
        # look at current pivot 
        low = check_low(R[:,i])
        
        # check if this is already a pivot element
        while (low in pivots) and (low != -1): 
            # while pivot is taken, perform matrix reduction on R
            
            j = pivots.index(low)
            R[:,i]= (R[:,i] + R[:,j]) % 2
            V[:,i]= (V[:,i] + V[:,j]) % 2
            
            # get new pivot
            low = check_low(R[:,i]) 
            
        pivots.append(low)
        
    
    return pivots, R, V



class TopologicalGroups:
    def __init__(self, dim=3):
        self.C = np.zeros(dim+1)
        self.Z = np.zeros(dim+1)
        self.B = np.zeros(dim+1)
        self.H = np.zeros(dim+1)
        self.dim = dim
        
    def calculate_group_dimensions(self, dim_list, pivots):
        m = 0
        for i in range(0,self.dim+1):
            # extract dimensions of vector spaces
            self.C[i]   = dim_list[i]
            self.Z[i]   = pivots[m:m+dim_list[i]].count(-1)
            self.B[i-1] = self.C[i] - self.Z[i]
            m += dim_list[i]

        for i in range(0,self.dim+1):
            self.H[i]   = self.Z[i] - self.B[i]

    
    
### Persistence Pairs

def calculate_int_persistence_pairs(D, pivots, simplex_objects_flat):
    r"""
    Calculates persistence pairs with integer timestep values.
    Formatted such that it can be plotted by gudhi: (dim, (a,b))
    """
    
    integer_pairs = IntegerPersistenceIntervals()
    integer_pairs.calculate_integer_intervals(D, pivots, simplex_objects_flat)
    
    return integer_pairs




class IntegerPersistenceIntervals:
    def __init__(self):
        self.list = []
        
    def calculate_integer_intervals(self, D, pivots, simplex_objects_flat):
        # calculate persistence pairs
        for j in range(D.shape[1]): # looking at the jth column
            # determine dimension of new simplex associated to jth row
            dim = simplex_objects_flat[j].dim

            # if jth column = 0 --> new cycle
            if pivots[j] == -1:
                self.append_infinite_pair(dim, simplex_objects_flat[j])
                
            # if jth column != 0 --> death of homology class (that must have been born previously)
            else:
                self.replace_infinite_pair(dim, simplex_objects_flat[pivots[j]], simplex_objects_flat[j])
        
        # sort by beginning of interval
        self.list.sort(key=lambda l: (l[1])[0])


    def append_infinite_pair(self, dim, simplex):
        a = simplex.index_total
        self.list.append((dim, (a,float("inf"))))

    def replace_infinite_pair(self, dim, birth_simplex, death_simplex):
        a = birth_simplex.index_total
        b = death_simplex.index_total
        
        try:
            self.list.remove((dim-1, (a, float("inf"))))
        except ValueError:
            print(f'Pair {(dim-1, (a, float("inf")))} has not yet been added to pp_list and hence cannot be removed.')
        else:
            self.list.append((dim-1, (a,b)))

            
            
            
            
def create_persistence_pairs_list(integer_pairs, torus_filtration, V, simplex_objects):
    persistence_pairs = []
    
    for (dim,(a,b)) in integer_pairs.list:
        pair = sc.Persistence_Pair(dim, (a,b))
        pair.int2cont_interval(torus_filtration)
        if dim == 0 or dim == 1:
            pair.calc_birth_rep(V, simplex_objects)
            pair.calc_cc()
        persistence_pairs.append(pair)
    
    del pair
    
    """
    ! ! !
    The Code below was created because at every time step, either a new pair is 
    created or dies. This was not the case, so I had to investigate the timsteps.
    I will leave this in for now, but it should be removed.
    
    Replace the text here with an error in case a number is skipped
    """
    dummy_list_1 = [pair.lifespan_int[0] for pair in persistence_pairs]
    dummy_list_2 = [pair.lifespan_int[1] for pair in persistence_pairs]
    dummy_list = dummy_list_1 + dummy_list_2
    dummy_list.sort()
    
    boo = False
    for i in range(len(torus_filtration)):
        if i not in dummy_list:
            print("this number was skipped:")
            print(i)
            boo = True
            break
    if boo == True:
        print("Number was skipped")
        return
        
    """
    Here this ends ...
    """
        
    persistence_pairs.sort(key=lambda pair: pair.lifespan_int[0])
    return persistence_pairs
  
    
    
def generate_persistence_pairs(D, V, pivots, torus_filtration, simplex_objects):
    simplex_objects_flat =  [simp for sublist in simplex_objects for simp in sublist]
    
    integer_pairs = calculate_int_persistence_pairs(D, pivots, simplex_objects_flat)
    persistence_pairs  = create_persistence_pairs_list(integer_pairs, torus_filtration, V, simplex_objects)
  
    return persistence_pairs


## The final PH()

def PH(torus_filtration, simplex_objects):
    """
    Takes list "torus_filtration" of naive torus filtration with continuous filtration values and
    list "simplex_objects" containing sublists of Simplex type objects, sorted by dimension.
    
    Returns the dimensions of Cn, Bn, Zn and Hn, 
    as well as persistence pairs "pp" and representatives "reps" of homology classes.
    """
    
    
    d=3 # this is the dimension we are working in
    
    D = boundary_matrix(simplex_objects)
    
    dim_list = [len(simplex_objects[i]) for i in range(0,d+1)]
    
    pivots, R, V = column_reduction(D)
    
    top_groups = TopologicalGroups()
    top_groups.calculate_group_dimensions(dim_list, pivots)
    
    persistence_pairs = generate_persistence_pairs(D, 
                                                   V, 
                                                   pivots, 
                                                   torus_filtration, 
                                                   simplex_objects)

    
    
    return top_groups, persistence_pairs



## Plotting torus persistent pairs


def plot_persistence_pairs(persistence_pairs):
    pair_list = [(pair.dim, pair.lifespan_cont) for pair in persistence_pairs]
    gd.plot_persistence_diagram(pair_list)
    
    
