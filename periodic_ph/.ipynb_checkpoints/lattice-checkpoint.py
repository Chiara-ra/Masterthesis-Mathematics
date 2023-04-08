"""
This module reduces a list of 0 up to 3 vectors, with one additional vector, 
to a spanning set of the smallest common superlattice. 

The expected input is a list of 3-coordinate numpy arrays. 
Internally, calculations are performed with sympy. 
The output is again a list of 3-coordinate numpy arrays.
"""


# libraries
import numpy as np
import numpy.linalg as la
from math import gcd
import sympy as sp



def reduce_spanning_set_3d(old_vectors, new_vector):
    """
    Takes list of 3-dim numpy integer vectors (old_vector)
    and a 3-dim numpy integer vector (new_vector).
    
    Outputs a minimal spanning set of 3-dim numpy integer vectors 
    of the the smallest common superlattice (spanning_set). 
    """
    
    # first check if there is anything to do at all
    if abs(la.norm(new_vector)) < .5:
        spanning_set = old_vectors
    
    elif len(old_vectors) == 0:
        spanning_set = [new_vector]
    
    elif len(old_vectors) == 1:
        spanned_area = la.norm(np.cross(old_vectors[0], new_vector))
        if spanned_area < .5: # collinear
            [x, y] = common_superlattice_1d(sp.Matrix(old_vectors[0]), sp.Matrix(new_vector))
            spanning_set = [x*old_vectors[0] + y*new_vector]
        else:
            spanning_set = [old_vectors[0], new_vector]
            
    elif len(old_vectors) == 2:
        all_vectors_matrix = np.transpose(np.array(old_vectors+[new_vector]))
        spanned_volume = abs(la.det(all_vectors_matrix))
        
        if spanned_volume < .5: #coplanar
            from sympy.abc import a, b
            from sympy.solvers.solvers import solve
            linear_equation = old_vectors[0]*a + old_vectors[1]*b - new_vector
            solution = solve(linear_equation, a, b)
            combination_integer = False
            try:
                if solution[a].q == 1 and solution[b].q == 1:
                    combination_integer = True
            except:
                pass

            if combination_integer:
                spanning_set = old_vectors
            else:
                coefficients = common_superlattice_2d(sp.Matrix(old_vectors[0]), 
                                                      sp.Matrix(old_vectors[1]), 
                                                      sp.Matrix(new_vector))
                
                spanning_set = build_spanning_set(coefficients, old_vectors, new_vector)
                
        else:
            spanning_set = old_vectors + [new_vector]
    
    elif len(old_vectors) == 3:
        coefficients = common_superlattice_3d(sp.Matrix(old_vectors[0]), 
                                              sp.Matrix(old_vectors[1]), 
                                              sp.Matrix(old_vectors[2]),
                                              sp.Matrix(new_vector))
      
        spanning_set = build_spanning_set(coefficients, old_vectors, new_vector)
                
    else:
        raise ValueError("Too many vectors in list of previous spanning set (old_vectors).")
        
    for i, vector in enumerate(spanning_set):
        spanning_set[i] = vector.astype(np.int32)
    return spanning_set


def build_spanning_set(coefficients, old_vectors, new_vector):
    n = len(old_vectors)
    spanning_set = []
    for i in range(n):
        vector = coefficients[i][n]*new_vector
        for j in range(n):
            vector = vector + coefficients[i][j]*old_vectors[j]
        spanning_set.append(vector)
    return spanning_set



    
def common_superlattice_1d(vector0, vector1):
    """
    Takes two sympy vectors (rational or integer) and returns coefficients of their gcd. 
    """
    
    # look for a non-zero coordinate
    index = nonzero_entry(vector0)
    
    # take least common denominator of the index'th entries
    lcd = lcd_matrix(sp.Matrix([vector0[index], vector1[index]]))

    a = vector0[index]*lcd
    b = vector1[index]*lcd
    x, y = gcdExtended(a, b)
    return [x, y]

    

def nonzero_entry(vector):
    for count, value in enumerate(vector):
        if value != 0:
            return count
    return 0 # in case the vector is the zero vector

def lcd_matrix(M):
    """
    Takes sympy matrix with rational coefficients and 
    returns lcm of denominators of all entries
    """
    lcm_denom = M[0,0].q
    for entry in M[:]:
        lcm_denom = lcm(lcm_denom, entry.q)
    return lcm_denom


def gcdExtended(a, b): 
    # Base Case 
    if a == 0 :  
        return 0,1

    x1, y1 = gcdExtended(b%a, a) 

    # Update x and y using results of recursive call 
    x = y1 - (b//a) * x1 
    y = x1 

    return x, y

def lcm(a, b):
    return abs(a*b) // gcd(a, b)



# -----------------------------------------------------



 
    
def common_superlattice_2d(vector0, vector1, new_vector):
    """
    Takes three sympy integer vectors, 
    the third one lying in the plane defined by the first two.
    
    Calculates coefficients of the integer basis spanning 
    the common superlattice of all three vectors. 
    """
    if len(new_vector) == 3:
        # skip one coordinate
        index  = np.argmax(np.abs(vector0.cross(vector1)))
        vector0 =    skip_coordinate(vector0, index)
        vector1 =    skip_coordinate(vector1, index)
        new_vector = skip_coordinate(new_vector, index)
    
    basis_matrix = sp.Matrix([vector0.T,vector1.T]).T
    primitive, primitive_coeff_basis, primitive_coeff_new_vector = find_parallel_primitive(basis_matrix, new_vector)
    
    # project basis_matrix to v_orth
    projected_basis = orthogonal_projection(basis_matrix, primitive)
    [x, y] = common_superlattice_1d(projected_basis[:,0], projected_basis[:,1])
    return [x, y, 0], list(primitive_coeff_basis) + [primitive_coeff_new_vector]

    
def skip_coordinate(vector, n):
    """
    Takes in 3-component vector and returns corresponding 2-component vector
    given by removing the nth coordinate (n = 0,1,2).
    """
    if n == 0:
        new_vector = [vector[1],vector[2]]
    elif n == 1:
        new_vector = [vector[0],vector[2]]
    elif n == 2:
        new_vector = [vector[0],vector[1]]
    
    return sp.Matrix(new_vector)



def orthogonal_projection(vectors, projector):
    r"""
    Projects vectors in sympy matrix 'vectors' orthogonaly onto 
    the orthogonal plane defined by the vector 'projector'.
    """
    projected_vectors =[]
    for i in range(vectors.shape[1]):
        projectee = vectors[:,i]
        projected_vector = projectee - (projectee.dot(projector)/projector.dot(projector))*projector
        projected_vectors.append(projected_vector[:])
    return sp.Matrix(projected_vectors).T


def gcd3(a,b,c):
    return gcd(gcd(a,b),c)


def find_parallel_primitive(basis, u):
    """
    Takes a basis of integer sympy vectors and an additional sympy vector u
    Outputs the primitive v in the lattice parallel to u and its coefficients in the basis and u.
    """
    # based on lemma 3.1
    G = basis
    #Gi = G**(-1)
    Gi = G.inv()
    denom_list = []
    p = 1
    for entry in Gi:
        if entry.q not in denom_list:
            denom_list.append(entry.q)
            p *= entry.q
    
    up = p*Gi*u
    v = G*up
    if len(u)==3:
        v = v / gcd3(up[0],up[1],up[2])
        coeff = up / gcd3(up[0],up[1],up[2]) #dividng by gcd of coefficients gives us the shortest parallel vector in the span of the original basis
    if len(u)==2:
        v = v / gcd(up[0],up[1])
        coeff = up / gcd(up[0],up[1])
    
    # to find shortest vector in the span of the new basis (four vectors), 
    # we find out the ratios of the two vectors 
    # get them to integer by multiplying with denominators
    # calculate the gcd of the resulting numbers
    # then divide the gcd (or the vector associated to it) by the denominator again
    for i in range(len(u)):
        if u[i]!=0:
            num1 = v[i]
            num2 = u[i]
            denom = num1.q*num2.q
            a, b = gcdExtended(num1*denom, num2*denom)
            new_gcd = int(a*num1*denom + b*num2*denom)
            v = v/num1*new_gcd
            coeff = coeff * a
            u_coeff = b
            break
    
    return v, coeff, u_coeff

# -----------------------------------------------------


def common_superlattice_3d(vector0, vector1, vector2, new_vector):
    """
    Takes three linearly independent sympy integer vectors, 
    with a fourth additional one whose not an integer combination of the others.
    
    Calculates coefficients of the integer basis spanning 
    the common superlattice of all four vectors. 
    """
    basis_matrix = sp.Matrix([vector0.T, vector1.T, vector2.T]).T
    primitive, primitive_coeff_basis, primitive_coeff_new_vector = find_parallel_primitive(basis_matrix, new_vector)
    projected_vectors = orthogonal_projection(basis_matrix, new_vector)
    
    basis0_coeff = [0,0,0,0]
    basis1_coeff = [0,0,0,0]
    basis2_coeff =  list(primitive_coeff_basis) + [primitive_coeff_new_vector]
    
    primitive_parallel_to_basis = False
    for i in range(3):
        if projected_vectors[:,i].norm() == 0:
            basis0_coeff[(i+1)%3] = 1
            basis1_coeff[(i+2)%3] = 1
            primitive_parallel_to_basis = True
            
    if not primitive_parallel_to_basis:
        # reduce to 2 component vectors and get new basis and new "new vector"
        vector0_2d, vector1_2d, new_vector_2d, re_index = setup_2d_induction(projected_vectors, new_vector)
        
        basis0_coeff_induct, basis1_coeff_induct = common_superlattice_2d(vector0_2d, vector1_2d, new_vector_2d)
        
        for i in range(3):
            basis0_coeff[re_index[i]] = basis0_coeff_induct[i]
            basis1_coeff[re_index[i]] = basis1_coeff_induct[i]
            # new_vector component stays 0 

    return basis0_coeff, basis1_coeff, basis2_coeff
    
    
    
def skip_coord(vector,n=1):
    """
    Takes in 3-component vector and returns corresponding 2-component vector
    given by removing the nth coordinate (n = 0,1,2).
    """
    if n == 0:
        new_vector = [vector[1],vector[2]]
    elif n == 1:
        new_vector = [vector[0],vector[2]]
    elif n == 2:
        new_vector = [vector[0],vector[1]]
    
    return sp.Matrix(new_vector)


def skip_correct_coord(matrix, skipping_direction):
    ind = np.argmax(np.abs(skipping_direction))
    column1_skipped = skip_coord(matrix[:,0],ind)
    column2_skipped = skip_coord(matrix[:,1],ind)
    column3_skipped = skip_coord(matrix[:,2],ind)
    
    return sp.Matrix([column1_skipped[:], column2_skipped[:], column3_skipped[:]]).T

    
def setup_2d_induction(projected_vectors, normal_vector):
    """
    Takes 3x3 sympy matrix with vectors projected to orthogonal complement of u.
    Skips one coordinate and shuffles vectors to be in the setup "basis + additional vector".
    Returns skipped vectors and shuffle dictionary. 
    """
    reduced_vectors = skip_correct_coord(projected_vectors, normal_vector)
    
    # search for vector to make new "extra"
    for i in range(3): #this just remember the variable "i"
        indices = [(i+1)%3,(i+2)%3]
        if reduced_vectors[:, indices].det() != 0:
            true_i=i
            break
        
        elif i==2: # this is outside of range, meaning no subset is basis
            raise ValueError("No linearly independant subset found in set of projected vectors.")
    
    # determine lcm of denomiators
    lcm_denom = lcd_matrix(reduced_vectors)
    
    re_index = {0: indices[0], 1: indices[1], 2: true_i}
    
    vector0    = reduced_vectors[:,re_index[0]]*lcm_denom
    vector1    = reduced_vectors[:,re_index[1]]*lcm_denom  
    new_vector = reduced_vectors[:,re_index[2]]*lcm_denom
                             
    return vector0, vector1, new_vector, re_index


