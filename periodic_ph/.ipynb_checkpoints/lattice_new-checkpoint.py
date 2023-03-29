"""
This module reduces a list of 0 up to 3 vectors, with one additional vector, 
to a spanning set of the smallest common superlattice. 

The expected input is a list of 3-coordinate numpy arrays. 
Internally, calculations are performed with sympy. 
The output is again a list of 3-coordinate numpy arrays.
"""




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
        spanned_area = abs(la.norm(np.cross(old_vectors[0], new_vector)))
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
            try:
                if solution[a].q == 1 and solution[b].q == 1:
                    combination_integer = True
            except:
                pass

            if combination_integer:
                spanning_set = old_vectors
            else:
                [x0, y0, z0], [x1, y1, z0] = common_superlattice_2d(sp.Matrix(old_vectors[0]), 
                                                                    sp.Matrix(old_vectors[1]), 
                                                                    sp.Matrix(new_vector))
                spanning_set = [x0*old_vectors[0] + y0*old_vectors[1] + z0*new_vector,
                                x1*old_vectors[0] + y1*old_vectors[1] + z1*new_vector]
                
        else:
            spanning_set = old_vectors + [new_vector]
    
    elif len(old_vectors) == 3:
        coefficients = common_superlattice_3d(sp.Matrix(old_vectors[0]), 
                                                        sp.Matrix(old_vectors[1]), 
                                                        sp.Matrix(old_vectors[2])
                                                        sp.Matrix(new_vector))
        spanning_set = []
        for i in range(3):
            vector = coefficients[i][3]*new_vector
            for j in range(3):
                vector = coefficients[i][j]*old_vectors[j]
            spanning_set.append(vector)
                
    else:
        raise ValueError("Too many vectors in list of previous spanning set (old_vectors).")
        
    return spanning_set
 



    
def common_superlattice_1d(vector1, vector2):
    """
    Takes two sympy vectors (rational or integer) and returns coefficients of their gcd. 
    """
    
    # look for a non-zero coordinate
    index = nonzero_entry(vector1)
    
    # take least common denominator of the index'th entries
    lcd = lcd_matrix(sp.Matrix([vector1[index], vector2[index]]))

    a = vector1[index]*lcd
    b = vector2[index]*lcd
    x, y = gcdExtended(a, b)
    return [x, y]

    

def nonzero_entry(vector):
    for count, value in enumerate(vector):
        if value != 0:
            return count

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
        return b,0,1

    x1, y1 = gcdExtended(b%a, a) 

    # Update x and y using results of recursive 
    # call 
    x = y1 - (b//a) * x1 
    y = x1 

    return x, y

def lcm(a, b):
    return abs(a*b) // gcd(a, b)



# -----------------------------------------------------



 
    
def common_superlattice_2d(vector0, vector1, new_vector):
    """
    Takes three sympy integer vectors, 
    the third one lying the the plane defined by the first two.
    
    Calculates coefficients of the integer basis spanning 
    the common superlattice of all three vectors. 
    """
    if len(new_vector) == 3:
        # skip one coordinate
        index  = np.argmax(np.abs(vector1.cross(vector2)))
        vector0 =    skip_coordinate(vector0, index)
        vector1 =    skip_coordinate(vector1, index)
        new_vector = skip_coordinate(new_vector, index)
    
    basis = sp.Matrix([vector0.T,vector1.T]).T
    primitive, primitive_coeff_basis, primitive_coeff_new_vector = find_parallel_primitive(basis, new_vector)
    
    # project basis to v_orth
    projected_basis = orthogonal_projection(basis, primitive)
    
    [x, y] = common_superlattice_1d(projected_basis[:,0], projected_basis[:,1])
    
    return [x, y], primitive_coeff_basis + [primitive_coeff_new_vector]

    
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
    
    return sp.Matrix(new_vec)



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
