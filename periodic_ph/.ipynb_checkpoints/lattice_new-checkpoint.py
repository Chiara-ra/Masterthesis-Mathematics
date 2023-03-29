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
 


