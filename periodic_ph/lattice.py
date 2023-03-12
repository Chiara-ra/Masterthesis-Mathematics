# libraries
import numpy as np
import numpy.linalg as la
from math import gcd
import sympy as sp



# math helper functions ------

def gcd3(a,b,c):
    return gcd(gcd(a,b),c)

def gcdExtended(a, b): 
    # Base Case 
    if a == 0 :  
        return b,0,1

    gcd, x1, y1 = gcdExtended(b%a, a) 

    # Update x and y using results of recursive 
    # call 
    x = y1 - (b//a) * x1 
    y = x1 

    return gcd,x,y

def lcm(a, b):
    return abs(a*b) // gcd(a, b)


# ------


def L31_parallel_vectors(basis, u):
    # function that takes basis (which generates sublattice Z) and vector v
    # outputs vector u as linear combination of basis elements, which is parallel to v
    # also outputs or at least explicitely calculates the coefficients involved
    # input is numpy arrays, output is sympy matrices
    G = basis
    Gi = G**(-1)
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
            new_gcd, a, b = gcdExtended(num1*denom, num2*denom)
            x = new_gcd / sp.sympify(num1.p*num2.q)
            v = v/num1*new_gcd
            coeff = coeff * a
            u_coeff = b
            break
    
    return v, coeff, u_coeff



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

    

def skip_corr_coord(Pv, u):
    ind = np.argmax(np.abs(u))
    Pv1_2 = skip_coord(Pv[:,0],ind)
    Pv2_2 = skip_coord(Pv[:,1],ind)
    Pv3_2 = skip_coord(Pv[:,2],ind)
    
    return sp.Matrix([Pv1_2[:],Pv2_2[:],Pv3_2[:]]).T



def skip_coord(vec,n=1):
    """
    Takes in 3-component vector and returns corresponding 2-component vector
    given by removing the nth coordinate (n = 0,1,2).
    """
    if n == 0:
        new_vec = [vec[1],vec[2]]
    elif n == 1:
        new_vec = [vec[0],vec[2]]
    elif n == 2:
        new_vec = [vec[0],vec[1]]
    
    return sp.Matrix(new_vec)



def lcm_mx(M):
    """
    Takes sympy 2x2 matrix with rational coefficients and 
    returns lcm of denominators of all entries
    """
    lcm_denom = M[0,0].q
    for entry in M[:]:
        lcm_denom = lcm(lcm_denom,entry.q)
    return lcm_denom







def proj2induct_2d(Pv,u):
    """
    Takes 3x3 sympy matrix with vectors projected to orthogonal complement
    and returns 
    - a 2x2 sympy matrix (picks two linearly independant column vectors, skips one coordinate) 
    - 'additional' vector to perform next orth projection
    """
    Pv_2d = skip_corr_coord(Pv,u)
    
    # search for vector to make new "extra"
    for i in range(3): #this just remember the variable "i"
        ind = [(i+1)%3,(i+2)%3]
        if Pv_2d[:,ind].det() != 0:
            true_i=i
            break
        
        elif i==2: # this is outside of range, meaning no subset is basis
            print("something went wrong! No linear ind. subset found.")
            """
            !!!!!
            Put in Real Error Message
            """
    
    
    u_2d = Pv_2d[:,true_i]
    basis_2d = Pv_2d[:,ind]
    
    # determine lcm of denomiators
    lcm_denom = lcm_mx(Pv_2d)
    
    # making stuff integer
    v_2d, coeff_basis_2d, coeff_u_2d = L31_parallel_vectors(basis_2d*lcm_denom, u_2d*lcm_denom)
    v_2d /= lcm_denom
    

    # calculate what coefficients of v are in the ORDER OF Pv, not Pv_2d!
    prim_coeff = [0,0,0]
    prim_coeff[ind[0]] = coeff_basis_2d[0]
    prim_coeff[ind[1]] = coeff_basis_2d[1]
    prim_coeff[true_i] = coeff_u_2d
    
    return basis_2d, u_2d, true_i, v_2d, prim_coeff
    

def nonzero_entry(vec):
    for i in range(len(vec)):
        if vec[i]!=0:
            break
    return i



def gcd_of_collinear_vectors(vec_mx):
    """
    Takes two sympy column vectors from sympy matrix and returns their gcd, 
    as well as the coefficients to build the gcd
    
    vec_mx       ... 2x2 matrix containing two column-vectors
    new_1d_basis ... gcd of vectors in vec_mx
    x, y         ... coefficients w.r.t. vectors in vec_mx
    """
    
    # look for a non-zero coordinate
    i = nonzero_entry(vec_mx[:,0])
    
    # take lcm of denominators of the i'th entries
    denoms = lcm_mx(vec_mx[i,:])

    a = vec_mx[i,0]*denoms
    b = vec_mx[i,1]*denoms
    c, x, y = gcdExtended(a, b)
    return x, y



def conv_np2sp(array):
    return sp.Matrix(array)

def conv_sp2np(matrix):
    return np.array(matrix).astype(np.int32)



def common_superlattice_3d(basis_np, u_np):
    """
    Takes a 3x3 numpy matrix containing an integer basis,
    as well as an additional numpy vector u.
    Calculates integer basis of the superlattice spanned
    by basis and u as 3x3 numpy matrix. 
    """
    
    basis = conv_np2sp(basis_np)
    u = conv_np2sp(u_np)
    
    # caculate primitive of u in superlattice
    v, coeff, u_coeff = L31_parallel_vectors(basis,u)
    
    # project basis to v_orth
    Pbasis = orthogonal_projection(basis,v)
    # If vector in P(basis) is 0, then original vector is parallel to u  
    # Pick gcd of both, which is be v.
    for i in range(3):
        if Pbasis[:,i].norm() == 0:
            base1_3d = basis[:,(i+1)%3]
            base2_3d = basis[:,(i+2)%3]
            break
            
        elif i==2:
            # reduce to 2 component vectors and get new basis and new u
            Pbasis_2d, u_2d, u_2d_index, v_2d, v_2d_coeff = proj2induct_2d(Pbasis,v) 
            
            # project basis_2d to v_2d_orth
            PPbasis = orthogonal_projection(Pbasis_2d,v_2d)
            
            # calculate gcd of projected vectors
            x, y = gcd_of_collinear_vectors(PPbasis)
            

            # calculate basis of 3d space
            base1_3d = basis[:,(u_2d_index+1)%3]*x + basis[:,(u_2d_index+2)%3]*y


            base2_3d = 0*u # <-- u-component is 0
            for i in range(3):
                base2_3d += v_2d_coeff[i] * basis[:,i]
    
    base3_3d = coeff[0]*basis[:,0]+coeff[1]*basis[:,1]+coeff[2]*basis[:,2]+u_coeff*u
    # is equivalent to: base3_3d=v

    new_basis = sp.Matrix([base1_3d[:], base2_3d[:], base3_3d[:]]).T
    
    return conv_sp2np(new_basis)
    
    
    
    
def common_superlattice_2d(basis_np, u_np):
    """
    Takes a 2x2 or 3x2 numpy matrix containing an integer spanning set,
    as well as an additional numpy vector u.
    Calculates integer basis of the superlattice spanned
    by basis and u as 2x2 or 3x2 numpy matrix. 
    """
    dim = len(u_np)
    
    basis = conv_np2sp(basis_np)
    u = conv_np2sp(u_np)
    
    if dim == 3:
        # skip one coordinate
        ind  = np.argmax(np.abs(basis[:,0].cross(basis[:,1])))
        vec1 = skip_coord(basis[:,0],ind)
        vec2 = skip_coord(basis[:,1],ind)
        u = skip_coord(u,ind)
        basis = sp.Matrix([vec1.T,vec2.T]).T
        
    v, coeff, coeff_u = L31_parallel_vectors(basis, u)
    
    # project basis to v_orth
    Pbasis = orthogonal_projection(basis,v)
    

    # calculate gcd of projected vectors
    x, y = gcd_of_collinear_vectors(Pbasis)
    # calculate new minimal spanning set
    new_vec1 = basis_np[:,0]*x + basis_np[:,1]*y
    new_vec2 = coeff[0]*basis_np[:,0] + coeff[1]*basis_np[:,1] + coeff_u*u_np

    span_set = np.transpose(np.array([new_vec1, new_vec2]))
    
    return span_set.astype(np.int32)
    
    
def common_superlattice_1d(vector1, vector2):
        """
        Takes vector1, which is basis of 1-dim lattice, and additional collinear vector2.
        Returns a new vector which spans the common suplattice.
        """
        vector_matrix  = sp.Matrix([vector1, vector2]).T
        x, y = gcd_of_collinear_vectors(vector_matrix)
        return x*vector1 + y*vector2






def reduce_spanning_set_3d(old_vecs, new_vec):
    """
    Takes in 3 dim vectors encoded as numpy integer vectors in list old_vecs
    and a 3 dim integer vector as numpy integer vector new_vec.
    
    Outputs a minimal set of spanning vectors of the lattice
    as numpy integer vectors in list span_set. 
    """
    
    """
    ! ! !
    At this point, our spanning set often gets updated, 
    even though we have just chosen a different representation of the vectors.
    Check that this does not happen (unnecessary work and confusion later on if change has happened)
    
    """
    
    l = len(old_vecs)
    old_vecs_mx = np.transpose(np.array(old_vecs))
    all_vecs_mx = np.transpose(np.array(old_vecs+[new_vec]))
    span_set = []
    
    
    if abs(la.norm(new_vec)) < 0.9:
        span_set = old_vecs
    
    elif l==0:
        span_set.append(new_vec)
    
    elif l==1:
        vec1 = old_vecs[0]
        
        
        if abs(la.norm(np.cross(vec1,new_vec))) < 0.1:
            gcd_vector = common_superlattice_1d(vec1, new_vec)
            span_set.append(gcd_vector)
    
        
        else:
            # else take [vec1, new_vec]
            span_set.append(vec1)
            span_set.append(new_vec)
        
    elif l==2:
        vec1 = old_vecs[0]
        vec2 = old_vecs[1]
        if abs(la.det(all_vecs_mx)) < 0.1:
            
            from sympy.abc import a, b
            coeffs = sp.solvers.solvers.solve(vec1*a+vec2*b-new_vec,a,b)
            is_int_comb = False
            for key,value in coeffs.items():
                try:
                    if value.q == 1:
                        is_int_comb = True
                except:
                    pass
            if is_int_comb:
                span_set = old_vecs
            else:
                # if new_vec lies in span of {vec1, vec2}, use lemma
                span_set_mx = common_superlattice_2d(old_vecs_mx, new_vec)
                span_set.append(span_set_mx[:,0])
                span_set.append(span_set_mx[:,1])
        
        else:
            # else take [vec1, vec2, new_vec]
            span_set.append(vec1)
            span_set.append(vec2)
            span_set.append(new_vec)
        
    elif l==3:
        # use common_superlattice_3d()
        span_set_mx = common_superlattice_3d(old_vecs_mx,new_vec)
        span_set.append(span_set_mx[:,0])
        span_set.append(span_set_mx[:,1])
        span_set.append(span_set_mx[:,2])
        
    else: 
        # this should not happen
        # INSERT REAL ERROR
        print("old_vecs contains too many vectors")
        
    return span_set



