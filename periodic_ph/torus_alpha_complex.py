# libraries
import numpy as np
import numpy.linalg as la
import sympy as sp
import gudhi as gd
from . import simplex_classes as sc


## Auxiliary functions for torus_filtration()


### Duplicating points to neighbouring cells
# Given points in a unit cell, this function duplicates them by translating to all neighbouring cells.  
# This is an auxiliary step to calculating the _periodic_ $\alpha$-filtration. 

def torus_copy(points, a=1,b=1,c=1):
    """
    Takes numpy array points of N points on axbxc cube
    and creates 8 copies surrounding it.
    
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


### Other functions


def setup_unit_complex(points,max_alpha_square=float("inf"),a=1,b=1,c=1):
    """
    Setup for periodic_fitration() where the gudhi package is used
    to create an alpha filtration on the duplicated points. 
    """
    points_3x3x3 = torus_copy(points,a,b,c)

    # alpha-complex gets generation by gudhi
    alpha_complex = gd.AlphaComplex(points_3x3x3)
    simplex_tree = alpha_complex.create_simplex_tree(max_alpha_square=max_alpha_square)
    filtration = simplex_tree.get_filtration()   
    coords_cuboid = [alpha_complex.get_point(i) for i in range(len(points_3x3x3))]

    # SWITCH: axbxc --> 1x1x1
    coords_unit = [[x/a, y/b, z/c] for [x,y,z] in coords_cuboid]

    return coords_unit, filtration



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


def check_bound(point):
    """
    Takes point with x,y,z coordinates.
    Checks if point lies in cuboid [0,1) x [0,1) x [0,1).
    If yes, outputs True, else False.
    """
    x = point[0]
    y = point[1]
    z = point[2]
    
    return (    x >= 0 and x < 1 
            and y >= 0 and y < 1 
            and z >= 0 and z < 1)

def equiv_num(x1,x2):
    """
    Calculates 'x1-x2 mod 1', where we always want to take 
    the representative with the smallest absolute value. 
    """
    z1 = abs((x1 - x2)%1)
    z2 = abs(z1 - 1)
    return min(z1,z2)


def identification_list(S0_list, S0, coords, eps = 1e-5):
    """
    Input:
        S0_list ... naive filtration list  of 0-simplices ([3],3) as given by gudhi
        S0      ... list of Simplex0D objects generated from S0

    Output:
        identify_list ... list of lists with entries [i,j], 
                           meaning that the ith vertex has to be matched to the jth Simplex0D object
    """
    # build identification list for later reference when building higher simplices
    identify_list = []
    
    """
    !
    This loop below could be made much more efficient!
    """

    for i in range(len(S0_list)):
        simp, filt_value = S0_list[i]
        coord = np.array(coords[i])

        for Simplex in S0: # looking for the point in the unit cell this corresponds to              
            other_coord = Simplex.coords

            if ((equiv_num(coord[0],other_coord[0])<eps) and 
                (equiv_num(coord[1],other_coord[1])<eps) and
                (equiv_num(coord[2],other_coord[2])<eps)):

                identify_index = Simplex.index_total

        identify_list.append([i, identify_index])
    return identify_list


def order_vertex(indices, coords):
    """
    Takes two vertices and outputs them by lexicographical order.
    
    Input:
        indices ... list [n0, n1], where ni is the integer filtration index of the ith vertex, identifying it uniquely 
        coords  ... list of lists [[x0, y0, z0], [x1, y1, z1]] containing the coordinates of both points.
        
    Output:
        [nl, nr]   ... list of indices in lexicographical order
        [xl,yl,zl] ... coordinates of left vertex
        [xr,yr,zl] ... coordinates of right vertex 
    
    """
    
    [n0, n1] = indices
    [x0, y0, z0] = np.around(coords[0],5)
    [x1, y1, z1] = np.around(coords[1],5)
    # we round to avoid ordering errors due to "same" being not the same
    
    
    if x0 < x1 or (x0 == x1 and y0 < y1) or (x0 == x1 and y0 == y1 and z0 < z1):
        # 0 is left
        left  = 0
        right = 1

    else:
        # 1 is left
        left  = 1
        right = 0
    
    # left vertex
    nl = indices[left]
    xl = coords[left][0]
    yl = coords[left][1]
    zl = coords[left][2]

    # right vertex
    nr = indices[right]
    xr = coords[right][0]
    yr = coords[right][1]
    zr = coords[right][2]
    return [nl, nr], [xl,yl,zl], [xr,yr,zr]


def crossing_vector(coord):
    """
    Takes coordinates coord=[x,y,z], which lie in unit_cube
    OR any of the 17 farther-right adjecent cells (up, up-right, right, low-right or low).
    
    Assumes that this coordinate corresponds to lexicographically second vertex of a 1-simplex and
    calculates respective crossing vector of 1-simplex. 
    """
   
    
    x = coord[0]
    y = coord[1]
    z = coord[2]
    
    dtype=np.int64
    vector = np.zeros(3,dtype=dtype)
    
    if x >= 1:
        vector += np.array([1,0,0],dtype=dtype)
    if y < 0 or y >= 1:
        vector += np.array([0,1,0],dtype=dtype) * int(np.sign(y))
    if z < 0 or z >= 1:
        vector += np.array([0,0,1],dtype=dtype) * int(np.sign(z))
    
    return np.array(vector)



### Creation of Simplex objects



def create_S0(S0_list, coords):
    """
    Input:
        S0_list ... List of 0-simplices as generated by gudhi, 
        after having split them by dimension
    Output: 
        S0 ... List of Simplex0D objects containing the combinatorical representation,
        the integer filtration value, the continuous filtration value 
        and the geometric coordinate of the point
    """

    int_filt_value = 0
    S0 = []

    for i in range(len(S0_list)):
        simp, filt_value = S0_list[i]

        # check if 0-simplex lies in main cell
        coord = coords[i]
        if check_bound(coord): # point lies inside main cell
            S0.append(sc.Simplex0D([int_filt_value], 
                                            int_filt_value, 
                                            filt_value, 
                                            coord))
            int_filt_value +=1
    return S0



def create_S1(S1_list, S0, coords, identify_list):
    """
    Input: 
        Input:
        S1_list ... List of 1-simplices as generated by gudhi, 
        after having split them by dimension
        S0 ...List of Simplex0D objects of the filtration
        coords ... list of coordinates of vertices ordered by integer filtration value
        identify_list ... identification list for periodically copied points as given by identification_list()
        
    Output: 
        S1 ... List of Simplex1D objects containing the combinatorical representation,
        the integer filtration value, the continuous filtration value,
        the crossing vector as numpy array and the vertices constituting the boundary
        in lexicographical order
    """


    # note that this is not the actual integer filtration value 
    # and this will get corrected later on
    int_filt_value = len(S0) 
    S1 = []


    for i in range(len(S1_list)):
        simp, filt_value = S1_list[i]

        # check what the leftmost vertex is
        [n_left, n_right], coord_left, coord_right = order_vertex(simp, [coords[simp[0]], coords[simp[1]]])
        
        # Check if leftmost vertex is in unit cell
        if check_bound(coord_left):

            # Calculate crossing vector
            cross_vec = crossing_vector(coord_right)
            
            # identify the old vertices with the newly numerated ones!!!!! 
            # both for simplex, as well as for ordered vertices

            n_left_new  = (identify_list[n_left])[1]
            n_right_new = (identify_list[n_right])[1]

            int_ordered_simp = sorted([n_left_new,n_right_new])
            lex_ordered_simp = [S0[n_left_new], S0[n_right_new]]
            
            S1.append(sc.Simplex1D(int_ordered_simp, 
                                   int_filt_value, 
                                   filt_value, 
                                   cross_vec,
                                   lex_ordered_simp
                                  ))

            int_filt_value +=1
    return S1


def find_2Simplex_boundary(points,coords,S1, eps):
    """
    Input:
        points ... integer filtration values of vertices
        coords ... coordinates (in unit cell and its duplicates) of vertices
        S1     ... list of Simplex1D objects
        
    Output:
        bound_1, bound_2, bound_3 ... three Simplex1D objects from S1 which are formed by these points
    """
    
    """
    ! ! ! 
    This has not yet been adapted to 3D (shift only consideres x and y)
    I just included the z coordinate into the old code so that everything runs. 
    Make sure this is done correctly!!
    ! ! !
    """
    
    
    p1,p2,p3 = points
    p1_coord, p2_coord, p3_coord = coords
    
    # first boundary element from p1 to p2
    verts_1  = sorted([p1, p2])
    cv_1 = crossing_vector(p2_coord)

    # second boundary element from p1 to p3
    verts_2  = sorted([p1, p3])
    cv_2 = crossing_vector(p3_coord)

    # third boundary element from p2 to p3
    verts_3  = sorted([p2, p3])
    cv_3 = crossing_vector(p3_coord - crossing_vector(p2_coord))
    
    
    for j in range(len(S1)):
        
        Simplex = S1[j]
        verts = Simplex.verts
        cv    = Simplex.cv

            
        if (verts[0] == verts_1[0]) and (verts[1] == verts_1[1]) and (np.linalg.norm(cv-cv_1)<eps):
            bound_1 = Simplex
        if (verts[0] == verts_2[0]) and (verts[1] == verts_2[1]) and (np.linalg.norm(cv-cv_2)<eps):
            bound_2 = Simplex
        if (verts[0] == verts_3[0]) and (verts[1] == verts_3[1]) and (np.linalg.norm(cv-cv_3)<eps):
            bound_3 = Simplex
       
    return [bound_1, bound_2, bound_3]


def create_S2(S2_list, S1, S0, coords, identify_list, eps):
    """
    Input: 
        Input:
        S2_list ... List of 2-simplices as generated by gudhi, 
        after having split them by dimension
        S1 ... List of Simplex1D objects of the filtration
        coords ... list of coordinates of vertices ordered by integer filtration value
        identify_list ... identification list for periodically copied points as given by identification_list()
        
    Output: 
        S2 ... List of Simplex2D objects containing the combinatorical representation,
        the integer filtration value, the continuous filtration value,
        and the vertices constituting the boundary
        in lexicographical order
    """

    S2 = []
    int_filt_value = len(S0) + len(S1)
    for i in range(len(S2_list)):
        simp, filt_value = S2_list[i]

        # how can we find the correct 2-simplices?

        # if we have the vertices of the 2-simplex, we can look for 1-simplices which have two of these boundary points
        # since we only keep 2-simplices with their left-most point in the main cell
        # we can calculate the crossing vectors and then uniquly identify 2 out of 3 boundary elements

        # for the last edge, we have to 
        # - take the middle and the rightmost point,
        # - shift them such that the middle point is now in the main cell
        # - calculate the crossing vector of these shifted points
        # this is then the crossing vector of the last edge, and we can again identify
        # the correct 1-simplex using the vertices and the crossing vector


        # ordering vertices
        a0 = simp[0]
        b0 = simp[1]
        c0 = simp[2]
        a0_coord = coords[a0]
        b0_coord = coords[b0]
        c0_coord = coords[c0]

        [a1,b1], a1_coord, b1_coord = order_vertex([a0,b0], [a0_coord, b0_coord])
        [a2,c1], a2_coord, c1_coord = order_vertex([a1,c0], [a1_coord, c0_coord])
        [b2,c2], b2_coord, c2_coord = order_vertex([b1,c1], [b1_coord, c1_coord])

        # a2 is the leftmost, b2 is the middle and c2 is the rightmost coordinate, 
        # or a2 < b2 < c2 in lexicographical order


        # only if a2 is in the main cell do we continue
        if check_bound(a2_coord): 

            # the vertices are not yet in the naming convention we have chosen
            # so we rename them using identify_list
            for i in range(len(identify_list)):
                new_name = (identify_list[i])[1]
                if i == a2:
                    a2 = new_name
                if i == b2:
                    b2 = new_name
                if i == c2:
                    c2 = new_name


            bound_1, bound_2, bound_3 = find_2Simplex_boundary([a2,b2,c2],[a2_coord,b2_coord,c2_coord], S1, eps = eps)

            S2.append(sc.Simplex2D(sorted([a2,b2,c2]), 
                                   int_filt_value, 
                                   filt_value, 
                                   [S0[a2],S0[b2],S0[c2]], 
                                   [bound_1, bound_2, bound_3]))
            int_filt_value += 1
    return S2




def boundary4_from_rest(boundary1,boundary2,boundary3):
    """
    Given a 3-simplex, it has 4 boundary elements. 
    If we know 4 boundary 2-simplices, we can compare their respective edges.
    Collecting those that only show up for one 2-simplex each, 
    we can uniquly identify forth last 2-simplex by its boundary.
    """

    boundary4 = []
    # they each have one edge (1-simplex) which was not in the original list we searched for
    for edge in boundary1:
        if (edge not in boundary2) and (edge not in boundary3):
            boundary4.append(edge)

    for edge in boundary2:
        if (edge not in boundary1) and (edge not in boundary3):
            boundary4.append(edge)

    for edge in boundary3:
        if (edge not in boundary1) and (edge not in boundary2):
            boundary4.append(edge)
    return boundary4



def order_4_vertices(simp, coords):
    """
    Input:
        simp   ... 3-simplex constituted of 4 vertices, for example simp=[1,2,3,4]
        coords ... coordinates of the 4 vertices in the same order as in simp
    Ouptut:
        Integer filtration values and coordinates of the 4 points in 
        lexicographical order as dictated by their coordinates.
    """
    [n1, n2, n3, n4] = simp
    coord1 = coords[n1]
    coord2 = coords[n2]
    coord3 = coords[n3]
    coord4 = coords[n4]

    # ordering vertices
    a0 = n1
    b0 = n2
    c0 = n3
    d0 = n4
    a0_coord = coord1
    b0_coord = coord2
    c0_coord = coord3
    d0_coord = coord4

    [a1,b1], a1_coord, b1_coord = order_vertex([a0,b0], [a0_coord, b0_coord])
    [a2,c1], a2_coord, c1_coord = order_vertex([a1,c0], [a1_coord, c0_coord])
    [a3,d1], a3_coord, d1_coord = order_vertex([a2,d0], [a2_coord, d0_coord])
    # a3 is minimum
    [b2,c2], b2_coord, c2_coord = order_vertex([b1,c1], [b1_coord, c1_coord])
    [b3,d2], b3_coord, d2_coord = order_vertex([b2,d1], [b2_coord, d1_coord])
    # b3 is second smallest
    [c3,d3], c3_coord, d3_coord = order_vertex([c2,d2], [c2_coord, d2_coord])

    return a3,b3,c3,d3,a3_coord,b3_coord,c3_coord,d3_coord



def create_S3(S3_list, S2, S1, S0, coords, identify_list, eps):
    """
    Input: 
        Input:
        S3_list ... List of 3-simplices as generated by gudhi, 
        after having split them by dimension
        S2 ... List of Simplex1D objects of the filtration
        coords ... list of coordinates of vertices ordered by integer filtration value
        identify_list ... identification list for periodically copied points as given by identification_list()
        
    Output: 
        S3 ... List of Simplex3D objects containing the combinatorical representation,
        the integer filtration value, the continuous filtration value,
        and the edges constituting the boundary,
        and all vertices in lexicographical order
    """
    
    """
    !
    This code is just copied from S2.
    The code for S3 is not yet written (optional, since we don't need 3 simplices in my thesis) ...
    """
    
    
    S3 = []
    int_filt_value = len(S0) + len(S1) + len(S2)
    for i in range(len(S3_list)):
        simp, filt_value = S3_list[i] 
        
        a,b,c,d,a_coord,b_coord,c_coord,d_coord = order_4_vertices(simp, coords)
        
        # only if a3 is in the main cell do we continue
        if check_bound(a_coord): 
            
            # the vertices are not yet in the naming convention we have chosen
            # so we rename them using identify_list
           
            a = identify_list[a][1]
            b = identify_list[b][1]
            c = identify_list[c][1]
            d = identify_list[d][1]
            
            

            # we calculate 3 of the 6 1-simplices making up the 1-dim faces of our simplex
            boundary1 = find_2Simplex_boundary([a,b,c],[a_coord,b_coord,c_coord],S1, eps = eps)
            boundary2 = find_2Simplex_boundary([a,b,d],[a_coord,b_coord,d_coord],S1, eps = eps)
            boundary3 = find_2Simplex_boundary([a,c,d],[a_coord,c_coord,d_coord],S1, eps = eps)
            boundary4 = boundary4_from_rest(boundary1,boundary2,boundary3)


            # these edges form the last face
            # we now search for this last face

            # then we search for the 3 2-simplices that can be idenfified using these faces
            for simp2 in S2:
                S_bound = simp2.boundary
                if boundary1 == S_bound:
                    face1 = simp2
                elif boundary2 == S_bound:
                    face2 = simp2
                elif boundary3 == S_bound:
                    face3 = simp2
                elif boundary4 == S_bound:
                    face4 = simp2 
                    """
                    !
                    In this particular case, since boundary 4 is not outputted using the function we are used to
                    it might be that the == does not detect the permutation we have. 
                    !
                    """

            S3.append(sc.Simplex3D(sorted([a,b,c,d]), 
                                   int_filt_value, 
                                   filt_value, 
                                   [S0[a],S0[b],S0[c],S0[d]], 
                                   [face1,face2,face3,face4]
                                  )
                     )
            int_filt_value += 1
            del face1,face2,face3,face4
    return S3


### Functions for generating the (correct) torus filtration



def generate_pfilt(S0,S1,S2,S3):
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
    
    for simplices in [S0, S1, S2, S3]:
        periodic_filt += [(simplex.verts, simplex.index_cont) for simplex in simplices]

    periodic_filt.sort(key = lambda some_tuple: some_tuple[1])

    return periodic_filt



def reorder_by_cont(S0,S1,S2,S3,periodic_filt):
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
    
    for simplices in [S0,S1,S2,S3]:
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
        

        
        
        
        
## Check legality of input data




def remove_duplicate_points(points):
    """
    Input: Nx3 numpy array
    Output: Mx3 numpy array, with all duplicate rows removed, hence M<N
    """
    # round inputs to 5 decimal places
    points = np.round(points, 5)
    return np.unique(points, axis = 0)



def check_domain(points, a, b, c):
    """
    Input: Nx3 numpy array, floats a,b,c
    Output: True if all points in array lie in [0,a)x[0,b)x[0,c)
    """
    in_domain = True
    for row in points:
        if (row[0] < 0) or (row[0] >= a):
            in_domain = False
            break
        elif (row[1] < 0) or (row[1] >= b):
            in_domain = False
            break
        elif (row[2] < 0) or (row[2] >= c):
            in_domain = False
            break
    return in_domain



def preprocess_points(points, a, b, c, eps = 1e-5):
    """
    Raise actual errors in this cases
    """
    if (not check_domain(points, a, b, c)):
        print("Some of the given points lie on the right boundary of the cell (have coordinate values x=a, y=b or c=c)")
        return
    
    points = remove_duplicate_points(points)
    
    # add noise in case the points are not in general position
    points += np.random.random((points.shape[0],3))*1e-3
    
    return points




## The final torus_filtration()


   
def torus_filtration(points, max_alpha_square=float("inf"), a=1, b=1, c=1):

    """
    Input:
        a, b, c            ... side lengths of cell [0,a) x [0,b) x [0,c)
        points             ... numpy array of size (N,3) containing N points in cell
        max_alpha_square   ... maximum alpha (squared) value of alpha-filtration
        
    Output:
        periodic_filt ... list of filtration elements of the form ([1,2,38],index_cont)
        S             ... list of lists [S0,S1,S2,S3], each sub-list containing all Simplex objects of given dimension
    """
   
    eps = 1e-5 # for identification purposes (rounding errors)
    
    
    
    points = preprocess_points(points, a, b, c, eps = eps)
    
    coords_unit, filtration = setup_unit_complex(points, max_alpha_square, a, b, c)
    
    # split unidentified simplices by dimension
    simp0, simp1, simp2, simp3 = dim_split(filtration)
    
    
    
    # 0-Simplices ---------------------------------
    S0 = create_S0(simp0, coords=coords_unit)
    identify_list = identification_list(S0_list=simp0, 
                                        S0=S0, 
                                        coords=coords_unit, 
                                        eps = eps)
    
    
    # 1-Simplices ---------------------------------
    S1 = create_S1(S1_list=simp1, 
                   S0=S0, 
                   coords=coords_unit, 
                   identify_list=identify_list) 
    
    # 2-Simplices ---------------------------------
    S2 = create_S2(S2_list=simp2, 
                   S1=S1, 
                   S0=S0, 
                   coords=coords_unit, 
                   identify_list=identify_list, 
                   eps = eps)
    
    
    # 3-Simplices
    S3 = create_S3(S3_list=simp3, 
                   S2=S2, 
                   S1=S1, 
                   S0=S0, 
                   coords=coords_unit, 
                   identify_list=identify_list, 
                   eps=eps) 

    
    
    """
    !!!
    We need a comment about what this step does
    """
    periodic_filt = generate_pfilt(S0=S0, S1=S1, S2=S2, S3=S3)
    
    
    # re-assign correct integer filtration value
    reorder_by_cont(S0=S0,S1=S1,S2=S2,S3=S3, periodic_filt=periodic_filt)
    
    
    # rescaling to axbxc cuboid
    for simp in S0:
        simp.transf_coord(a,b,c)
    
    
    simplex_objects = [S0, S1, S2, S3]
    
    
    # Calculate connected components of 0 simplices
    calc_cc(simplex_objects)
    
    
    

    
    return periodic_filt, simplex_objects
    
