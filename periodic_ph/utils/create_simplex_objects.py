import numpy as np
import numpy.linalg as la
from . import simplex_classes as sc
from operator import itemgetter



eps = 1e-5 # for identification purposes (rounding errors)
   



# auxiliary functions




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




def order_vertices_lexicographically(vertex_list):
    """
    Input: vertex_list ... list of vertices [index, [x,y,z]]
    Output: sorted_list ... list of vertices ordered by x,y,z
    """
    return sorted(vertex_list, key=itemgetter(1))
    
    
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







def create_crossing_vector(coord):
    """
    Takes coordinates coord=[x,y,z], which lie in unit_cube
    OR any of the 17 farther-right adjecent cells (up, up-right, right, low-right or low).
    
    Assumes that this coordinate corresponds to lexicographically second vertex of a 1-simplex and
    calculates respective crossing vector of 1-simplex. 
    """
   
    
    x = coord[0]
    y = coord[1]
    z = coord[2]
    
    dtype = np.int64
    vector = np.zeros(3,dtype=dtype)
    
    if x >= 1:
        vector += np.array([1,0,0],dtype=dtype)
    if y < 0 or y >= 1:
        vector += np.array([0,1,0],dtype=dtype) * int(np.sign(y))
    if z < 0 or z >= 1:
        vector += np.array([0,0,1],dtype=dtype) * int(np.sign(z))
    
    return np.array(vector)











# create simplex objects

def create_torus_complex(torus_complex_object):
    for dim in range(4):
        torus_complex_object.simplex_objects[dim] = create_simplex_objects(dim, torus_complex_object)

        if dim == 0:
            torus_complex_object.identification_list = create_identification_list(torus_complex_object)
            
def create_identification_list(torus_complex_object):
    """
    Input:
        S0_list ... naive filtration list  of 0-simplices ([3],3) as given by gudhi
        S0      ... list of Simplex0D objects generated from S0

    Output:
        identification_list ... list of lists with entries [i,j], 
                           meaning that the ith vertex has to be matched to the jth Simplex0D object
    """
    S0_list = torus_complex_object.auxiliary_filtration[0]
    S0 = torus_complex_object.simplex_objects[0]
    coords = torus_complex_object.coordinates
    
    # build identification list for later reference when building higher simplices
    identification_list = []
    

    for i in range(len(S0_list)):
        simp, filt_value = S0_list[i]
        coord = np.array(coords[i])

        for Simplex in S0: # looking for the point in the unit cell this corresponds to              
            other_coord = Simplex.coords

            if equivalent_coordinates_mod1(coord, other_coord):
                identify_index = Simplex.index_total
                break

        identification_list.append(identify_index)
    return identification_list


def equivalent_coordinates_mod1(coordinate1, coordinate2):
    return (equiv_num(coordinate1[0], coordinate2[0]) and 
            equiv_num(coordinate1[1], coordinate2[1]) and
            equiv_num(coordinate1[2], coordinate2[2])
            )


def equiv_num(x1, x2):
    """
    Evaluates if 'x1==x2 mod 1', w.r.t rounding errors. 
    """
    z1 = abs((x1 - x2)%1)
    z2 = abs(z1 - 1)
    return (min(z1, z2) < eps)






def create_simplex_objects(dimension, torus_complex):

        simp0, simp1, simp2, simp3 = torus_complex.auxiliary_filtration

        if dimension == 0:
            simplices = create_S0(simp0, 
                                  torus_complex.coordinates)
        elif dimension == 1:
            simplices = create_S1(simp1, 
                                  torus_complex.simplex_objects[0], 
                                  torus_complex.coordinates, 
                                  torus_complex.identification_list) 

        elif dimension == 2:
            simplices = create_S2(simp2, 
                                  torus_complex.simplex_objects[0], 
                                  torus_complex.simplex_objects[1], 
                                  torus_complex.coordinates, 
                                  torus_complex.identification_list)
        elif dimension == 3:
            simplices = create_S3(simp3, 
                                  torus_complex.simplex_objects[0], 
                                  torus_complex.simplex_objects[1], 
                                  torus_complex.simplex_objects[2],
                                  torus_complex.coordinates, 
                                  torus_complex.identification_list)
        return simplices

    
    

    
    
def create_S0(S0_list, coordinates):
    """
    Input:
        S0_list ... List of 0-simplices as generated by gudhi, 
        after having split them by dimension
        coordinates ... Coordinates of auxiliary 0-simpleces
    Output: 
        S0 ... List of Simplex0D objects containing the combinatorical representation,
        the integer filtration value, the continuous filtration value 
        and the geometric coordinate of the point
    """

    
    S0 = []
    
    int_filt_value = 0
    for i, [simp, birth_time] in enumerate(S0_list):
        coordinate = coordinates[i]
        if check_bound(coordinate): # point lies inside main cell
            S0.append(sc.Simplex0D([int_filt_value], 
                                    int_filt_value,
                                    birth_time, 
                                    coordinate))
            int_filt_value +=1
            
    return S0







def create_S1(S1_list, S0, coords, identification_list):
    """
    Input: 
        Input:
        S1_list ... List of 1-simplices as generated by gudhi, 
        after having split them by dimension
        S0 ...List of Simplex0D objects of the filtration
        coords ... list of coordinates of vertices ordered by integer filtration value
        identification_list ... identification list for periodically copied points as given by identification_list()
        
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
        #[vertex0, vertex1], first_coordinate, second_coordinate = order_vertex(simp, [coords[simp[0]], coords[simp[1]]])
        #POC
        [vertex0, vertex0_coords], [vertex1, vertex1_coords] = order_vertices_lexicographically([[simp[0], coords[simp[0]]], [simp[1], coords[simp[1]]]])
        
        # Check if leftmost vertex is in unit cell
        if check_bound(vertex0_coords):

            # Calculate crossing vector
            cross_vec = create_crossing_vector(vertex1_coords)
            
            # identify the old vertices with the newly numerated ones!!!!! 
            # both for simplex, as well as for ordered vertices

            vertex0_new  = (identification_list[vertex0])
            vertex1_new = (identification_list[vertex1])

            int_ordered_simp = sorted([vertex0_new, vertex1_new])
            lex_ordered_simp = [S0[vertex0_new], S0[vertex1_new]]
            
            S1.append(sc.Simplex1D(int_ordered_simp, 
                                   int_filt_value, 
                                   filt_value, 
                                   cross_vec,
                                   lex_ordered_simp
                                  ))

            int_filt_value +=1
    return S1





def find_2Simplex_boundary(points, coords, S1):
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
    cv_1 = create_crossing_vector(p2_coord)

    # second boundary element from p1 to p3
    verts_2  = sorted([p1, p3])
    cv_2 = create_crossing_vector(p3_coord)

    # third boundary element from p2 to p3
    verts_3  = sorted([p2, p3])
    cv_3 = create_crossing_vector(p3_coord - create_crossing_vector(p2_coord))
    
    
    for j in range(len(S1)):
        
        Simplex = S1[j]
        verts = Simplex.verts
        cv    = Simplex.cv

        edges_agree = lambda vertex0, vertex1, cv_0, cv_1: (vertex0[0] == vertex1[0] and
                                                            vertex0[1] == vertex1[1] and
                                                            la.norm(cv-cv_1) < eps)
        if edges_agree(verts, verts_1, cv, cv_1):
            bound_1 = Simplex
        if edges_agree(verts, verts_2, cv, cv_2):
            bound_2 = Simplex
        if edges_agree(verts, verts_3, cv, cv_3):
            bound_3 = Simplex
       
    return [bound_1, bound_2, bound_3]





def create_S2(S2_list, S0, S1, coords, identification_list):
    """
    Input: 
        Input:
        S2_list ... List of 2-simplices as generated by gudhi, 
        after having split them by dimension
        S1 ... List of Simplex1D objects of the filtration
        coords ... list of coordinates of vertices ordered by integer filtration value
        identification_list ... identification list for periodically copied points as given by identification_list()
        
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
        a = simp[0]
        b = simp[1]
        c = simp[2]
        a_coord = coords[a]
        b_coord = coords[b]
        c_coord = coords[c]
        
        [vertex0, vertex0_coord], [vertex1, vertex1_coord], [vertex2, vertex2_coord] = order_vertices_lexicographically([[a, a_coord], [b, b_coord], [c, c_coord]])

        # only if vertex0 is in the main cell do we continue
        if check_bound(vertex0_coord): 

            # the vertices are not yet in the naming convention we have chosen
            # so we rename them using identification_list
            for i in range(len(identification_list)):
                new_name = (identification_list[i])
                if i == vertex0:
                    vertex0 = new_name
                if i == vertex1:
                    vertex1 = new_name
                if i == vertex2:
                    vertex2 = new_name


            bound_1, bound_2, bound_3 = find_2Simplex_boundary([vertex0, vertex1, vertex2],
                                                               [vertex0_coord, vertex1_coord, vertex2_coord], 
                                                               S1)

            S2.append(sc.Simplex2D(sorted([vertex0,vertex1,vertex2]), 
                                   int_filt_value, 
                                   filt_value, 
                                   [S0[vertex0],S0[vertex1],S0[vertex2]], 
                                   [bound_1, bound_2, bound_3]))
            int_filt_value += 1
    return S2



def boundary4_from_rest(boundary1, boundary2, boundary3):
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



def create_S3(S3_list, S0, S1, S2, coords, identification_list):
    """
    Input: 
        Input:
        S3_list ... List of 3-simplices as generated by gudhi, 
        after having split them by dimension
        S2 ... List of Simplex1D objects of the filtration
        coords ... list of coordinates of vertices ordered by integer filtration value
        identification_list ... identification list for periodically copied points as given by identification_list()
        
    Output: 
        S3 ... List of Simplex3D objects containing the combinatorical representation,
        the integer filtration value, the continuous filtration value,
        and the edges constituting the boundary,
        and all vertices in lexicographical order
    """
    
    
    S3 = []
    int_filt_value = len(S0) + len(S1) + len(S2)
    for i in range(len(S3_list)):
        simp, filt_value = S3_list[i] 
        
        
        vertices = [[simp[i], coords[simp[i]]] for i in range(len(simp))]

        ([vertex0, vertex0_coord], 
         [vertex1, vertex1_coord], 
         [vertex2, vertex2_coord],
         [vertex3, vertex3_coord]) = order_vertices_lexicographically(vertices)
        
        # only if vertex0 is in the main cell do we continue
        if check_bound(vertex0_coord): 
            
            # the vertices are not yet in the naming convention we have chosen
            # so we rename them using identification_list
           
            vertex0 = identification_list[vertex0]
            vertex1 = identification_list[vertex1]
            vertex2 = identification_list[vertex2]
            vertex3 = identification_list[vertex3]
            
            

            # we calculate 3 of the 6 1-simplices making up the 1-dim faces of our simplex
            boundary1 = find_2Simplex_boundary([vertex0, vertex1, vertex2],
                                               [vertex0_coord, vertex1_coord, vertex2_coord], 
                                               S1)
            boundary2 = find_2Simplex_boundary([vertex0, vertex1, vertex3],
                                               [vertex0_coord, vertex1_coord, vertex3_coord], 
                                               S1)
            boundary3 = find_2Simplex_boundary([vertex0, vertex2, vertex3],
                                               [vertex0_coord, vertex2_coord, vertex3_coord], 
                                               S1)
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
                    In this particular case, since boundary 4 is not outputted 
                    using the function we are used to
                    it might be that the == does not detect the permutation we have. 
                    !
                    """

            S3.append(sc.Simplex3D(sorted([vertex0,vertex1,vertex2,vertex3]), 
                                   int_filt_value, 
                                   filt_value, 
                                   [S0[vertex0],S0[vertex1],S0[vertex2],S0[vertex3]], 
                                   [face1,face2,face3,face4]
                                  )
                     )
            int_filt_value += 1
            del face1,face2,face3,face4
    return S3

# a b c a_coord b_coord c_coord