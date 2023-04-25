# libraries
import numpy as np
import random as rd


# This file contains examples for calculating torus persistence /
# proof of concept PPH computations



def random_points(N=5, a=1, b=1, c=1):
    """
    Generates N random points in [0,a]x[0,b]x[0,c] cuboid,
    outputts Nx3 numpy array, each row containing x, y,z coordinates of a point.
    """
    return np.array([[rd.random()*a, rd.random()*b, rd.random()*c] for i in range(N)])



def torus_knot(p: int, q: int, dim: int, res = 20):
    r""" Outputs array of points sampled from (p,q)-torus knot
    with a resolution of res.
    
    Points either lie in unit square on R2
    or in unit square in R3 embedded at z=.5
    """
               
    if (not isinstance(dim, int)) or (dim not in [2,3]):
        raise ValueError(f"dim needs to be integer 2 or 3, but is {dim}.")
    else:
        spacing = np.linspace(0,1,res,endpoint=False)
        x = np.mod(- q*spacing, 1)
        y = np.mod(  p*spacing, 1)

        if dim == 2:
            a = np.array([x,y]).transpose()

        elif dim == 3:
            z = np.array([1/2 for dummy in range(res)])
            a = np.array([x,y,z]).transpose()
            
    return a


def knot_3D(p: int, q: int, l: int, res = 20):
    r""" Points lying along diagonal line with resolution 20. 
    Depending on projection a (p,q)-knot, (p,l)-knot or (q,l)-knot.
    """
               
    if (not isinstance(dim, int)) or (dim not in [2,3]):
        raise ValueError(f"dim needs to be integer 2 or 3, but is {dim}.")
    else:
        spacing = np.linspace(0,1,res,endpoint=False)
        x = np.mod(q*spacing, 1)
        y = np.mod(p*spacing, 1)
        y = np.mod(l*spacing, 1)
            
    return np.array([x,y,z]).transpose()



def interwoven_grid_211(num_steps=10):
    """
    interwoven 3D grid with index 2*1*1
    """
    #Now we don't want a for-loop but instead only one copy of the motif, thus fixed values here.

    x_direction_first_part_x_vals = np.linspace(0, 0.3, num_steps, endpoint=False) 
    x_direction_first_part_y_vals = np.linspace(0, 0.4, num_steps, endpoint=False)
    x_direction_first_part_z_vals = np.linspace(0, 0,   num_steps, endpoint=False)

    x_direction_second_part_x_vals = np.linspace(0.3, 1.2, num_steps, endpoint=False) 
    x_direction_second_part_y_vals = np.linspace(0.4, 0.2, num_steps, endpoint=False)
    x_direction_second_part_z_vals = np.linspace(0, 0.4, num_steps, endpoint=False)

    x_direction_third_part_x_vals = np.linspace(1.2,  2,num_steps, endpoint=False) 
    #I end a bit away from the end, so that no two points are identitcal.
    x_direction_third_part_y_vals = np.linspace(0.2, 0, num_steps, endpoint=False)
    x_direction_third_part_z_vals = np.linspace(0.4, 0, num_steps, endpoint=False)


    y_direction_first_part_x_vals = np.linspace(0, 0, num_steps, endpoint=False) 
    y_direction_first_part_y_vals = np.linspace(0, 2,num_steps, endpoint=False) 
    #I start and end a bit away from the ends, so that no two points are identitcal.
    y_direction_first_part_z_vals = np.linspace(0, 0, num_steps, endpoint=False)


    z_direction_first_part_x_vals = np.linspace(0, 0, num_steps, endpoint=False) 
    z_direction_first_part_y_vals = np.linspace(0, 0, num_steps, endpoint=False)
    z_direction_first_part_z_vals = np.linspace(0, 1, num_steps, endpoint=False) 
    #I start and end a bit away from the ends, so that no two points are identitcal.


    x_vals = np.concatenate((x_direction_first_part_x_vals, 
                             x_direction_second_part_x_vals, 
                             x_direction_third_part_x_vals, 
                             y_direction_first_part_x_vals, 
                             z_direction_first_part_x_vals))
    y_vals = np.concatenate((x_direction_first_part_y_vals, 
                             x_direction_second_part_y_vals, 
                             x_direction_third_part_y_vals, 
                             y_direction_first_part_y_vals, 
                             z_direction_first_part_y_vals))
    z_vals = np.concatenate((x_direction_first_part_z_vals, 
                             x_direction_second_part_z_vals, 
                             x_direction_third_part_z_vals, 
                             y_direction_first_part_z_vals, 
                             z_direction_first_part_z_vals))


    #To bring it into one unit cell (the unit cell is the standard 1x1x1 cubic unit cell)
    x_vals = x_vals - np.floor(x_vals)
    y_vals = y_vals - np.floor(y_vals) 
    z_vals = z_vals - np.floor(z_vals)
    
    a = np.array([x_vals,y_vals,z_vals]).transpose()
    
    return a








def interwoven_grid_222(step_size = 0.04):
    """
    interwoven 3D grid with index 2*2*2
    """
    
    
    #interwoven 3D grid with index 2*2*2, just one motif, as example point set to input in periodic persistence software:


    length_edge_1 = np.sqrt(0.3**2 + 0.4**2)
    length_edge_2 = np.sqrt(0.9**2 + 0.2**2 + 0.8**2)
    length_edge_3 = np.sqrt(0.8**2 + 0.4**2 + 0.2**2)

    num_points_along_edge_1 = int( np.ceil(length_edge_1 / step_size) + 1 )
    num_points_along_edge_2 = int( np.ceil(length_edge_2 / step_size) + 1 )
    num_points_along_edge_3 = int( np.ceil(length_edge_3 / step_size) + 1 )

    x_iter=0
    y_iter=0
    z_iter=0

    x_direction_first_part_x_vals = np.linspace(x_iter,      x_iter+0.3, num_points_along_edge_1) 
    x_direction_first_part_y_vals = np.linspace(y_iter,      y_iter+0.4, num_points_along_edge_1)
    x_direction_first_part_z_vals = np.linspace(z_iter,      z_iter,     num_points_along_edge_1)

    x_direction_second_part_x_vals = np.linspace(x_iter+0.3, x_iter+1.2, num_points_along_edge_2) 
    x_direction_second_part_y_vals = np.linspace(y_iter+0.4, y_iter+0.6, num_points_along_edge_2)
    x_direction_second_part_z_vals = np.linspace(z_iter,     z_iter+0.8, num_points_along_edge_2)

    x_direction_third_part_x_vals = np.linspace(x_iter+1.2,  x_iter+2,   num_points_along_edge_3) 
    x_direction_third_part_y_vals = np.linspace(y_iter+0.6,  y_iter,     num_points_along_edge_3)
    x_direction_third_part_z_vals = np.linspace(z_iter+0.8,  z_iter,     num_points_along_edge_3)

    ##Now we do the same kind of trick in y- and z-direction as well.
    ##I.e. Also in y-direction we skip an integer point by connecting (0,0,0) directly with (0,2,0).
    ##But not with a straight line but with a curved line consisting of 3 pieces instead.
    ##The curve is the same as before. I just permuted the coordinates.
    y_direction_first_part_x_vals = np.linspace(x_iter,      x_iter,     num_points_along_edge_1) 
    y_direction_first_part_y_vals = np.linspace(y_iter,      y_iter+0.3, num_points_along_edge_1) 
    y_direction_first_part_z_vals = np.linspace(z_iter,      z_iter+0.4, num_points_along_edge_1)

    y_direction_second_part_x_vals = np.linspace(x_iter,     x_iter+0.8, num_points_along_edge_2) 
    y_direction_second_part_y_vals = np.linspace(y_iter+0.3, y_iter+1.2, num_points_along_edge_2)
    y_direction_second_part_z_vals = np.linspace(z_iter+0.4, z_iter+0.6, num_points_along_edge_2)

    y_direction_third_part_x_vals = np.linspace(x_iter+0.8,  x_iter,     num_points_along_edge_3) 
    y_direction_third_part_y_vals = np.linspace(y_iter+1.2,  y_iter+2,   num_points_along_edge_3) 
    y_direction_third_part_z_vals = np.linspace(z_iter+0.6,  z_iter,     num_points_along_edge_3)

    ##And in z-direction as well:
    z_direction_first_part_x_vals = np.linspace(x_iter,      x_iter+0.4, num_points_along_edge_1) 
    z_direction_first_part_y_vals = np.linspace(y_iter,      y_iter,     num_points_along_edge_1)
    z_direction_first_part_z_vals = np.linspace(z_iter,      z_iter+0.3, num_points_along_edge_1) 

    z_direction_second_part_x_vals = np.linspace(x_iter+0.4, x_iter+0.6, num_points_along_edge_2) 
    z_direction_second_part_y_vals = np.linspace(y_iter,     y_iter+0.8, num_points_along_edge_2)
    z_direction_second_part_z_vals = np.linspace(z_iter+0.3, z_iter+1.2, num_points_along_edge_2)

    z_direction_third_part_x_vals = np.linspace(x_iter+0.6,  x_iter,     num_points_along_edge_3) 
    z_direction_third_part_y_vals = np.linspace(y_iter+0.8,  y_iter,     num_points_along_edge_3)
    z_direction_third_part_z_vals = np.linspace(z_iter+1.2,  z_iter+2,   num_points_along_edge_3) 



    x_vals = np.concatenate((x_direction_first_part_x_vals, x_direction_second_part_x_vals, x_direction_third_part_x_vals, y_direction_first_part_x_vals, y_direction_second_part_x_vals, y_direction_third_part_x_vals, z_direction_first_part_x_vals, z_direction_second_part_x_vals, z_direction_third_part_x_vals))
    y_vals = np.concatenate((x_direction_first_part_y_vals, x_direction_second_part_y_vals, x_direction_third_part_y_vals, y_direction_first_part_y_vals, y_direction_second_part_y_vals, y_direction_third_part_y_vals, z_direction_first_part_y_vals, z_direction_second_part_y_vals, z_direction_third_part_y_vals))
    z_vals = np.concatenate((x_direction_first_part_z_vals, x_direction_second_part_z_vals, x_direction_third_part_z_vals, y_direction_first_part_z_vals, y_direction_second_part_z_vals, y_direction_third_part_z_vals, z_direction_first_part_z_vals, z_direction_second_part_z_vals, z_direction_third_part_z_vals))


    #To bring it into one unit cell (the unit cell is the standard 1x1x1 cubic unit cell)
    x_vals = x_vals - np.floor(x_vals)
    y_vals = y_vals - np.floor(y_vals) 
    z_vals = z_vals - np.floor(z_vals)
    
    a = np.array([x_vals,y_vals,z_vals]).transpose()

    return a





