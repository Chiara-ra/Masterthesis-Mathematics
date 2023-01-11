# This file contains examples for calculating torus persistence /
# proof of concept PPH computations

import numpy as np

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








def interwoven_grid_222(num_steps=10):
    """
    interwoven 3D grid with index 2*2*2
    """
    
    
    x_direction_first_part_x_vals = np.linspace(0, 0.3, num_steps, endpoint = False)
    x_direction_first_part_y_vals = np.linspace(0,  0.4, num_steps, endpoint = False)
    x_direction_first_part_z_vals = np.linspace(0,  0, num_steps, endpoint = False)

    x_direction_second_part_x_vals = np.linspace(0.3, 1.2, num_steps, endpoint = False) 
    x_direction_second_part_y_vals = np.linspace(0.4, 0.2, num_steps, endpoint = False)
    x_direction_second_part_z_vals = np.linspace(0, 0.4, num_steps, endpoint = False)

    x_direction_third_part_x_vals = np.linspace(1.2,  2,num_steps, endpoint = False)
    x_direction_third_part_y_vals = np.linspace(0.2,  0, num_steps, endpoint = False)
    x_direction_third_part_z_vals = np.linspace(0.4,  0, num_steps, endpoint = False)


    y_direction_first_part_x_vals = np.linspace(0,  0, num_steps, endpoint = False) 
    y_direction_first_part_y_vals = np.linspace(0, 0.3, num_steps, endpoint = False) 
    y_direction_first_part_z_vals = np.linspace(0,  0.4, num_steps, endpoint = False)

    y_direction_second_part_x_vals = np.linspace(0, 0.4, num_steps, endpoint = False) 
    y_direction_second_part_y_vals = np.linspace(0.3, 1.2, num_steps, endpoint = False)
    y_direction_second_part_z_vals = np.linspace(0.4, 0.2, num_steps, endpoint = False)

    y_direction_third_part_x_vals = np.linspace(0.4,  0, num_steps, endpoint = False) 
    y_direction_third_part_y_vals = np.linspace(1.2,  2, num_steps, endpoint = False) 
    y_direction_third_part_z_vals = np.linspace(0.2,  0, num_steps, endpoint = False)


    z_direction_first_part_x_vals = np.linspace(0,  0.4, num_steps, endpoint = False) 
    z_direction_first_part_y_vals = np.linspace(0,  0, num_steps, endpoint = False)
    z_direction_first_part_z_vals = np.linspace(0, 0.3, num_steps, endpoint = False) 

    z_direction_second_part_x_vals = np.linspace(0.4, 0.2, num_steps, endpoint = False) 
    z_direction_second_part_y_vals = np.linspace(0, 0.4, num_steps, endpoint = False)
    z_direction_second_part_z_vals = np.linspace(0.3, 1.2, num_steps, endpoint = False)

    z_direction_third_part_x_vals = np.linspace(0.2,  0, num_steps, endpoint = False) 
    z_direction_third_part_y_vals = np.linspace(0.4,  0, num_steps, endpoint = False)
    z_direction_third_part_z_vals = np.linspace(1.2,  2, num_steps, endpoint = False) 



    x_vals = np.concatenate((x_direction_first_part_x_vals, 
                             x_direction_second_part_x_vals, 
                             x_direction_third_part_x_vals, 
                             y_direction_first_part_x_vals, 
                             y_direction_second_part_x_vals, 
                             y_direction_third_part_x_vals, 
                             z_direction_first_part_x_vals, 
                             z_direction_second_part_x_vals, 
                             z_direction_third_part_x_vals))
    
    y_vals = np.concatenate((x_direction_first_part_y_vals, 
                             x_direction_second_part_y_vals, 
                             x_direction_third_part_y_vals, 
                             y_direction_first_part_y_vals, 
                             y_direction_second_part_y_vals, 
                             y_direction_third_part_y_vals, 
                             z_direction_first_part_y_vals, 
                             z_direction_second_part_y_vals, 
                             z_direction_third_part_y_vals))
    
    z_vals = np.concatenate((x_direction_first_part_z_vals, 
                             x_direction_second_part_z_vals, 
                             x_direction_third_part_z_vals, 
                             y_direction_first_part_z_vals, 
                             y_direction_second_part_z_vals, 
                             y_direction_third_part_z_vals, 
                             z_direction_first_part_z_vals, 
                             z_direction_second_part_z_vals, 
                             z_direction_third_part_z_vals))


    #To bring it into one unit cell (the unit cell is the standard 1x1x1 cubic unit cell)
    x_vals = x_vals - np.floor(x_vals)
    y_vals = y_vals - np.floor(y_vals) 
    z_vals = z_vals - np.floor(z_vals)

    a = np.array([x_vals,y_vals,z_vals]).transpose()

    return a





