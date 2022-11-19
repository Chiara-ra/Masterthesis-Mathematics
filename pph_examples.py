# This file contains examples for calculating torus persistence /
# proof of concept PPH computations

import numpy as np

def torus_knot(p: int, q: int, dim: int, res = 20):
    r""" Outputs array of points sampled from (p,q)-torus knot
    with a resolution of res.
    
    Points either lie in unit square on R2
    or in unit square in R3 embedded at z=.5
    """
               
    if (type(dim) != int) or (dim not in [2,3]):
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