# libraries
import numpy as np       



def remove_duplicate_points(points):
    """
    Input: Nx3 numpy array
    Output: Mx3 numpy array, with all duplicate rows removed, hence M<N
    """
    # round inputs to 5 decimal places
    points = np.round(points, 5)
    return np.unique(points, axis = 0)


## Check legality of input data
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



def preprocess_points(points, a, b, c):
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

