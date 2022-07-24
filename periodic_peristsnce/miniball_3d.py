import math
import numpy as np

import scipy.spatial
import itertools

import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
import pylab as pl

import draw


# numerical error margin
EPS = 1e-12


class BallException(Exception):
    """Raised when there is no excluding-including ball

    Attributes:
        message -- boiler-plate message
    """
    def __init__(self, message):
        self.message = message


def normal(points):
    p = points[0]
    q = points[1]

    n = np.array([p[1]*q[2] - p[2]*q[1],    p[2]*q[0] - p[0]*q[2],     p[0]*q[1] - p[1]*q[0]])
    return n
#print(normal([[1,1,1], [1,1,-1]]))


# cc of 2 points
def circumsphere_3d_2(pointsf):
    points = np.array(pointsf)
    return list((points[0] + points[1]) / 2.0), np.linalg.norm((points[0] - points[1]) / 2.0)


def circumsphere_3d_3(pointsf):
    points = np.array(pointsf)
    #print(points)
    d1 = points[1] - points[0]
    d2 = points[2] - points[0]

    # compute normal vector
    n = normal([d1,d2])

    # third eqn is: cc is in the plane defined by normal eqn
    genmat = np.array([d1,d2, n])
    #print(genmat)

    normvec = np.array([np.linalg.norm(genmat[0])**2, np.linalg.norm(genmat[1])**2, 0])

    cc = np.linalg.solve(2*genmat, normvec)
    return list(cc + points[0]), np.linalg.norm(cc)
#TODO:test
#print(circumsphere_3d_3(np.array([[1,1,1],[2,2,2], [2,1,2]])))


def circumsphere_3d_4(pointsf):
    points = np.array(pointsf)
    #print(points)
    # generator matrix D: contains the three direction vectors of tetrahedron as rows
    genmat = np.array([points[i] - points[0] for i in range(1,4)])
    #print(genmat)

    # vector n containing the square norms of the 3 vectors from genmat
    normvec = np.array([np.linalg.norm(genmat[i])**2 for i in range(3)])
    #print(normvec)

    # 0-based circumsphere_3d_ is the solution x to 2Dx = n
    cc = np.linalg.solve(2*genmat, normvec)
    #print(cc)
    #print(cc + points[0])
    return list(cc + points[0]), np.linalg.norm(cc)


def circumsphere_3d(points):
    if len(points) == 0:
        return None, 0
    elif len(points) == 1:
        return points[0], 0
    elif len(points) == 2:
        return circumsphere_3d_2(points)
    elif len(points) == 3:
        return circumsphere_3d_3(points)
    elif len(points) == 4:
        return circumsphere_3d_4(points)
    else:
        raise BallException("You want too many points on your ball.")



def miniball_3d(pin, pon):
    """Compute smallest enclosing ball of pin that has pon
    on its surface.

    Args:
        pin: List of points to be inside the ball
        pon: List of points to be on the ball

    Returns:
        tuple: the center (x, y, z) of the enclosing ball
        int: the radius of the enclosing ball

    """

    if pin == [] or len(pon) == 3:
        cc, cr = circumsphere_3d(pon)
    else:
        p = pin.pop()
        # compute smallest enclosing disk of pin-p, pon
        cc, cr = miniball_3d(list(pin), list(pon))
        # if p is outside the disk
        if cc is None or np.linalg.norm(np.array(p) - np.array(cc)) > cr:
            cc, cr = miniball_3d(list(pin), list(pon + [p]))
    return cc, cr



def miniexball_3d(pin, pon, pout):
    """Compute smallest enclosing ball of pin that has pon
    on its surface.

    Args:
        pin: List of points to be inside the ball
        pon: List of points to be on the ball
        pout: List of points to be outside the ball

    Returns:
        tuple: the center (x, y, z) of the enclosing ball
        int: the radius of the enclosing ball

    """
    if len(pon) == 4:
        cc, cr = circumsphere_3d(pon)
        valid = True
        for p in pin:
            if np.linalg.norm(np.array(p) - np.array(cc)) > cr + EPS:
                valid = False
        for p in pout:
            if np.linalg.norm(np.array(p) - np.array(cc)) < cr - EPS:
                valid = False
        if not valid:
            raise BallException("No sphere including pin and excluding pout exists.")
    elif pout != []:
        p = pout.pop()
        # compute smallest enclosing disk of pin-p, pon
        cc, cr = miniexball_3d(list(pin), list(pon), list(pout))
        # if p is inside the disk
        if cc is None or np.linalg.norm(np.array(p) - np.array(cc)) < cr:
            cc, cr = miniexball_3d(list(pin), list(pon + [p]), list(pout))
    elif pin != []:
        p = pin.pop()
        # compute smallest enclosing disk of pin-p, pon
        cc, cr = miniexball_3d(list(pin), list(pon), list(pout))
        # if p is outside the disk
        if cc is None or np.linalg.norm(np.array(p) - np.array(cc)) > cr:
            cc, cr = miniexball_3d(list(pin), list(pon + [p]), list(pout))
    else:
        cc, cr = circumsphere_3d(pon)
    return cc, cr


def run_example():
    points = np.random.rand(5,3)
    points_out = np.random.rand(2,3)

    # points = np.array([[0,0,0], [0,0,1], [0,1,0], [-0.1,0,-1]])
    # points_out = np.array([[-0.5,0,0], [-0.3,0,-1]])

    # print(points)
    # print(points_out)


    cc, cr = miniball_3d(np.ndarray.tolist(points), [])
    print(cc, cr)

    success = True
    try:
        ccx, crx = miniexball_3d(np.ndarray.tolist(points), [], np.ndarray.tolist(points_out))
    except BallException:
        print("No ball including the desired points while excluding the others exists.")
        success = False

    if success:
        print(ccx, crx)


    # cc, cr = circumsphere_3d_4(points)
    # ax = a3.Axes3D(pl.figure())
    # ax.scatter(points[0:4,0], points[0:4,1], points[0:4,2], marker='o')
    # draw.plot_sphere(ax, cc, cr, 'b')
    # pl.show()

    # visualize the thing
    ax = a3.Axes3D(pl.figure())
    ax.scatter(points[:,0], points[:,1], points[:,2], marker='o')
    ax.scatter(points_out[:,0], points_out[:,1], points_out[:,2], marker='x', color='r')
    draw.plot_sphere(ax, cc, cr, 'b')
    if success:
        draw.plot_sphere(ax, ccx, crx, 'r')
    pl.show()


#run_example()
