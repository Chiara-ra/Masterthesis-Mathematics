import matplotlib.pyplot as plt
import math
import numpy as np

# numerical error margin
EPS = 1e-12

class BallException(Exception):
    """Raised when there is no excluding-including ball

    Attributes:
        message -- boiler-plate message
    """
    def __init__(self, message):
        self.message = message



def circumsphere_2d(points):
    if len(points) == 0:
        return None, 0
    elif len(points) == 1:
        return points[0], 0
    elif len(points) == 2:
        return (np.array(points[0]) + np.array(points[1])) / 2, 0.5*np.linalg.norm(np.array(points[0]) - np.array(points[1]))
    elif len(points) == 3:
        # Mathematical algorithm from Wikipedia: Circumscribed circle
        ax = points[0][0]; ay = points[0][1]
        bx = points[1][0]; by = points[1][1]
        cx = points[2][0]; cy = points[2][1]
        d = (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)) * 2.0
        if d == 0.0:
            return None
        x = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
        y = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
        return (x, y), math.hypot(x - ax, y - ay)
    else:
        raise BallException("You want too many points on your ball.")



def miniball_2d(pin, pon):
    """Compute smallest enclosing ball of pin that has pon
    on its surface.

    Args:
        pin: List of points to be inside the ball
        pon: List of points to be on the ball

    Returns:
        tuple: the center (x, y) of the enclosing ball
        int: the radius of the enclosing ball

    """

    if pin == [] or len(pon) == 3:
        cc, cr = circumsphere_2d(pon)
    else:
        p = pin.pop()
        # compute smallest enclosing disk of pin-p, pon
        cc, cr = miniball_2d(list(pin), list(pon))
        # if p is outside the disk
        if cc is None or np.linalg.norm(np.array(p) - np.array(cc)) > cr:
            cc, cr = miniball_2d(list(pin), list(pon + [p]))
    return cc, cr



def miniexball_2d(pin, pon, pout):
    """Compute smallest enclosing ball of pin that has pon
    on its surface and pout outside or on the surface.

    Args:
        pin: List of points to be inside the ball
        pon: List of points to be on the ball
        pout: List of points to be outside the ball

    Returns:
        tuple: the center (x, y) of the enclosing ball
        int: the radius of the enclosing ball

    """
    if len(pon) == 3:
        cc, cr = circumsphere_2d(pon)
        valid = True
        for p in pin:
            if np.linalg.norm(np.array(p) - np.array(cc)) > cr + EPS:
                valid = False
        for p in pout:
            if np.linalg.norm(np.array(p) - np.array(cc)) < cr - EPS:
                valid = False
        if not valid:
            raise BallException("No sphere including pin and excluding pout exists.")
    # I believe the two elifs can be swapped, but this way it seems to be
    # a bit faster
    elif pout != []:
        p = pout.pop()
        # compute smallest enclosing disk of pin-p, pon
        cc, cr = miniexball_2d(list(pin), list(pon), list(pout))
        # if p is inside the disk
        if cc is None or np.linalg.norm(np.array(p) - np.array(cc)) < cr:
            cc, cr = miniexball_2d(list(pin), list(pon + [p]), list(pout))
    elif pin != []:
        p = pin.pop()
        # compute smallest enclosing disk of pin-p, pon
        cc, cr = miniexball_2d(list(pin), list(pon), list(pout))
        # if p is outside the disk
        if cc is None or np.linalg.norm(np.array(p) - np.array(cc)) > cr:
            cc, cr = miniexball_2d(list(pin), list(pon + [p]), list(pout))
    else:
        cc, cr = circumsphere_2d(pon)
    return cc, cr


def run_example():
    points = np.random.rand(5,2)
    points_out = np.random.rand(2,2)

    cc, cr = miniball_2d(np.ndarray.tolist(points), [])
    print(cc, cr)
    circle = plt.Circle(cc, cr, color='b', fill=False)

    success = True
    try:
        ccx, crx = miniexball_2d(np.ndarray.tolist(points), [], np.ndarray.tolist(points_out))
    except BallException:
        print("No ball including the desired points while excluding the others exists.")
        success = False

    if success:
        print(ccx, crx)
        circlex = plt.Circle(ccx, crx, color='r', fill=False)

    ax = plt.gca()
    ax.cla() # clear things for fresh plot

    # change default range so that new circles will work
    plt.axis('equal')
    ax.set_xlim((-1, 2))
    ax.set_ylim((-1, 2))
    # some data
    ax.plot(points[:,0], points[:,1], 'o', color='black')
    ax.plot(points_out[:,0], points_out[:,1], 'x', color='red')
    # key data point that we are encircling

    ax.add_artist(circle)
    if success:
        ax.add_artist(circlex)

    plt.show()


#run_example()