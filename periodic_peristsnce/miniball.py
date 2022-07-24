import miniball_2d
import miniball_3d


def miniball(pin):
    """Compute smallest enclosing ball of pin in 2 or 3 dimensions.

    Args:
        pin: List of points to be inside/on the ball

    Returns:
        tuple: the center (x, y) of the enclosing ball
        int: the radius of the enclosing ball
    """
    if len(pin) != 0:
        dim = len(pin[0])
    else:
        return None, 0

    if dim == 2:
        return miniball_2d.miniball_2d(pin, [])
    elif dim == 3:
        return miniball_3d.miniball_3d(pin, [])
    else:
        raise ValueError("Miniball is not implemented for dimension %d" % dim)


def miniexball(pin, pout):
    """Compute smallest enclosing ball of pin which does not
       contain any points from pout in its interior.

    Args:
        pin: List of points to be inside/on the ball
        pout: List of points to be outside/on the ball

    Returns:
        tuple: the center (x, y) of the enclosing ball
        int: the radius of the enclosing ball

    Raises:
        BallException: if no such ball exists
    """
    if len(pin) != 0:
        dim = len(pin[0])
    else:
        return None, 0

    if dim == 2:
        return miniball_2d.miniexball_2d(pin, [], pout)
    elif dim == 3:
        return miniball_3d.miniexball_3d(pin, [], pout)
    else:
        raise ValueError("Miniball is not implemented for dimension %d" % dim)


def miniexonball(pin, pon, pout):
    """Compute smallest enclosing ball of pin which does not
       contain any points from pout in its interior.

    Args:
        pin: List of points to be inside/on the ball
        pon: List of points to be on the ball
        pout: List of points to be outside/on the ball

    Returns:
        tuple: the center (x, y) of the enclosing ball
        int: the radius of the enclosing ball

    Raises:
        BallException: if no such ball exists
    """
    if len(pin) != 0:
        dim = len(pin[0])
    elif len(pon) != 0:
        dim = len(pon[0])
    else:
        return None, 0
    # TODO: To deal with degeneracies, we should use the largest
    # affinely independent subset of pon rather than pon itself
    if dim == 2:
        return miniball_2d.miniexball_2d(pin, pon, pout)
    elif dim == 3:
        return miniball_3d.miniexball_3d(pin, pon, pout)
    else:
        raise ValueError("Miniball is not implemented for dimension %d" % dim)
