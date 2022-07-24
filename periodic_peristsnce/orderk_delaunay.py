import itertools
import numpy as np
import scipy.spatial

'''
Compute order-k Delaunay diagram for a set of points in any dimension

Parameters:
    points - list of points
    order  - order up to which to compute the higher order Delaunay diagrams
                                                                           
Output:
    diagrams_vertices: 
        List of vertex lists.
        For each order-k Delaunay diagram from 1 up to order,
        the list of vertices where each vertex is a k-tuple of point indices.
    diagrams_simplices:
        List of simplex lists.
        For each order-k Delaunay diagram from 1 up to order,
        the list of top-dimensional simplices of a triangulation of the diagram,
        where each simplex is a (d+1)-tuple of indices into the corresponding
        vertex list.
    diagrams_cells:
        List of cell lists.
        For each order-k Delaunay diagram from 1 up to order,
        the list of top-dimensional cells of the diagram,
        where each cell is a tuple of vertices (i.e. tuple of k-tuples of point indices)
        which spans the cell (i.e. the cell is the convex hull of the vertices).
    diagrams_generations:
        List of generation lists.
        For each order-k Delaunay diagram from 1 up to order,
        the list of generations of the top-dimensional cells of the diagram,
        i.e. each cell from diagrams_cells is associated a generation.
'''


class Cell:
    # list of vertices of the cell, which themselves are k-tuples of points
    # k: the k for which this cell appears
    def __init__(self, vertices, k):
        self.vertices = vertices
        self.k = k

    def __str__(self):
        return str(self.vertices) + "[%d]" % self.k


def orderk_delaunay(points, order):
    # Dimension of the ambient space.
    dimension = len(points[0])

    # List of the order-k Delaunay diagrams:
    # vertices (tuples of point indices),
    # simplices from the triangulated version (with indices into the vertex lists),
    # cells (tuples of vertices)
    # generations (generation for each cell)
    diagrams_vertices = []
    diagrams_simplices = []
    diagrams_cells = []
    diagrams_generations = []


    # Step 1: Compute order-1 Delaunay diagram

    # We're interested in the centroids of lifts of ktuples.
    lifts = [tuple(np.append(c, np.linalg.norm(c)**2)) for c in points]

    chull = scipy.spatial.ConvexHull(lifts, qhull_options="Qs")
    # (chull.equations[i][dimension] < 0 mean only take the lower convex hull of the lifts)
    simplices = [chull.simplices[i] for i in range(len(chull.simplices)) if chull.equations[i][dimension] < 0]

    tupled_simplices = [[(vertex,) for vertex in simplex] for simplex in simplices]
    diagrams_vertices.append([(i,) for i in range(len(points))])
    diagrams_simplices.append(simplices)
    diagrams_cells.append(tupled_simplices)
    diagrams_generations.append([1]*len(simplices))
    # Make a list of the order 1 cells.
    # cell_queue will be used as the list of cells who haven't gone through
    # all their barycentric polytopes yet.
    cell_queue = list(zip(tupled_simplices, [1]*len(simplices)))
    cell_queue = [Cell(*cell) for cell in cell_queue]


    # Step 2: Compute order-k Delaunay diagrams for k >= 2
    for k in range(2, order+1):
        # Step 2.1: Compute the vertices and generation >= 2 cells of the order-k
        # Delaunay diagram.
        new_nextgen_cells = []
        new_generations = []
        new_vertices = []
        cell_queue_new = []
        # Go over all cells whose cycle of barycentric polytopes we haven't completed yet.
        for cell in cell_queue:
            # k - cell.k is the generation of the new barycentric cell we get.
            generation = k - cell.k
            if generation < dimension:
                # Take all (generation+1)-tuples of vertices
                # --> list of gtuples of vertices (which are tuples of points)
                gtuples = list(itertools.combinations(cell.vertices, generation + 1))
                # For each of these tuples of vertices, we take the union of the vertices to get a new vertex.
                # The outer 'sorted' is not necessary, but ensures output is entirely sorted,
                # which is useful for testing and comparing.
                nvs = sorted([tuple(sorted(set.union(*[set(vertex) for vertex in gtuple]))) for gtuple in gtuples])
                # The number of points in each new vertex should be k+1,
                # otherwise something went wrong.
                new_nextgen_cells.append(nvs)
                new_generations.append(generation + 1)
                new_vertices += nvs
            else:
                pass
            
            # Once the generation is dimension - 1, we've cycled through all barycentric
            # polytopes and thus don't need to add the cell to the queue again
            if k - cell.k < dimension - 1:
                cell_queue_new.append(cell)

        # List of new vertices without repetitions.
        new_vertices = list(set(new_vertices))
        # For each tuple that we identified as vertex, compute the centroid of its lifts.
        new_lifts = np.array([np.sum([lifts[i] for i in new_vertex], axis=0)/k for new_vertex in new_vertices])

        chull = scipy.spatial.ConvexHull(new_lifts, qhull_options="Qs")
        # Compute the simplices of the triangulated order-k Delaunay diagram.
        # Each simplex is a tuple of integers, these integers are indices into new_vertices,
        # which contains the k-tuples of original points which are vertices.
        # (chull.equations[i][dimension] < 0 mean only take the lower convex hull of the lifts)
        simplices = [sorted(chull.simplices[i]) for i in range(len(chull.simplices)) if chull.equations[i][dimension] < 0]


        # Step 2.2: Compute the remaining cells of the order-k Delaunay diagram
        new_firstgen_cells = []
        for simplex in simplices:
            # Get the k-tuples that are the vertices of the simplex.
            vertices = [set(new_vertices[i]) for i in simplex]
            inters = set.intersection(*vertices)
            # Simplices are first generation if the intersection of their vertices is k-1.
            if len(inters) == k-1:
                # The outer 'sorted' is not necessary, but ensures output is entirely sorted,
                # which is useful for testing and comparing.
                cell_vertices = sorted([new_vertices[i] for i in simplex])
                new_firstgen_cells.append(cell_vertices)
                cell_queue_new.append(Cell(cell_vertices, k))

        # Use our compiled queue for the next iteration
        cell_queue = cell_queue_new

        # Save the computed stuff
        diagrams_vertices.append(new_vertices)
        diagrams_simplices.append(simplices)
        diagrams_cells.append(new_nextgen_cells + new_firstgen_cells)
        diagrams_generations.append(new_generations + [1]*len(new_firstgen_cells))


    return diagrams_vertices, diagrams_simplices, diagrams_cells, diagrams_generations

