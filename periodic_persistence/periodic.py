import numpy as np
import matplotlib.pyplot as plt

from itertools import chain, combinations, product
from collections import defaultdict
import math
from math import sqrt, cos, sin, pi

from orderk_delaunay import orderk_delaunay
from miniball import miniexonball
# pip3 install phat
import phat

# from the documentation:
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


dimension = 3 # d
npoints = 2 # n
# sample some random points to triangulate
points = np.random.random((npoints,dimension))*10-np.array([4]*dimension)

draw_labels = True
# How to choose representative cells for each class.
# 0 = like in Car10
# 1 = if the lexicographically (by coordinates) first vertex has offset 0.
# 2 = if the lexicographically (by offset) first vertex has offset 0.
# 3 = random
# Note: only 0 currently works for computing persistence.
representative_mode = 0


# Selling's algorithm to reduce a basis in 3D
# Input: list of basis vectors with their negative sum appended at the end
def reduce3d(b):
    coeffs = np.zeros((4,4))
    reduced = True
    for i,j,h,k in [[0,1,2,3], [0,2,1,3], [0,3,1,2], [1,2,0,3], [1,3,0,2], [2,3,0,1]]:
        coeffs[i,j] = np.dot(b[i], b[j])
        if coeffs[i,j] > 0:
            reduced = False
            break

    while not reduced:
        b[h] += b[i]
        b[k] += b[i]
        b[i] = -b[i]

        reduced = True
        # lattice is reduced if all coefficients are negative
        for i,j,h,k in [[0,1,2,3], [0,2,1,3], [0,3,1,2], [1,2,0,3], [1,3,0,2], [2,3,0,1]]:
            coeffs[i,j] = np.dot(b[i], b[j])
            if coeffs[i,j] > 0:
                reduced = False
                break

    return b


'''
Below are the d translation vectors of the lattice.
They have to be reduced vectors, meaning that the
angles between them are pairwise >= 90 degrees.
Use the function reduce3d for this purpose
'''

# 2D test data
t1 = np.array([-0.5, 1])
t2 = np.array([1.5, 0])
# the last one is always negative sum of all the other ones
tn = -(t1+t2)
basevecs = [t1,t2,tn]
points = np.array([[0, 0], [-0.2, -0.6]]) #[0.20, 0.68]

# #3D test data
# t1 = np.array([-0.5, 1, 0])
# t2 = np.array([1.5, 0, 0.5])
# t3 = np.array([1, 0.5, 1.2])
# t1 = np.array([-3.8, 1.2, 0])
# t2 = np.array([1.7, 4, 0.5])
# t3 = np.array([0.8, 2.3, 1.4])
# tn = -(t1+t2+t3)
# basevecs = [t1,t2,t3,tn]
# basevecs = reduce3d(basevecs)
# basevecs = sorted(basevecs, key=lambda x:np.linalg.norm(x))
# # points = np.array([[0, 0, 0]])
# points = np.random.uniform(-3, 3, (5000,3))

#Standard Li
# t1 = np.array([3.507, -1e-6, -1e-6])
# t2 = np.array([-1e-6, 3.507, -1e-6])
# t3 = np.array([-1e-6, -1e-6, 3.507])
# tn = -(t1+t2+t3)
# basevecs = [t1,t2,t3,tn]
# points = np.array([[0. , 0. , 0. ], [1.7535, 1.7535, 1.7535]])

# Standard Li 3-cell
# t1 = 3*np.array([3.507, -1e-6, -1e-6])
# t2 = 3*np.array([-1e-6, 3.507, -1e-6])
# t3 = 3*np.array([-1e-6, -1e-6, 3.507])
# tn = -(t1+t2+t3)
# basevecs = [t1,t2,t3,tn]
# points = np.array([[0. , 0. , 0. ], [0. , 0. , 3.507 ], [0. , 0. , 7.014 ], [0. , 3.507 , 0. ], [0. , 3.507 , 3.507 ], [0. , 3.507 , 7.014 ], [0. , 7.014 , 0. ], [0. , 7.014 , 3.507 ], [0. , 7.014 , 7.014 ], [3.507 , 0. , 0. ], [3.507 , 0. , 3.507 ], [3.507 , 0. , 7.014 ], [3.507 , 3.507 , 0. ], [3.507 , 3.507 , 3.507 ], [3.507 , 3.507 , 7.014 ], [3.507 , 7.014 , 0. ], [3.507 , 7.014 , 3.507 ], [3.507 , 7.014 , 7.014 ], [7.014 , 0. , 0. ], [7.014 , 0. , 3.507 ], [7.014 , 0. , 7.014 ], [7.014 , 3.507 , 0. ], [7.014 , 3.507 , 3.507 ], [7.014 , 3.507 , 7.014 ], [7.014 , 7.014 , 0. ], [7.014 , 7.014 , 3.507 ], [7.014 , 7.014 , 7.014 ], [1.7535, 1.7535, 1.7535], [1.7535, 1.7535, 5.2605], [1.7535, 1.7535, 8.7675], [1.7535, 5.2605, 1.7535], [1.7535, 5.2605, 5.2605], [1.7535, 5.2605, 8.7675], [1.7535, 8.7675, 1.7535], [1.7535, 8.7675, 5.2605], [1.7535, 8.7675, 8.7675], [5.2605, 1.7535, 1.7535], [5.2605, 1.7535, 5.2605], [5.2605, 1.7535, 8.7675], [5.2605, 5.2605, 1.7535], [5.2605, 5.2605, 5.2605], [5.2605, 5.2605, 8.7675], [5.2605, 8.7675, 1.7535], [5.2605, 8.7675, 5.2605], [5.2605, 8.7675, 8.7675], [8.7675, 1.7535, 1.7535], [8.7675, 1.7535, 5.2605], [8.7675, 1.7535, 8.7675], [8.7675, 5.2605, 1.7535], [8.7675, 5.2605, 5.2605], [8.7675, 5.2605, 8.7675], [8.7675, 8.7675, 1.7535], [8.7675, 8.7675, 5.2605], [8.7675, 8.7675, 8.7675]])

# # Low temp phase Li
# t1 = np.array([4.404, -1e-5, -1e-5])
# t2 = np.array([-1e-5, 4.404, -1e-5])
# t3 = np.array([-1e-5, -1e-5, 4.404])
# tn = -(t1+t2+t3)
# basevecs = [t1,t2,t3,tn]
# points = np.array([[0. , 0. , 0. ], [0. , 2.202, 2.202], [2.202, 0. , 2.202], [2.202, 2.202, 0. ]])
# points += 1e-4*np.random.random(points.shape) - 5e-5

# # High pressure phase Li
# t1 = 2.4023*np.array([1, -1e-5, -1e-5])
# t2 = 2.4023*np.array([cos(2*pi/3), sin(2*pi/3), -1e-5])  # 120 degree angle to t1
# t3 = np.array([-1e-5, -1e-5, 5.51592])
# tn = -(t1+t2+t3)
# basevecs = np.array([t1,t2,t3,tn])
# basevecs = basevecs + 1e-4*np.random.random(basevecs.shape) - 5e-5
# basevecs = reduce3d(basevecs)
# points = np.array([[ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00], [ 1.38696855e+00, -4.44514795e-17, 1.83864000e+00], [ 6.93484276e-01, 1.20115000e+00, 3.67728000e+00]])
# points += 1e-4*np.random.random(points.shape) - 5e-5

# #4.2K phase Li
# t1 = 3.111*np.array([1, -1e-5, -1e-5])
# t2 = 3.111*np.array([cos(2*pi/3), sin(2*pi/3), -1e-5])  # 120 degree angle to t1
# t3 = 22.86*np.array([-1e-6, -1e-6, 1])
# tn = -(t1+t2+t3)
# basevecs = np.array([t1,t2,t3,tn])
# basevecs = basevecs + 1e-4*np.random.random(basevecs.shape) - 5e-5
# basevecs = reduce3d(basevecs)
# points = np.array([[0.00000000e+00, 0.00000000e+00, 0.00000000e+00], [1.79613669e+00, 9.04646728e-17, 7.62000000e+00], [8.98068344e-01, 1.55550000e+00, 1.52400000e+01], [0.00000000e+00, 0.00000000e+00, 1.77805080e+01], [0.00000000e+00, 0.00000000e+00, 5.07949200e+00], [1.79613669e+00, 9.04646728e-17, 2.54050800e+00], [8.98068344e-01, 1.55550000e+00, 1.01605080e+01], [1.79613669e+00, 9.04646728e-17, 1.26994920e+01], [8.98068344e-01, 1.55550000e+00, 2.03194920e+01]])
# points += 1e-4*np.random.random(points.shape) - 5e-5

# # Mg with a similarish structure
# t1 = 3.2085*np.array([1, -1e-5, -1e-5])
# t2 = 3.2085*np.array([cos(2*pi/3), sin(2*pi/3), -1e-5])  # 120 degree angle to t1
# t3 = 5.2106*np.array([-1e-6, -1e-6, 1])
# tn = -(t1+t2+t3)
# basevecs = np.array([t1,t2,t3,tn])
# basevecs = basevecs + 1e-4*np.random.random(basevecs.shape) - 5e-5
# basevecs = reduce3d(basevecs)
# points = np.array([[ 9.26121548e-01, 1.60408958e+00, 1.30265000e+00], [ 9.26399412e-01, 1.60425000e+00, 1.30265000e+00], [ 9.26121548e-01, 1.60441042e+00, 1.30265000e+00], [ 1.85252096e+00, 1.60425000e-04, 3.90795000e+00], [ 1.85224310e+00, -7.84711851e-17, 3.90795000e+00], [ 1.85252096e+00, -1.60425000e-04, 3.90795000e+00]])
# points += 1e-4*np.random.random(points.shape) - 5e-5

# High pressure Nitrogen
t1 = 3.595*np.array([1, 0, 0])
t2 = 3.595*np.array([cos(2*pi/3), sin(2*pi/3), 0])  # 120 degree angle to t1
t3 = 5.845*np.array([0, 0, 1])
tn = -(t1+t2+t3)
basevecs = np.array([t1,t2,t3,tn])
basevecs = basevecs + 2e-5*np.random.random(basevecs.shape) - 1e-6
basevecs = reduce3d(basevecs)
points = np.array([[ 0.65256053, 2.0200305 , 4.0605215 ], [ 1.03768333, 1.35261875, 4.0605215 ], [ 1.42311746, 2.01985075, 4.0605215 ], [ 1.03768333, 2.24238125, 4.0605215 ], [ 1.42311746, 1.57514925, 4.0605215 ], [ 0.65256053, 1.5749695 , 4.0605215 ], [ 2.46080079, -0.2225305 , 1.7844785 ], [ 2.075678 , 0.44488125, 1.7844785 ], [ 1.69024386, -0.22235075, 1.7844785 ], [ 2.075678 , -0.44488125, 1.7844785 ], [ 1.69024386, 0.22235075, 1.7844785 ], [ 2.46080079, 0.2225305 , 1.7844785 ], [ 2.46080079, -0.2225305 , 1.1380215 ], [ 2.075678 , 0.44488125, 1.1380215 ], [ 1.69024386, -0.22235075, 1.1380215 ], [ 2.075678 , -0.44488125, 1.1380215 ], [ 1.69024386, 0.22235075, 1.1380215 ], [ 2.46080079, 0.2225305 , 1.1380215 ], [ 0.65256053, 2.0200305 , 4.7069785 ], [ 1.03768333, 1.35261875, 4.7069785 ], [ 1.42311746, 2.01985075, 4.7069785 ], [ 1.03768333, 2.24238125, 4.7069785 ], [ 1.42311746, 1.57514925, 4.7069785 ], [ 0.65256053, 1.5749695 , 4.7069785 ]])
points += 1e-5*np.random.random(points.shape) - 5e-6

# # High temp Scandium
# t1 = 3.3098*np.array([1, -1e-5, -1e-5])
# t2 = 3.3098*np.array([cos(2*pi/3), sin(2*pi/3), -1e-5])  # 120 degree angle to t1
# t3 = 5.266*np.array([-1e-6, -1e-6, 1])
# tn = -(t1+t2+t3)
# basevecs = np.array([t1,t2,t3,tn])
# basevecs = basevecs + 1e-4*np.random.random(basevecs.shape) - 5e-5
# basevecs = reduce3d(basevecs)
# points = np.array([[ 9.55361415e-01, 1.65473451e+00, 1.31650000e+00], [ 9.55648052e-01, 1.65490000e+00, 1.31650000e+00], [ 9.55361415e-01, 1.65506549e+00, 1.31650000e+00], [ 1.91100947e+00, 1.65490000e-04, 3.94950000e+00], [ 1.91072283e+00, -1.98162908e-17, 3.94950000e+00], [ 1.91100947e+00, -1.65490000e-04, 3.94950000e+00]])
# points += 1e-4*np.random.random(points.shape) - 5e-5

print(basevecs)

# FCC
# t1 = np.array([   1,          0,         0])
# t2 = np.array([-0.4999,  sqrt(3)/2,         0])
# t3 = np.array([-0.4999, -sqrt(3)/6, sqrt(2/3)])
# # t1 = np.array([1, 1, 0])
# # t2 = np.array([1, 0, 1])
# # t3 = np.array([0, 1, 1])
# tn = -(t1+t2+t3)
# basevecs = [t1,t2,t3,tn]
# points = np.array([[0, 0, 0]])

# # HCP
# t1 = np.array([   1,          0,         0])
# t2 = np.array([-0.5,  sqrt(3)/2,         0])
# t3 = np.array([   0,          0, sqrt(8/3)])
# tn = -(t1+t2+t3)
# basevecs = [t1,t2,t3,tn]
# points = np.array([[0, 0, 0], [-0.5, -sqrt(3)/6, sqrt(2/3)]])

# # Cube
# t1 = np.array([    1, -1e-5, -1e-5])
# t2 = np.array([-1e-5,     1, -1e-5])
# t3 = np.array([-1e-5, -1e-5,     1])
# tn = -(t1+t2+t3)
# basevecs = [t1,t2,t3,tn]
# points = np.array([[0, 0, 0]])
# points = np.array(list(product(np.arange(0, 1, 0.33334), repeat=dimension)))
# points = points + np.random.random(points.shape)*1e-5


# in case we overwrite the random points, we need to get the
# correct number of points
npoints, dimension = points.shape


for v1,v2 in combinations(basevecs, 2):
    scal_prod = np.dot(v1,v2)
    magn_prod = np.linalg.norm(v1)*np.linalg.norm(v2)
    angle = np.arccos(scal_prod/magn_prod)
    if angle < np.pi/2:
        print("Lattice basis not reduced!")


# we compute all 0-1 linear combinations of these d+1 vectors.
ts = [sum(ps) for ps in list(powerset(basevecs))[1:-1]]
#print(np.array(ts))

'''
Throw out the vectors which are just negates of another one.
This can be done by just taking the first half of them.
This is because the second half are the negatives
of the vectors in the first half, in reverse order,
i.e they are paired up last with first, 2nd-last with 2nd, ...
In such a pair, the two indices are binary
complements of each other. Thus one is exactly the sum
of those vectors not in the other. As the sum of all vectors
together is 0, this implies that they are negates of each other.
'''
fs = ts[:len(ts)//2]



# Translate all points into the FD.
# The FD is $\bigcap_i \{|<x, f_i>| \leq \frac{1}{2} <f_i, f_i>\}$
fdpoints = []
for x in points:
    i = 0
    '''
    We go through all the faces. If for one face the point is
    not in the corresponding strip, we translate it, and start
    from the beginning checking all faces.
    TODO: 
    (1) Does this always terminate? Or at least, can
    the number of iterations explode?
    (2) Is there a smarter way of doing this?
    '''
    while i < len(fs):
        f = fs[i]
        scal = np.dot(x,f) / np.dot(f,f)
        if abs(scal) > 0.5:
            x = x - math.floor(scal + 0.5) * f
            i = 0
        else:
            i += 1

    fdpoints.append(x)

fdpoints = np.array(fdpoints)


if dimension == 2:
    '''
    I want to compute the Voronoi domain to draw it later, 
    so I do the following quick and dirty hack:
    I sort the 6 translation vectors by angle, then for each pair
    of consecutive vectors I compute the circumsphere of the two
    together with the origin to get my Voronoi vertex.
    '''
    ts_by_angle = sorted(ts, key=lambda p:np.arctan2(p[0], p[1]))
    # compute circumspheres
    ccs = [miniexonball([], [np.array([0,0]), ts_by_angle[i], ts_by_angle[(i+1)%6]], [])[0] for i in range(6)]
    # now the Voronoi domain is the polygon spanned by these circumcenteres


"""
if dimension == 2:
    # plot the translates of the points into FD
    ax = plt.gca()
    ax.cla()
    plt.xlim([-3,3])
    plt.ylim([-3,3])
    for t in ts:
        line = plt.Line2D([0, t[0]], [0, t[1]])
        ax.add_artist(line)
    ax.plot(fdpoints[:,0], fdpoints[:,1], 'o', color='red')
    vdomain = plt.Polygon(np.array(ccs), alpha=0.1)
    v3domain = plt.Polygon(3*np.array(ccs), alpha=0.1)
    ax.add_artist(vdomain)
    ax.add_artist(v3domain)
    plt.show()
"""


'''
We now want all the points' translates which are within 3*FD
3*FD is $\bigcup_i \{|<x, f_i>| \leq \frac{3}{2} <f_i, f_i>\}$

We do this by simply translating the points around and
intersecting with 3*FD

TODO: Is there a more systematic way of finding exactly those
translates of the FD that overlap 3*FD?
Or does the reducedness give us at least a guarantee that
a certain interval is sufficient?

At least we can do a sanity check by counting the number of points.
'''
mpoints = np.array(fdpoints)
# equivalence class the point is in
mpoints_classes = np.arange(0, len(fdpoints))
# how many translation wrt t1, ..., td to apply
mpoints_offsets = np.zeros((len(fdpoints), dimension), dtype=int)
#print(mpoints_offsets)



'''
The loop below is the same as

for i1 in range(-2,3):
    for i2 in range(-2,3):
        for ...
'''
for indices in product(range(-3,4), repeat=dimension):
    # skip this iff all indices are 0, as these are the original points
    # which we already have.
    if not any(indices):
        continue

    # same as i1*t1 + i2*t2 + ...
    shift = sum(basevecs[i]*indices[i] for i in range(dimension))

    ttpoints = fdpoints + shift
    good = np.empty(ttpoints.shape[0])
    good[:] = True
    for f in fs:
        # AND good with whether the points are in the 3*strip belonging to f
        good = np.logical_and(good, np.abs(np.dot(ttpoints,f) / np.dot(f,f)) <= 1.5)
    # ttpoints[good] are those translates that are within 3*FD
    new_mpoints = ttpoints[good]
    mpoints = np.concatenate((mpoints, new_mpoints), axis=0)
    # for each of these points, get the ID of the point in the FD it is equivalent to
    new_mpoints_classes = np.arange(0, len(fdpoints))[good]
    mpoints_classes = np.concatenate((mpoints_classes, new_mpoints_classes), axis=0)
    # also store the translation indices for each of these points
    if np.count_nonzero(good):
        new_mpoints_offsets = np.array([indices]*np.count_nonzero(good))
        mpoints_offsets = np.concatenate((mpoints_offsets, new_mpoints_offsets), axis=0)
    if len(new_mpoints) != 0:
        print(indices)

# t1 = np.array([-0.8, 1.2, 0])
# t2 = np.array([1.5, 0, 0.5])
# t3 = np.array([0.8, 0.3, 1.4])
print(len(mpoints_classes), 27*len(points))
# d = defaultdict(int)
# for offset in mpoints_offsets:
#     d[tuple(offset)] += 1

# for k,v in sorted(d.items(), key=lambda x:x[1]):
#     print(k,v)

# dc = defaultdict(int)
# for clas in mpoints_classes:
#     dc[clas] += 1

# for k,v in sorted(dc.items(), key=lambda x:x[1]):
#     if v != 27:
#         print(k,v)


# dictionary mapping a representative + offset to the index of that points
rep_offset_to_index = dict()
for index in range(len(mpoints)):
    rep = mpoints_classes[index]
    offset = tuple(mpoints_offsets[index])
    key = (rep, offset)
    rep_offset_to_index[key] = index


'''
Compute the Delaunay triangulation.
'''
_, simplices, _, _ = orderk_delaunay(mpoints, 1)
simplices = np.array(simplices[0]) # we only care about first-order

# print(mpoints)
# print(mpoints_classes)
# print(mpoints_offsets)
# print(simplices)



'''
Let's filter out the simplices that don't contain any
of the points in the FD.

As mpoints was constructed as the points in the FD,
followed by various translates, we know that the first
npoints points of mpoints are the ones in the FD,
so we can just check whether a vertex ID in a simplex
is in the original FD by checking "< npoints".
'''
simplices = simplices[np.any(simplices < npoints, axis=1)]
print(simplices)


'''
Now we sort the cells into equivalence classes.
These classes are the cells of our abstract simplicial complex
used to compute the alpha-shapes and persistence.

Two cells are equivalent if their vertices pairwise have
the same representative vertex, and if the relative translation
vectors for the first vertex to the other ones matches pairwise.

To ensure we pairwise compare the correct vertices and
relative translation vectors, we exploit that there's a linear
order on the vertices of a cell defined by index of
representative vertex, and (tiebreaker) absolute translation vector,
which is invariant under translation of the cell.
'''

# equivalence classes for the cells.
# key: representative vertices + relative translation
# value: list of cells represented by this class
rel_reprs = defaultdict(set)

for tri in simplices:
    # for each point in the triangle, we take the point class and translation
    representative = [(mpoints_classes[i], tuple(mpoints_offsets[i]), i) for i in tri]
    # sort this, by vertex class first with translation as tiebreaker
    # this ensures that equivalent triangles will have their vertices in the same order
    rs = sorted(representative)
    # translations for each vertex
    translations = np.array([r[1] for r in rs])

    # generate all the faces, including the whole cell itself
    for face_size in range(dimension+1, 0, -1):
        # Enumerate all faces of face_size vertices
        for face in combinations(range(dimension+1), face_size):
            # compute the relative translation for 2nd-1st, 3rd-1st, ... vertex of the face
            rtrnss = tuple(tuple(translations[face[i]]-translations[face[0]]) for i in range(1, face_size))
            # use the vertex equivalence classes, and the relative translations as index into our dict
            rel_repr = (tuple(rs[v][0] for v in face), rtrnss)
            # add our face as cell represented by this class
            rel_reprs[rel_repr].add(tuple(sorted(rs[v][2] for v in face)))



# TODO: we still need to establish the face/coface relations,
# in order to convert the data into a boundary matrix, and to
# compute radius values for each cell.


'''
Now pick one cell per equivalence class.
We follow the convention from Car10 to choose a canonical representative,
as follows:
Each vertex of the cell has an offset vector. 
For the d+1 vertices, these vectors are (coordinates x1, ... xd): 
(o_x1,0,  o_x2,0,  ...,  o_xd,0)
(o_x1,1,  o_x2,1,  ...,  o_xd,1)
 ...
(o_x1,d,  o_x2,d,  ...,  o_xd,d)
Example, in 2D, if we name the coordinates x and y, they are:
(o_x0, o_y0)
(o_x1, o_y1)
(o_x2, o_y2)
We take the representative such that
min_i {o_x1,i} = 0
For now, this is only for visualization purposes.

TODO: Prove that there the representative chosen like this 
always exists and is unique.

Question: We might in fact get an edge/vertex whose co-faces
are not in the set of representatives?
Or does this scheme solve the question below?
Question: If done more elaborately, we could get a nicer
set of representatives that is nicely connected (i.e. the
underlying space restriced to d and d-1 simplices' interiors
is already connected). Is there a simple way how to do that?
'''
final_simplices = []
if representative_mode == 0:
    for cell_class in rel_reprs.values():
        for cell in cell_class:
            # get the translation vectors for each vertex, 
            # and put them all into a matrix.
            tr_vecs = np.array([mpoints_offsets[v] for v in cell])
            if all(np.min(tr_vecs, axis=0) == np.zeros(dimension)):
                # over all coordinates, the minimum of the offsets for
                # all vertices is 0 in that coordinate 
                final_simplices.append(cell)
                # --> canonical representative
elif representative_mode == 1:
    for cell_class in rel_reprs.values():
        for cell in cell_class:
            canonical_vertices = [(list(mpoints[v]), v) for v in cell]
            canonical_vertices.sort(key=lambda x:x[0])
            # if the first point with respect to 1st coordinate
            # (other coordinates as tiebreakers) is canonical, take the cell
            if canonical_vertices[0][1] < npoints:
                final_simplices.append(cell)
elif representative_mode == 2:
    for cell_class in rel_reprs.values():
        for cell in cell_class:
            canonical_vertices = [(list(mpoints_offsets[v]), v) for v in cell]
            canonical_vertices.sort(key=lambda x:x[0])
            # if the first point with respect to 1st coordinate
            # (other coordinates as tiebreakers) is canonical, take the cell
            if canonical_vertices[0][1] < npoints:
                final_simplices.append(cell)
else:
    # Simply pick a random representative:
    final_simplices = [list(t)[0] for t in rel_reprs.values()]

# for rep,cells in rel_reprs.items():
#     print("Vertices: {:<10}, Translations: {:<16} -- Cells: {}".format(str(rep[0]), str(rep[1]), cells))


# plot the translates of the points into 3*FD and their Delaunay 
# triangulation restricted to triangles having a vertex in the FD.
# Highlight a representative set of triangles.
def cells(triangles, alpha):
    for tri in triangles:
        dim = len(tri) - 1
        if dim == 0:
            vx = tri[0]
            ax.plot(mpoints[vx][0], mpoints[vx][1], 'o', color='black')
        elif dim == 1:
            xs = [mpoints[vx][0] for vx in tri]
            ys = [mpoints[vx][1] for vx in tri]
            line = plt.Line2D(xs, ys, color='black', lw=3.0, alpha=alpha)
            ax.add_artist(line)
        elif dim == 2:
            vxs = []
            for vx in tri:
                vxs.append([mpoints[vx][0], mpoints[vx][1]])
            pol = plt.Polygon(vxs, color='black', alpha=alpha)
            ax.add_artist(pol)


if dimension == 2:
    ax = plt.gca()
    ax.cla()
    ax.axis('equal')
    plt.xlim([-3,3])
    plt.ylim([-3,3])
    ax.plot(mpoints[:,0], mpoints[:,1], 'o', color='red')
    # # draw lattice points
    # ax.plot([i*basevecs[0][0]+j*basevecs[1][0] for i in range(-5,5) for j in range(-5,5)], [i*basevecs[0][1]+j*basevecs[1][1] for i in range(-5,5) for j in range(-5,5)], 'o')
    # # draw basis vectors
    # for i in range(len(basevecs)):
    #     ax.arrow(0, 0, basevecs[i][0], basevecs[i][1], head_width=0.1, head_length=0.1, fc='k', ec='k')
    cells(simplices, 0.2)
    cells(final_simplices, 0.5)
    vdomain = plt.Polygon(np.array(ccs), alpha=0.1, edgecolor='b')
    ax.add_artist(vdomain)
    # draw all adjacent copies of the voronoi domain
    for t in ts:
        ax.add_artist(plt.Polygon(np.array(ccs)+t, alpha=0.1, edgecolor='b'))
    v3domain = plt.Polygon(3*np.array(ccs), alpha=0.1, edgecolor='b')
    ax.add_artist(v3domain)
    if draw_labels:
        for i in range(len(mpoints)):
            ax.text(mpoints[i][0], mpoints[i][1], str(i))
    plt.show()





# This is a lazy implementation of boundary and coboundary map
boundary = defaultdict(list)
# vertices that are in some coface, but not in the cell itself
coboundary_vertices = defaultdict(list)

for cell in final_simplices:
    # cell is a tuple of vertex indices
    if len(cell) > 1:
        # 
        for vertex in cell:
            # face is the cell with that one vertex removed
            face = tuple(sorted(set(cell) - set([vertex])))
            cbd_vertex = vertex # to be changed if we have a shift
            tr_vecs = np.array([mpoints_offsets[v] for v in face])
            # offset vector by which to shift the face to get canonical representative
            to_shift = -np.min(tr_vecs, axis=0)
            # if the face is not canonical, we need to find the equivalent canonical one
            if any(to_shift != np.zeros(dimension)):
                canonical_face = []
                # we do the same process for the face vertices nad the coboundary vertex
                # this is quick and dirty
                for i,v in enumerate(face):
                    # for each vertex, get its desired offset
                    new_offset = tuple(tr_vecs[i] + to_shift)
                    # and its representative
                    rep_v = mpoints_classes[v]
                    # find the index of the shifted vertex
                    shifted_vx = rep_offset_to_index[(rep_v, new_offset)]
                    # this is the corresponding vertex in the canonical face
                    canonical_face.append(shifted_vx)
                face = tuple(sorted(canonical_face))

                # Also translate the vertex that is missing from the face,
                # so we get the correct index for the coboundary vertex
                new_offset = tuple(mpoints_offsets[vertex] + to_shift)
                rep_v = mpoints_classes[vertex]
                shifted_vx = rep_offset_to_index[(rep_v, new_offset)]
                cbd_vertex = shifted_vx

            boundary[cell].append(face)
            coboundary_vertices[face].append(cbd_vertex)


# print(np.array(final_simplices))
# print(boundary)
# print(coboundary_vertices)

radius_values = []
dimensions = []
for cell in final_simplices:
    dim = len(cell)
    dimensions.append(dim-1)
    if dim == 1:
        r = 0
    elif 1 < dim < dimension+1:
        #print(np.array(cell), np.array(coboundary_vertices[cell]))
        # smallest enclosing ball of cell vertices excluding coface vertices
        cc, r = miniexonball([], list(mpoints[np.array(cell)]), list(mpoints[np.array(coboundary_vertices[cell])]))
    else:
        cc, r = miniexonball([], list(mpoints[np.array(cell)]), [])
    radius_values.append(r)

# sort cells by dimension with radius value as tiebreaker
# dimension first to ensure that faces can't have bigger index than cofaces
sorted_cells_with_radius = sorted(zip(final_simplices, zip(dimensions, radius_values)), key=lambda x:x[1])
cell_indices = dict()
for i,cell in enumerate(sorted_cells_with_radius):
    cell_indices[cell[0]] = i
    print("{:2d}: {:<15} dim: {}, rad: {:.4f}".format(i, str(cell[0]), cell[1][0], cell[1][1]))


boundary_matrix = phat.boundary_matrix(representation = phat.representations.vector_vector)
bdmx = []
for cell in sorted_cells_with_radius:
    dim = cell[1][0]
    if dim == 0:
        # If it's a vertex, it doesn't have a boundary
        bd = []
    else:
        # for each face (select all but 1 point), look up the index in
        # the filtration dictionary and put the list of these indices as boundary
        bd = sorted([cell_indices[face] for face in boundary[cell[0]]])
        # remove pairs of duplicates, e.g. edges forming a cycle alone
        bd_no_duplicates = []
        i = 0
        while i < len(bd):
            if i == len(bd) - 1:
                bd_no_duplicates.append(bd[i])
            elif bd[i] == bd[i+1]:
                i += 1
            else:
                bd_no_duplicates.append(bd[i])
            i += 1
        #print(bd, bd_no_duplicates)
        bd = bd_no_duplicates
    bdmx.append((dim, sorted(bd)))
boundary_matrix.columns = bdmx


ppairs = boundary_matrix.compute_persistence_pairs()
ppairs.sort()

unpaired_cells = set(range(len(final_simplices)))
pdict = defaultdict(int)
print("\nThere are %d persistence pairs: " % len(ppairs))
for pair in ppairs:
    unpaired_cells.remove(pair[0])
    unpaired_cells.remove(pair[1])
    dim = sorted_cells_with_radius[pair[0]][1][0]
    birth = round(sorted_cells_with_radius[pair[0]][1][1], 4)
    death = round(sorted_cells_with_radius[pair[1]][1][1], 4)
    if death - birth > 1e-6:
        pdict[(birth, death, dim)] += 1
    print("(%d, %d) - Birth: %.4f, Death: %.4f, Dim: %d" % (pair + (birth, death, dim)))

print("\nNontrivial pairs:")
for p,m in sorted(pdict.items()):
    print("Birth: %f, Death: %f, Dim: %d, Multiplicity: %d" % (p[0], p[1], p[2], m))
print("Cells without death:")
for i in unpaired_cells:
    cell = sorted_cells_with_radius[i]
    print("dim: {}, radius: {}".format(cell[1][0], cell[1][1]))
# Remark: rank of the k-th homology group for the d-dimensional torus
# is d choose k. 




'''
I MIGHT NOT ACTUALLY NEED THIS :D :D :D

Currently, our simplicial complex has some cells that have
the same combinatorial representation. This prevents the application
of the boundary matrix reduction algorithm to compute persistence.
We need to split cells with the same combinatorial representation
to get a subdivision of our complex where all combinatorial
representations are unique. It is sufficient to scan edges for
duplicates, as higher dimensional duplicates will have an a
duplicate edge as a face.

Make a dictionary of 'duplicate' edges:
key is a sorted pair of representative vertices.

Now values are also dictionaries:
for each types of representative edge, it can occur with
different relative translations.

For each relative translation, we store the cells that have
such an edge.


We now proceed to subdivide edges. If a combinatorial edge occurs in 
multiple relative translations, we split all these different edges,
one by one, into two (or three if it's a self edge) parts.

We need to split all affected simplices having this edge as a face.
Note that doing it only for top-dimensional simplices would suffice,
if we weren't also interested in the radius values of each cell.
We can give the new vertex a negative value. Its radius value is the
same as that of the original edge. These new vertices strictly speaking
also have a representative and translation vector associated to them.
However, we don't need these as they are only required for dictionary
lookup, which we don't need as we won't split edges involving this 
vertex anymore.

Splitting a 2-cell in 2D: we take the vertex which is not part of the
edge. We connect it to the new vertex, and together with each of the
old vertices it forms two new cells. For the first of these, there is
an edge not involving the new vertex. We need to look this one up in
the dictionary, find the original cell that we just split in its
co-face list, and modify it. Similarly for the second new cell.


# dictionary of combinatorial edges
cedict = defaultdict(lambda:defaultdict(list))


for tri in final_simplices:
    for edge in combinations(tri, 2):
        # sort edge vertices by ID of their class representative, 
        # with translation vector as tiebreaker
        edge = sorted(edge, key=lambda x:(mpoints_classes[x], tuple(mpoints_offsets[x])))
        # get relative translation
        trns = mpoints_offsets[edge[1]] - mpoints_offsets[edge[0]]
        trns = tuple(trns)
        # get representative vertices of this edge, sorted due to edge being sorted
        e = tuple(mpoints_classes[v] for v in edge)
        cedict[e][trns].append(list(tri))


for k,v in cedict.items():
    print(k)
    for kk,vv in v.items():
        print(kk, vv)
    print()

'''

