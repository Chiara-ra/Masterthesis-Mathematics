# This file creates a csv with points sampled from a (p,q)-torus knot 
# These points can be chosen either to lie on the unit square in R2
# or in the unit cube on a unit plane at 1/2 of the width of the cube

"""
BECAUSE I AM AN IDIOT, 
AND WAS NOT SURE HOW A (P,Q) KNOT IS FORMALLY DEFINED,
I DERIVED IT AGAIN USING WIKIPEDIA REFERENCES

IGNORE THIS

parametrisation of knot in R^3 (torus with radii 1,2),
with theta from (0,2pi):
https://en.wikipedia.org/wiki/Torus_knot

r =   cos(q*theta) + 2
x = r*cos(p*theta)
y = r*sin(p*theta)
z = - sin(q*theta)

parametrisation of surface in R^3:
https://en.wikipedia.org/wiki/Torus#Geometry

x = (2+cos(alpha))*cos(beta)
y = (2+cos(alpha))*sin(beta)
z = sin(alpha)

mapping from cylinder to (0,2pi)-square
alpha = arcsin(z)
beta  = arctan(y/x)

mapping from cylinder to (0,1)-square
alpha = arcsin(z)/(2*pi)
beta  = arctan(y/x)/(2*pi)

parametrisation of (p,q)-knot on unit square,
with theta from (0,2pi):
alpha = -q*theta/(2*pi)
beta  =  p*theta/(2*pi)


FINALLY 
parametrisation of (p,q)-knot on unit square,
with theta from (0,1):
alpha = -q*theta
beta  =  p*theta


"""


import numpy as np

dim = 3
p = 2
q = 3
res = 100


spacing = np.linspace(0,1,res,endpoint=False)
x = np.mod(- q*spacing, 1)
y = np.mod(  p*spacing, 1)

if dim == 2:
    a = np.array([x,y]).transpose()
    
elif dim == 3:
    z = np.array([1/2 for dummy in range(res)])
    a = np.array([x,y,z]).transpose()
    
np.savetxt(f"torus_knot_{dim}d_p{p}_q{q}.csv", a, delimiter=",")