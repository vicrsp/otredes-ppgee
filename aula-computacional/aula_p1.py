# -*- coding: utf-8 -*-
# This example formulates and solves the following simple MIP model:
#  maximize
#        x + 2y
#  subject to
#        x + y  <= 4
#        x      <= 2
#            y  <= 3
#        x, y > 0

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

#%%  GUROBI

try:

    # Create a new model
    m = gp.Model("lp1")


    # Create variables
    x = m.addVar(vtype=GRB.CONTINUOUS, name="x")
    y = m.addVar(vtype=GRB.CONTINUOUS, name="y")

    # Set objective
    m.setObjective(x + 2*y, GRB.MAXIMIZE)


    # Add constraints
    m.addConstr(x + y <= 4, "c1")
    m.addConstr(x <= 2, "c2")
    m.addConstr(y <= 3, "c3")
    m.addConstr(x >= 0, "c4")
    m.addConstr(y >= 0, "c5")

    # Optimize model
    m.optimize()

    print('Sol: x={}, y={}'.format(x.X,y.X))
    print('Obj: %g' % m.objVal)

except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))

except AttributeError:
    print('Encountered an attribute error')


#%%  GUROBI - MATRIX

try:

    # Create a new model
    m = gp.Model("lp2")


     # Create variables
    x = m.addMVar(shape=2, vtype=GRB.CONTINUOUS, name="x")

    # Set objective
    obj = np.array([1.0, 2.0])
    m.setObjective(obj @ x, GRB.MAXIMIZE)

    # Build (sparse) constraint matrix
    data = np.array([[1,1], [1,0], [0,1]])

    A = csr_matrix(data)

    # Build rhs vector
    rhs = np.array([4, 2, 3])

    # Add constraints
    m.addConstr(A @ x <= rhs, name="c")

    # Optimize model
    m.optimize()

    print(x.X)
    print('Obj: %g' % m.objVal)

except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))

except AttributeError:
    print('Encountered an attribute error')



#%%  SCIPY
#import numpy as np
#from scipy.optimize import linprog
f = np.array([-1, -2])
A = np.array([[1,1], [1,0], [0,1]])
b = np.array([4, 2, 3])

res = linprog(f, A, b, method='simplex')
print(res)


