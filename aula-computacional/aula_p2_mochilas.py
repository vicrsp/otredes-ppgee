# -*- coding: utf-8 -*-
import numpy as np
import gurobipy as gp
from gurobipy import GRB, quicksum
import matplotlib.pyplot as plt


#%%  DESIGNAÇÂO GENERALIZADA
try:

    # Create a new model
    model = gp.Model("mochilas")
    
    # Max weight
    wmax = 10
    cap = wmax
    # Number of items
    n = 10
    # Weights
    w = np.random.randint(low=1, high=wmax, size=n)
    # Create variables
    # The maximum amount of backpacks is n, meaning each item has w=wmax.
    # So the number of columns is:
    # => n*n (item i can be in n backpacks) + n (if backpack is in use)
    c = np.append(np.zeros((1, n*n)), np.ones((1,n)))
    
    x = model.addVars((n*n + n), obj = c, vtype=GRB.BINARY, name="x")
    
    # All items must be placed in a backpack
    for j in range(n):
        model.addConstr(quicksum(x[i*n + j] for i in range(n)) == 1)
        
    # The sum of all items in a backpack cannot exceed its capacity
    for j in range(n):
        model.addConstr(quicksum(x[j*n + i] * w[i] for i in range(n)) <= cap * x[n*n + j])
    
        
    # Constraint 
    model.write('mochilas.lp')

    # Optimize model
    model.optimize()
    
    line = 'Mochilas:'
    for i in range(n):
        line = line + '{},'.format(model.getVarByName('x[{}]'.format(n*n + i)).x)
        
    print(line)
        
        
#    for v in model.getVars():
#       print('%s %g' % (v.varName, v.x))
        
    print('Obj: %g' % model.objVal)

except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))

except AttributeError:
    print('Encountered an attribute error')



#%%  SCIPY


