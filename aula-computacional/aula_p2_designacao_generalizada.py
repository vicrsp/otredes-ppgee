# -*- coding: utf-8 -*-
import numpy as np
import gurobipy as gp
from gurobipy import GRB, quicksum
import matplotlib.pyplot as plt


#%%  DESIGNAÇÂO GENERALIZADA
try:

    # Create a new model
    model = gp.Model("dg")
    
    # Number of tasks
    n = 50
    # Number of agents
    m = 5
    # Create variables
    c = np.random.randint(low=1, high=10, size=(n,m))
    x = model.addVars(n, m, obj=c, vtype=GRB.BINARY, name="x")
    
    # Constraint #1 
    for i in range(n):
        model.addConstr(quicksum(x[i,j] for j in range(m)) == 1)
    
    # Constraint #2
    for j in range(m):
        aj = np.random.randint(low=1, high=10, size=n)
        model.addConstr(quicksum(x[i,j] * aj[i] for i in range(n)) <= 44)
             
    model.write('dg.lp')

    # Optimize model
    model.optimize()

#    for i in range(n):
#        line = 'Line {}:'.format(i)
#        for j in range(m):
#            line = line + '{},'.format(model.getVarByName('x[{},{}]'.format(i,j)).x)
#        
#        print(line)
        
    image = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            image[i,j] = model.getVarByName('x[{},{}]'.format(i,j)).x
    
    plt.matshow(image)
    plt.xticks(range(m), range(m))
    plt.yticks(range(n), range(n))

    plt.show()
        
#    for v in model.getVars():
#        print('%s %g' % (v.varName, v.x))
        
    print('Obj: %g' % model.objVal)

except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))

except AttributeError:
    print('Encountered an attribute error')



#%%  SCIPY


