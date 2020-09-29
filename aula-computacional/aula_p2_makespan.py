# -*- coding: utf-8 -*-
import numpy as np
import gurobipy as gp
from gurobipy import GRB, quicksum
import matplotlib.pyplot as plt

#%%  MAKESPAN - GUROBI
try:

    # Create a new model
    model = gp.Model("makespan")
    
    # Number of tasks
    n = 20
    # Number of machines
    m = 5
    # Processing times
    p = np.random.randint(low=1, high=25, size=(m,n))

    # Create variables
    x = model.addVars(m, n, vtype=GRB.BINARY, name="x")
    y = model.addVar(vtype=GRB.INTEGER, name="y")
    
    model.setObjective(y, GRB.MINIMIZE)

    # Constraint #1: machine allocation should be below makespan
    for i in range(m):
        model.addConstr(quicksum(x[i,j] * p[i,j] for j in range(n)) <= y)

    # Constrain #2: each job should be scheduled once
    for j in range(n):
        model.addConstr(quicksum(x[i,j] for i in range(m)) == 1)

    model.write('makespan.lp')

    # Optimize model
    model.optimize()

    image = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            image[i,j] = model.getVarByName('x[{},{}]'.format(i,j)).x
    
    plt.matshow(image)
    plt.yticks(range(m), range(m))
    plt.xticks(range(n), range(n))

    plt.show()
        
#    for v in model.getVars():
#        print('%s %g' % (v.varName, v.x))
        
    print('Obj: %g' % model.objVal)

except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))

except AttributeError:
    print('Encountered an attribute error')


#%%  MAKESPAN - CPLEX
import cplex
from cplex.exceptions import CplexError
from docplex.mp.model import Model

makespan_prob = Model(name='makespan')

x = makespan_prob.binary_var_matrix(range(m), range(n), name='x')
y = makespan_prob.integer_var(lb=0, name='y')

# Constraint #1: machine allocation should be below makespan
for i in range(m):
    makespan_prob.add_constraint(makespan_prob.sum(x[i,j] * p[i,j] for j in range(n)) <= y)

# Constrain #2: each job should be scheduled once
for j in range(n):
    makespan_prob.add_constraint(makespan_prob.sum(x[i,j] for i in range(m)) == 1)

makespan_prob.minimize(y)

makespan_prob.solve()
makespan_prob.print_information()
makespan_prob.print_solution()

makespan_prob.export_as_lp('D:\\otredes-ppgee\\aula-computacional\\makespan_cplex.lp')
image = np.zeros((m, n))
for i in range(m):
    for j in range(n):
        image[i,j] = makespan_prob.get_var_by_name('x_{}_{}'.format(i,j))
    
plt.matshow(image)
plt.yticks(range(m), range(m))
plt.xticks(range(n), range(n))

plt.show()



# %%
