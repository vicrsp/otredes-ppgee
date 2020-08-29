#%%  CPLEX imports
from docplex.mp.model import Model
import numpy as np
import matplotlib.pyplot as plt

model = Model(name='mine_schedule')
#%% INSTANCE DATA
n_trucks = 34 # The number of trucks
n_years = 5 # The time period
n_bins = 10 # The number of age bins

# Discounted cost value for truck T at age bin B and period T
C = 100 * np.random.random((n_trucks, n_bins, n_years))
# Engine rebuild cost
FE = [750000] * n_trucks
# Available truck hours per period T
A = np.random.randint(low=7000, high=10000, size = (n_trucks, n_years))
# Maximum available truck hours at age bin V
M = np.tile(np.arange(start=0, stop=n_bins, step=5000), n_trucks)
# The cumulative used hours for truck t at time period t
H = np.random.randint(low=0, high=1000, size = (n_trucks, n_years))
# The required truck hours for a given time period y
R = np.random.randint(low=200000, high=240000, size = n_years)
# The initial truck ages
IAge = np.random.randint(low=0, high=40000, size = n_trucks)
# The critical age bin
c_critical = 5


#%% CPLEX VARS
x = model.integer_var_cube(range(n_trucks), range(n_bins), range(n_years), name='x')
y_bin = model.binary_var_cube(range(n_trucks), range(n_bins), range(n_years), name='y_b')
y_critical = model.binary_var_cube(range(n_trucks), range(n_bins), range(n_years), name='y_c')
# h = model.integer_var_matrix(range(n_trucks), range(n_years + 1), name='h')

#%% CPLEX MODEL
# Objective: TODO: revisar se não está adicionando FE a mais
model.minimize(model.sum(x[t,b,y] * C[t,b,y] + y_critical[t,c_critical,y]*FE[t] for t in range(n_trucks) for b in range(n_bins) for y in range(n_years)))

# Constraints
# (1) - Trucks maximum availability
for y in range(n_years):
    for t in range(n_trucks):
        model.add_constraint(model.sum(x[t,b,y] for b in range(n_bins)) <= A[t,y])
# (2) - Ensure move to upper bin 
for t in range(n_trucks):
    for b in range(n_bins): 
        model.add_constraint(model.sum(x[t,b,y] for y in range(n_years)) <= M[b])
# (3) - Accumulated truck age
# for y in range(n_years):
#     for t in range(n_trucks):
#         if(y == 0):
#             model.add_constraint(model.sum(x[t,b,y] for b in range(n_bins)) + h[y] == h[y+1])
#         else:
#             model.add_constraint(model.sum(x[t,b,y] for b in range(n_bins)) + h[y-1] == h[y])

# (4) - Correct bin order (lower bound)
for t in range(n_trucks):
    for b in range(n_bins): 
        for y in range(n_years): 
            model.add_constraint(model.sum(x[t,b,k] for k in range(y)) - M[b] * y_bin[t,b,y] >= 0)

# (5) - Correct bin order (upper bound)
for t in range(n_trucks):
    for b in range((n_bins - 1)): # we can skip the last bin: TODO: validate this assumption
        for y in range(n_years): 
            model.add_constraint(x[t,(b+1),y] - M[(b+1)]* model.sum(y_bin[t,b,k] for k in range(y)) <= 0)

# (6) - Required truck yours per year
for y in range(n_years):
    model.add_constraint(model.sum(x[t,b,y] for t in range(n_trucks) for b in range(n_bins)) == R[y])
    # model.add_constraint(model.sum(h[t,b,y] for t in range(n_trucks) for b in range(n_bins)) == R[y])
    
# (7) - Initial ages
for t in range(n_trucks):
    model.add_constraint(model.sum(x[t,b,0] for b in range(n_bins)) >= IAge[t])

model.export_as_lp('D:\\otredes-ppgee\\trabalho_final\\test.lp')

#%% SOLVE