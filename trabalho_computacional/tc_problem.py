
#%%  Imports
from docplex.mp.model import Model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

model = Model(name='equipment_maintenance_sched')
#model.parameters.mip.tolerances.mipgap = 0.05

#%% Load data
ClusterDB = pd.read_csv('ClusterDB.csv', header=None, sep=',', names=['ID', 'eta', 'beta']).set_index('ID')
EquipDB = pd.read_csv('EquipDB.csv', header=None, sep=',', names=['ID', 't0', 'cluster', 'failCost']).set_index('ID')
MPDB = pd.read_csv('MPDB.csv', header=None, sep=',', names=['ID', 'riskFactor', 'planCost']).set_index('ID')

EquipDB = EquipDB.join(ClusterDB, on='cluster', sort=False)

n_equipment = EquipDB.shape[0]
n_plans = MPDB.shape[0]
deltaT = 5 # anos
#%% Calculate the failure probabilities

def calculateFailureProbability(t0, k, eta, beta):
    dcf = lambda x: 1 - np.exp(-((x/eta)**beta))
    Ft0 = dcf(t0)
    FtdT = dcf(t0 + k*deltaT)
    return (FtdT - Ft0)/(1 - Ft0)

failProbs = np.zeros((n_equipment, n_plans))
for i in range(n_equipment):
    for j in range(n_plans):
        equipment = EquipDB.loc[(i+1)]
        eta = equipment.eta
        beta = equipment.beta
        t0 = equipment.t0
        k = MPDB.loc[(j+1)].riskFactor

        failProbs[i,j] = calculateFailureProbability(t0, k, eta, beta)

#%% Build optimization problem
# Decision vars
y = model.binary_var_matrix(range(n_equipment), range(n_plans), name='y')

# Objectives
# 1: maintenace cost
maintenance_cost = model.sum(y[i,j] * MPDB.loc[(j+1)].planCost for i in range(n_equipment) for j in range(n_plans))

# 2: expected failure cost
expected_failure_cost = model.sum(failProbs[i,j] * y[i,j] * EquipDB.loc[(i+1)].failCost for i in range(n_equipment) for j in range(n_plans))

# Restrictions
# 1: only one plan can be used for each equipment
for i in range(n_equipment):
    model.add_constraint(model.sum(y[i,j] for j in range(n_plans)) == 1)

#%% Solve
# TEST: Plambda scalarization
n_pareto = 200
results = np.zeros((n_pareto, n_equipment))
for index, value in enumerate(np.linspace(0, 1, n_pareto)):
    model.minimize(value * maintenance_cost + (1-value)*expected_failure_cost)
    solution = model.solve()

    for i in range(n_equipment):   
        plans = []     
        for j in range(n_plans):
            plans.append(int(model.get_var_by_name('y_{}_{}'.format(i,j))))
        
        results[index, i] = np.argmax(plans) + 1

np.savetxt("Solution01.csv", results, delimiter=",", fmt='%d' )

#%%
model.minimize(maintenance_cost + expected_failure_cost)
solution = model.solve(log_output=True)

if(solution):
    model.report()
    model.print_information()
    model.print_solution()
# %% Eval HVI
from oct2py import octave
octave.eval('pkg load statistics')
hvi = octave.EvalParetoApp('Solution01.csv')


# %%
