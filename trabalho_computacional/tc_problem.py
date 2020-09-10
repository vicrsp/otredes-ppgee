
#%%  Imports
from docplex.mp.model import Model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

model = Model(name='equipment_maintenance_sched')
#model.parameters.mip.tolerances.mipgap = 0.05

#%% Load data
ClusterDB = pd.read_csv('ClusterDB.csv', header=None, sep=',', names=['ID', 'eta', 'beta'])
EquipDB = pd.read_csv('EquipDB.csv', header=None, sep=',', names=['ID', 't0', 'cluster', 'failCost'])
MPDB = pd.read_csv('ClusterDB.csv', header=None, sep=',', names=['ID', 'riskFactor', 'planCost'])

EquipDB = pd.merge(EquipDB, ClusterDB, left_on='cluster', right_on='ID', suffixes=('_equip', '_cluster'))

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
        equipment = EquipDB.loc[EquipDB.ID_equip == (i+1)]
        eta = equipment.eta.values[0]
        beta = equipment.beta.values[0]
        t0 = equipment.t0.values[0]
        k = MPDB.loc[MPDB.ID == (j+1)].riskFactor.values[0]

        failProbs[i,j] = calculateFailureProbability(t0, k, eta, beta)

#%% Build optimization problem
# Decision vars
y = model.binary_var_matrix(range(n_equipment), range(n_plans), name='y')

# Objectives
# 1: maintenace cost
maintenance_cost = model.sum(y[i,j] * MPDB.loc[MPDB.ID == (j+1)].planCost.values[0] for i in range(n_equipment) for j in range(n_plans))

# 2: expected failure cost
expected_failure_cost = model.sum(failProbs[i,j] * y[i,j] * EquipDB.loc[EquipDB.ID_equip == (i+1)].failCost.values[0] for i in range(n_equipment) for j in range(n_plans))

# Restrictions
# 1: only one plan can be used for each equipment
for i in range(n_equipment):
    model.add_constraint(model.sum(y[i,j] for j in range(n_plans)) == 1)

#%% Solve
# TEST: Plambda scalarization
model.minimize(maintenance_cost + expected_failure_cost)
solution = model.solve(log_output=True)

if(solution):
    model.report()
    model.print_information()
    model.print_solution()
# %%
