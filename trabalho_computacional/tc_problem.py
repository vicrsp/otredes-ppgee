
#%%  Imports
from docplex.mp.model import Model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#model = Model(name='equipment_maintenance_sched')
#model.parameters.mip.tolerances.mipgap = 0.05

#%% Model class
class EquipmentMaintenanceProblem:
    def __init__(self, deltaT = 5):
        self.ClusterDB = pd.read_csv('ClusterDB.csv', header=None, sep=',', names=['ID', 'eta', 'beta']).set_index('ID')
        self.EquipDB = pd.read_csv('EquipDB.csv', header=None, sep=',', names=['ID', 't0', 'cluster', 'failCost']).set_index('ID')
        self.MPDB = pd.read_csv('MPDB.csv', header=None, sep=',', names=['ID', 'riskFactor', 'planCost']).set_index('ID')
        self.EquipDB = self.EquipDB.join(self.ClusterDB, on='cluster', sort=False)

        self.n_equipment = self.EquipDB.shape[0]
        self.n_plans = self.MPDB.shape[0]
        self.deltaT = deltaT 
        self.failProbs = self.get_failure_probabilities()

    def calculateFailureProbability(self, t0, k, eta, beta):
        dcf = lambda x: 1 - np.exp(-((x/eta)**beta))
        Ft0 = dcf(t0)
        FtdT = dcf(t0 + k*self.deltaT)
        return (FtdT - Ft0)/(1 - Ft0)
    
    def get_failure_probabilities(self):
        failProbs = np.zeros((self.n_equipment, self.n_plans))
        for i in range(self.n_equipment):
            for j in range(self.n_plans):
                equipment = self.EquipDB.loc[(i+1)]
                eta = equipment.eta
                beta = equipment.beta
                t0 = equipment.t0
                k = self.MPDB.loc[(j+1)].riskFactor
                failProbs[i,j] = self.calculateFailureProbability(t0, k, eta, beta)
        
        return failProbs

    def reset_model(self):
        self.model = Model(name='equipment_maintenance_sched')
        self.y = self.set_decision_vars()
        self.set_default_constraints()

    def set_default_constraints(self):
        for i in range(self.n_equipment):
            self.model.add_constraint(self.model.sum(self.y[i,j] for j in range(self.n_plans)) == 1)

    def set_decision_vars(self):
        return self.model.binary_var_matrix(range(self.n_equipment), range(self.n_plans), name='y')

    def get_objectives(self):
        maintenance_cost = self.model.sum(self.y[i,j] * self.MPDB.loc[(j+1)].planCost for i in range(self.n_equipment) for j in range(self.n_plans))
        expected_failure_cost = self.model.sum(self.failProbs[i,j] * self.y[i,j] * self.EquipDB.loc[(i+1)].failCost for i in range(self.n_equipment) for j in range(self.n_plans))

        return maintenance_cost, expected_failure_cost



    def get_solution_values(self):
        values = np.zeros((self.n_equipment,self.n_plans))
        for i in range(self.n_equipment):   
            for j in range(self.n_plans):
                values[i,j] = int(self.y[i,j]) 
        return values

    def eval_maintenance_cost(self, x):
        value = 0
        for i in range(self.n_equipment):
            for j in range(self.n_plans):
                value = value + self.MPDB.loc[(j+1)].planCost * x[i,j]

        return value

    def eval_expected_failure_cost(self, x):
        value = 0
        for i in range(self.n_equipment):
            for j in range(self.n_plans):
                value = value + self.failProbs[i,j] * x[i,j] * self.EquipDB.loc[(i+1)].failCost

        return value

    def solve_plambda(self, n_pareto = 200, output_file = None):
        self.reset_model()
        fobj1, fobj2 = self.get_objectives()

        solutions = np.zeros((n_pareto, self.n_equipment))
        pareto_values = np.zeros(n_pareto)

        w = np.linspace(0, 1, n_pareto)
        for ip in range(n_pareto):
            self.model.minimize(w[ip] * fobj1 + (1-w[ip])*fobj2)
            solution = self.model.solve()

            solution_values = self.get_solution_values()

            for i in range(self.n_equipment):   
                solutions[ip, i] = np.argmax(solution_values[i,:]) + 1
            
            pareto_values[ip] = solution.get_objective_value()
            print('Iteration Plambda {}: {}'.format(ip, pareto_values[ip]))

        if(output_file != None):
            np.savetxt("{}.csv".format(output_file), solutions, delimiter=",", fmt='%d' )
        
        return solutions, pareto_values

    def solve_pepsilon(self, n_pareto = 200, output_file = None):
        self.reset_model()
        maint_cost, fail_cost = self.get_objectives()

        # First vertex
        self.model.minimize(maint_cost)
        self.model.solve()
        x1 = self.get_solution_values()

        # Second vertex
        self.model.minimize(fail_cost)
        self.model.solve()
        x2 = self.get_solution_values()

        # Calulate the eps bounds
        epsilon_min = self.eval_expected_failure_cost(x2)
        epsilon_max = self.eval_expected_failure_cost(x1)
        epsilon = np.linspace(epsilon_min, epsilon_max, n_pareto)

        solutions = np.zeros((n_pareto, self.n_equipment))
        pareto_values = np.zeros(n_pareto)
        # pareto_values[0] = epsilon_max
        # solutions[0, :] = np.argmax(x2[i,:]) + 1

        ## TODO: aumentar a resolução do epslon próximo das bordas
        # Parece que encontrar mais soluções nessa região aumenta o HVI
        # da fronteira pareto inicial

        for ip in range(n_pareto):
            self.reset_model()
            maint_cost, fail_cost = self.get_objectives()
            
            # Add the p-epsilon constraint
            self.model.add_constraint(fail_cost - epsilon[ip] <= 0)

            # Add the objective
            self.model.minimize(maint_cost)

            solution = self.model.solve()
            solution_values = self.get_solution_values()

            for i in range(self.n_equipment):   
                solutions[ip, i] = np.argmax(solution_values[i,:]) + 1
            
            pareto_values[ip] = solution.get_objective_value()
            print('Iteration Pepsilon {}: {}'.format(ip, pareto_values[ip]))
        
        # pareto_values[-1] = epsilon_min
        
        if(output_file != None):
            np.savetxt("{}.csv".format(output_file), solutions, delimiter=",", fmt='%d' )

        return solutions, pareto_values
#%% SOLVE
problem = EquipmentMaintenanceProblem()
_, pareto_values_plambda = problem.solve_plambda(n_pareto = 50, output_file='Solution03')
_, pareto_values_pepsilon = problem.solve_pepsilon(n_pareto = 50, output_file='Solution02')

plt.plot(pareto_values_pepsilon, 'r.-')
plt.plot(pareto_values_plambda, 'b.-')

# %% Eval HVI
from oct2py import octave
octave.eval('pkg load statistics')
hvi_pe = octave.EvalParetoApp('Solution02.csv')
hvi_plam = octave.EvalParetoApp('Solution03.csv')


# %%
