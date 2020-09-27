
#%%  Imports
from docplex.mp.model import Model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
#%% Model class
class EquipmentMaintenanceProblem:
    def __init__(self, deltaT = 5):
        self.ClusterDB = pd.read_csv('ClusterDB.csv', header=None, sep=',', names=['ID', 'eta', 'beta']).set_index('ID')
        self.EquipDB = pd.read_csv('EquipDB.csv', header=None, sep=',', names=['ID', 't0', 'cluster', 'failCost']).set_index('ID')
        self.MPDB = pd.read_csv('MPDB.csv', header=None, sep=',', names=['ID', 'riskFactor', 'planCost']).set_index('ID')
        self.EquipDB = self.EquipDB.join(self.ClusterDB, on='cluster', sort=False)

        self.equipDBArray = self.EquipDB.to_numpy()
        self.MPDBArray = self.MPDB.to_numpy()

        self.n_equipment = self.EquipDB.shape[0]
        self.n_plans = self.MPDB.shape[0]
        self.deltaT = deltaT 
        self.failProbs = self.get_failure_probabilities()

    def calculate_failure_probability(self, t0, k, eta, beta):
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
                failProbs[i,j] = self.calculate_failure_probability(t0, k, eta, beta)
        
        return failProbs

    def init_model(self):
        self.model = Model(name='equipment_maintenance_sched')
        self.y = self.set_decision_vars()
        self.set_default_constraints()

    def reset_model(self):
        self.model.clear_constraints()
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
        for j in range(self.n_plans):
            planCost = self.MPDB.loc[(j+1)].planCost
            for i in range(self.n_equipment):
                value = value + planCost * x[i,j]

        return value

    def eval_expected_failure_cost(self, x):
        value = 0
        for i in range(self.n_equipment):
            failCost = self.EquipDB.loc[(i+1)].failCost
            for j in range(self.n_plans):
                value = value + self.failProbs[i,j] * x[i,j] * failCost

        return value

    def eval_hvi(self, front, minref, maxref):
        b = np.argsort(front[:, 0])
        front = front[b, :]
        for i in range(front.shape[1]):
            front[:,i] = (front[:,i] - minref[i]) / (maxref[i] - minref[i])
        nadir = np.array([1, 1])
        area = 0
        front_areas = []

        for i in range(front.shape[0]):
            if(i < (front.shape[0] - 1)):
                hi = ((front[i+1,0]-front[i,0])*(nadir[1]-front[i,1]))
            else:
                hi = ((nadir[0]-front[i,0])*(nadir[1]-front[i,1]))
            area = area + hi
            front_areas.append(hi)
        return area, front_areas


    def solve_pepsilon(self, output_file = None):
        # start the timer
        start = time.time()

        self.init_model()
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
        epsilon_min = self.eval_maintenance_cost(x1)
        epsilon_max = self.eval_maintenance_cost(x2)

        utopic = np.array([self.eval_maintenance_cost(x1), self.eval_expected_failure_cost(x2)])
        nadir = np.array([self.eval_maintenance_cost(x2), self.eval_expected_failure_cost(x1)])

        epsilon = np.arange(epsilon_min, epsilon_max + 1)

        n_pareto = len(epsilon)
        solutions = np.zeros((n_pareto, self.n_equipment))
        pareto_values = np.zeros((n_pareto, 2))
        
        for index, eps_value in enumerate(epsilon):
            self.reset_model()            
            # Add the p-epsilon constraint
            self.model.add_constraint(maint_cost - eps_value <= 0)

            # Add the objective
            self.model.minimize(fail_cost)

            self.model.solve()
            solution_values = self.get_solution_values()

            for i in range(self.n_equipment):   
                solutions[index, i] = np.argmax(solution_values[i,:]) + 1
            
            pareto_values[index, 0] = self.eval_maintenance_cost(solution_values)
            pareto_values[index, 1] = self.eval_expected_failure_cost(solution_values)

            print('Iteration solution {}: {}'.format(index, pareto_values[index]))

        end = time.time()
        print('Elapsed time: {} sec'.format(end - start))

        hvi, _ = self.eval_hvi(pareto_values, utopic, nadir)
        print('Final HVI = {}'.format(hvi))

        if(output_file != None):
            np.savetxt("{}.csv".format(output_file), solutions, delimiter=",", fmt='%d' )

        return solutions, pareto_values

    def pareto_analysis(self, front):
        
        minref = front.min(axis=0).reshape(2,1)
        maxref = front.max(axis=0).reshape(2,1)

        _, ax = plt.subplots()

        ax.plot(front[:,0], front[:,1] , 'r.')
        ax.set_xlabel('Custo de manutenção')
        ax.set_ylabel('Custo esperado de Falha')

        hvi_original, areas = self.eval_hvi(front, minref, maxref)
        ax.legend(['HVI: {}'.format(hvi_original)])

        plt.show()

        df = pd.DataFrame(front, columns=['fobj1','fobj2'])
        df['area'] = areas

        return df
    
    def eval_file(self, filename):
        file_data = pd.read_csv(filename, sep=',', header=None).to_numpy()
        
        n, m = file_data.shape
        pareto_values = np.zeros((n, 2))

        for index in range(n):
            solution_file = file_data[index, :]
            solution_values = np.zeros((m, self.n_plans))

            for i in range(m):
                if(solution_file[i] == 1):
                    solution_values[i,:] = [1,0,0]
                elif(solution_file[i] == 2):
                    solution_values[i,:] = [0,1,0]
                else:
                    solution_values[i,:] = [0,0,1]

            pareto_values[index, 0] = self.eval_maintenance_cost(solution_values)
            pareto_values[index, 1] = self.eval_expected_failure_cost(solution_values)

            print('Processed row {}'.format(index))
        
        return self.pareto_analysis(pareto_values)



#%% SOLVE
problem = EquipmentMaintenanceProblem()
_, pareto_values_pepsilon = problem.solve_pepsilon(output_file='TC_Solution')

