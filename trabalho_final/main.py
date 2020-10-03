#%%  CPLEX imports
from docplex.mp.model import Model
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from os import path

#%% PROBLEM DEFINITION
class TruckMaintenanceProblem:
    def __init__(self, n_trucks = 10, n_bins = 20, n_years = 5):
        self.n_trucks = n_trucks # The number of trucks
        self.n_years = n_years # The time period
        self.n_bins = n_bins # The number of age bins

    def init_model(self, C, c_critical, FE, A, M, R):
        self.model = Model(name='mine_schedule')
        self.x, self.y_bin = self.set_decision_vars()
        self.set_objective(C, c_critical, FE)
        self.set_constraints(A, M, R)
    
    def set_decision_vars(self):
        x = self.model.integer_var_cube(range(self.n_trucks), range(self.n_bins), range(self.n_years), name='x')
        y_bin = self.model.binary_var_cube(range(self.n_trucks), range(self.n_bins), range(self.n_years), name='y_bin')
        return x, y_bin

    def set_objective(self, C, c_critical, FE):
        hour_costs = self.model.sum(self.x[t,b,y] * C[t,b,y] for t in range(self.n_trucks) for b in range(self.n_bins) for y in range(self.n_years))
        repair_costs = self.model.sum(self.y_bin[t,c_critical[t],y]*FE[t] for t in range(self.n_trucks) for y in range(self.n_years))
        self.model.minimize(self.model.sum(hour_costs + repair_costs))    

    def set_constraints(self, A, M, R):
        # (1) - Trucks maximum availability
        for y in range(self.n_years):
            for t in range(self.n_trucks):
               self.model.add_constraint(self.model.sum(self.x[t,b,y] for b in range(self.n_bins)) <= A[t,y])

        # (2) - Ensure move to upper bin 
        for t in range(self.n_trucks):
            for b in range(self.n_bins): 
                self.model.add_constraint(self.model.sum(self.x[t,b,y] for y in range(self.n_years)) <= M)

        # (4) - Correct bin order (lower bound)
        for t in range(self.n_trucks):
            for b in range(self.n_bins): 
                for y in range(self.n_years): 
                    self.model.add_constraint(self.model.sum(self.x[t,b,k] for k in range(y+1)) - M * self.y_bin[t,b,y] >= 0)

        # (5) - Correct bin order (upper bound)
        for t in range(self.n_trucks):
            for b in range((self.n_bins - 1)): # we can skip the last bin
                for y in range(self.n_years): 
                    self.model.add_constraint(self.x[t,(b+1),y] - M * self.model.sum(self.y_bin[t,b,k] for k in range(y+1)) <= 0)

        # (6) - Required truck yours per year
        for y in range(self.n_years):
            self.model.add_constraint(self.model.sum(self.x[t,b,y] for t in range(self.n_trucks) for b in range(self.n_bins)) == R[y])


    def solve(self, log=True, gap = 0.01, max_time = 60 * 10):
        self.model.parameters.mip.tolerances.mipgap = gap
        self.model.parameters.timelimit = max_time
        self.solution = self.model.solve(log_output=log)

    def report_results(self, c_critical, InitialAge):
        if(self.solution):
            self.model.report()
            self.model.print_information()
            #model.print_solution()

            # Plot the accumulated hours
            fig, ax = plt.subplots(3,1, figsize=(16,10))

            image_h = np.zeros((self.n_years, self.n_trucks))
            image_hours = np.zeros((self.n_years, self.n_trucks))
            image_bins = np.zeros((self.n_bins, self.n_trucks))
            image_y_critical = np.zeros((self.n_years, self.n_trucks))
        
            for i in range(self.n_years):        
                for j in range(self.n_trucks):
                    hours = 0
                    for b in range(self.n_bins):
                        hours = hours + int(self.model.get_var_by_name('x_{}_{}_{}'.format(j,b,i)))
                    image_hours[i,j] = hours

            for i in range(self.n_bins):
                for j in range(self.n_trucks):
                    hours = 0
                    for y in range(self.n_years):
                        hours = hours + int(self.model.get_var_by_name('x_{}_{}_{}'.format(j,i,y)))
                    image_bins[i,j] = hours

            for i in range(self.n_years):
                for j in range(self.n_trucks):
                    image_y_critical[i,j] = int(self.model.get_var_by_name('y_bin_{}_{}_{}'.format(j,c_critical[j],i)))

            #ch = ax[0].matshow(image_h, cmap='Greys')
            #cho = ax[1].matshow(image_hours,cmap='Reds')
            image_h = np.tile(InitialAge, (self.n_years, 1)) + np.cumsum(image_hours, axis=0)
            sns.heatmap(image_h / 1000, annot=True, ax=ax[0])
            sns.heatmap(image_hours / 1000, annot=True,ax=ax[1])
            sns.heatmap(image_bins / 1000, annot=True, ax=ax[2])
            # sns.heatmap(image_y_critical , annot=True, ax=ax[2])

            ax[0].set_ylabel('Horas')
            ax[1].set_ylabel('Horas')
            ax[2].set_ylabel('Horas')

            ax[0].set_xlabel('# Caminhão')
            ax[1].set_xlabel('# Caminhão')
            ax[2].set_xlabel('# Caminhão')


            #fig.colorbar(ch, ax=ax[0])
            #fig.colorbar(cho, ax=ax[1])
            #fig.colorbar(chy, ax=ax[2])


            plt.tight_layout()
            plt.show()

#%% INSTANCE FACTORY
class TruckMaintenanceProblemInstanceFactory: 
    def __init__(self):
        pass

    def get_small_instance(self):
        n_trucks = 4 # The number of trucks
        n_years = 1 # The time period
        n_bins = 40 # The number of age bins
        max_planned_production = 365 * 24 * n_trucks # the entire year non stop
        min_truck_availability = max_planned_production / n_trucks
        M = 1000
        
        # Engine rebuild cost
        FE = [750000] * n_trucks
        # Available truck hours per period T
        if(path.isfile('instances/small_availability.csv')):
            A = pd.read_csv('instances/small_availability.csv', sep=',', header=None).to_numpy()
        else:
            A = np.random.randint(low=min_truck_availability*0.9, high=min_truck_availability, size = (n_trucks, n_years))
            np.savetxt('instances/small_availability.csv', A, delimiter=",")

        # The required truck hours for a given time period y
        if(path.isfile('instances/small_targets.csv')):
            R = pd.read_csv('instances/small_targets.csv', sep=',', header=None).to_numpy().reshape(-1, )
        else:
            R = self.get_production_targets(n_years, max_planned_production, target_type='random')
            np.savetxt('instances/small_targets.csv', R, delimiter=",")
        
        # The initial truck ages
        if(path.isfile('instances/small_truck_ages.csv')):
            InitialAge = pd.read_csv('instances/small_truck_ages.csv', sep=',', header=None).to_numpy().reshape(-1, )
        else:
            InitialAge = self.get_initial_ages(n_trucks, max_age=20000, ages_type='random')
            np.savetxt('instances/small_truck_ages.csv', InitialAge, delimiter=",")

        # The critical age bin adjusted for each truck
        if(path.isfile('instances/small_critical_bins.csv')):
            c_critical = pd.read_csv('instances/small_critical_bins.csv', sep=',', header=None).to_numpy().reshape(-1, )
        else:
            c_critical = self.get_critical_bins(InitialAge, default_critical_bin=20, bin_size=M)
            np.savetxt('instances/small_critical_bins.csv', c_critical, delimiter=",", fmt='%d')
    
        # Discounted cost value for truck T at age bin B and period T
        if(path.isfile('instances/small_maintenance_cost.csv')):
            C_file = pd.read_csv('instances/small_maintenance_cost.csv', sep=',', header=None).to_numpy()
            C = np.zeros((n_trucks, n_bins, n_years))
            for i in range(C_file.shape[0]):
                t, b, y, value = C_file[i, :]
                C[int(t), int(b), int(y)] = value
        else:
            C = self.get_cost_matrix(n_trucks, n_bins, n_years, cost_type='increasing', critical_bins=c_critical)
            C_toFile = []
            for t in range(n_trucks):
                for b in range(n_bins):
                    for y in range(n_years):
                        C_toFile.append([t,b,y,C[t,b,y]])

            np.savetxt('instances/small_maintenance_cost.csv', np.array(C_toFile), delimiter=",")
        
        return n_trucks, n_bins, n_years, C, c_critical, FE, A, M, R, InitialAge

    def get_large_instance(self):
        n_trucks = 40 # The number of trucks
        n_years = 10 # The time period
        n_bins = 20 # The number of age bins
        max_planned_production = 365 * 24 * n_trucks # the entire year non stop
        min_truck_availability = max_planned_production / n_trucks
        M = 5000
        
        # Engine rebuild cost
        FE = [750000] * n_trucks
        # Available truck hours per period T
        if(path.isfile('instances/large_availability.csv')):
            A = pd.read_csv('instances/large_availability.csv', sep=',', header=None).to_numpy()
        else:
            A = np.random.randint(low=min_truck_availability*0.9, high=min_truck_availability, size = (n_trucks, n_years))
            np.savetxt('instances/large_availability.csv', A, delimiter=",")

        # The required truck hours for a given time period y
        if(path.isfile('instances/large_targets.csv')):
            R = pd.read_csv('instances/large_targets.csv', sep=',', header=None).to_numpy().reshape(-1, )
        else:
            R = self.get_production_targets(n_years, max_planned_production, target_type='random')
            np.savetxt('instances/large_targets.csv', R, delimiter=",")
        
        # The initial truck ages
        if(path.isfile('instances/large_truck_ages.csv')):
            InitialAge = pd.read_csv('instances/large_truck_ages.csv', sep=',', header=None).to_numpy().reshape(-1, )
        else:
            InitialAge = self.get_initial_ages(n_trucks, ages_type='random')
            np.savetxt('instances/large_truck_ages.csv', InitialAge, delimiter=",")

        # The critical age bin adjusted for each truck
        if(path.isfile('instances/large_critical_bins.csv')):
            c_critical = pd.read_csv('instances/large_critical_bins.csv', sep=',', header=None).to_numpy().reshape(-1, )
        else:
            c_critical = self.get_critical_bins(InitialAge, bin_size=M)
            np.savetxt('instances/large_critical_bins.csv', c_critical, delimiter=",", fmt='%d')
    

        # Discounted cost value for truck T at age bin B and period T
        if(path.isfile('instances/large_maintenance_cost.csv')):
            C_file = pd.read_csv('instances/large_maintenance_cost.csv', sep=',', header=None).to_numpy()
            C = np.zeros((n_trucks, n_bins, n_years))
            for i in range(C_file.shape[0]):
                t, b, y, value = C_file[i, :]
                C[int(t), int(b), int(y)] = value
        else:
            C = self.get_cost_matrix(n_trucks, n_bins, n_years, cost_type='increasing', critical_bins=c_critical)
            C_toFile = []
            for t in range(n_trucks):
                for b in range(n_bins):
                    for y in range(n_years):
                        C_toFile.append([t,b,y,C[t,b,y]])

            np.savetxt('instances/large_maintenance_cost.csv', np.array(C_toFile), delimiter=",")

        return n_trucks, n_bins, n_years, C, c_critical, FE, A, M, R, InitialAge

    def get_paper_instance(self):
        n_trucks = 34 # The number of trucks
        n_years = 10 # The time period
        n_bins = 20 # The number of age bins
        max_planned_production = 365 * 24 * n_trucks # the entire year non stop
        min_truck_availability = max_planned_production / n_trucks
        M = 5000
        
        # Engine rebuild cost
        FE = [750000] * n_trucks
        # Available truck hours per period T
        if(path.isfile('instances/paper_availability.csv')):
            A = pd.read_csv('instances/paper_availability.csv', sep=',', header=None).to_numpy()
        else:
            A = np.random.randint(low=min_truck_availability*0.9, high=min_truck_availability, size = (n_trucks, n_years))
            np.savetxt('instances/paper_availability.csv', A, delimiter=",")

        # The required truck hours for a given time period y
        if(path.isfile('instances/paper_targets.csv')):
            R = pd.read_csv('instances/paper_targets.csv', sep=',', header=None).to_numpy().reshape(-1, )
        else:
            R = self.get_production_targets(n_years, max_planned_production, target_type='paper')
            np.savetxt('instances/paper_targets.csv', R, delimiter=",")
        
        # The initial truck ages
        if(path.isfile('instances/paper_truck_ages.csv')):
            InitialAge = pd.read_csv('instances/paper_truck_ages.csv', sep=',', header=None).to_numpy().reshape(-1, )
        else:
            InitialAge = self.get_initial_ages(n_trucks, ages_type='paper')
            np.savetxt('instances/paper_truck_ages.csv', InitialAge, delimiter=",")

        # The critical age bin adjusted for each truck
        if(path.isfile('instances/paper_critical_bins.csv')):
            c_critical = pd.read_csv('instances/paper_critical_bins.csv', sep=',', header=None).to_numpy().reshape(-1, )
        else:
            c_critical = self.get_critical_bins(InitialAge, bin_size=M)
            np.savetxt('instances/paper_critical_bins.csv', c_critical, delimiter=",", fmt='%d')
    

        # Discounted cost value for truck T at age bin B and period T
        if(path.isfile('instances/paper_maintenance_cost.csv')):
            C_file = pd.read_csv('instances/paper_maintenance_cost.csv', sep=',', header=None).to_numpy()
            C = np.zeros((n_trucks, n_bins, n_years))
            for i in range(C_file.shape[0]):
                t, b, y, value = C_file[i, :]
                C[int(t), int(b), int(y)] = value
        else:
            C = self.get_cost_matrix(n_trucks, n_bins, n_years, cost_type='increasing', critical_bins=c_critical)
            C_toFile = []
            for t in range(n_trucks):
                for b in range(n_bins):
                    for y in range(n_years):
                        C_toFile.append([t,b,y,C[t,b,y]])

            np.savetxt('instances/paper_maintenance_cost.csv', np.array(C_toFile), delimiter=",")
      
        return n_trucks, n_bins, n_years, C, c_critical, FE, A, M, R, InitialAge

    def get_cost_matrix(self, n_trucks, n_bins, n_years, cost_type='random', default_critical_bin = 15, critical_bins = []):
        if(cost_type == 'random'):
            return 100 * np.random.random((n_trucks, n_bins, n_years))
        if(cost_type == 'increasing'):
            means = np.linspace(1, 10, n_bins)
            C = np.zeros((n_trucks, n_bins, n_years))

            for t in range(n_trucks):
                critical_bin = int(critical_bins[t])
                offset = int(default_critical_bin - critical_bins[t])
                for b in range(n_bins):
                    for y in range(n_years): 
                        if(b > critical_bin):
                            mean_val = means[b-critical_bin] + np.random.standard_normal()
                        else:
                            mean_val = means[b + offset] + np.random.standard_normal() 
                        C[t,b,y] = mean_val 
            
        return C

    def get_production_targets(self,n_years,planned_production, target_type='random'):
        if(target_type=='random'):
            return np.random.randint(low=0.7*planned_production, high=planned_production*0.8, size=n_years)
        if((target_type=='paper') & (n_years == 10)):
            return [221050, 220300, 232500, 231500, 232600, 230000, 220000, 200000, 106300, 25000]

    def get_initial_ages(self, n_trucks, max_age=20000, ages_type='random'):
        if(ages_type == 'random'):
            return np.random.randint(low=0, high=max_age, size = n_trucks)
        if(ages_type == 'zero'):
            return np.zeros(n_trucks)
        if(ages_type == 'paper'): 
            return [43055, 43864, 42595, 43141, 43570, 42659, 42603, 42162, 42214, 42555, 42213, 41259, 42180, 41122, 41216, 41472, 41495, 41571, 37766, 37936, 32033, 32503, 32479, 30384, 21762, 21686, 21310, 16585, 16734, 16311, 15682, 0, 0, 0]

    def get_critical_bins(self, ages, default_critical_bin = 14, bin_size = 5000):
        return np.array([ int(default_critical_bin - np.floor(age / bin_size)) for age in ages])
# %% SMALL INSTANCE
factory = TruckMaintenanceProblemInstanceFactory()
n_trucks, n_bins, n_years, C, c_critical, FE, A, M, R, InitialAges = factory.get_small_instance()

# Plot the cost matrix
fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12,8))
ax[0,0].plot(C[0,:,0])
ax[0,1].plot(C[1,:,0])
ax[1,0].plot(C[2,:,0])
ax[1,1].plot(C[3,:,0])

ax[0,0].set_title('Caminhão #1')
ax[0,1].set_title('Caminhão #2')
ax[1,0].set_title('Caminhão #3')
ax[1,1].set_title('Caminhão #4')

fig.suptitle('Custo de manutenção [$/hora] por faixa de idade')
plt.show()

# Solve
small_instance = TruckMaintenanceProblem(n_trucks, n_bins, n_years)
small_instance.init_model(C, c_critical, FE, A, M, R)
small_instance.solve(gap=0.00001)
small_instance.report_results(c_critical, InitialAges)
# %%
# %% LARGE INSTANCE
n_trucks, n_bins, n_years, C, c_critical, FE, A, M, R, InitialAges = factory.get_large_instance()

# Solve
large_instance = TruckMaintenanceProblem(n_trucks, n_bins, n_years)
large_instance.init_model(C, c_critical, FE, A, M, R)
large_instance.solve()
large_instance.report_results(c_critical, InitialAges)
# %%
