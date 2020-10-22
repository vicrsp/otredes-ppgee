#%%  CPLEX imports
from docplex.mp.model import Model
from docplex.mp.progress import ProgressListener, ProgressClock, TextProgressListener
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from os import path

sns.set(font_scale=1.2)

#%% PROBLEM DEFINITION

class MIPProgress(ProgressListener):
    def __init__(self, clock=ProgressClock.Gap, gap_fmt=None, obj_fmt=None,
                absdiff=None, reldiff=None):
        ProgressListener.__init__(self, clock, absdiff, reldiff)
        self._gap_fmt = gap_fmt or "{:.2%}"
        self._obj_fmt = obj_fmt or "{:.4f}"
        self._count = 0
        self.mip_gap = []
        self.current_objective = []
        self.time = []

    def notify_start(self):
        super(MIPProgress, self).notify_start()
        self._count = 0

    def notify_progress(self, progress_data):
        self._count += 1
        pdata_has_incumbent = progress_data.has_incumbent
        if pdata_has_incumbent:
            self.current_objective.append(progress_data.current_objective)
            self.mip_gap.append(progress_data.mip_gap * 100)
            self.time.append(round(progress_data.time, 1))
    
    def plot_progress(self):
        if(len(self.mip_gap) > 0):
            _, ax = plt.subplots(2, 1, figsize=(10,8))
            ax[0].plot(self.time, self.mip_gap)
            ax[1].plot(self.time, self.current_objective)

            ax[0].set_xlabel('Time (sec)')
            ax[1].set_xlabel('Time (sec)')

            ax[0].set_ylabel('Relative gap (%)')
            ax[1].set_ylabel('Objective')

class TruckMaintenanceProblem:
    def __init__(self, n_trucks = 10, n_bins = 20, n_years = 5):
        self.n_trucks = n_trucks # The number of trucks
        self.n_years = n_years # The time period
        self.n_bins = n_bins # The number of age bins
        self.progress = MIPProgress()

    def init_model(self, C, c_critical, FE, A, M, R, InitialAge):
        self.model = Model(name='mine_schedule')
        self.model.add_progress_listener(self.progress)
        self.x, self.y_bin = self.set_decision_vars()
        self.set_objective(C, c_critical, FE)
        self.set_constraints(A, M, R, InitialAge)
    
    def set_decision_vars(self):
        x = self.model.integer_var_cube(range(self.n_trucks), range(self.n_bins), range(self.n_years), name='x')
        y_bin = self.model.binary_var_cube(range(self.n_trucks), range(self.n_bins), range(self.n_years), name='y_bin')
        return x, y_bin

    def set_objective(self, C, c_critical, FE):
        hour_costs = self.model.sum(self.x[t,b,y] * C[t,b,y] for t in range(self.n_trucks) for b in range(self.n_bins) for y in range(self.n_years))
        repair_costs = self.model.sum(self.y_bin[t,c_critical[t],y]*FE[t] for t in range(self.n_trucks) for y in range(self.n_years))
        self.model.minimize(self.model.sum(hour_costs + repair_costs))    

    def set_constraints(self, A, M, R, InitialAge):
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

        # (7) Ensure a truck does not operate more than 100,000 hours
        for t in range(self.n_trucks):
            self.model.add_constraint(self.model.sum(self.x[t,b,y] for b in range(self.n_bins) for y in range(self.n_years)) <= M * self.n_bins - InitialAge[t])
    
    def solve(self, log=True, gap = None, max_time = 60 * 10, aggressive=False):
        if (gap != None):
            self.model.parameters.mip.tolerances.mipgap = gap

        if(aggressive):
            self.model.parameters.mip.strategy.probe = 3 # agressive probing
            self.model.parameters.emphasis.mip = 3 # ephasis on solution quality
            self.model.parameters.mip.strategy.lbheur = 1
            # self.model.parameters.mip.cuts.cliques = 3
            # self.model.parameters.mip.cuts.covers = 3
            # self.model.parameters.mip.cuts.disjunctive = 3
            # self.model.parameters.mip.cuts.flowcovers = 2
            # self.model.parameters.mip.cuts.gomory = 2
            # self.model.parameters.mip.cuts.pathcut = 2

        self.model.parameters.timelimit = max_time
        self.solution = self.model.solve(log_output=log)

    def report_results(self, c_critical, InitialAge, prefix='instance'):
        if(self.solution):
            self.model.report()
            self.model.print_information()
            
            # Plot the accumulated hours
            fig, ax = plt.subplots(1,1, figsize=(16,10))
            fig2, ax2 = plt.subplots(1,1, figsize=(8,6))
            fig3, ax3 = plt.subplots(1,1, figsize=(16,10))
            fig4, ax4 = plt.subplots(1,1, figsize=(16,10))

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

            image_h = np.tile(InitialAge, (self.n_years, 1)) + np.cumsum(image_hours, axis=0)
            
            sns.heatmap((image_h/1000).astype(int), xticklabels=list(range(1,self.n_trucks+1)), yticklabels=list(range(1,self.n_years+1)), annot=True, ax=ax, fmt='d', annot_kws={"size": 8})
            sns.heatmap((image_hours/1000).astype(int), xticklabels=list(range(1,self.n_trucks+1)), yticklabels=list(range(1,self.n_years+1)),annot=True,ax=ax3, fmt='d', annot_kws={"size": 8})
            sns.heatmap((image_bins/1000).astype(int), xticklabels=list(range(1,self.n_trucks+1)), yticklabels=list(range(1,self.n_bins+1)),annot=True, ax=ax4, fmt='d', annot_kws={"size": 8})
            sns.heatmap(image_y_critical, xticklabels=list(range(1,self.n_trucks+1)), yticklabels=list(range(1,self.n_years+1)),annot=True, ax=ax2, annot_kws={"size": 8})

            ax.set_ylabel('Ano')
            ax.set_title('Horas acumuladas')
            
            ax3.set_ylabel('Ano')
            ax3.set_title('Horas alocadas')
            
            ax4.set_ylabel('Faixa de idade')
            ax4.set_title('Horas por faixa')
            ax4.set_xlabel('Caminhão')

            ax2.set_ylabel('Ano')
            ax2.set_xlabel('Caminhão')
            ax2.set_title('Faixa crítica atingida')


            #ax[0].set_xlabel('# Caminhão')
            #ax[3].set_xlabel('Truck #')

            #fig.colorbar(ch, ax=ax[0])
            #fig.colorbar(cho, ax=ax[1])
            #fig.colorbar(chy, ax=ax[2])
            # plt.tight_layout()
            plt.show()

            fig.tight_layout()
            fig2.tight_layout()
            fig3.tight_layout()
            fig4.tight_layout()

            fig.savefig('images/{}_solution_accumulated.png'.format(prefix))
            fig2.savefig('images/{}_critical.png'.format(prefix))
            fig3.savefig('images/{}_solution_values.png'.format(prefix))
            fig4.savefig('images/{}_solutions_bins.png'.format(prefix))

#%% INSTANCE FACTORY
class TruckMaintenanceProblemInstanceFactory: 
    def __init__(self):
        pass

    def get_small_instance(self):
        n_trucks = 4 # The number of trucks
        n_years = 3 # The time period
        n_bins = 20 # The number of age bins
        max_planned_production = 365 * 24 * n_trucks # the entire year non stop
        min_truck_availability = max_planned_production / n_trucks
        M = 2000
        
        # Engine rebuild cost
        FE = [750000] * n_trucks
        # Available truck hours per period T
        A = self.load_available_truck_hours('small_availability.csv', n_trucks, n_years, min_truck_availability)
        # The required truck hours for a given time period y
        R = [25000, 25000, 25000, 25000]#self.load_production_targets('small_targets.csv', n_years, max_planned_production, target_type='random') 
        # The initial truck ages
        InitialAge = [0, 0, 8000, 8000] #self.load_initial_truck_ages('small_truck_ages.csv', n_trucks, max_age=10000, ages_type='random')
        # The critical age bin adjusted for each truck
        c_critical = self.load_critical_bins('small_critical_bins.csv', InitialAge, M, 8)
        # Discounted cost value for truck T at age bin B and period T
        C = self.load_cost_matrix('small_maintenance_cost.csv', n_trucks, n_bins, n_years, cost_type='increasing', critical_bins=c_critical)

        
        return n_trucks, n_bins, n_years, C, c_critical, FE, A, M, R, InitialAge

    def get_average_instance(self):
        n_trucks = 20 # The number of trucks
        n_years = 5 # The time period
        n_bins = 20 # The number of age bins
        max_planned_production = 365 * 24 * n_trucks # the entire year non stop
        min_truck_availability = max_planned_production / n_trucks
        M = 4000
        
        # Engine rebuild cost
        FE = [750000] * n_trucks
        # Available truck hours per period T
        A = self.load_available_truck_hours('average_availability.csv', n_trucks, n_years, min_truck_availability)
        # The required truck hours for a given time period y
        R = self.load_production_targets('average_targets.csv', n_years, max_planned_production, target_type='random') 
        # The initial truck ages
        InitialAge = self.load_initial_truck_ages('average_truck_ages.csv', n_trucks, ages_type='random')
        # The critical age bin adjusted for each truck
        c_critical = self.load_critical_bins('average_critical_bins.csv', InitialAge, M, 10)
        # Discounted cost value for truck T at age bin B and period T
        C = self.load_cost_matrix('average_maintenance_cost.csv', n_trucks, n_bins, n_years, cost_type='increasing', critical_bins=c_critical)
        
        return n_trucks, n_bins, n_years, C, c_critical, FE, A, M, R, InitialAge

    def get_large_instance(self):
        n_trucks = 45 # The number of trucks
        n_years = 10 # The time period
        n_bins = 20 # The number of age bins
        max_planned_production = 365 * 24 * n_trucks # the entire year non stop
        min_truck_availability = max_planned_production / n_trucks
        M = 5000
        
        # Engine rebuild cost
        FE = [750000] * n_trucks
        # Available truck hours per period T
        A = self.load_available_truck_hours('large_availability.csv', n_trucks, n_years, min_truck_availability)
        # The required truck hours for a given time period y
        R = self.load_production_targets('large_targets.csv', n_years, max_planned_production, target_type='random') 
        # The initial truck ages
        InitialAge = self.load_initial_truck_ages('large_truck_ages.csv', n_trucks, ages_type='random')
        # The critical age bin adjusted for each truck
        c_critical = self.load_critical_bins('large_critical_bins.csv', InitialAge, M)
        # Discounted cost value for truck T at age bin B and period T
        C = self.load_cost_matrix('large_maintenance_cost.csv', n_trucks, n_bins, n_years, cost_type='increasing', critical_bins=c_critical)

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
        A = self.load_available_truck_hours('paper_availability.csv', n_trucks, n_years, min_truck_availability)
        # The required truck hours for a given time period y
        R = self.load_production_targets('paper_targets.csv', n_years, max_planned_production, target_type='paper') 
        # The initial truck ages
        InitialAge = self.load_initial_truck_ages('paper_truck_ages.csv', n_trucks, ages_type='paper')
        # The critical age bin adjusted for each truck
        c_critical = self.load_critical_bins('paper_critical_bins.csv', InitialAge, M)
        # Discounted cost value for truck T at age bin B and period T
        C = self.load_cost_matrix('paper_maintenance_cost.csv', n_trucks, n_bins, n_years, cost_type='increasing', critical_bins=c_critical)
        return n_trucks, n_bins, n_years, C, c_critical, FE, A, M, R, InitialAge

    def load_available_truck_hours(self, file_name, n_trucks, n_years, min_truck_availability):
        full_path = 'instances/{}'.format(file_name)
        if(path.isfile(full_path)):
            A = pd.read_csv(full_path, sep=',', header=None).to_numpy()
        else:
            A = np.random.randint(low=min_truck_availability*0.9, high=min_truck_availability*0.95, size = (n_trucks, n_years))
            np.savetxt(full_path, A, delimiter=",")
        return A

    def load_production_targets(self, file_name, n_years, max_planned_production, target_type):
        full_path = 'instances/{}'.format(file_name)
        if(path.isfile(full_path)):
            R = pd.read_csv(full_path, sep=',', header=None).to_numpy().reshape(-1, )
        else:
            R = self.get_production_targets(n_years, max_planned_production, target_type=target_type)
            np.savetxt(full_path, R, delimiter=",")
        return R

    def load_critical_bins(self, file_name, InitialAge, M, default_critical_bin = 14):
        full_path = 'instances/{}'.format(file_name)
        if(path.isfile(full_path)):
            c_critical = pd.read_csv(full_path, sep=',', header=None).to_numpy().reshape(-1, )
        else:
            c_critical = self.get_critical_bins(InitialAge, default_critical_bin=default_critical_bin, bin_size=M)
            np.savetxt(full_path, c_critical, delimiter=",", fmt='%d')

        return c_critical

    def load_initial_truck_ages(self, file_name, n_trucks, ages_type, max_age = 20000):
        full_path = 'instances/{}'.format(file_name)
        if(path.isfile(full_path)):
            InitialAge = pd.read_csv(full_path, sep=',', header=None).to_numpy().reshape(-1, )
        else:
            InitialAge = self.get_initial_ages(n_trucks, max_age=max_age, ages_type=ages_type)
            np.savetxt(full_path, InitialAge, delimiter=",")
        return InitialAge

    def load_cost_matrix(self, file_name, n_trucks, n_bins, n_years, cost_type, critical_bins):
        full_path = 'instances/{}'.format(file_name)
        if(path.isfile(full_path)):
            C_file = pd.read_csv(full_path, sep=',', header=None).to_numpy()
            C = np.zeros((n_trucks, n_bins, n_years))
            for i in range(C_file.shape[0]):
                t, b, y, value = C_file[i, :]
                C[int(t), int(b), int(y)] = value
        else:
            C = self.get_cost_matrix(n_trucks, n_bins, n_years, cost_type=cost_type, critical_bins=critical_bins)
            C_toFile = []
            for t in range(n_trucks):
                for b in range(n_bins):
                    for y in range(n_years):
                        C_toFile.append([t,b,y,C[t,b,y]])

            np.savetxt(full_path, np.array(C_toFile), delimiter=",")
        
        return C

    def get_cost_matrix(self, n_trucks, n_bins, n_years, cost_type='random', default_critical_bin = 15, critical_bins = []):
        if(cost_type == 'random'):
            return 100 * np.random.random((n_trucks, n_bins, n_years))
        if(cost_type == 'increasing'):
            means = np.linspace(5, 15, n_bins) ** 2
            C = np.zeros((n_trucks, n_bins, n_years))

            for t in range(n_trucks):
                critical_bin = int(critical_bins[t])
                offset = int(default_critical_bin - critical_bins[t])
                for b in range(n_bins):
                    for y in range(n_years): 
                        if(b > critical_bin):
                            mean_val = means[b-critical_bin] + np.random.standard_normal() * 10
                        else:
                            mean_val = means[b + offset] + np.random.standard_normal() * 10
                        C[t,b,y] = mean_val 
            
        return C

    def get_production_targets(self,n_years,planned_production, target_type='random'):
        if(target_type=='random'):
            scale = np.ones(n_years)
            # reduce the value of the last years
            if(n_years > 3):
                scale[-2] = 0.5
                scale[-1] = 0.1
            return (np.random.randint(low=0.7*planned_production, high=planned_production*0.8, size=n_years) * scale).astype(int)
        if((target_type=='paper') & (n_years == 10)):
            return [221050, 220300, 232500, 231500, 232600, 230000, 220000, 200000, 106300, 25000]

    def get_initial_ages(self, n_trucks, max_age=20000, ages_type='random'):
        if(ages_type == 'random'):
            return np.random.randint(low=0, high=max_age, size = n_trucks)
        if(ages_type == 'zero'):
            return np.zeros(n_trucks)
        if((ages_type == 'paper') & (n_trucks == 34)): 
            return [43055, 43864, 42595, 43141, 43570, 42659, 42603, 42162, 42214, 42555, 42213, 41259, 42180, 41122, 41216, 41472, 41495, 41571, 37766, 37936, 32033, 32503, 32479, 30384, 21762, 21686, 21310, 16585, 16734, 16311, 15682, 0, 0, 0]

    def get_critical_bins(self, ages, default_critical_bin = 14, bin_size = 5000):
        return np.array([ int(default_critical_bin - np.floor(age / bin_size)) for age in ages])
# %% INSTANCE FACTORY
factory = TruckMaintenanceProblemInstanceFactory()
# %% SMALL INSTANCE
n_trucks, n_bins, n_years, C, c_critical, FE, A, M, R, InitialAges = factory.get_small_instance()

# Plot the cost matrix
fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12,8))

[ax[0,0].plot(np.arange(n_bins) * M + InitialAges[0], C[0,:,i]) for i in range(n_years)]
[ax[0,1].plot(np.arange(n_bins) * M + InitialAges[1], C[1,:,i]) for i in range(n_years)]
[ax[1,0].plot(np.arange(n_bins) * M + InitialAges[2], C[2,:,i]) for i in range(n_years)]
[ax[1,1].plot(np.arange(n_bins) * M + InitialAges[3], C[3,:,i]) for i in range(n_years)]

ax[0,0].set_title('Caminhão #1')
ax[0,1].set_title('Caminhão #2')
ax[1,0].set_title('Caminhão #3')
ax[1,1].set_title('Caminhão #4')

ax[0,0].set_ylabel('Custo de manutenção [$/hora]')
ax[1,0].set_ylabel('Custo de manutenção [$/hora]')

ax[1,0].set_xlabel('Idade [horas]')
ax[1,1].set_xlabel('Idade [horas]')

# ax[0, 0].axvline(InitialAges[0], color='red')
# ax[0, 1].axvline(InitialAges[1], color='red')
# ax[1, 0].axvline(InitialAges[2], color='red')
# ax[1, 1].axvline(InitialAges[3], color='red')
ax[0,0].legend(['Ano 1','Ano 2','Ano 3'])
ax[0,1].legend(['Ano 1','Ano 2','Ano 3'])
ax[1,0].legend(['Ano 1','Ano 2','Ano 3'])
ax[1,1].legend(['Ano 1','Ano 2','Ano 3'])
plt.tight_layout()
plt.show()

# ax[1,0].set_xticks(range(n_bins))
# ax[1,1].set_xticks(range(n_bins))

# Solve
small_instance = TruckMaintenanceProblem(n_trucks, n_bins, n_years)
small_instance.init_model(C, c_critical, FE, A, M, R, InitialAges)
small_instance.solve()
small_instance.report_results(c_critical, InitialAges, 'small')
small_instance.progress.plot_progress()

# %% AVERAGE INSTANCE
n_trucks, n_bins, n_years, C, c_critical, FE, A, M, R, InitialAges = factory.get_average_instance()

# Solve
avg_instance = TruckMaintenanceProblem(n_trucks, n_bins, n_years)
avg_instance.init_model(C, c_critical, FE, A, M, R, InitialAges)
avg_instance.solve()

# Report
avg_instance.report_results(c_critical, InitialAges, 'average')
avg_instance.progress.plot_progress()

# %% LARGE INSTANCE
n_trucks, n_bins, n_years, C, c_critical, FE, A, M, R, InitialAges = factory.get_large_instance()

# Solve
large_instance = TruckMaintenanceProblem(n_trucks, n_bins, n_years)
large_instance.init_model(C, c_critical, FE, A, M, R, InitialAges)
large_instance.solve(gap=0.001, max_time=60 * 30)
large_instance.report_results(c_critical, InitialAges, 'large')
large_instance.progress.plot_progress()

# %% PAPER INSTANCE
n_trucks, n_bins, n_years, C, c_critical, FE, A, M, R, InitialAges = factory.get_paper_instance()

# Solve
paper_instance = TruckMaintenanceProblem(n_trucks, n_bins, n_years)
paper_instance.init_model(C, c_critical, FE, A, M, R, InitialAges)
paper_instance.solve(gap=0.01, max_time=60 * 30, aggressive=False)

# Report
paper_instance.report_results(c_critical, InitialAges, 'paper')
paper_instance.progress.plot_progress()

# %%


# %%
