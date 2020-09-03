#%%  CPLEX imports
from docplex.mp.model import Model
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

model = Model(name='mine_schedule')
model.parameters.mip.tolerances.mipgap = 0.1
#%% ARTIFICIAL INSTANCE DATA
n_trucks = 34 # The number of trucks
n_years = 10 # The time period
n_bins = 20 # The number of age bins
planned_production = 365 * 24 * n_trucks # the entire year non stop
#age_bin_size = 5000
M = 5000
min_truck_availability = planned_production / n_trucks

def get_cost_matrix(n_trucks, n_bins, n_years, cost_type='random', critical_bin = 15):
    if(cost_type == 'random'):
        return 100 * np.random.random((n_trucks, n_bins, n_years))
    if(cost_type == 'increasing'):
        means = np.linspace(5, 20, n_bins)
        C = np.zeros((n_trucks, n_bins, n_years))
        for t in range(n_trucks):
            for b in range(n_bins):
                for y in range(n_years): 
                    if(b > critical_bin):
                        mean_val = means[b-critical_bin] + np.random.standard_normal()
                    else:
                        mean_val = means[b] + np.random.standard_normal() 
                    C[t,b,y] = mean_val 
        
        return C

def get_production_targets(n_years, target_type='random'):
    if(target_type=='random'):
        return np.random.randint(low=0.7*planned_production, high=planned_production*0.8, size=n_years)
    if(target_type=='paper' & n_years == 10):
        return [221050, 220300, 232500, 231500, 232600, 230000, 220000, 200000, 106300, 25000]

def get_initial_ages(n_trucks, ages_type='random'):
    if(ages_type == 'random'):
        return np.random.randint(low=0, high=20000, size = n_trucks)
    if(ages_type == 'zero'):
        return np.zeros(n_trucks)
    # if(ages_type == 'paper'): TODO: copy from paper

# Discounted cost value for truck T at age bin B and period T
C = get_cost_matrix(n_trucks, n_bins, n_years, cost_type='increasing')
# Engine rebuild cost
FE = [750000] * n_trucks
# Available truck hours per period T
A = np.random.randint(low=min_truck_availability*0.9, high=min_truck_availability, size = (n_trucks, n_years))
#np.ones((n_trucks, n_years)) * min_truck_availability
# Maximum available truck hours at age bin V
#M = np.arange(start=1, stop=n_bins+1) * age_bin_size
# The cumulative used hours for truck t at time period t
# H = np.random.randint(low=0, high=1000, size = (n_trucks, n_years))
# The required truck hours for a given time period y
R = get_production_targets(n_years)
# The initial truck ages
InitialAge = get_initial_ages(n_trucks, 'zero')
# The critical age bin
c_critical = 15

#%% CPLEX VARS
x = model.integer_var_cube(range(n_trucks), range(n_bins), range(n_years), name='x')
y_bin = model.binary_var_cube(range(n_trucks), range(n_bins), range(n_years), name='y_bin')
h = model.integer_var_matrix(range(n_trucks), range(n_years), name='h')

#%% CPLEX MODEL
# Objective: TODO: revisar se não está adicionando FE a mais
hour_costs = model.sum(x[t,b,y] * C[t,b,y] for t in range(n_trucks) for b in range(n_bins) for y in range(n_years))
repair_costs = model.sum(y_bin[t,c_critical,y]*FE[t] for t in range(n_trucks) for y in range(n_years))
model.minimize(model.sum(hour_costs + repair_costs))

# Constraints
# (1) - Trucks maximum availability
for y in range(n_years):
    for t in range(n_trucks):
        model.add_constraint(model.sum(x[t,b,y] for b in range(n_bins)) <= A[t,y])
# (2) - Ensure move to upper bin 
for t in range(n_trucks):
    for b in range(n_bins): 
        model.add_constraint(model.sum(x[t,b,y] for y in range(n_years)) <= M)
# (3) - Accumulated truck age

# (4) - Correct bin order (lower bound)
for t in range(n_trucks):
    for b in range(n_bins): 
        for y in range(n_years): 
            model.add_constraint(model.sum(x[t,b,k] for k in range(y+1)) - M * y_bin[t,b,y] >= 0)

# (5) - Correct bin order (upper bound)
for t in range(n_trucks):
    for b in range((n_bins - 1)): # we can skip the last bin
        for y in range(n_years): 
            model.add_constraint(x[t,(b+1),y] - M * model.sum(y_bin[t,b,k] for k in range(y+1)) <= 0)

# (6) - Required truck yours per year
for y in range(n_years):
    model.add_constraint(model.sum(x[t,b,y] for t in range(n_trucks) for b in range(n_bins)) == R[y])
    
# (7) - Initial ages
for y in range(n_years):
    for t in range(n_trucks):
        if(y == 0):
            model.add_constraint(model.sum(x[t,b,y] for b in range(n_bins)) + InitialAge[t] == h[t,y])
        else:
            model.add_constraint(model.sum(x[t,b,y] for b in range(n_bins)) + h[t,y-1] == h[t,y])

#model.export_as_lp('D:\\otredes-ppgee\\trabalho_final\\test.lp')
#model.export_as_mps('D:\\otredes-ppgee\\trabalho_final\\test_mps.mps')

#%% SOLVE
solution = model.solve(log_output=True)
#%% RESULTS
if(solution):
    model.report()
    model.print_information()
    #model.print_solution()

    # Plot the accumulated hours
    fig, ax = plt.subplots(3,1, figsize=(16,10))

    image_h = np.zeros((n_years, n_trucks))
    image_hours = np.zeros((n_years, n_trucks))
    image_bins = np.zeros((n_bins, n_trucks))
    image_y_critical = np.zeros((n_years, n_trucks))
    for i in range(n_years):
        for j in range(n_trucks):
            image_h[i,j] = model.get_var_by_name('h_{}_{}'.format(j,i))

    for i in range(n_years):        
        for j in range(n_trucks):
            hours = 0
            for b in range(n_bins):
                hours = hours + int(model.get_var_by_name('x_{}_{}_{}'.format(j,b,i)))
            image_hours[i,j] = hours

    for i in range(n_bins):
        for j in range(n_trucks):
            hours = 0
            for y in range(n_years):
                hours = hours + int(model.get_var_by_name('x_{}_{}_{}'.format(j,i,y)))
            image_bins[i,j] = hours

    for i in range(n_years):
        for j in range(n_trucks):
            image_y_critical[i,j] = int(model.get_var_by_name('y_bin_{}_{}_{}'.format(j,c_critical,i)))

    #ch = ax[0].matshow(image_h, cmap='Greys')
    #cho = ax[1].matshow(image_hours,cmap='Reds')
    sns.heatmap(image_h / 1000, annot=True, ax=ax[0])
    sns.heatmap(image_hours / 1000, annot=True,ax=ax[1])
    sns.heatmap(image_bins / 1000, annot=True, ax=ax[2])

    #fig.colorbar(ch, ax=ax[0])
    #fig.colorbar(cho, ax=ax[1])
    #fig.colorbar(chy, ax=ax[2])


    plt.tight_layout()
    plt.show()

# %%
