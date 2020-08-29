#%%  CPLEX imports
from docplex.mp.model import Model
import numpy as np
import matplotlib.pyplot as plt

model = Model(name='mine_schedule')
model.parameters.mip.tolerances.mipgap = 0.01
#%% ARTIFICIAL INSTANCE DATA
n_trucks = 34 # The number of trucks
n_years = 10 # The time period
n_bins = 20 # The number of age bins
planned_production = 200000
age_bin_size = 5000
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
                    
# Discounted cost value for truck T at age bin B and period T
C = get_cost_matrix(n_trucks, n_bins, n_years, cost_type='increasing')
# Engine rebuild cost
FE = [750000] * n_trucks
# Available truck hours per period T
A = np.random.randint(low=min_truck_availability, high=min_truck_availability * 1.5, size = (n_trucks, n_years))
#np.ones((n_trucks, n_years)) * min_truck_availability
# Maximum available truck hours at age bin V
M = np.arange(start=1, stop=n_bins+1) * age_bin_size
# The cumulative used hours for truck t at time period t
# H = np.random.randint(low=0, high=1000, size = (n_trucks, n_years))
# The required truck hours for a given time period y
R = np.ones(n_years) * planned_production #p.random.randint(low=200000, high=240000, size = n_years)
# The initial truck ages
InitialAge = np.random.randint(low=0, high=20000, size = n_trucks)
# The critical age bin
c_critical = 15

#%% CPLEX VARS
x = model.integer_var_cube(range(n_trucks), range(n_bins), range(n_years), name='x')
y_bin = model.binary_var_cube(range(n_trucks), range(n_bins), range(n_years), name='y_bin')
h = model.integer_var_matrix(range(n_trucks), range(n_years), name='h')

#%% CPLEX MODEL
# Objective: TODO: revisar se não está adicionando FE a mais
model.minimize(model.sum(x[t,b,y] * C[t,b,y] + y_bin[t,c_critical,y]*(FE[t]/(n_trucks*n_bins*n_years)) for t in range(n_trucks) for b in range(n_bins) for y in range(n_years)))

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

# (4) - Correct bin order (lower bound)
for t in range(n_trucks):
    for b in range(n_bins): 
        for y in range(n_years): 
            model.add_constraint(model.sum(x[t,b,k] for k in range(y+1)) - M[b] * y_bin[t,b,y] >= 0)

# (5) - Correct bin order (upper bound)
for t in range(n_trucks):
    for b in range((n_bins - 1)): # we can skip the last bin
        for y in range(n_years): 
            model.add_constraint(x[t,(b+1),y] - M[(b+1)]* model.sum(y_bin[t,b,k] for k in range(y+1)) <= 0)

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

model.export_as_lp('D:\\otredes-ppgee\\trabalho_final\\test.lp')
model.export_as_mps('D:\\otredes-ppgee\\trabalho_final\\test_mps.mps')

#%% SOLVE
solution = model.solve(log_output=True)
#%% RESULTS
if(solution):
    model.report()
    model.print_information()
    #model.print_solution()

    # Plot the accumulated hours
    fig, ax = plt.subplots(2,1, figsize=(16,10))

    image_h = np.zeros((n_years, n_trucks))
    image_hours = np.zeros((n_years, n_trucks))
    for i in range(n_years):
        for j in range(n_trucks):
            image_h[i,j] = model.get_var_by_name('h_{}_{}'.format(j,i))

    for i in range(n_years):        
        for j in range(n_trucks):
            hours = 0
            for b in range(n_bins):
                hours = hours + int(model.get_var_by_name('x_{}_{}_{}'.format(j,b,i)))
            image_hours[i,j] = hours

    ch = ax[0].matshow(image_h)
    cho = ax[1].matshow(image_hours)

    fig.colorbar(ch, ax=ax[0])
    fig.colorbar(cho, ax=ax[1])

    plt.tight_layout()
    plt.show()

# %%
