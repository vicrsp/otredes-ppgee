import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ClusterDB = pd.read_csv('ClusterDB.csv', header=None, sep=',', names=['ID', 'eta', 'beta']).set_index('ID')
EquipDB = pd.read_csv('EquipDB.csv', header=None, sep=',', names=['ID', 'Idade', 'Cluster', 'Custo de Falha']).set_index('ID')
MPDB = pd.read_csv('MPDB.csv', header=None, sep=',', names=['ID', 'riskFactor', 'planCost']).set_index('ID')

EquipDB_ClusterDB = EquipDB.join(ClusterDB, on='Cluster', sort=False)


# EquipDB
sns.pairplot(EquipDB, hue='Cluster')
EquipDB.groupby('Cluster').aggregate([np.mean, np.std])

EquipDB_ClusterDB.groupby('Cluster').count()['Idade']

# ClusterDB
def calculateFailureProbability(t0, delta, eta, beta):
    dcf = lambda x: 1 - np.exp(-((x/eta)**beta))
    Ft0 = dcf(t0)
    FtdT = dcf(t0 + delta)
    return (FtdT - Ft0)/(1 - Ft0)

for i in range(ClusterDB.shape[0]):
    row = ClusterDB.iloc[i]    
    pw_c = [calculateFailureProbability(0, age, row['eta'], row['beta']) for age in np.linspace(0, 40, 100)]
    plt.plot((pw_c))

plt.legend(['1', '2', '3', '4'])
plt.xlabel('Idade')
plt.ylabel('Probabilidade de falha')
    

# Analyze solution
data = pd.read_csv('Pareto_Solution05.csv')
fig, ax = plt.subplots()
ax.plot(data.loc[:,'fobj1'], data.loc[:,'fobj2'], 'r.')
ax.set_xlabel('Custo de manutenção')
ax.set_ylabel('Custo esperado de Falha')
ax.fill_between(data.loc[:,'fobj1'], data.loc[:,'fobj2'], data.loc[:,'fobj2'].max(), alpha=0.5)


# Solutions heatmap
solution = pd.read_csv('Solution05.csv', header=None).to_numpy()
# plt.figure(figsize=(10,12))
# sns.heatmap(solution, yticklabels=100, xticklabels=100, cmap=['red','blue','green'], fmt='%d', cbar_kws={'ticks': [1, 2, 3], 'orientation': 'horizontal', 'label': 'Plano de manutenção'})
# plt.xlabel('Equipamento')
# plt.ylabel('Solução')
# plt.savefig('solutions_heatmap.png')

# Solutions proportion
plt.figure()
n_sol, n_cols = solution.shape
proportions = np.zeros((n_cols, 3))
for i in range(n_cols):
    proportions[i,0] = 100 * np.count_nonzero(solution[:,i] == 1) / n_sol
    proportions[i,1] = 100 * np.count_nonzero(solution[:,i] == 2) / n_sol
    proportions[i,2] = 100 * np.count_nonzero(solution[:,i] == 3) / n_sol

solutions_cluster = pd.DataFrame(np.transpose(solution))
solutions_cluster['id'] = EquipDB.index
solutions_cluster['cluster'] = EquipDB['Cluster']

s_melted = pd.melt(solutions_cluster, id_vars=['id','cluster'], var_name='solution_index')
#sns.barplot(x='id',y='value', hue='cluster', data=s_melted)

s_cluster_count = s_melted.groupby(['cluster', 'value']).count()['id']
s_cluster_count.groupby(level=0).sum()
values_per_cluster = 100 * s_cluster_count / s_cluster_count.groupby(level=0).sum()
