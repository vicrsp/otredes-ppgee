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
solution_r = solution[np.arange(0, 1001, 2)]
plt.figure(figsize=(10,12))
sns.heatmap(solution_r, yticklabels=100, xticklabels=100, cmap='viridis')
plt.xlabel('Equipamento')
plt.ylabel('Solução')
plt.savefig('solutions_heatmap.png')