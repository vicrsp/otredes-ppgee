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
    