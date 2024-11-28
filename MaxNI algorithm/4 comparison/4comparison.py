import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 18
def load_tpm(file_path):
    return pd.read_csv(file_path, index_col=0)

def simulate_demands(tpm, alphas, time_steps=50):
    results = []
    for alpha in alphas:
        total_exceeds = 0
        queue_count = 0
        for i in tpm.index:
            for j in tpm.columns:
                mu = tpm.at[i, j]
                if mu > 0:
                    lambda_ = alpha * mu
                    for _ in range(time_steps):
                        demand = poisson.rvs(lambda_)
                        if demand > mu:
                            total_exceeds += demand - mu
                    queue_count += 1
        results.append(total_exceeds/100)
    return results

file_path_MaxNI = 'multiplied_PCN_MaxNI_matrix.csv'
file_path_Random = 'multiplied_PCN_Random_matrix.csv'
file_path_LowestDegree = 'multiplied_PCN_LowestDegree_matrix.csv'
file_path_MaxClustering = 'multiplied_PCN_MaxClustering_matrix.csv'


df = pd.read_csv(file_path_MaxNI)
print(f'{file_path_MaxNI}: {df.shape}')

df = pd.read_csv(file_path_Random)
print(f'{file_path_Random}: {df.shape}')

df = pd.read_csv(file_path_LowestDegree)
print(f'{file_path_LowestDegree}: {df.shape}')

df = pd.read_csv(file_path_MaxClustering)
print(f'{file_path_MaxClustering}: {df.shape}')
def load_tpm(file_path):
    return pd.read_csv(file_path, index_col=0)

MaxNI = load_tpm(file_path_MaxNI)
Random = load_tpm(file_path_Random)
LowestDegree = load_tpm(file_path_LowestDegree)
MaxClustering = load_tpm(file_path_MaxClustering)

alphas = np.arange(0.5, 1, 0.05)

results_MaxNI = simulate_demands(MaxNI, alphas)
results_Random = simulate_demands(Random, alphas)
results_LowestDegree = simulate_demands(LowestDegree, alphas)
results_MaxClustering = simulate_demands(MaxClustering, alphas)

df = pd.DataFrame({
    "Alpha": alphas,
    "MaxNI": results_MaxNI,
    "Random": results_Random,
    "LowestDegree": results_LowestDegree,
    "MaxClustering": results_MaxClustering
})

df.to_csv('4compare.csv', index=False)

plt.figure(figsize=(8, 5))
plt.plot(alphas, results_Random, label='Random', color='#1f70a9', marker='s')
plt.plot(alphas, results_LowestDegree, label='Lowest Degree', color='#ea7827', marker='o')
plt.plot(alphas, results_MaxClustering, label='MaxClustering', color='#c22f2f', marker='d', markersize=8)
plt.plot(alphas, results_MaxNI, label='MaxNI', color='#449945', marker='*', markersize=11)

ax = plt.gca()
ax.set_xlabel('α', fontdict={'family': 'Times New Roman', 'style': 'italic', 'size': 24})
ax.set_ylabel('Balance deficit', fontsize=24)

plt.legend(fontsize=22)
plt.grid(True)

plt.savefig(fname='MaxNI-4results.png', dpi=500)
plt.savefig('MaxNI-4results.pdf', format='pdf', bbox_inches='tight', dpi=500)

# Display plot
plt.show(dpi=300)