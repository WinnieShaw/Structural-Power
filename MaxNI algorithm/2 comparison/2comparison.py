import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson
plt.rcParams['font.size'] = 20
plt.rcParams['font.family'] = 'Times New Roman'
def load_tpm(file_path):
    return pd.read_csv(file_path, index_col=0)

def simulate_demands(tpm, alphas, time_steps=50):
    results = []
    for alpha in alphas:
        alpha_results = []
        for _ in range(time_steps):
            total_exceeds = 0
            queue_count = 0
            for i in tpm.index:
                for j in tpm.columns:
                    mu = tpm.at[i, j]
                    if mu > 0:
                        lambda_ = alpha * mu
                        demand = poisson.rvs(lambda_)
                        if demand > mu:
                            total_exceeds += demand - mu
                        queue_count += 1
            alpha_results.append((total_exceeds, queue_count))
        results.extend([(alpha, res[0], res[1]) for res in alpha_results])
    return results

file_path = 'PCN_matrix.csv'
data = pd.read_csv(file_path, header=None)

data = data * 100
data.columns = range(200)
data.index = range(200)

output_file = 'PCN_matrix_multiplied.csv'
data.to_csv(output_file, index=True, header=True)
print(data)

file_path_remote = output_file
file_path_greedy1 = 'multiplied_PCN_MaxNI_matrix.csv'

tpm_remote = load_tpm(file_path_remote)
tpm_greedy1 = load_tpm(file_path_greedy1)

alphas = np.arange(0.5, 1, 0.05)

results_remote = simulate_demands(tpm_remote, alphas)
results_greedy1 = simulate_demands(tpm_greedy1, alphas)

df_remote = pd.DataFrame(results_remote, columns=['Alpha', 'Exceedances', 'Queue_Count'])
df_greedy1 = pd.DataFrame(results_greedy1, columns=['Alpha', 'Exceedances', 'Queue_Count'])

df = pd.concat([
    df_remote.assign(Method="Original Network"),
    df_greedy1.assign(Method="MaxNI Network")
])

mean_df = df.groupby(['Alpha', 'Method']).mean().reset_index()

df.to_csv('2compare.csv', index=False)