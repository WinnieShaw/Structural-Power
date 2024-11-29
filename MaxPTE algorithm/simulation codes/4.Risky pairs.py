import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

f = 10
thresholds = np.arange(0, 101, 5)
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 18

def count_pairs_below_threshold(file_path, threshold):
    flow_matrix = pd.read_csv(file_path, header=0, index_col=0)
    num_pairs_below_threshold = 0
    for u in range(200):
        for v in range(200):
            if u == v:
                continue
            flow_value = flow_matrix.at[u, str(v)]
            if flow_value <= threshold:
                num_pairs_below_threshold += 1
    return num_pairs_below_threshold/(39800)*100

file_paths = {
    'Original': f'MF output\{f}\PCN_{f}_multiplied_matrix.csv_flow_matrix.csv',
    'MaxPTE': f'MF output\{f}\PCN_{f}_MaxPTE_multiplied_matrix.csv_flow_matrix.csv',
    'Random': f'MF output\{f}\PCN_{f}_Random_multiplied_matrix.csv_flow_matrix.csv',
    'MaxOut': f'MF output\{f}\PCN_{f}_MaxOut_multiplied_matrix.csv_flow_matrix.csv',
    'MinOut': f'MF output\{f}\PCN_{f}_MinOut_multiplied_matrix.csv_flow_matrix.csv',
    'MaxBetweeness': f'MF output\{f}\PCN_{f}_MaxBetweeness_multiplied_matrix.csv_flow_matrix.csv',
    'MaxClustering': f'MF output\{f}\PCN_{f}_MaxClustering_multiplied_matrix.csv_flow_matrix.csv'
}

results_below_threshold = {method: [] for method in file_paths}

for threshold in thresholds:
    for method, file_path in file_paths.items():
        count = count_pairs_below_threshold(file_path, threshold)
        results_below_threshold[method].append(count)

results_df = pd.DataFrame(results_below_threshold, index=thresholds)
results_df.index.name = 'Deficit thresholds'
bad = 'bad'
results_df.to_csv(f'..\summary results\{bad} nodepairs\{bad}-nodepairs{f}.csv')
print(f'save to ..\summary results\{bad} nodepairs\{bad}-nodepairs{f}.csv')