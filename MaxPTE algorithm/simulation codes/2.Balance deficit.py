import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 18

f = 10
def calculate_maxflow_demands(file_path, demands):
    flow_matrix = pd.read_csv(file_path, header=0, index_col=0)

    results = []
    num_pairs = 0
    for demand in demands:
        total_deficit = 0
        for u in range(200):
            for v in range(200):
                if u == v:
                    continue
                flow_value = flow_matrix.at[u, str(v)]
                total_deficit += max(demand - flow_value, 0)
                num_pairs += 1
        average_deficit = total_deficit / num_pairs
        results.append(average_deficit)
    return results

file_paths = {
    'Original': f'MF output\{f}\PCN_{f}_multiplied_matrix.csv_flow_matrix.csv',
    'MaxPTE': f'MF output\{f}\PCN_{f}_MaxPTE_multiplied_matrix.csv_flow_matrix.csv',
    'Random': f'MF output\{f}\PCN_{f}_Random_multiplied_matrix.csv_flow_matrix.csv',
    'MaxOut': f'MF output\{f}\PCN_{f}_MaxOut_multiplied_matrix.csv_flow_matrix.csv',
    'MinOut': f'MF output\{f}\PCN_{f}_MinOut_multiplied_matrix.csv_flow_matrix.csv',
    'MaxBetweeness': f'MF output\{f}\PCN_{f}_MaxBetweeness_multiplied_matrix.csv_flow_matrix.csv',
    'MaxClustering': f'MF output\{f}\PCN_{f}_MaxClustering_multiplied_matrix.csv_flow_matrix.csv'
}

demands = np.arange(0, 100, 5)
results_dict = {}

for key, file_path in file_paths.items():
    results_dict[key] = calculate_maxflow_demands(file_path, demands)
    print(f"{key}: {results_dict[key]}")

results_df = pd.DataFrame(results_dict, index=demands)
results_df.index.name = 'Demand'
results_df.to_csv(f'..\summary results\demand results\maxflow_demands_results{f}.csv')

