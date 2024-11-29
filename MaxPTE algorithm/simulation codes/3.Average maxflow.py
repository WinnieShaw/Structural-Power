import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

f =10
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 18

def calculate_averageflow(file_path):
    flow_matrix = pd.read_csv(file_path, header=0, index_col=0)
    total_flow = 0
    num_pairs = 0
    for u in range(200):
        for v in range(200):
            if u == v:
                continue
            flow_value = flow_matrix.at[u, str(v)]
            total_flow += flow_value
            num_pairs += 1
    average_maxflow = total_flow / num_pairs
    return average_maxflow


file_paths = {
    'Original': f'MF output\{f}\PCN_{f}_multiplied_matrix.csv_flow_matrix.csv',
    'MaxPTE': f'MF output\{f}\PCN_{f}_MaxPTE_multiplied_matrix.csv_flow_matrix.csv',
    'Random': f'MF output\{f}\PCN_{f}_Random_multiplied_matrix.csv_flow_matrix.csv',
    'MaxOut': f'MF output\{f}\PCN_{f}_MaxOut_multiplied_matrix.csv_flow_matrix.csv',
    'MinOut': f'MF output\{f}\PCN_{f}_MinOut_multiplied_matrix.csv_flow_matrix.csv',
    'MaxBetweeness': f'MF output\{f}\PCN_{f}_MaxBetweeness_multiplied_matrix.csv_flow_matrix.csv',
    'MaxClustering': f'MF output\{f}\PCN_{f}_MaxClustering_multiplied_matrix.csv_flow_matrix.csv'
}

results = {}
for method, file_path in file_paths.items():
    results[method] = calculate_averageflow(file_path)
    print(f"{method}: {results[method]}")

results_df = pd.DataFrame(list(results.items()), columns=['Method', 'Average Maximum Flow'])
average = 'average'
results_df.to_csv(f'..\summary results\{average} maxflow\{average}_maxflow_results{f}.csv', index=False)
print(f'save to ..\summary results\{average} maxflow\{average}_maxflow_results{f}.csv')
