import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

f = 10 #1~10
file_path_tpm = f'..\datasets\PCN_TPM_matrix{f}.csv'
df = pd.read_csv(file_path_tpm, header=0, index_col=0)

df_full = pd.read_csv(file_path_tpm, header=None, index_col=None)

initial_balance = 100
multiplied_data = df_full * initial_balance

multiplied_data.index = range(multiplied_data.shape[0])
multiplied_data.columns = range(multiplied_data.shape[1])

multiplied_data.to_csv(f'output\PCN_{f}_multiplied_matrix.csv', index=True, header=True)
file_path_BA_matrix = f'output\PCN_{f}_multiplied_matrix.csv'
df = pd.read_csv(file_path_BA_matrix, header=0, index_col=0)

df.index = df.columns = range(df.shape[0])
BA_graph = nx.from_pandas_adjacency(df, create_using=nx.DiGraph())
if not nx.is_strongly_connected(BA_graph):
    raise ValueError("Graph is not strongly connected")

for i in range(df.shape[0]):
    for j in range(df.shape[1]):
        if i != j and df.at[i, j] != 0:
            BA_graph[i][j]['capacity'] = df.at[i, j]

def calculate_clustering_coefficient(G):
    return nx.average_clustering(G)

def merge_edge(node):
    global BA_graph
    current_clustering = calculate_clustering_coefficient(BA_graph)
    best_clustering = current_clustering
    best_merge = None

    neighbors = list(BA_graph.neighbors(node))
    if len(neighbors) < 2:
        return

    for i in neighbors:
        for j in neighbors:
            if i == j:
                continue
            newG = BA_graph.copy()
            capacity_to_add = newG[node][j]['capacity']
            newG[node][i]['capacity'] += capacity_to_add
            newG.remove_edge(node, j)
            new_clustering = calculate_clustering_coefficient(newG)
            if new_clustering > best_clustering and nx.is_strongly_connected(newG):
                best_clustering = new_clustering
                best_merge = (i, j)

    if best_merge:
        i, j = best_merge
        capacity_to_add = BA_graph[node][j]['capacity']
        BA_graph[node][i]['capacity'] += capacity_to_add
        BA_graph.remove_edge(node, j)
        print(f'Node {node}: Merged edge ({node}->{j}) into ({node}->{i}) to increase clustering coefficient to {best_clustering}')
    else:
        print(f'Node {node}: No edge merging increased clustering coefficient.')

# initialize clustering coefficient
original_clustering = calculate_clustering_coefficient(BA_graph)
print("Initial Clustering Coefficient:", original_clustering)

for node in BA_graph.nodes():
    merge_edge(node)

max_clustering = calculate_clustering_coefficient(BA_graph)
print("Maximized Clustering Coefficient:", max_clustering)

result_df = nx.to_pandas_adjacency(BA_graph, weight='capacity')
result_df.to_csv(f'output\PCN_{f}_MaxClustering_multiplied_matrix.csv')

print(result_df)
