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

for i in range(df.shape[0]):
    for j in range(df.shape[1]):
        if i != j and df.at[i, j] != 0:
            BA_graph[i][j]['capacity'] = df.at[i, j]

target_edges = 1300

def merge_edge(node):
    global BA_graph
    neighbors = list(BA_graph.neighbors(node))
    if len(neighbors) < 2:
        return
    # Sort by the degrees of the neighbors and select the top two with the lowest degrees
    sorted_neighbors = sorted(neighbors, key=lambda x: BA_graph.degree[x], reverse=False)
    max_neighbor = sorted_neighbors[0]
    second_neighbor = sorted_neighbors[1]

    newG = BA_graph.copy()
    newG[node][max_neighbor]['capacity'] += newG[node][second_neighbor]['capacity']
    newG.remove_edge(node, second_neighbor)

    if not nx.is_strongly_connected(newG):
        print('Graph is not strongly connected.')
        return
    BA_graph = newG

while target_edges < BA_graph.number_of_edges():
    print(f'channels count: {BA_graph.number_of_edges()}')
    for node in BA_graph.nodes():
        merge_edge(node)
        if(target_edges == BA_graph.number_of_edges()):
            break

result_df = nx.to_pandas_adjacency(BA_graph, weight='capacity')

row_sums = result_df.sum(axis=1)
if not all(np.isclose(row_sums, 100)):
    raise ValueError("Not all rows sum to 100")

print(BA_graph)
if not nx.is_strongly_connected(BA_graph):
    raise ValueError("Graph is not strongly connected")
result_df.to_csv(f'output\PCN_{f}_MinOut_multiplied_matrix.csv')

total_sum = result_df.values.sum()

print("Total sum of all elements (excluding row and column labels):", total_sum)
