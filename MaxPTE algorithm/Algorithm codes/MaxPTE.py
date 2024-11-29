import pandas as pd
import numpy as np
import networkx as nx

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

df.index = df.columns = range(df.shape[0])
BA_graph = nx.from_pandas_adjacency(df, create_using=nx.DiGraph())
if not nx.is_strongly_connected(BA_graph):
    raise ValueError("Graph is not strongly connected")

for i in range(df.shape[0]):
    for j in range(df.shape[1]):
        if i != j and df.at[i, j] != 0:
            BA_graph[i][j]['capacity'] = df.at[i, j]

data = {
    'n': [],
    'virtual_node': [],
    'entropy1': [],
    'entropy2': [],
    'PTE': []
}

def calPTE(BA_graph):
    df = nx.to_pandas_adjacency(BA_graph, weight='capacity')

    row_sums = df.sum(axis=1)
    df_normalized = df.div(row_sums, axis=0)
    df_normalized[row_sums == 0] = 0

    def calculate_entropy(dist):
        return -np.sum(dist * np.log2(dist + np.finfo(float).eps))

    column_sums = df_normalized.sum(axis=0)
    probabilities = column_sums / column_sums.sum()
    entropy1 = calculate_entropy(probabilities)

    def calculate_average_row_entropy(df_normalized):
        row_entropies = df_normalized.apply(lambda row: calculate_entropy(row), axis=1)
        return row_entropies.mean()

    entropy2 = calculate_average_row_entropy(df_normalized)

    PTE = entropy1 - entropy2
    total_edges = BA_graph.number_of_edges()

    new_data = {
        'n': df.shape[0],
        'entropy1': entropy1,
        'entropy2': entropy2,
        'PTE': PTE
    }
    return new_data

used_neighbors = set()

def merge_edge(node):
    global BA_graph
    PTE = calPTE(BA_graph)['PTE']
    tmpPTE = PTE
    tmpN = node
    tmpM = node
    print(f'checking {node},tmpPTE={PTE}')
    neighbors = list(BA_graph.neighbors(node))
    if (len(neighbors) < 2):
        return
    for i in neighbors:
        for j in neighbors:
            if i == j:
                continue
            newG = BA_graph.copy()
            capacity_to_add = newG[node][j]['capacity']
            newG[node][i]['capacity'] += capacity_to_add
            newG.remove_edge(node, j)
            if tmpPTE < calPTE(newG)['PTE'] and nx.is_strongly_connected(newG):
                tmpPTE = calPTE(newG)['PTE']
                tmpN = i
                tmpM = j
    #find the maximum PTE
    if tmpN != node:
        newG = BA_graph.copy()
        capacity_to_add = newG[node][tmpM]['capacity']
        newG[node][tmpN]['capacity'] += capacity_to_add
        newG.remove_edge(node,tmpM)
        BA_graph = newG
        newPTE = calPTE(BA_graph)['PTE']
        print(f'after combining node {node} can achieve the max PTE: {newPTE}')
    else:
        print(f'Higher PTE of combining node {node} is not found.')

original_PTE = calPTE(BA_graph)['PTE']
print("Initial PTE:", original_PTE)

for node in BA_graph.nodes():
    merge_edge(node)


max_PTE = calPTE(BA_graph)['PTE']
print("Maximized PTE:", max_PTE)


result_df = nx.to_pandas_adjacency(BA_graph, weight='capacity')

row_sums = result_df.sum(axis=1)
print(result_df)
print("Sum of each row:")
print(row_sums)
if not all(np.isclose(row_sums, 100, atol=0.001)):
    raise ValueError("Not all rows sum to 100 within a tolerance of 0.001")

print(BA_graph)

result_df.to_csv(f'PCN_{f}_MaxPTE_multiplied_matrix.csv')

total_sum = result_df.values.sum()

print("Total sum of all elements (excluding row and column labels):", total_sum)

file_path = f'output\PCN_{f}_MaxPTE_multiplied_matrix.csv'
data = pd.read_csv(file_path, index_col=0)

count_positive = (data > 0).sum().sum()

print("Number of elements greater than 0:", count_positive)