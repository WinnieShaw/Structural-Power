import pandas as pd
import networkx as nx
import queue
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

file_path_BA_matrix = 'PCN_matrix.csv'
df = pd.read_csv(file_path_BA_matrix, header=None)
BA_graph = nx.from_pandas_adjacency(df, create_using=nx.DiGraph())
G = nx.karate_club_graph()
visited = {i: 0 for i in range(200)}

data = {
    'n': [],
    'virtual_node':[],
    'entropy1':[], #H(I)
    'entropy2':[], #H(O)
    'NI': [],
    'PTE': []
}

def calNI(BA_graph, node):

    df = nx.to_pandas_adjacency(BA_graph, weight='weight')

    row_sums = df.sum(axis=1)
    df_normalized = df.div(row_sums, axis=0)
    df_normalized[row_sums == 0] = 0
    # print(df_normalized)

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
    n = df.shape[0]
    NI = PTE * np.log2(n) if n > 1 else 0

    new_data = {
        'n': n,
        'virtual_node': node,
        'entropy1': entropy1,
        'entropy2': entropy2,
        'NI': NI,
        'PTE': PTE
    }
    return new_data



def calculate_node_weight(node):
    if isinstance(node, int):
        return 1
    elif isinstance(node, str):
        return node.count('_') + 1
    else:
        raise ValueError("Node name must be either an int or a str.")

def merge_nodes(graph, node_n, node_m, virtual_node):
    new_graph = graph.copy()

    in_edges = {}
    out_edges = {}

    for node in [node_n, node_m]:
        for pred in graph.predecessors(node):
            if pred not in [node_n, node_m]:
                weight = graph[pred][node]['weight']
                if pred in in_edges:
                    in_edges[pred] += weight
                else:
                    in_edges[pred] = weight

    for pred, weight in in_edges.items():
        pred_weight = calculate_node_weight(pred)
        new_graph.add_edge(pred, virtual_node, weight=weight * pred_weight)

    for node in [node_n, node_m]:
        for succ in graph.successors(node):
            if succ not in [node_n, node_m]:
                weight = graph[node][succ]['weight']
                if succ in out_edges:
                    out_edges[succ] += weight
                else:
                    out_edges[succ] = weight

    virtual_node_weight = calculate_node_weight(virtual_node)
    for succ, weight in out_edges.items():
        new_graph.add_edge(virtual_node, succ, weight=weight * virtual_node_weight)

    new_graph.remove_nodes_from([node_n, node_m])

    # Normalization of edge weights
    total_in_weight = sum(data['weight'] for _, _, data in new_graph.in_edges(virtual_node, data=True))
    total_out_weight = sum(data['weight'] for _, _, data in new_graph.out_edges(virtual_node, data=True))
    for pred, _, data in new_graph.in_edges(virtual_node, data=True):
        data['weight'] /= total_in_weight
    for _, succ, data in new_graph.out_edges(virtual_node, data=True):
        data['weight'] /= total_out_weight

    return new_graph

def count_original_nodes(node_name):
    if isinstance(node_name, int):
        return 1  # If the node name is an integer, assume it is an unmerged original node.
    elif isinstance(node_name, str):
        return node_name.count('_') + 1  # Count the number of underscores in the string node name, and add 1 to determine how many nodes were merged to form the combined node.
    else:
        raise ValueError("Node name must be either an int or a str.")

def addNeighbor(n, queueNodes, q, BA_graph):
    # Iterate through and add all outgoing neighbors
    if BA_graph.has_node(n):
        for neighbor in BA_graph.successors(n):
            if neighbor not in queueNodes:

                count_nodes = count_original_nodes(neighbor)
                q.put(neighbor)
                queueNodes.add(neighbor)

        # add all neighbors
        for neighbor in BA_graph.predecessors(n):
            if neighbor not in queueNodes:
                count_nodes = count_original_nodes(neighbor)
                q.put(neighbor)
                queueNodes.add(neighbor)


new_data = calNI(BA_graph,'all')
maxNI = new_data['NI']
i = 0
print(BA_graph,f'Original NI = {maxNI}',f'{BA_graph}',f'{file_path_BA_matrix}')

def check(n):
    q = queue.Queue()
    queueNodes = set()
    q.put(n)
    queueNodes.add(n)
    global BA_graph
    global maxNI
    while not q.empty():
        n = q.get()
        print(f'node {n} is visited')
        visited[n] = 1
        new_graph = BA_graph.copy()

        addNeighbor(n,queueNodes,q,new_graph)
        if not q.empty():
            m = q.get()
            print(f'node {m} is visited')
            visited[m] = 1
        else:
            break

        virtual_node = f"{n}_{m}"
        new_graph = merge_nodes(BA_graph, n, m, virtual_node)

        new_data = calNI(new_graph, virtual_node)
        NIvalue = new_data['NI']
        print(NIvalue)
        if (NIvalue > maxNI):
            addNeighbor(m,queueNodes,q,new_graph)
            print(f'combine({n},{m}),update tmpNI = {NIvalue}')
            BA_graph = new_graph
            maxNI = NIvalue
            for key, value in new_data.items():
                data[key].append(value)

for key, value in list(visited.items()):
    print(f"checking node {key}")
    if value == 1:
        continue
    check(key)

file_path = f'{file_path_BA_matrix}_Combined.csv'
df = pd.DataFrame(data)
df.to_csv(file_path, index=False)

print(maxNI)
print(BA_graph)
print(BA_graph,f'Original NI={maxNI}',f'{BA_graph}',f'{file_path_BA_matrix}')

adj_matrix = nx.to_pandas_adjacency(BA_graph, weight='weight', dtype=float)

row_sums = adj_matrix.sum(axis=1)
tpm_matrix = adj_matrix.div(row_sums, axis=0)
tpm_matrix[row_sums == 0] = 0

file_path_tpm = 'PCN_MaxNI_matrix.csv'
tpm_matrix.to_csv(file_path_tpm,index=True)

print(f"MaxNI outcome is saved to {file_path_tpm}.")

df_full = pd.read_csv('PCN_MaxNI_matrix.csv', index_col=0)  # 保留行列标签
def count_original_nodes(combined_node):
    return len(combined_node.split('_'))

row_counts = df_full.index.to_series().apply(count_original_nodes)

multiplier_matrix = np.tile(row_counts.values.reshape(-1, 1), (1, len(df_full.columns)))

initial_balance = 100  # coins in every nodes(the sum of channels' balance of every node)
multiplied_data = df_full * multiplier_matrix * initial_balance

# 保存新的DataFrame到CSV文件
multiplied_data.to_csv(f'multiplied_{file_path_tpm}')