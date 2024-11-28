import pandas as pd
import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
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

    total_in_weight = sum(data['weight'] for _, _, data in new_graph.in_edges(virtual_node, data=True))
    total_out_weight = sum(data['weight'] for _, _, data in new_graph.out_edges(virtual_node, data=True))
    for pred, _, data in new_graph.in_edges(virtual_node, data=True):
        data['weight'] /= total_in_weight
    for _, succ, data in new_graph.out_edges(virtual_node, data=True):
        data['weight'] /= total_out_weight

    if new_graph.has_edge(virtual_node, virtual_node):
        new_graph.remove_edge(virtual_node, virtual_node)

    new_graph = nx.DiGraph(new_graph)  # digraph
    new_graph.remove_edges_from(nx.selfloop_edges(new_graph))

    return new_graph

def random_merge(graph, target_node_count):
    while len(graph.nodes()) > target_node_count:
        nodes = list(graph.nodes())
        node_n = random.choice(nodes)
        nodes.remove(node_n)
        node_m = random.choice(nodes)
        virtual_node = f"{node_n}_{node_m}"

        if graph.has_edge(node_n, node_m):
            graph.remove_edge(node_n, node_m)
        if graph.has_edge(node_m, node_n):
            graph.remove_edge(node_m, node_n)

        graph = merge_nodes(graph, node_n, node_m, virtual_node)
    return graph

file_path_BA_matrix = 'PCN_matrix.csv'
df = pd.read_csv(file_path_BA_matrix, header=None)
BA_graph = nx.from_pandas_adjacency(df, create_using=nx.DiGraph())
data = pd.read_csv(file_path_BA_matrix, index_col=0, header=0)
n = data.shape[0]
print(n)

merged_graph = random_merge(BA_graph, 139)

def generate_tpm(graph):
    adj_matrix = nx.to_pandas_adjacency(graph, weight='weight')
    row_sums = adj_matrix.sum(axis=1)
    tpm_matrix = adj_matrix.div(row_sums, axis=0)
    tpm_matrix[row_sums == 0] = 0 # avoid division by zero
    return tpm_matrix

tpm_matrix = generate_tpm(merged_graph)
print(tpm_matrix)

tpm_matrix.to_csv(f'{file_path_BA_matrix}_random_Matrix.csv')
print(f"Successfully save to {file_path_BA_matrix}_random_Matrix.csv")

df_full = pd.read_csv(f'{file_path_BA_matrix}_random_Matrix.csv', index_col=0)
file_path_BA_matrix = 'PCN_Random_matrix.csv'
def count_original_nodes(combined_node):
    return len(combined_node.split('_'))

row_counts = df_full.index.to_series().apply(count_original_nodes)

multiplier_matrix = np.tile(row_counts.values.reshape(-1, 1), (1, len(df_full.columns)))
initial_balance = 100
multiplied_data = df_full * multiplier_matrix * initial_balance
multiplied_data.to_csv(f'multiplied_{file_path_BA_matrix}')
print(multiplied_data)
