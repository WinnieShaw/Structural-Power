import pandas as pd
import math
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np

file_path_BA_matrix = 'PCN_matrix.csv'
df = pd.read_csv(file_path_BA_matrix, header=None)
BA_graph = nx.from_pandas_adjacency(df, create_using=nx.DiGraph())
G = nx.karate_club_graph()
visited = {i: 0 for i in range(200)}

def load_graph(file_path):
    df = pd.read_csv(file_path, header=None)
    return nx.from_pandas_adjacency(df, create_using=nx.DiGraph())

def calculate_node_weight(node):
    if isinstance(node, int):
        return 1
    elif isinstance(node, str):
        return node.count('_') + 1
    else:
        raise ValueError("Node name must be either an int or a str.")

def merge_nodes(graph, node_n, node_m, virtual_node):
    new_graph = graph.copy()
    # 合并入边和出边
    for node in [node_n, node_m]:
        for pred in graph.predecessors(node):
            if pred not in [node_n, node_m]:
                weight = graph[pred][node]['weight']
                if new_graph.has_edge(pred, virtual_node):
                    new_graph[pred][virtual_node]['weight'] += weight
                else:
                    new_graph.add_edge(pred, virtual_node, weight=weight)
        for succ in graph.successors(node):
            if succ not in [node_n, node_m]:
                weight = graph[node][succ]['weight']
                if new_graph.has_edge(virtual_node, succ):
                    new_graph[virtual_node][succ]['weight'] += weight
                else:
                    new_graph.add_edge(virtual_node, succ, weight=weight)
    # 删除原始节点
    new_graph.remove_nodes_from([node_n, node_m])
    return new_graph

def optimize_clustering(graph):
    n = len(graph.nodes())
    log_n = math.log(n)
    max_clustering = nx.average_clustering(graph.to_undirected())
    best_graph = graph.copy()

    improved = True
    while improved and len(best_graph.nodes()) > 139:
        improved = False
        node_list = list(best_graph.nodes())
        clustering = nx.clustering(best_graph.to_undirected())
        # 根据聚类系数从高到低排序节点
        node_list.sort(key=lambda x: clustering[x], reverse=True)

        for i in range(len(node_list)):
            for j in range(i + 1, len(node_list)):
                node1, node2 = node_list[i], node_list[j]
                virtual_node = f"{node1}_{node2}"
                new_graph = merge_nodes(best_graph, node1, node2, virtual_node)
                new_clustering = nx.average_clustering(new_graph.to_undirected())
                if new_clustering > max_clustering:
                    max_clustering = new_clustering
                    best_graph = new_graph
                    improved = True
                    print(f"Updated max clustering coefficient: {max_clustering} after merging {node1} and {node2}")
                    break
            if improved:
                break

    return best_graph

optimized_graph = optimize_clustering(BA_graph)


print(optimized_graph)

adj_matrix = nx.to_pandas_adjacency(optimized_graph, weight='weight', dtype=float)

row_sums = adj_matrix.sum(axis=1)
tpm_matrix = adj_matrix.div(row_sums, axis=0)
tpm_matrix[row_sums == 0] = 0

file_path = 'PCN_MaxClustering_matrix.csv'
tpm_matrix.to_csv(file_path,index=True)

df_full = pd.read_csv(file_path, index_col=0)
def count_original_nodes(combined_node):
    return len(combined_node.split('_'))

row_counts = df_full.index.to_series().apply(count_original_nodes)

multiplier_matrix = np.tile(row_counts.values.reshape(-1, 1), (1, len(df_full.columns)))

initial_balance = 100
multiplied_data = df_full * multiplier_matrix * initial_balance

multiplied_data.to_csv(f'multiplied_{file_path}')

print(multiplied_data)