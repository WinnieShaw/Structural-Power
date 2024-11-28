import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

n = 139

file_path = 'PCN_matrix.csv'
df = pd.read_csv(file_path, header=None)

BA_graph = nx.from_pandas_adjacency(df, create_using=nx.DiGraph())
initial_labels = {node: str(node) for node in BA_graph.nodes()}
nx.set_node_attributes(BA_graph, initial_labels, 'label')

def calculate_node_weight(node):
    if isinstance(node, int):
        return 1
    elif isinstance(node, str):
        return node.count('_') + 1
    else:
        raise ValueError("Node name must be either an int or a str.")
def merge_nodes_simple(graph, source, target):
    source_weight = calculate_node_weight(graph.nodes[source]['label'])
    target_weight = calculate_node_weight(graph.nodes[target]['label'])
    new_weight = source_weight + target_weight

    new_label = f"{graph.nodes[target]['label']}_{graph.nodes[source]['label']}"
    nx.set_node_attributes(graph, {target: new_label}, 'label')

    for neighbor in list(graph.successors(source)):
        weight = graph[source][neighbor]['weight'] if 'weight' in graph[source][neighbor] else 1
        weight *= source_weight
        if graph.has_edge(target, neighbor):
            graph[target][neighbor]['weight'] += weight
        else:
            graph.add_edge(target, neighbor, weight=weight)

    for neighbor in list(graph.predecessors(source)):
        weight = graph[neighbor][source]['weight'] if 'weight' in graph[neighbor][source] else 1
        weight *= source_weight
        if graph.has_edge(neighbor, target):
            graph[neighbor][target]['weight'] += weight
        else:
            graph.add_edge(neighbor, target, weight=weight)

    graph.remove_node(source)

    total_in_weight = sum(graph[pred][target]['weight'] for pred in graph.predecessors(target))
    total_out_weight = sum(graph[target][succ]['weight'] for succ in graph.successors(target))
    for pred in graph.predecessors(target):
        graph[pred][target]['weight'] /= total_in_weight
    for succ in graph.successors(target):
        graph[target][succ]['weight'] /= total_out_weight

def merge_until_n_nodes(graph, n):
    while graph.number_of_nodes() > n:
        nodes_by_degree = sorted(graph.nodes(), key=lambda x: graph.degree(x))
        if len(nodes_by_degree) < 2:
            break
        source = nodes_by_degree[0]
        target = nodes_by_degree[1] if nodes_by_degree[1] != source else nodes_by_degree[2]
        merge_nodes_simple(graph, source, target)

merge_until_n_nodes(BA_graph, n)
def remove_self_loops(graph):
    loops = list(nx.selfloop_edges(graph))
    graph.remove_edges_from(loops)

remove_self_loops(BA_graph)
labels = nx.get_node_attributes(BA_graph, 'label')
final_adj_matrix = nx.to_pandas_adjacency(BA_graph, weight='weight', dtype=float)
final_adj_matrix = final_adj_matrix.rename(index=labels, columns=labels)
row_sums = final_adj_matrix.sum(axis=1)
tpm_matrix_final = final_adj_matrix.div(row_sums, axis=0)
tpm_matrix_final[row_sums == 0] = 0

final_tpm_path = 'PCN_LowestDegree_matrix.csv'
tpm_matrix_final.to_csv(final_tpm_path, index=True)

print(BA_graph)

df_full = pd.read_csv(final_tpm_path, index_col=0)

def count_original_nodes(combined_node):
    return len(combined_node.split('_'))

row_counts = df_full.index.to_series().apply(count_original_nodes)

multiplier_matrix = np.tile(row_counts.values.reshape(-1, 1), (1, len(df_full.columns)))

initial_balance = 100
multiplied_data = df_full * multiplier_matrix * initial_balance

multiplied_data.to_csv(f'multiplied_{final_tpm_path}')

print(multiplied_data)
