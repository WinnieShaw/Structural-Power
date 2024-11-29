import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 18

f = 10
def load_tpm(file_path):
    df = pd.read_csv(file_path, header=0, index_col=0)  # 修改部分
    df.index = range(df.shape[0])
    df.columns = range(df.shape[1])
    return df.astype(float)


def create_graph_from_tpm(file_path):
    df = load_tpm(file_path)
    G = nx.from_pandas_adjacency(df, create_using=nx.DiGraph())
    for u, v, data in G.edges(data=True):
        data['capacity'] = data['weight']
    return G

def calculate_average_max_flow(file_path):
    BA_graph = create_graph_from_tpm(file_path)
    if not nx.is_strongly_connected(BA_graph):
        raise ValueError("Graph is not strongly connected")
    print(file_path)
    total_flow = 0
    num_pairs = 0

    flow_matrix = pd.DataFrame(0.0, index=range(200), columns=range(200), dtype=float)

    for u in range(200):
        print(f'caculating node {u} maximum flow to others')
        for v in range(200):
            if u == v:
                continue
            flow_value, flow_dict = nx.maximum_flow(BA_graph, u, v, capacity='capacity')
            flow_matrix.at[u, v] = flow_value
            total_flow += flow_value
            num_pairs += 1

    average_flow = total_flow / num_pairs

    print(f'{file_path}: total_flow={total_flow}, average flow={average_flow}')
    flow_matrix.to_csv(f'output\{f}\{file_path}_flow_matrix.csv')
    return average_flow

file_path_original = f'..\Algorithm codes\PCN_{f}_multiplied_matrix.csv'
file_path_greedy1 = f'..\Algorithm codes\output\PCN_{f}_MaxPTE_multiplied_matrix.csv'
file_path_random = f'..\Algorithm codes\output\PCN_{f}_Random_multiplied_matrix.csv'
file_path_maxout = f'..\Algorithm codes\output\PCN_{f}_MaxOut_multiplied_matrix.csv'
file_path_minout = f'..\Algorithm codes\output\PCN_{f}_MinOut_multiplied_matrix.csv'
file_path_maxbt = f'..\Algorithm codes\output\PCN_{f}_MaxBetweeness_multiplied_matrix.csv'
file_path_maxClustering = f'..\Algorithm codes\output\PCN_{f}_MaxClustering_multiplied_matrix.csv'

avg_max_flow_original = calculate_average_max_flow(file_path_original)
avg_max_flow_greedy1 = calculate_average_max_flow(file_path_greedy1)
avg_max_flow_random = calculate_average_max_flow(file_path_random)
avg_max_flow_maxout = calculate_average_max_flow(file_path_maxout)
avg_max_flow_minout = calculate_average_max_flow(file_path_minout)
avg_max_flow_maxbt = calculate_average_max_flow(file_path_maxbt)
avg_max_flow_maxClustering = calculate_average_max_flow(file_path_maxClustering)