import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import rc

# 强制使用 Times New Roman 字体并设置样式
rc('font', family='serif', serif='Times New Roman')
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 22

# 定义文件范围
file_list = [f"./bad-nodepairs{i}.csv" for i in range(1, 11)]

# 检查文件是否存在
existing_files = [file for file in file_list if os.path.exists(file)]
if not existing_files:
    raise FileNotFoundError("未找到任何符合条件的文件！请检查文件是否存在并命名正确。")

# 读取文件并计算均值
dfs = [pd.read_csv(file, index_col=0) for file in existing_files]
averages = pd.concat(dfs).groupby("Deficit thresholds").mean()

# 定义颜色
colors = {
    'Original': '#ea7827',
    'Random': 'skyblue',
    'MaxOut': '#1f70a9',
    'MinOut': '#c22f2f',
    'MaxBetweeness': '#af8fd0',
    'MaxClustering': '#f6bd21',
    'MaxPTE': '#449945'
}

# 创建图表
fig, ax = plt.subplots(figsize=(10, 6))
thresholds = averages.index

# 绘制每种方法的数据
for method in averages.columns:
    if method in colors:  # 确保颜色映射
        ax.plot(thresholds, averages[method], label=method, color=colors[method], marker='o')

# 添加图例、网格、标签
ax.legend(fontsize=21)
ax.grid(True, linestyle='-', linewidth=1.0, color='#999999', alpha=0.35)
ax.set_xticks(thresholds)
ax.set_xlabel('Ω', fontdict={'family': 'Times New Roman', 'style': 'italic', 'size': 26})
ax.set_ylabel("Proportion%", fontsize=26)

# 调整布局并保存图表
plt.tight_layout()
plt.savefig('badpairs.pdf', format='pdf', bbox_inches='tight', dpi=500)

# 保存均值结果到 CSV 文件
averages.to_csv('badpairs.csv')

