import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os  # 确保导入 os 模块

# 设置绘图风格
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 22

# 定义文件范围为 1-10 的文件
file_list = [f"./maxflow_demands_results{i}.csv" for i in range(1, 11)]

# 确保找到文件
existing_files = [file for file in file_list if os.path.exists(file)]
if not existing_files:
    print("未找到任何符合条件的文件！请检查文件是否存在并命名正确。")
else:
    print(f"找到以下文件: {existing_files}")

# 读取文件并计算平均值
dfs = [pd.read_csv(file) for file in existing_files]
averages = pd.concat(dfs).groupby("Demand").mean()

# 定义颜色（只绘制颜色字典中定义的列）
colors = {
    'Original': '#ea7827',
    'Random': 'skyblue',
    'MaxOut': '#1f70a9',
    'MinOut': '#c22f2f',
    'MaxBetweeness': '#af8fd0',
    'MaxClustering': '#f6bd21',
    'MaxPTE': '#449945'
}

# 绘制结果
plt.figure(figsize=(8, 5))
demands = averages.index

for key in averages.columns:
    if key in colors:  # 仅绘制已定义颜色的列
        plt.plot(demands, averages[key], label=key, color=colors[key], marker='o', markersize=6, linewidth=2)
    else:
        print(f"未定义颜色的列已跳过绘制: {key}")

plt.xlim(0, 100)
plt.grid(True, linestyle='-', linewidth=1.0, color='#999999', alpha=0.35)
plt.legend(fontsize=16)
#plt.title("Average Values Across Algorithms")
plt.xlabel("Demand")
plt.ylabel("Balance deficit")
plt.savefig(fname='Balance deficit.pdf', format='pdf', bbox_inches='tight', dpi=500)

# 打印完成提示
print("绘图完成，文件已保存为 Balance_Deficit.pdf")
