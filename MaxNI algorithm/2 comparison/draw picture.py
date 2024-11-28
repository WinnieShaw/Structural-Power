import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.size'] = 20
plt.rcParams['font.family'] = 'Times New Roman'

df = pd.read_csv('2compare.csv')

df['Method'] = df['Method'].str.replace('Network', 'network', regex=False)

print("Modified Methods:", df['Method'].unique())

mean_df = df.groupby(['Alpha', 'Method']).mean().reset_index()

palette = {'Original network': '#c22f2f', 'MaxNI network': '#449945'}
markers = {'Original network': 's', 'MaxNI network': '*'}

plt.figure(figsize=(7, 4))
ax = sns.lineplot(data=df, x='Alpha', y='Exceedances', hue='Method', style='Method',
                  dashes=False, errorbar='sd', estimator='mean', palette=palette)

for method, marker in markers.items():
    subset = mean_df[mean_df['Method'] == method]
    plt.plot(subset['Alpha'], subset['Exceedances'], linestyle='', marker=marker,
             ms=10 if marker == '*' else 6, color=palette[method])

handles, labels = ax.get_legend_handles_labels()
new_labels, new_handles = [], []
for handle, label in zip(handles, labels):
    if label not in new_labels:
        new_labels.append(label)
        new_handles.append(handle)
ax.legend(handles=new_handles, labels=new_labels, fontsize=20)

plt.grid(True)
ax.set_xlabel('α', fontdict={'family': 'Times New Roman', 'style': 'italic', 'size': 24})
plt.ylabel('Balance deficit', fontsize=24)

plt.savefig('2compare.png', format='png', bbox_inches='tight', dpi=500)
plt.savefig('2compare.pdf', format='pdf', bbox_inches='tight', dpi=500)

# 显示图表
plt.show(dpi=300)
