import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 24
# 数据组织
data = {
    "Method": ["Original", "MaxPTE", "Random", "MaxOut", "MinOut", "MaxBetweenness", "MaxClustering"] * 10,
    "Average Maximum Flow": [
        68.60488249977226,77.5335114416974,63.75937924983774,64.16204147003869,62.71290300053632,60.79252752142344,60.53251124699007,
        69.94738105696719,78.80740070447271,64.74419580884549,64.73939214653447,64.47945784842766,60.52953986166326,60.60801813767186,
        70.07688051404777, 78.71089905762754, 63.354717146599114, 62.929178310723685, 62.929178310723685, 61.244683554859556, 60.98275373476431,
        70.0748922169261, 78.57349337253999, 64.34718275146237, 65.09475193111425, 64.60884339663075, 62.007009078531006, 60.98275373476431,
        71.45238096577647, 80.5089653259327, 65.98582300343307, 65.98335657587431, 66.23367837452022, 63.31604463363637, 64.00028484915485,
        71.32706614752867, 80.69222775589635, 67.03548125743583, 66.96755880256015, 65.94322432142295, 63.68103121724999, 63.923890574929324,
        72.03853799854055, 82.18838838861406, 66.48433300955121, 66.74289495996489, 67.0216966976714, 65.01412619020387, 62.84378935149288,
        71.86990391205232, 81.95702138992542, 66.13835709102455, 66.45475040258384, 64.96406171458764, 63.50511305606626, 61.94958459358605,
        68.35399010767365, 76.84191570876315, 62.9073432279229, 64.39565927530025, 62.98059033888027, 62.44732997684652, 61.00565461684646,
        70.01385774917875, 79.3138465937889, 64.59172055353746, 64.67532441364772, 63.30199295014494, 62.77869280263736, 61.30179406348338
    ]
}

# 创建 DataFrame
df = pd.DataFrame(data)

# 按方法分组数据
grouped_data = [df[df['Method'] == method]['Average Maximum Flow'] for method in df['Method'].unique()]

# 设置图形大小
fig,ax=plt.subplots(figsize=(10, 6))

# 马卡龙色列表
colors = ['#ea7827', '#449945', 'skyblue', '#1f70a9', '#c22f2f', '#af8fd0', '#f6bd21']

# 设置 alpha 值（透明度），范围在0到1之间，1为不透明，0为完全透明
alpha = 0.5

# 绘制横向箱线图
box = plt.boxplot(grouped_data, patch_artist=True, labels=df['Method'].unique(), vert=False)

# 为每个箱子设置颜色，并应用透明度
for patch, color in zip(box['boxes'], colors):
    rgba_color = to_rgba(color, alpha)  # 添加透明度
    patch.set_facecolor(rgba_color)

# 设置字体为Times New Roman
plt.xticks(fontname='Times New Roman')
plt.yticks(fontname='Times New Roman')

# 添加标题和标签
#plt.title('Box Plot of Average Maximum Flow by Method', fontname='Times New Roman')
plt.xlabel('Average Maximum Flow', fontname='Times New Roman')
#plt.ylabel('Method', fontname='Times New Roman')

# 调整图表布局
plt.tight_layout()

# 显示图表
plt.savefig('box.png',dpi=300)
plt.show()

from matplotlib.backends.backend_pdf import PdfPages
with PdfPages("box.pdf") as pdf:
    pdf.savefig(fig)  # 显式传递 fig 对象
    plt.close(fig)  # 关闭图形
