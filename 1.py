import matplotlib.pyplot as plt
import numpy as np

# 前三个柱子更紧，最后一个柱子隔开
x = np.array([0.80, 1.08, 1.36, 3.10])
heights = [4., 3.3, 3.1, 4.8]

# 全部使用同一种淡蓝色
bar_color = '#9ec9ff'

fig, ax = plt.subplots(figsize=(3.2, 2.0), dpi=200)

# 透明背景
fig.patch.set_alpha(0)
ax.set_facecolor('none')

# 柱状图
ax.bar(
    x,
    heights,
    width=0.22,
    color=bar_color,
    edgecolor=bar_color,
    linewidth=0.8,
    alpha=0.95
)

# 去掉默认边框和刻度
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

# 手动画坐标轴
# 横轴
ax.plot([0.35, 4.0], [0, 0], color='black', lw=1.0)

# 纵轴箭头
ax.annotate(
    '',
    xy=(0.35, 5.5), xytext=(0.35, 0),
    arrowprops=dict(arrowstyle='->', lw=1.0, color='black')
)

# 显示范围
ax.set_xlim(0, 4.1)
ax.set_ylim(0, 5.7)

# 保存透明背景图片
plt.savefig(
    'bar_chart_transparent.png',
    dpi=300,
    transparent=True,
    bbox_inches='tight',
    pad_inches=0.02
)

plt.show()