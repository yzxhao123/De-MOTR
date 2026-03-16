import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import numpy as np

# 创建图形
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 12)
ax.axis('off')

# 定义颜色方案
color_input = '#E8F4FD'
color_tcn = '#BBE1FA'
color_dcn = '#A8D5E5'
color_primary = '#FFE5B4'
color_auxiliary = '#FFD4A3'
color_ffn = '#C8E6C9'
color_output = '#FFF9C4'
color_query = '#F8BBD0'
color_memory = '#E1BEE7'

# 绘制主要模块
def draw_module(ax, x, y, width, height, text, color, fontsize=10):
    rect = FancyBboxPatch((x, y), width, height,
                          boxstyle="round,pad=0.05",
                          facecolor=color,
                          edgecolor='#333333',
                          linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, text,
           ha='center', va='center', fontsize=fontsize, fontweight='bold')

# 绘制箭头
def draw_arrow(ax, x1, y1, x2, y2, style='->', lw=1.5, color='#333333', label=''):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle=style,
                           linewidth=lw,
                           color=color,
                           connectionstyle="arc3,rad=0")
    ax.add_patch(arrow)
    if label:
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        ax.text(mid_x + 0.2, mid_y, label, fontsize=9, style='italic')

# 1. 输入特征
draw_module(ax, 5.5, 0.5, 3, 0.8, 'Sparse Text Features', color_input, 10)

# 2. Multi-Scale Depthwise TCN
draw_module(ax, 5, 2, 4, 1.2, 'Multi-Scale Depthwise TCN\n(Temporal Modeling)', color_tcn, 10)
draw_arrow(ax, 7, 1.3, 7, 2)

# 3. DCNv4
draw_module(ax, 5, 4, 4, 1.2, 'DCNv4\n(Spatial Alignment)', color_dcn, 10)
draw_arrow(ax, 7, 3.2, 7, 4)

# 4. Primary Modulation Layer
draw_module(ax, 4.5, 6, 5, 1.5, 'Primary Modulation Layer\n(CrossAttention)', color_primary, 10)
draw_arrow(ax, 7, 5.2, 7, 6, label='F_aligned')

# Tracking Query输入
draw_module(ax, 0.5, 6.3, 2, 0.9, 'Tracking Query\n(Q_tracking)', color_query, 9)
draw_arrow(ax, 2.5, 6.75, 4.5, 6.75, style='->', color='#E91E63')

# 5. Auxiliary Modulation Layer
draw_module(ax, 4, 8.2, 6, 1.5, 'Auxiliary Modulation Layer\n(Adaptive Gating + Attention)', color_auxiliary, 10)
draw_arrow(ax, 7, 7.5, 7, 8.2, label='F_primary')

# Memory Bank输入
draw_module(ax, 0.5, 8.5, 2, 0.9, 'Memory Bank\n(M_t-1)', color_memory, 9)
draw_arrow(ax, 2.5, 8.95, 4, 8.95, style='->', color='#9C27B0')

# Gate机制标注
ax.text(11, 8.95, 'α = σ(W_gate·[F_primary; M_t-1])',
        fontsize=8, style='italic', bbox=dict(boxstyle="round,pad=0.3",
        facecolor='white', edgecolor='gray', alpha=0.8))

# 6. FFN + Residual
draw_module(ax, 5, 10.3, 4, 1, 'FFN + Residual', color_ffn, 10)
draw_arrow(ax, 7, 9.7, 7, 10.3, label='F_auxiliary')

# 残差连接
draw_arrow(ax, 10, 8.95, 10, 10.8, style='-', lw=1, color='#666666')
draw_arrow(ax, 10, 10.8, 9, 10.8, style='->', lw=1, color='#666666')
ax.text(10.5, 9.8, 'Residual', fontsize=8, rotation=90)

# 7. 输出
draw_module(ax, 5.5, 11.5, 3, 0.5, 'F_output', color_output, 10)
draw_arrow(ax, 7, 11.3, 7, 11.5)

# 添加公式标注
formulas = [
    (11.5, 6.75, 'F_primary = CrossAttention(F_aligned, Q_tracking)\n              + F_aligned', 9),
    (11.5, 4.5, 'Deformable convolution with\nlarge receptive field', 8),
    (11.5, 2.5, 'Multi-scale temporal\ndependency capture', 8)
]

for x, y, text, size in formulas:
    ax.text(x, y, text, fontsize=size,
           bbox=dict(boxstyle="round,pad=0.3",
           facecolor='#FFFEF7', edgecolor='gray', alpha=0.9))

# 添加标题
ax.text(7, 12.5, 'TextBlock Module with Hierarchical Modulation Strategy',
        fontsize=14, fontweight='bold', ha='center')

# 添加模块分组标注
# 时序建模部分
rect1 = Rectangle((4.5, 1.8), 5, 3.6,
                  fill=False, edgecolor='blue', linewidth=1.5, linestyle='--', alpha=0.5)
ax.add_patch(rect1)
ax.text(4.3, 3.5, 'Temporal-Spatial\nModeling', fontsize=8, color='blue', rotation=90, va='center')

# 调制部分
rect2 = Rectangle((3.5, 5.8), 7, 4,
                  fill=False, edgecolor='red', linewidth=1.5, linestyle='--', alpha=0.5)
ax.add_patch(rect2)
ax.text(3.3, 7.8, 'Hierarchical\nModulation', fontsize=8, color='red', rotation=90, va='center')

plt.title('TextBlock Architecture with Hierarchical Modulation', fontsize=16, pad=20)
plt.tight_layout()
plt.show()

