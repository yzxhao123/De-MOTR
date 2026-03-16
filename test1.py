import torch
import torch.nn as nn
import torch.nn.functional as F

class MambaSS2DBlock(nn.Module):
    """
    轻量 SS2D 模块：
    - 输入：x (B, C, H, W)
    - 输出：x (B, C, H, W)
    - 原理：沿行列分别用 1D Mamba 处理，再融合
    """
    def __init__(self, dim, expand=2, dropout=0.0, use_local_conv=True):
        super().__init__()
        self.dim = dim
        self.expand = expand
        self.use_local_conv = use_local_conv

        hidden_dim = dim * expand

        # 行向 Mamba
        self.row_mamba = nn.Sequential(
            nn.Conv1d(dim, hidden_dim, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, dim, kernel_size=1, bias=False),
        )

        # 列向 Mamba
        self.col_mamba = nn.Sequential(
            nn.Conv1d(dim, hidden_dim, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, dim, kernel_size=1, bias=False),
        )

        if self.use_local_conv:
            self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        else:
            self.local_conv = nn.Identity()

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape

        # 行处理
        x_row = x.permute(0, 2, 1, 3).reshape(B*H, C, W)  # (B*H, C, W)
        x_row = self.row_mamba(x_row)
        x_row = x_row.reshape(B, H, C, W).permute(0, 2, 1, 3)

        # 列处理
        x_col = x.permute(0, 3, 1, 2).reshape(B*W, C, H)  # (B*W, C, H)
        x_col = self.col_mamba(x_col)
        x_col = x_col.reshape(B, W, C, H).permute(0, 2, 3, 1)

        # 融合 + 局部卷积
        x = x_row + x_col
        x = self.local_conv(x)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)

        return x

# 测试
if __name__ == "__main__":
    x = torch.randn(2, 64, 32, 32)  # B, C, H, W
    ss2d = MambaSS2DBlock(dim=64)
    y = ss2d(x)
    print(y.shape)  # (2, 64, 32, 32)
