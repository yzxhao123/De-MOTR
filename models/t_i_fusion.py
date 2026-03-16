import torch
from torch import nn
from util.misc import NestedTensor

import torch
import torch.nn as nn
import torch.nn.functional as F
from density_map import DensityModule

class SparseTextFusion(nn.Module):
    def __init__(self, in_channels=512, embed_dim=256, text_dim=768, num_topk=100):
        super().__init__()
        self.num_topk = num_topk
        self.density_module = DensityModule(in_channels)   # 用 density map 生成权重
        self.sparse_proj = nn.Linear(in_channels, embed_dim)
        self.text_proj = nn.Linear(text_dim, embed_dim)
        self.fused_to_original_proj = nn.Linear(embed_dim, in_channels)

    def forward(self, feat, text_emb):
        """
        feat: (B, C, H, W)  来自 backbone 的最后一层特征
        text_emb: (text_dim,) or (1, text_dim) 文本特征
        """
        B, C, H, W = feat.shape

        # --------- 1. 生成 density map ---------
        density_map = self.density_module(feat)          # (B, 1, H, W)
        density_map = density_map.view(B, -1)            # (B, HW)

        # --------- 2. Top-k 稀疏选择 ---------
        topk_vals, topk_idx = torch.topk(density_map, self.num_topk, dim=1)  # (B, topk)

        feat_flat = feat.view(B, C, -1).permute(0, 2, 1)                      # (B, HW, C)
        sparse_feat = torch.gather(feat_flat, 1, topk_idx.unsqueeze(-1).expand(-1, -1, C))  # (B, topk, C)

        # --------- 3. 融合文本特征 ---------
        text_feat = self.text_proj(text_emb.reshape(1, 1, -1)).to(sparse_feat.device)
        text_feat = text_feat.expand(B, self.num_topk, -1)                    # (B, topk, embed_dim)

        fused_sparse = self.sparse_proj(sparse_feat) + text_feat              # (B, topk, embed_dim)
        fused_sparse = self.fused_to_original_proj(fused_sparse)              # (B, topk, C)

        # --------- 4. 将稀疏特征写回原 feature map ---------
        tensor_flat = feat_flat.scatter(1, topk_idx.unsqueeze(-1).expand(-1, -1, C), fused_sparse)
        feat_new = tensor_flat.permute(0, 2, 1).view(B, C, H, W)              # (B, C, H, W)

        return feat_new
