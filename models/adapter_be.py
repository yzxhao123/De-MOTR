import torch
import torch.nn as nn
import torch.nn.functional as F
from util.misc import NestedTensor
from .text_feature import DefaultTextEmb
from torch.nn.functional import normalize

class DensityModule(nn.Module):
    """生成密度图，用于稀疏采样"""
    def __init__(self, in_channels, dilation=2, reduction=4):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.dilated_conv = nn.Conv2d(1, 1, kernel_size=3, padding=dilation, dilation=dilation)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1x1(x)
        x1 = self.dilated_conv(x)
        x = self.relu(x1+x)


        return x

class SparseTextFusion(nn.Module):
    """安全版本：DensityModule 引导稀疏点 + 文本特征融合"""
    def __init__(self, in_channels=512, embed_dim=256, text_dim=768,
                 topk_ratio=0.5, fusion_scale=0.1, use_density=True):
        super().__init__()
        self.topk_ratio = topk_ratio      # 保留比例
        self.fusion_scale = fusion_scale  # 控制文本融合影响
        self.use_density = use_density

        # DensityModule
        self.density_module = DensityModule(in_channels) if use_density else None

        # 融合相关线性层
        self.sparse_proj = nn.Linear(in_channels, embed_dim)
        self.text_proj = nn.Linear(text_dim, embed_dim)
        self.fused_to_original_proj = nn.Linear(embed_dim, in_channels)
        self.gate_param = nn.Parameter(torch.zeros(1))

    def forward(self, tensor, text_emb):
        B, C, H, W = tensor.shape
        feat_flat = tensor.view(B, C, -1).permute(0, 2, 1)  # (B, HW, C)

        # ---------- 稀疏采样 ----------
        num_topk = max(1, int(H * W * self.topk_ratio))
        density_map = None  # 新增

        if self.use_density and self.density_module is not None:
            density_map = self.density_module(tensor)         # (B, 1, H, W)
            density_flat = density_map.view(B, -1)
            topk_vals, topk_idx = torch.topk(density_flat, num_topk, dim=1)
        else:
            # 原始均值排序
            scores = feat_flat.mean(-1)
            topk_vals, topk_idx = torch.topk(scores, num_topk, dim=1)

        self.last_density_map = density_map

        sparse_feat = torch.gather(feat_flat, 1, topk_idx.unsqueeze(-1).expand(-1, -1, C))

        # ---------- 文本融合 ----------
        text_feat = self.text_proj(text_emb.to(sparse_feat.device))
        if text_feat.dim() == 3 and text_feat.size(1) == 1:
            text_feat = text_feat.squeeze(1)
        text_feat = torch.relu(text_feat) * self.fusion_scale

        # ---------- 梯度安全融合 ----------
        fused_sparse = self.sparse_proj(sparse_feat)
        fused_sparse_norm = F.layer_norm(fused_sparse, fused_sparse.size()[-1:])
        text_feat_norm = F.layer_norm(text_feat, text_feat.size()[-1:]).unsqueeze(1).expand(-1, num_topk, -1)
        gate = torch.sigmoid(self.gate_param)  # nn.Parameter, init 0
        fused_sparse = fused_sparse_norm + gate * text_feat_norm

        # alpha = 0.5
        # fused_sparse = fused_sparse_norm + alpha * text_feat_norm
        fused_sparse = self.fused_to_original_proj(fused_sparse)

        # ---------- 安全残差写回 ----------
        fused_sparse = normalize(fused_sparse, dim=-1) * sparse_feat.norm(dim=-1, keepdim=True)
        tensor_flat = feat_flat.scatter(1, topk_idx.unsqueeze(-1).expand(-1, -1, C), fused_sparse)
        tensor = tensor_flat.permute(0, 2, 1).view(B, C, H, W)
        # 在 SparseTextFusion.forward 方法末尾，return 前添加
        return tensor, {
            'density_map': density_map,
            'sparse_indices': topk_idx,
            'original_features': feat_flat.view(B, H, W, C).permute(0, 3, 1, 2),
            'fused_features': fused_sparse
        }




class BackboneEncoderAdaptor(nn.Module):
    """安全版 BackboneEncoderAdaptor，只对 conv3 做 DensityModule 引导稀疏 + 文本融合"""
    def __init__(self, backbone, encoder, embed_dim=256, fusion_scale=0.1, topk_ratio=0.2):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.coord_proj = nn.Linear(2, embed_dim)

        # 只对 conv3 创建 SparseTextFusion
        self.sparse_fusion = SparseTextFusion(
            in_channels=backbone.num_channels[0],
            embed_dim=embed_dim,
            topk_ratio=topk_ratio,      # 动态 top_k 比例
            fusion_scale=fusion_scale,
            use_density=True             # 使用 DensityModule 指导稀疏采样
        )

    def forward(self, features, text_emb):
        out_features = []
        out_pos = []
        fused_features=None
        density_map_out = None
        fusion_info = None

        for i, feat in enumerate(features):
            tensor, mask = feat.decompose()
            B, C, H, W = tensor.shape

            # 仅对 conv3 做稀疏 + 文本融合
            if i == 0:
                tensor,fusion_info = self.sparse_fusion(tensor, text_emb)
                density_map_out = fusion_info['density_map']
                fused_features = NestedTensor(tensor, mask)


            out_features.append(NestedTensor(tensor, mask))
            out_pos.append(self.backbone[1](NestedTensor(tensor, mask)).to(tensor.dtype))

        return out_features, out_pos,fused_features,density_map_out,fusion_info


