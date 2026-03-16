import torch
import torch.nn as nn
import torch.nn.functional as F

class DensityModule(nn.Module):
    """1×1 conv + dilated conv + ReLU"""
    def __init__(self, in_channels, dilation=2):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.dilated_conv = nn.Conv2d(1, 1, kernel_size=3, padding=dilation, dilation=dilation)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1x1(x)           # (B, 1, H, W)
        x = self.dilated_conv(x)      # (B, 1, H, W)
        x = self.relu(x)
        return x


class JoinerWithDensity(nn.Sequential):
    def __init__(self, backbone, position_embedding, num_topk=100, dilation=2):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

        # 为每一层特征准备一个密度模块
        self.density_modules = nn.ModuleList([
            DensityModule(c, dilation=dilation) for c in self.num_channels
        ])
        self.num_topk = num_topk

    def forward(self, tensor_list):
        # 1. Backbone 输出多尺度特征
        xs = self[0](tensor_list)  # dict {layer_name: NestedTensor}
        out_features = [xs[k] for k in sorted(xs.keys())]

        # 2. 位置编码
        pos_encodings = [self[1](feat).to(feat.tensors.dtype) for feat in out_features]

        # 3. 对每一层特征生成密度图并Top-K筛选
        sparse_feats = []
        sparse_pos = []
        for idx, feat in enumerate(out_features):
            feat_map = feat.tensors        # (B, C, H, W)
            B, C, H, W = feat_map.shape


            density_map = self.density_modules[idx](feat_map)  # (B, 1, H, W)


            density_flat = density_map.view(B, -1)             # (B, H*W)
            topk_vals, topk_idx = torch.topk(density_flat, self.num_topk, dim=1)


            ys = (topk_idx // W).float() / H   # 归一化 y
            xs_coord = (topk_idx % W).float() / W   # 归一化 x
            coords = torch.stack([xs_coord, ys], dim=-1)  # (B, K, 2)


            feat_flat = feat_map.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
            selected_feats = torch.gather(feat_flat, 1, topk_idx.unsqueeze(-1).expand(-1, -1, C))

            sparse_feats.append(selected_feats)  # list of (B, K, C)
            sparse_pos.append(coords)             # list of (B, K, 2)

        return sparse_feats, sparse_pos, pos_encodings
